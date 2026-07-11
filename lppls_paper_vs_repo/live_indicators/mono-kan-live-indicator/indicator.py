"""MONO-KAN Live LPPLS Indicator - command-line live indicator.

Usage:
    python indicator.py --csv PRICES.csv [--series price|equity] [--out out.json]

CSV: either plain columns (Date, Close[, Volume]) or a yfinance export with
the 3-row header. --series equity analyses ln(price x volume).
Output: JSON with the current state and, for HLPPL-family models, the
paper's threshold trading action.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lppls_engine.core import minmax_scale, solve_linear, design_matrix

MODEL = 'mono-kan-live-indicator'
FAMILY = 'mono'
ENGINE = 'kan_raw'
W_ROLL = 126
LIVE_WINDOWS = list(range(60, 241, 20))
N = 252


def load_csv(path, series="price"):
    head = open(path).readline()
    if head.startswith("Price,"):
        df = pd.read_csv(path, skiprows=[1, 2])
        df = df.rename(columns={df.columns[0]: "Date"})
    else:
        df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if series == "equity":
        if "Volume" not in df.columns:
            raise SystemExit("equity mode needs a Volume column")
        df["Close"] = df["Close"] * df["Volume"]
    df["logp"] = np.log(df["Close"].astype(float))
    return df


def get_dae():
    from lppls_engine import dae, synthetic
    wpath = Path(__file__).resolve().parent / "dae_weights.npz"
    if not wpath.exists():
        print("training DAE cleaner on synthetic LPPLS shapes (~20 s, once)...", file=sys.stderr)
        def pairs(n, seed):
            rng = np.random.default_rng(seed)
            tc, m, w, c1, c2 = synthetic.sample_params(rng, n)
            clean = np.stack([synthetic.clean_series(*p) for p in zip(tc, m, w, c1, c2)]).astype(np.float32)
            noisy = clean.copy()
            half = n // 2
            noisy[:half] += synthetic.white_noise(rng, noisy[:half].shape).astype(np.float32)
            noisy[half:] += synthetic.ar1_noise(rng, noisy[half:].shape).astype(np.float32)
            return noisy, clean
        Xn, Xc = pairs(60000, 11)
        Vn, Vc = pairs(8000, 12)
        params, _ = dae.train(Xn, Xc, Vn, Vc, log=lambda s: None)
        dae.save_params(params, wpath)
    return dae, dae.load_params(wpath)


def clean_scaled(x_scaled):
    dae, params = get_dae()
    mirrored = x_scaled[-1] < x_scaled[0]
    x_in = 1.0 - x_scaled if mirrored else x_scaled
    out = dae.clean(params, x_in).astype(np.float64)
    return ((1.0 - out) if mirrored else out), mirrored


def window_endpoint(win, seed):
    """Fitted trajectory endpoint (log units) for the residual, per ENGINE."""
    from lppls_engine import hlppl
    if ENGINE in ("trf", "trf_dae"):
        w_in = win
        if ENGINE == "trf_dae":
            x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(win)), win)
            xs, (mn, rng_) = minmax_scale(x)
            xc, _ = clean_scaled(xs)
            w_in = np.interp(np.linspace(0, 1, len(win)), np.linspace(0, 1, N), xc) * rng_ + mn
        theta, _ = hlppl.fit_lppl_trf(w_in, seed=seed)
        if theta is None:
            return np.nan
        return hlppl.lppl7(theta, np.array([float(len(win))]))[0]
    from lppls_engine import kan
    x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(win)), win)
    xs, (mn, rng_) = minmax_scale(x)
    mirrored = False
    if ENGINE == "kan_dae":
        xs, mirrored = clean_scaled(xs)
        if mirrored:
            xs = 1.0 - xs
    r = kan.fit_mlnn_kan(xs, seed=seed)
    t_norm = np.linspace(0.0, 1.0, N)
    beta = solve_linear(t_norm, xs, r["tc"], r["m"], r["w"])
    yend = float((design_matrix(t_norm[-1:], r["tc"], r["m"], r["w"]) @ beta)[0])
    if mirrored:
        yend = 1.0 - yend
    return yend * rng_ + mn


def run_hlppl(df):
    from lppls_engine import hlppl
    logp = df["logp"].values
    hype = hlppl.hype_proxy(df.get("Volume", pd.Series(np.ones(len(df)))).values)
    n_scored = min(len(df) - W_ROLL + 1, 160)
    grid = list(range(len(df) - n_scored, len(df)))
    eps = [logp[t] - window_endpoint(logp[t - W_ROLL + 1 : t + 1], seed=t) for t in grid]
    eps_norm = hlppl.causal_normalize(np.nan_to_num(np.array(eps), nan=0.0))
    score = hlppl.bubble_score(eps_norm, hype[np.array(grid)])
    episodes = hlppl.label_episodes(score, [str(df.Date.iloc[i].date()) for i in grid])
    # naive persistence forecast when history is short; ridge otherwise
    if len(score) >= 60:
        from numpy.linalg import lstsq
        LAGS = 10
        Xf = [list(score[i - LAGS + 1 : i + 1]) for i in range(LAGS, len(score) - 5)]
        Yf = [score[i + 1 : i + 6] for i in range(LAGS, len(score) - 5)]
        X, Y = np.array(Xf), np.array(Yf)
        A = np.vstack([X, np.sqrt(3.0) * np.eye(X.shape[1])])
        fc = []
        for h in range(5):
            wgt = lstsq(A, np.concatenate([Y[:, h], np.zeros(X.shape[1])]), rcond=None)[0]
            fc.append(float(np.array(score[-LAGS:]) @ wgt))
    else:
        fc = [float(score[-1])] * 5
    action, reason = hlppl.trading_decision(np.array(fc), position=0)
    return dict(model=MODEL, asof=str(df.Date.iloc[-1]), close=float(df.Close.iloc[-1]),
                score=float(score[-1]), forecasts_1_5=[round(float(v), 3) for v in fc],
                episodes=episodes[-3:], action=action, reason=reason)


def run_mono(df):
    from lppls_engine import mlnn
    logp = df["logp"].values
    rows = []
    t_norm = np.linspace(0.0, 1.0, N)
    for L in LIVE_WINDOWS:
        if L >= len(df):
            continue
        win = logp[len(df) - L : len(df)]
        x = np.interp(t_norm, np.linspace(0, 1, L), win)
        xs, (mn, rng_) = minmax_scale(x)
        mirrored = False
        if ENGINE == "dae":
            xs, mirrored = clean_scaled(xs)
        if ENGINE == "kan_raw":
            from lppls_engine import kan
            r = kan.fit_mlnn_kan(xs, seed=L)
        else:
            r = mlnn.fit_mlnn(xs, seed=L)
        beta = solve_linear(t_norm, xs, r["tc"], r["m"], r["w"])
        b, a_sc = float(beta[1]), float(beta[0])
        if mirrored:
            b, a_sc = -b, 1.0 - a_sc
        rows.append(dict(L=L, sign="positive" if b < 0 else "negative",
                         tc_bars=(r["tc"] - 1.0) * (L - 1),
                         level_c=float(np.exp(a_sc * rng_ + mn))))
    tcs = np.array([r["tc_bars"] for r in rows])
    lvl = np.array([r["level_c"] for r in rows])
    neg = np.mean([r["sign"] == "negative" for r in rows])
    return dict(model=MODEL, asof=str(df.Date.iloc[-1]), close=float(df.Close.iloc[-1]),
                n_fits=len(rows), negative_share=round(float(neg), 2),
                tc_bars_after_now=dict(zip(["q25", "median", "q75"],
                                           np.round(np.percentile(tcs, [25, 50, 75]), 1).tolist())),
                critical_level=dict(zip(["q25", "median", "q75"],
                                        np.round(np.percentile(lvl, [25, 50, 75]), 3).tolist())),
                fits=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--series", choices=["price", "equity"], default="price")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load_csv(args.csv, args.series)
    res = run_hlppl(df) if FAMILY == "hlppl" else run_mono(df)
    res["series"] = args.series
    js = json.dumps(res, indent=2, default=str)
    print(js)
    if args.out:
        Path(args.out).write_text(js)


if __name__ == "__main__":
    main()
