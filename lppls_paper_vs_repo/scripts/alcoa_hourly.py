"""ALCOA hourly live analysis — full HLPPL engine matrix, last-month focus.

Engines (rolling window W = 126 hourly bars ~ 3.6 weeks):
  HLPPL          paper TRF calibration on raw log-price
  HLPPL-DAE      TRF on DAE-cleaned windows (mirror trick for declines)
  HLPPL-KAN      M-LNN-KAN trajectory on raw windows
  HLPPL-DAE-KAN  M-LNN-KAN on DAE-cleaned windows
plus MONO / MONO-KAN / MONO-DAE ensembles at the live bar.

Careful-live-system details:
  * hourly volume has strong intraday seasonality -> the hype proxy divides
    each bar's volume by the trailing 20-day mean OF THE SAME CLOCK HOUR;
  * rolling scores are computed from 2026-03-02 (pre-peak run-up onward) so
    the causal normalisation sees the crash extreme; fine stride (1-2 bars)
    inside the last month, coarse stride before it (disclosed);
  * ML decision layer uses the paper's day-based horizons mapped to bars
    (h = 7, 14, 21, 28, 35 bars = 1..5 trading days).

Outputs: results/fig_alcoa_hourly_{scores,densities,decision}.png and
alcoa_hourly_{scores,ensemble,forecast}.csv
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import dae, hlppl, kan, mlnn
from deep_lppls.core import minmax_scale, solve_linear

W_ROLL = 126
BARS_PER_DAY = 7
LAST_MONTH_BARS = 22 * BARS_PER_DAY
ROLL_START_DATE = "2026-03-02"
LIVE_WINDOWS = [60, 90, 126, 168, 210, 252, 294, 336, 378, 420]
N = 252
HORIZON_BARS = [7, 14, 21, 28, 35]  # 1..5 trading days

ENGINES = ["HLPPL", "HLPPL-DAE", "HLPPL-KAN", "HLPPL-DAE-KAN"]
COLORS = {"HLPPL": "tab:blue", "HLPPL-DAE": "tab:green", "HLPPL-KAN": "tab:orange",
          "HLPPL-DAE-KAN": "tab:red", "MONO": "tab:purple", "MONO-KAN": "tab:cyan",
          "MONO-DAE": "tab:brown"}


def load_hourly():
    df = pd.read_csv(ROOT / "data" / "AA_hourly.csv", skiprows=[1, 2])
    df = df.rename(columns={"Price": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(None)
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    df["hour"] = df["Datetime"].dt.hour
    return df


def hourly_hype(df, days=20):
    """Abnormal volume vs trailing mean of the SAME clock hour (kills the
    intraday U-shape), squashed to (0, 1)."""
    v = df["Volume"].values.astype(float)
    hours = df["hour"].values
    out = np.full(len(df), 0.5)
    for i in range(len(df)):
        mask = (hours[:i] == hours[i])
        hist = v[:i][mask][-days:]
        if len(hist) >= 5:
            rel = v[i] / max(hist.mean(), 1e-9)
            out[i] = rel / (1.0 + rel)
    return out


def clean_window_logspace(win):
    """DAE-clean a log-price window (mirroring declines), back in log units."""
    x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(win)), win)
    x_scaled, (mn, rng_) = minmax_scale(x)
    mirrored = x_scaled[-1] < x_scaled[0]
    x_in = 1.0 - x_scaled if mirrored else x_scaled
    out = dae.clean(hlppl._dae_params(), x_in).astype(np.float64)
    if mirrored:
        out = 1.0 - out
    return np.interp(np.linspace(0, 1, len(win)), np.linspace(0, 1, N), out) * rng_ + mn


def kan_endpoint(win, seed):
    """KAN trajectory endpoint value (log units) for the residual."""
    x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(win)), win)
    x_scaled, (mn, rng_) = minmax_scale(x)
    mirrored = x_scaled[-1] < x_scaled[0]
    x_in = 1.0 - x_scaled if mirrored else x_scaled
    r = kan.fit_mlnn_kan(x_in, seed=seed)
    t_norm = np.linspace(0.0, 1.0, N)
    beta = solve_linear(t_norm, x_in, r["tc"], r["m"], r["w"])
    from deep_lppls.core import design_matrix
    yend = float((design_matrix(t_norm[-1:], r["tc"], r["m"], r["w"]) @ beta)[0])
    if mirrored:
        yend = 1.0 - yend
    return yend * rng_ + mn


def rolling_scores(df, hype):
    """Four engines on a variable-stride grid: fine inside the last month."""
    logp = df["logp"].values
    start_idx = int(df.index[df["Datetime"] >= ROLL_START_DATE][0])
    start_idx = max(start_idx, W_ROLL - 1)
    fine_from = len(df) - LAST_MONTH_BARS
    grids = {
        "HLPPL": list(range(start_idx, fine_from, 3)) + list(range(fine_from, len(df))),
        "HLPPL-DAE": list(range(start_idx, fine_from, 3)) + list(range(fine_from, len(df))),
        "HLPPL-KAN": list(range(start_idx, fine_from, 6)) + list(range(fine_from, len(df), 2)),
        "HLPPL-DAE-KAN": list(range(start_idx, fine_from, 6)) + list(range(fine_from, len(df), 2)),
    }
    series = {}
    for eng, grid in grids.items():
        t0 = time.time()
        eps = []
        for k, t2 in enumerate(grid):
            win = logp[t2 - W_ROLL + 1 : t2 + 1]
            if eng == "HLPPL":
                theta, _ = hlppl.fit_lppl_trf(win, seed=t2)
                yend = hlppl.lppl7(theta, np.array([float(W_ROLL)]))[0] if theta is not None else np.nan
            elif eng == "HLPPL-DAE":
                wc = clean_window_logspace(win)
                theta, _ = hlppl.fit_lppl_trf(wc, seed=t2)
                yend = hlppl.lppl7(theta, np.array([float(W_ROLL)]))[0] if theta is not None else np.nan
            elif eng == "HLPPL-KAN":
                x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, W_ROLL), win)
                x_scaled, (mn, rng_) = minmax_scale(x)
                r = kan.fit_mlnn_kan(x_scaled, seed=t2)
                t_norm = np.linspace(0.0, 1.0, N)
                beta = solve_linear(t_norm, x_scaled, r["tc"], r["m"], r["w"])
                from deep_lppls.core import design_matrix
                yend = float((design_matrix(t_norm[-1:], r["tc"], r["m"], r["w"]) @ beta)[0]) * rng_ + mn
            else:  # HLPPL-DAE-KAN
                yend = kan_endpoint(win, seed=t2)
            eps.append(win[-1] - yend if np.isfinite(yend) else np.nan)
            if (k + 1) % 60 == 0:
                print(f"  [{eng}] {k + 1}/{len(grid)} ({time.time() - t0:.0f}s)", flush=True)
        eps = np.array(eps)
        eps_norm = hlppl.causal_normalize(np.nan_to_num(eps, nan=0.0))
        score = hlppl.bubble_score(eps_norm, hype[np.array(grid)])
        series[eng] = pd.DataFrame(dict(idx=grid, dt=df["Datetime"].values[np.array(grid)],
                                        eps=eps, eps_norm=eps_norm, score=score))
        print(f"  {eng} done ({time.time() - t0:.0f}s, {len(grid)} fits)", flush=True)
    return series


def live_ensembles(df):
    logp = df["logp"].values
    t2_idx = len(df) - 1
    rows = []
    t_norm = np.linspace(0.0, 1.0, N)
    bar_dt = pd.Timedelta(days=1) / BARS_PER_DAY * (7 / 5)  # calendar time per bar

    def rec(method, tc, m, w, x_fit, scale, mirrored, L):
        mn, rng_ = scale
        beta = solve_linear(t_norm, x_fit, tc, m, w)
        b, a_sc = float(beta[1]), float(beta[0])
        if mirrored:
            b, a_sc = -b, 1.0 - a_sc
        tc_bars = (tc - 1.0) * (L - 1)
        rows.append(dict(method=method, L=L, sign="positive" if b < 0 else "negative",
                         tc_bars_after_now=tc_bars, tc_days=tc_bars / BARS_PER_DAY,
                         tc_dt=df["Datetime"].iloc[-1] + bar_dt * tc_bars,
                         price_c=float(np.exp(a_sc * rng_ + mn)), m=m, w=w))

    for L in LIVE_WINDOWS:
        win = logp[t2_idx - L + 1 : t2_idx + 1]
        x = np.interp(t_norm, np.linspace(0, 1, L), win)
        x_scaled, scale = minmax_scale(x)
        mirrored = x_scaled[-1] < x_scaled[0]
        x_mir = 1.0 - x_scaled if mirrored else x_scaled
        x_clean_mir = dae.clean(hlppl._dae_params(), x_mir).astype(np.float64)

        r = mlnn.fit_mlnn(x_scaled, seed=L); rec("MONO", r["tc"], r["m"], r["w"], x_scaled, scale, False, L)
        r = kan.fit_mlnn_kan(x_scaled, seed=L); rec("MONO-KAN", r["tc"], r["m"], r["w"], x_scaled, scale, False, L)
        r = mlnn.fit_mlnn(x_clean_mir, seed=L); rec("MONO-DAE", r["tc"], r["m"], r["w"], x_clean_mir, scale, mirrored, L)
        r = kan.fit_mlnn_kan(x_scaled, seed=L + 1); rec("HLPPL-KAN", r["tc"], r["m"], r["w"], x_scaled, scale, False, L)
        r = kan.fit_mlnn_kan(x_clean_mir, seed=L); rec("HLPPL-DAE-KAN", r["tc"], r["m"], r["w"], x_clean_mir, scale, mirrored, L)
        for name, w_in in [("HLPPL", win), ("HLPPL-DAE", clean_window_logspace(win))]:
            theta, _ = hlppl.fit_lppl_trf(w_in, seed=L)
            if theta is not None:
                A, B, C, m7, w7, phi, tc7 = theta
                tc_bars = tc7 - L
                rows.append(dict(method=name, L=L, sign="positive" if B < 0 else "negative",
                                 tc_bars_after_now=tc_bars, tc_days=tc_bars / BARS_PER_DAY,
                                 tc_dt=df["Datetime"].iloc[-1] + bar_dt * tc_bars,
                                 price_c=float(np.exp(A)), m=m7, w=w7))
        print(f"  live L={L} done", flush=True)
    return pd.DataFrame(rows)


def ml_forecast(sc, df):
    """Ridge forecasts of the HLPPL score at 1..5 trading-day horizons
    (7..35 bars), walk-forward. Returns forecasts + val correlations."""
    from numpy.linalg import lstsq

    full = sc.set_index("idx")["score"]
    grid = np.arange(full.index.min(), full.index.max() + 1)
    s = np.interp(grid, full.index.values, full.values)  # interp coarse->hourly
    LAGS = 14
    feats, targs = [], []
    for i in range(LAGS, len(s) - HORIZON_BARS[-1]):
        feats.append(list(s[i - LAGS + 1 : i + 1]))
        targs.append([s[i + h] for h in HORIZON_BARS])
    X, Y = np.array(feats), np.array(targs)
    n_train = int(len(X) * 0.8)
    Xm, Xs = X[:n_train].mean(0), X[:n_train].std(0) + 1e-9
    Xn = (X - Xm) / Xs
    A = np.vstack([Xn[:n_train], np.sqrt(3.0) * np.eye(X.shape[1])])
    fc, val_corr = [], []
    for h in range(len(HORIZON_BARS)):
        b = np.concatenate([Y[:n_train, h], np.zeros(X.shape[1])])
        w = lstsq(A, b, rcond=None)[0]
        pv = Xn[n_train:] @ w
        val_corr.append(float(np.corrcoef(pv, Y[n_train:, h])[0, 1]))
        f_live = (s[-LAGS:] - Xm) / Xs
        fc.append(float(f_live @ w))
    return np.array(fc), val_corr


def main():
    df = load_hourly()
    print(f"AA hourly {df.Datetime.iloc[0]} -> {df.Datetime.iloc[-1]} ({len(df)} bars), "
          f"last {df.Close.iloc[-1]:.2f}", flush=True)
    hype = hourly_hype(df)

    series = rolling_scores(df, hype)
    pd.concat([s.assign(engine=k) for k, s in series.items()]).to_csv(
        ROOT / "results" / "alcoa_hourly_scores.csv", index=False)

    for eng, sc in series.items():
        month = sc[sc["idx"] >= len(df) - LAST_MONTH_BARS]
        lo, hi = month["score"].min(), month["score"].max()
        print(f"  {eng}: last-month score range [{lo:+.2f}, {hi:+.2f}], now {sc.score.iloc[-1]:+.2f}, "
              f"bars<=-0.7: {(month.score <= -0.7).sum()}", flush=True)

    print("live ensembles...", flush=True)
    ens = live_ensembles(df)
    ens.to_csv(ROOT / "results" / "alcoa_hourly_ensemble.csv", index=False)
    print(ens.groupby("method").agg(n=("sign", "size"),
                                    neg_share=("sign", lambda s: float((s == "negative").mean())),
                                    med_tc_days=("tc_days", "median"),
                                    med_price_c=("price_c", "median")).round(2).to_string(), flush=True)

    fc, val_corr = ml_forecast(series["HLPPL"], df)
    action, reason = hlppl.trading_decision(fc, position=0)
    pd.DataFrame(dict(h_days=[1, 2, 3, 4, 5], forecast=fc)).to_csv(
        ROOT / "results" / "alcoa_hourly_forecast.csv", index=False)
    print("day-horizon forecasts: " + " ".join(f"{v:+.2f}" for v in fc), flush=True)
    print("val corr: " + " ".join(f"{v:.2f}" for v in val_corr), flush=True)
    print(f"DECISION: {action} — {reason}", flush=True)

    plot(df, series, ens, fc, action, reason)
    print("figures saved", flush=True)


def plot(df, series, ens, fc, action, reason):
    # --- scores figure ---
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.6], "hspace": 0.05})
    seg = df[df["Datetime"] >= ROLL_START_DATE]
    ax.plot(seg["Datetime"], seg["Close"], color="black", lw=0.8)
    ax.set_ylabel("AA close (hourly)")
    ax.grid(alpha=0.25)
    month_start = df["Datetime"].iloc[len(df) - LAST_MONTH_BARS]
    ax.axvspan(month_start, df["Datetime"].iloc[-1], color="gold", alpha=0.10)
    ax.set_title("ALCOA hourly: price and HLPPL engine scores (last month highlighted)")
    for eng, sc in series.items():
        ax2.plot(sc["dt"], sc["score"], color=COLORS[eng], lw=1.0, label=eng, alpha=0.9)
    for y, ls in [(0.8, "-"), (-0.8, "-"), (0.7, "--"), (-0.7, "--"), (0.3, ":"), (-0.3, ":")]:
        ax2.axhline(y, color="grey", ls=ls, lw=0.7)
    ax2.axvspan(month_start, df["Datetime"].iloc[-1], color="gold", alpha=0.10)
    ax2.set_ylim(-1.35, 1.35)
    ax2.set_ylabel("Bubble Score")
    ax2.legend(loc="lower left", fontsize=8, ncols=4)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.savefig(ROOT / "results" / "fig_alcoa_hourly_scores.png", dpi=140, bbox_inches="tight")

    # --- densities figure ---
    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 3.4], hspace=0.04, wspace=0.03)
    ax_t = fig.add_subplot(gs[0, 0]); axm = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_p = fig.add_subplot(gs[1, 1], sharey=axm)
    seg = df.iloc[-40 * BARS_PER_DAY:]
    axm.plot(seg["Datetime"], seg["Close"], color="black", lw=0.9)
    last_dt = df["Datetime"].iloc[-1]
    axm.axvline(last_dt, color="red", ls="-.", lw=1.4)
    x_hi = last_dt + pd.Timedelta(days=45)
    axm.axvspan(last_dt, x_hi, color="grey", alpha=0.08)
    p_lo, p_hi = seg["Close"].min() * 0.7, seg["Close"].max() * 1.15
    axm.set_xlim(seg["Datetime"].iloc[0], x_hi)
    axm.set_ylim(p_lo, p_hi)
    t_grid = pd.date_range(last_dt - pd.Timedelta(days=10), x_hi, freq="6h")
    t_num = mdates.date2num(t_grid)
    p_grid = np.linspace(p_lo, p_hi, 300)
    conf = []
    for method, g in ens.groupby("method"):
        conf.append(f"{method} {int(100 * (g['sign'] == 'negative').mean())}%")
        tcs = mdates.date2num(pd.to_datetime(g["tc_dt"]))
        tcs = tcs[(tcs > t_num[0]) & (tcs < t_num[-1])]
        pcs = g["price_c"].values
        pcs = pcs[(pcs > p_lo) & (pcs < p_hi)]
        if len(np.unique(tcs)) >= 3:
            pdf = gaussian_kde(tcs)(t_num)
            ax_t.plot(t_grid, pdf, color=COLORS[method], lw=1.5)
            ax_t.fill_between(t_grid, 0, pdf, color=COLORS[method], alpha=0.2)
        if len(np.unique(pcs)) >= 3:
            pdfp = gaussian_kde(pcs)(p_grid)
            ax_p.plot(pdfp, p_grid, color=COLORS[method], lw=1.5)
            ax_p.fill_betweenx(p_grid, 0, pdfp, color=COLORS[method], alpha=0.2)
    ax_t.axvline(last_dt, color="red", ls="-.", lw=1.2)
    ax_t.set_yticks([]); ax_t.set_ylabel("PDF($t_c$)")
    ax_t.set_title(f"ALCOA hourly live indicator @ {last_dt} — negative-fit share: " + " | ".join(conf), fontsize=8.5)
    ax_p.set_xticks([]); ax_p.set_xlabel("PDF(price$_c$)")
    plt.setp(ax_p.get_yticklabels(), visible=False); plt.setp(ax_t.get_xticklabels(), visible=False)
    axm.set_ylabel("AA close")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in COLORS.values()]
    axm.legend(handles, COLORS.keys(), loc="upper right", fontsize=8)
    axm.grid(alpha=0.25)
    axm.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.savefig(ROOT / "results" / "fig_alcoa_hourly_densities.png", dpi=140, bbox_inches="tight")

    # --- decision figure ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = series["HLPPL"]
    month = sc[sc["idx"] >= len(df) - LAST_MONTH_BARS]
    ax.plot((month["idx"] - (len(df) - 1)) / BARS_PER_DAY, month["score"], color="tab:blue", lw=1.2,
            label="HLPPL score (hourly)")
    sck = series["HLPPL-DAE-KAN"]
    monthk = sck[sck["idx"] >= len(df) - LAST_MONTH_BARS]
    ax.plot((monthk["idx"] - (len(df) - 1)) / BARS_PER_DAY, monthk["score"], color="tab:red", lw=1.2,
            alpha=0.8, label="HLPPL-DAE-KAN score")
    ax.plot([1, 2, 3, 4, 5], fc, "o--", color="tab:purple", ms=8, lw=1.6, label="ML forecast (1..5 days)")
    for y, lbl in [(0.7, "short entry"), (-0.7, "long entry"), (0.3, "short exit"), (-0.3, "long exit")]:
        ax.axhline(y, color="grey", ls="--", lw=0.8)
        ax.text(-21, y + 0.02, lbl, fontsize=7, color="grey")
    ax.axvline(0, color="red", ls="-.", lw=1.2)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("trading days relative to now")
    ax.set_ylabel("Bubble Score")
    ax.set_title(f"ALCOA hourly — ML decision: {action}\n({reason})", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(ROOT / "results" / "fig_alcoa_hourly_decision.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
