"""Rolling LPPLS bubble scan of SPY 1998-2010: detect every bubble episode
and report the predicted critical time t_c AND critical price p_c = exp(A)
(the LPPLS log-price at t_c, since O(t_c) = A in Eq. 1 of the paper).

Scan design (LPPLS confidence-indicator style, cf. the Boulder repo and
Sornette et al.):
  * t2 slides over 1998-2010 in steps of 5 trading days;
  * at each t2 the model is fitted on windows of {60, 90, 126, 189, 252, 378}
    trading days ending at t2 (Boulder package fit, 25 NM searches - the
    fastest calibration, 5k+ fits);
  * each fit is qualified with the repo's default filter conditions
    (0 < m < 1, 2 < w < 15, O > 2.5, D > 0.5, tc inside the repo's window-
    relative range); b < 0 -> positive bubble, b > 0 -> negative bubble;
  * pos/neg confidence at t2 = share of qualified fits among that sign;
  * consecutive scan points with confidence >= CONF_THR are merged into
    episodes (gaps <= MERGE_GAP scan points allowed).

For each detected episode we report the median predicted t_c and p_c over the
qualified fits, and the realised outcome (peak/trough that followed). On top,
the paper's per-series methods (LM, M-LNN, M-LNN-KAN) and P-LNN-100K are run
once per episode - on the 252-day window ending at the episode's maximum-
confidence date - each contributing its own t_c and p_c prediction.

Outputs:
  results/fig_spy_bubble_scan.png
  results/spy_scan_confidence.csv    per-t2 confidence + median tc/price_c
  results/spy_bubble_episodes.csv    episode table (detector ensemble)
  results/spy_episode_methods.csv    per-episode predictions by each method
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import kan, lm, mlnn, plnn
from deep_lppls.core import minmax_scale, solve_linear
from lppls.lppls import LPPLS

T2_STEP = 5
WINDOWS = [40, 60, 80, 100, 126, 152, 189, 226, 252, 300, 350]
CONF_THR = 0.30
MIN_RUN = 2         # scan points above threshold to open an episode
MERGE_GAP = 6       # scan points (= 30 trading days) allowed inside an episode
N = 252             # resampled length for the paper-method fits

DATASETS = {
    "spy": dict(csv="SPY_daily_1998_2021.csv", end="2010-12-31", label="SPY 1998-2010"),
    # validation anchor: the Boulder repo's own bundled Nasdaq dot-com series
    "nasdaq": dict(csv=None, end="2002-12-31", label="Nasdaq 1994-2002 (repo sample)"),
}


def load_prices(key):
    if key == "spy":
        df = pd.read_csv(ROOT / "data" / "SPY_daily_1998_2021.csv", parse_dates=["Date"])
    else:
        import lppls as _lppls_pkg
        nasdaq = Path(_lppls_pkg.__file__).parent / "data" / "nasdaq_dotcom.csv"
        df = pd.read_csv(nasdaq, parse_dates=["Date"])
        df = df.rename(columns={"Adj Close": "Close"})[["Date", "Close"]]
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df.Date <= DATASETS[key]["end"]].reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    return df


def qualify(tc, m, w, b, c, O, D, t1, t2):
    """Boulder repo default filter conditions (compute_indicators)."""
    tc_lo = max(t2 - 60.0, t2 - 0.5 * (t2 - t1))
    tc_hi = min(t2 + 252.0, t2 + 0.5 * (t2 - t1))
    O_eff = O if (b != 0 and c != 0) else np.inf
    return (
        tc_lo < tc < tc_hi
        and 0.0 < m < 1.0
        and 2.0 < w < 15.0
        and O_eff > 2.5
        and D > 0.5
    )


def run_scan(df):
    t_all = np.arange(len(df), dtype=float)
    logp = df["logp"].values
    model = LPPLS(observations=np.array([t_all, logp]))
    first_t2 = WINDOWS[0]  # need at least the smallest window
    t2_grid = list(range(first_t2, len(df), T2_STEP))
    fits, conf_rows = [], []
    t_start = time.time()
    for k, t2_idx in enumerate(t2_grid):
        pos_q = neg_q = pos_n = neg_n = 0
        tcs, pcs, tcs_n, pcs_n = [], [], [], []
        for L in WINDOWS:
            t1_idx = t2_idx - L + 1
            if t1_idx < 0:
                continue
            obs = np.array([t_all[t1_idx : t2_idx + 1], logp[t1_idx : t2_idx + 1]])
            tc, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=25, obs=obs)
            ok = tc != 0 and qualify(tc, m, w, b, c, O, D, float(t1_idx), float(t2_idx))
            price_c = float(np.exp(a)) if tc != 0 else np.nan
            fits.append(dict(t2_idx=t2_idx, L=L, tc=tc, m=m, w=w, b=b, a=a,
                             price_c=price_c, O=O, D=D, qualified=ok))
            if tc == 0:
                continue
            if b < 0:
                pos_n += 1
                if ok:
                    pos_q += 1
                    tcs.append(tc)
                    pcs.append(price_c)
            elif b > 0:
                neg_n += 1
                if ok:
                    neg_q += 1
                    tcs_n.append(tc)
                    pcs_n.append(price_c)  # floor price for negative bubbles
        conf_rows.append(dict(
            t2_idx=t2_idx, date=df.loc[t2_idx, "Date"], price=df.loc[t2_idx, "Close"],
            pos_conf=pos_q / pos_n if pos_n else 0.0,
            neg_conf=neg_q / neg_n if neg_n else 0.0,
            med_tc_idx=np.median(tcs) if tcs else np.nan,
            med_price_c=np.median(pcs) if pcs else np.nan,
            med_tc_idx_neg=np.median(tcs_n) if tcs_n else np.nan,
            med_price_c_neg=np.median(pcs_n) if pcs_n else np.nan,
            n_pos_qual=pos_q,
        ))
        if (k + 1) % 100 == 0:
            print(f"  scan {k + 1}/{len(t2_grid)} t2={df.loc[t2_idx, 'Date'].date()} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    return pd.DataFrame(conf_rows), pd.DataFrame(fits)


def find_episodes(conf, col):
    """Group scan points with conf[col] >= CONF_THR into episodes."""
    flags = (conf[col] >= CONF_THR).values
    episodes, start, last_hit, run = [], None, None, 0
    for i, f in enumerate(flags):
        if f:
            if start is None:
                start, run = i, 0
            last_hit = i
            run += 1
        elif start is not None and i - last_hit > MERGE_GAP:
            if run >= MIN_RUN:
                episodes.append((start, last_hit))
            start, last_hit, run = None, None, 0
    if start is not None and run >= MIN_RUN:
        episodes.append((start, last_hit))
    return episodes


def realised_outcome(df, start_idx, end_idx, positive=True, horizon=130):
    """Realised peak (or trough) between the episode start and `horizon`
    trading days past the episode end (confidence usually stays elevated
    through the actual extreme), plus the drawdown/rebound that followed."""
    seg = df.iloc[start_idx : min(end_idx + horizon, len(df))]
    if positive:
        ext_idx = int(seg["Close"].idxmax())
        after = df.iloc[ext_idx : min(ext_idx + horizon, len(df))]
        move = 100.0 * (after["Close"].min() / df.loc[ext_idx, "Close"] - 1.0)
    else:
        ext_idx = int(seg["Close"].idxmin())
        after = df.iloc[ext_idx : min(ext_idx + horizon, len(df))]
        move = 100.0 * (after["Close"].max() / df.loc[ext_idx, "Close"] - 1.0)
    return ext_idx, float(df.loc[ext_idx, "Close"]), move


def idx_to_date(df, fidx):
    fidx = float(np.clip(fidx, 0, len(df) - 1))
    lo = int(np.floor(fidx))
    hi = min(lo + 1, len(df) - 1)
    return df.loc[lo, "Date"] + (df.loc[hi, "Date"] - df.loc[lo, "Date"]) * (fidx - lo)


def paper_methods_on_episode(df, t2_idx, plnn_params):
    """Run LM / M-LNN / M-LNN-KAN / P-LNN-100K on the 252-day window ending
    at t2_idx; return their tc (index units) and price_c."""
    L = min(N, t2_idx + 1)
    win = df["logp"].iloc[t2_idx - L + 1 : t2_idx + 1].values
    x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, L), win)
    x_scaled, (mn, rng) = minmax_scale(x)
    t = np.linspace(0.0, 1.0, N)
    out = {}
    r = lm.fit_lm(t, x_scaled, seed=int(t2_idx))
    out["LM"] = (r["tc"], r["m"], r["w"])
    r = mlnn.fit_mlnn(x_scaled, seed=int(t2_idx))
    out["M-LNN"] = (r["tc"], r["m"], r["w"])
    r = kan.fit_mlnn_kan(x_scaled, seed=int(t2_idx))
    out["M-LNN-KAN"] = (r["tc"], r["m"], r["w"])
    p = plnn.predict(plnn_params, x_scaled.astype(np.float32))[0]
    out["P-LNN-100K"] = (float(p[0]), float(p[1]), float(p[2]))
    res = {}
    for name, (tc, m, w) in out.items():
        a = solve_linear(t, x_scaled, tc, m, w)[0]
        tc_idx = t2_idx + (tc - 1.0) * (L - 1)
        res[name] = dict(tc_idx=tc_idx, price_c=float(np.exp(a * rng + mn)), m=m, w=w)
    return res


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="spy")
    args = ap.parse_args()
    key = args.data

    df = load_prices(key)
    print(f"[{key}] scan range {df.Date.iloc[0].date()} -> {df.Date.iloc[-1].date()} "
          f"({len(df)} trading days)", flush=True)
    conf, fits = run_scan(df)
    conf.to_csv(ROOT / "results" / f"{key}_scan_confidence.csv", index=False)
    fits.to_csv(ROOT / "results" / f"{key}_scan_fits.csv", index=False)

    plnn_params = plnn.load_params(ROOT / "models" / "P-LNN-100K.npz")
    # warm-up JITs so per-episode fits are steady-state
    paper_methods_on_episode(df, WINDOWS[-1], plnn_params)

    ep_rows, method_rows = [], []
    for sign, col in [("positive", "pos_conf"), ("negative", "neg_conf")]:
        for (i0, i1) in find_episodes(conf, col):
            c = conf.iloc[i0 : i1 + 1]
            peak_conf_row = c.loc[c[col].idxmax()]
            t2_star = int(peak_conf_row["t2_idx"])
            start_idx = int(c["t2_idx"].iloc[0])
            end_idx = int(c["t2_idx"].iloc[-1])
            ext_idx, ext_price, move = realised_outcome(df, start_idx, end_idx, positive=(sign == "positive"))
            tc_col = "med_tc_idx" if sign == "positive" else "med_tc_idx_neg"
            pc_col = "med_price_c" if sign == "positive" else "med_price_c_neg"
            med_tc = c[tc_col].median()
            med_pc = c[pc_col].median()
            ep_rows.append(dict(
                sign=sign,
                flag_start=c["date"].iloc[0].date(), flag_end=c["date"].iloc[-1].date(),
                max_conf=round(float(c[col].max()), 2),
                pred_tc=(idx_to_date(df, med_tc).date() if np.isfinite(med_tc) else None),
                pred_price_c=(round(med_pc, 2) if np.isfinite(med_pc) else None),
                realised_extreme=df.loc[ext_idx, "Date"].date(),
                realised_price=round(ext_price, 2),
                subsequent_move_pct=round(move, 1),
            ))
            if sign == "positive":
                for name, r in paper_methods_on_episode(df, t2_star, plnn_params).items():
                    method_rows.append(dict(
                        episode=f"{c['date'].iloc[0].date()}..{c['date'].iloc[-1].date()}",
                        t2=df.loc[t2_star, "Date"].date(), method=name,
                        pred_tc=idx_to_date(df, r["tc_idx"]).date(),
                        pred_price_c=round(r["price_c"], 2),
                        realised_peak=df.loc[ext_idx, "Date"].date(),
                        realised_price=round(ext_price, 2),
                        tc_err_days=round(r["tc_idx"] - ext_idx, 1),
                        price_c_err_pct=round(100.0 * (r["price_c"] / ext_price - 1.0), 1),
                    ))
    eps = pd.DataFrame(ep_rows)
    eps.to_csv(ROOT / "results" / f"{key}_bubble_episodes.csv", index=False)
    mdf = pd.DataFrame(method_rows)
    mdf.to_csv(ROOT / "results" / f"{key}_episode_methods.csv", index=False)
    print(eps.to_string(index=False), flush=True)
    print(mdf.to_string(index=False), flush=True)

    plot(df, conf, eps, key)
    print("figure saved", flush=True)


def plot(df, conf, eps, key):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.05},
    )
    ax.plot(df["Date"], df["Close"], color="black", lw=0.9, label=f"{key.upper()} close")
    ax.set_yscale("log")

    # median predicted critical price, coloured by confidence
    ok = conf[conf["med_price_c"].notna() & (conf["pos_conf"] >= CONF_THR)]
    sc = ax.scatter(
        [idx_to_date(df, v) for v in ok["med_tc_idx"]], ok["med_price_c"],
        c=ok["pos_conf"], cmap="Reds", vmin=CONF_THR, vmax=1.0, s=22,
        edgecolors="none", zorder=5,
        label="predicted ($t_c$, price$_c$), qualified fits",
    )
    fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03, label="pos. confidence")

    for _, e in eps.iterrows():
        color = "red" if e["sign"] == "positive" else "green"
        ax.axvspan(pd.Timestamp(e["flag_start"]), pd.Timestamp(e["flag_end"]), color=color, alpha=0.15)
        if e["sign"] == "positive":
            ax.axvline(pd.Timestamp(e["realised_extreme"]), color="black", ls="--", lw=1.0, alpha=0.7)

    ax.set_ylabel(f"{key.upper()} close (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(f"{DATASETS[key]['label']}: LPPLS bubble scan - flagged episodes (red=positive, "
                 "green=negative), predicted critical price, realised peaks (dashed)")

    ax2.plot(conf["date"], conf["pos_conf"], color="tab:red", lw=1.2, label="positive-bubble confidence")
    ax2.plot(conf["date"], conf["neg_conf"], color="tab:green", lw=1.2, label="negative-bubble confidence")
    ax2.axhline(CONF_THR, color="grey", ls=":", lw=1)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("confidence")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(ROOT / "results" / f"fig_{key}_bubble_scan.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
