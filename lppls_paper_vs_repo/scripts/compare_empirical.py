"""Empirical bubble comparison, replicating the protocol of the paper
(Sec. 3.2, Figs. 4-5) on real bubble episodes:

  --data tasi   Tadawul All Share Index 2021-22 bubble (peak 2022-05-08)
  --data spy    SPY COVID melt-up (peak 2020-02-19, crash into 2020-03-23)

For a set of calibration windows [t1, t2] that all end strictly before the
realised peak (both endpoints shift, as in the paper), each window is
resampled to 252 observations (paper: "each empirical dataset is resampled
such that there are 252 observations preceding the critical time"), the
log-price is min-max scaled, and (tc, m, w) is estimated with:

  LM (paper appendix A.1)     multistart Levenberg-Marquardt
  lppls-repo (NM)             Boulder-Investment-Technologies/lppls, fit(25)
  M-LNN                       fresh 2-hidden-layer ReLU network per window
  M-LNN-KAN                   same protocol, KAN (B-spline) layers [extension]
  P-LNN-100K / -AR1 / -BOTH   pre-trained supervised networks

Outputs per dataset key: results/fig_<key>_fits.png (Fig. 4 analogue),
results/<key>_estimates.csv, results/timing_<key>.csv, results/<key>_summary.csv.
"""

import argparse
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

from deep_lppls import kan, lm, mlnn, plnn
from deep_lppls.core import design_matrix, minmax_scale, solve_linear
from lppls.lppls import LPPLS

N = 252
T2_OFFSETS = [5, 10, 15, 20, 25, 30]        # trading days before the peak
WINDOW_LENGTHS = [150, 200, 252, 300, 350]  # trading days
PLNN_NAMES = ["P-LNN-100K", "P-LNN-100K-AR1", "P-LNN-100K-BOTH"]

DATASETS = {
    # peak dates pinned to the realised bubble peak of each episode (for SPY
    # the global max sits at the data edge, so argmax would pick the wrong one)
    "tasi": dict(csv="TASI_daily_2020_2024.csv", peak="2022-05-08", label="TASI 2021-22 bubble"),
    "spy": dict(csv="SPY_daily_1998_2021.csv", peak="2020-02-19", label="SPY COVID melt-up"),
}

COLORS = {
    "LM": "tab:blue",
    "lppls-repo (NM)": "tab:red",
    "M-LNN": "tab:orange",
    "M-LNN-KAN": "tab:cyan",
    "P-LNN-100K": "tab:purple",
    "P-LNN-100K-AR1": "tab:green",
    "P-LNN-100K-BOTH": "tab:brown",
}
PLOT_METHODS = ["LM", "lppls-repo (NM)", "M-LNN", "M-LNN-KAN", "P-LNN-100K"]


def load_data(key):
    cfg = DATASETS[key]
    df = pd.read_csv(ROOT / "data" / cfg["csv"], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    peak_idx = int(df.index[df.Date == cfg["peak"]][0])
    return df, peak_idx, cfg["label"]


def resample_window(logp_window):
    """Linear-interpolate a window of arbitrary length onto 252 points."""
    src = np.linspace(0.0, 1.0, len(logp_window))
    dst = np.linspace(0.0, 1.0, N)
    return np.interp(dst, src, logp_window)


def critical_price(x_scaled, mn, rng, tc, m, w):
    """Critical price p_c = exp(A): the LPPLS log-price at tc (O(tc) = A,
    Eq. 1), with A obtained from the analytic solve on the scaled window and
    mapped back through the min-max scaling of the log-price."""
    if not np.isfinite(tc):
        return np.nan
    a = solve_linear(np.linspace(0.0, 1.0, N), x_scaled, tc, m, w)[0]
    return float(np.exp(a * rng + mn))


def fit_all_methods(x_scaled, scale, plnn_models, seed):
    """Fit one min-max-scaled 252-point window with every method.
    Returns {method: (result dict, seconds)}; tc in normalised window units,
    price_c in original price units. scale = (mn, rng) of the log-price."""
    mn, rng = scale
    t = np.linspace(0.0, 1.0, N)
    t_days = np.arange(N, dtype=float)
    out = {}

    t0 = time.perf_counter()
    r = lm.fit_lm(t, x_scaled, seed=seed)
    out["LM"] = (dict(tc=r["tc"], m=r["m"], w=r["w"]), time.perf_counter() - t0)

    t0 = time.perf_counter()
    model = LPPLS(observations=np.array([t_days, x_scaled]))
    tc_d, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=25)
    dt = time.perf_counter() - t0
    tc_norm = 1.0 + (tc_d - t_days[-1]) / (N - 1) if tc_d != 0 else np.nan
    out["lppls-repo (NM)"] = (dict(tc=tc_norm, m=m if tc_d != 0 else np.nan, w=w if tc_d != 0 else np.nan), dt)

    t0 = time.perf_counter()
    r = mlnn.fit_mlnn(x_scaled, seed=seed)
    out["M-LNN"] = (dict(tc=r["tc"], m=r["m"], w=r["w"]), time.perf_counter() - t0)

    t0 = time.perf_counter()
    r = kan.fit_mlnn_kan(x_scaled, seed=seed)
    out["M-LNN-KAN"] = (dict(tc=r["tc"], m=r["m"], w=r["w"]), time.perf_counter() - t0)

    for name, params in plnn_models.items():
        t0 = time.perf_counter()
        p = plnn.predict(params, x_scaled.astype(np.float32))[0]
        out[name] = (dict(tc=float(p[0]), m=float(p[1]), w=float(p[2])), time.perf_counter() - t0)

    # critical price for every method (outside the timed sections: it is a
    # post-processing step shared by all calibrations)
    for name, (r, dt) in out.items():
        r["price_c"] = critical_price(x_scaled, mn, rng, r["tc"], r["m"], r["w"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="tasi")
    args = ap.parse_args()
    key = args.data

    df, peak_idx, label = load_data(key)
    peak_date = df.loc[peak_idx, "Date"]
    after = df.iloc[peak_idx : peak_idx + 130]  # ~6 months
    trough_idx = int(after["Close"].idxmin())
    trough_date = df.loc[trough_idx, "Date"]
    print(f"[{key}] peak {peak_date.date()} close={df.loc[peak_idx, 'Close']:.2f} | "
          f"drawdown trough {trough_date.date()} close={df.loc[trough_idx, 'Close']:.2f}", flush=True)

    plnn_models = {n: plnn.load_params(ROOT / "models" / f"{n}.npz") for n in PLNN_NAMES
                   if (ROOT / "models" / f"{n}.npz").exists()}
    print(f"loaded P-LNN models: {list(plnn_models)}", flush=True)

    # warm-up so one-time JIT compilation doesn't pollute per-fit timings
    dummy = np.linspace(0.0, 1.0, N) + 0.01 * np.random.default_rng(0).standard_normal(N)
    fit_all_methods(dummy, (0.0, 1.0), plnn_models, seed=0)

    rows, timing = [], {}
    for t2_off in T2_OFFSETS:
        t2_idx = peak_idx - t2_off
        for L in WINDOW_LENGTHS:
            t1_idx = t2_idx - L + 1
            if t1_idx < 0:
                continue
            win = df["logp"].iloc[t1_idx : t2_idx + 1].values
            x_res = resample_window(win)
            x_scaled, scale = minmax_scale(x_res)
            res = fit_all_methods(x_scaled, scale, plnn_models, seed=t2_off * 1000 + L)
            peak_price = float(df.loc[peak_idx, "Close"])
            for method, (r, dt) in res.items():
                # normalised tc -> trading days after t2 (window spans L
                # trading days mapped onto [0, 1])
                tc_days_after_t2 = (r["tc"] - 1.0) * (L - 1) if np.isfinite(r["tc"]) else np.nan
                tc_idx = t2_idx + tc_days_after_t2 if np.isfinite(tc_days_after_t2) else np.nan
                rows.append(dict(
                    t2_off=t2_off, L=L, method=method,
                    t1=df.loc[t1_idx, "Date"], t2=df.loc[t2_idx, "Date"],
                    tc_norm=r["tc"], m=r["m"], w=r["w"],
                    tc_days_after_t2=tc_days_after_t2, tc_idx=tc_idx,
                    tc_days_vs_peak=(tc_idx - peak_idx) if np.isfinite(tc_idx) else np.nan,
                    price_c=r["price_c"],
                    price_c_vs_peak_pct=100.0 * (r["price_c"] / peak_price - 1.0) if np.isfinite(r["price_c"]) else np.nan,
                    seconds=dt,
                ))
                timing.setdefault(method, []).append(dt)
        print(f"  windows ending {t2_off}d before peak done", flush=True)

    est = pd.DataFrame(rows)
    est.to_csv(ROOT / "results" / f"{key}_estimates.csv", index=False)

    tdf = pd.DataFrame([dict(method=m, mean_s=np.mean(v), std_s=np.std(v), n=len(v)) for m, v in timing.items()])
    tdf.to_csv(ROOT / "results" / f"timing_{key}.csv", index=False)
    print(tdf.to_string(index=False), flush=True)

    # summary: median predicted tc (days relative to realised peak) per method
    summ = est.groupby("method").agg(
        median_tc_vs_peak_days=("tc_days_vs_peak", "median"),
        iqr_lo=("tc_days_vs_peak", lambda s: s.quantile(0.25)),
        iqr_hi=("tc_days_vs_peak", lambda s: s.quantile(0.75)),
        median_price_c=("price_c", "median"),
        median_price_c_vs_peak_pct=("price_c_vs_peak_pct", "median"),
        median_m=("m", "median"), median_w=("w", "median"),
        n_valid=("tc_days_vs_peak", lambda s: s.notna().sum()),
    ).round(2)
    summ.to_csv(ROOT / "results" / f"{key}_summary.csv")
    print(summ.to_string(), flush=True)

    plot(df, est, peak_idx, trough_idx, key, label)
    print("figure saved", flush=True)


def idx_to_date(df, fidx):
    """Fractional trading-day index -> calendar date (clipped to data)."""
    fidx = float(np.clip(fidx, 0, len(df) - 1))
    lo = int(np.floor(fidx))
    hi = min(lo + 1, len(df) - 1)
    frac = fidx - lo
    return df.loc[lo, "Date"] + (df.loc[hi, "Date"] - df.loc[lo, "Date"]) * frac


def plot(df, est, peak_idx, trough_idx, key, label):
    """Paper Fig. 4 analogue: tc PDFs strip on top, log-price + fits below."""
    fig, (ax_pdf, ax) = plt.subplots(
        2, 1, figsize=(14, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.04},
    )
    lo_idx = max(peak_idx - 420, 0)
    hi_idx = min(peak_idx + 160, len(df) - 1)
    seg = df.iloc[lo_idx:hi_idx]
    ax.plot(seg["Date"], seg["logp"], color="black", lw=1.0, label=f"{key.upper()} ln(price)")

    # representative window: t2 = 20 trading days before peak, L = 252
    t2_off, L = 20, 252
    t2_idx = peak_idx - t2_off
    t1_idx = t2_idx - L + 1
    win = df["logp"].iloc[t1_idx : t2_idx + 1].values
    x_res = resample_window(win)
    x_scaled, (mn, rng) = minmax_scale(x_res)
    ax.axvspan(df.loc[t1_idx, "Date"], df.loc[t2_idx, "Date"], color="grey", alpha=0.15, label="calibration window")
    ax.axvline(df.loc[t1_idx, "Date"], color="green", ls="-.", lw=1.2, label="$t_1$")
    ax.axvline(df.loc[t2_idx, "Date"], color="red", ls="-.", lw=1.4, label="$t_2$ (present)")
    ax.axvspan(df.loc[peak_idx, "Date"], df.loc[trough_idx, "Date"], color="red", alpha=0.12, label="realised peak→trough")
    ax.axvline(df.loc[peak_idx, "Date"], color="black", ls="-.", lw=1.2)
    ax.axvline(df.loc[trough_idx, "Date"], color="black", ls="--", lw=1.2)

    # fits of the representative window, extended past t2 towards tc
    rep = est[(est.t2_off == t2_off) & (est.L == L) & est.method.isin(PLOT_METHODS)]
    for _, r in rep.iterrows():
        if not np.isfinite(r["tc_norm"]):
            continue
        tc, m, w = r["tc_norm"], r["m"], r["w"]
        beta = solve_linear(np.linspace(0, 1, N), x_scaled, tc, m, w)
        t_ext = np.linspace(0.0, min(max(tc - 0.004, 1.0), 1.30), 700)
        yhat = design_matrix(t_ext, tc, m, w) @ beta
        dates = [idx_to_date(df, t1_idx + ti * (L - 1)) for ti in t_ext]
        ax.plot(dates, yhat * rng + mn, color=COLORS[r["method"]], lw=1.6, alpha=0.9, label=f"{r['method']} fit")

    # y-limits driven by the data, so runaway fits don't distort the panel
    ax.set_ylim(seg["logp"].min() - 0.03, seg["logp"].max() + 0.06)

    # PDFs of predicted tc across all windows (dedicated strip, as in Fig. 4)
    x_dates = pd.date_range(df.loc[lo_idx, "Date"], df.loc[hi_idx, "Date"], freq="D")
    x_num = mdates.date2num(x_dates)
    for method in PLOT_METHODS:
        vals = est[(est.method == method)]["tc_idx"].dropna().values
        vals = vals[(vals > lo_idx) & (vals < hi_idx)]  # shown range only
        n_total = est[(est.method == method)]["tc_idx"].notna().sum()
        if len(vals) < 3:
            continue
        d_num = np.array([mdates.date2num(idx_to_date(df, v)) for v in vals])
        if np.std(d_num) < 1e-6:
            continue
        kde = gaussian_kde(d_num)
        pdf = kde(x_num)
        label_m = f"{method} ({len(vals)}/{n_total} in range)"
        ax_pdf.plot(x_dates, pdf, color=COLORS[method], lw=1.6, label=label_m)
        ax_pdf.fill_between(x_dates, 0, pdf, color=COLORS[method], alpha=0.25)
    ax_pdf.axvspan(df.loc[peak_idx, "Date"], df.loc[trough_idx, "Date"], color="red", alpha=0.12)
    ax_pdf.axvline(df.loc[peak_idx, "Date"], color="black", ls="-.", lw=1.2)
    ax_pdf.axvline(df.loc[trough_idx, "Date"], color="black", ls="--", lw=1.2)
    ax_pdf.axvline(df.loc[peak_idx - 20, "Date"], color="red", ls="-.", lw=1.0)
    ax_pdf.set_ylabel("PDF of $t_c$")
    ax_pdf.set_yticks([])
    ax_pdf.legend(fontsize=8, loc="upper left")
    ax_pdf.set_title(f"{label}: LPPLS fits and PDFs of predicted $t_c$ across "
                     f"{est.groupby('method').size().max()} calibration windows (paper Fig. 4 protocol)")

    ax.set_ylabel(f"ln({key.upper()} close)")
    ax.legend(loc="upper left", fontsize=9, ncols=2)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.savefig(ROOT / "results" / f"fig_{key}_fits.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
