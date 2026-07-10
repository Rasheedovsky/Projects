"""Synthetic benchmark replicating the paper's experimental design (Sec. 2.3
and 3.1): 250 random scenarios from the Table 1 ranges, evaluated with
 - LM (paper Appendix A.1: multistart Levenberg-Marquardt)
 - Boulder-Investment-Technologies/lppls package (multistart Nelder-Mead)
 - M-LNN (fresh network per series)
 - P-LNN variants (pre-trained)

Produces:
 - results/fig_synth_cdf_white.png / _ar1.png  (paper Fig. 3 analogue)
 - results/timing_synthetic.csv                (paper Table 2 analogue)
 - results/synthetic_estimates.csv             (raw per-scenario estimates)

Usage: python scripts/benchmark_synthetic.py [--n 250]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import lm, mlnn, plnn, synthetic
from deep_lppls.core import f1_loss
from lppls.lppls import LPPLS

DAY = 1.0 / (synthetic.N_POINTS - 1)


def make_scenarios(n, noise, seed):
    rng = np.random.default_rng(seed)
    tc, m, w, c1, c2 = synthetic.sample_params(rng, n)
    X = np.empty((n, synthetic.N_POINTS), dtype=np.float64)
    for i in range(n):
        X[i] = synthetic.clean_series(tc[i], m[i], w[i], c1[i], c2[i])
    if noise == "white":
        X += synthetic.white_noise(rng, X.shape)
    else:
        X += synthetic.ar1_noise(rng, X.shape)
    return X, np.column_stack([tc, m, w])


def fit_boulder(t_days, x, seed=None):
    """Fit via the reference repo's public API (Nelder-Mead, 25 searches).
    Time axis in days (0..251) to match its unnormalised-time convention."""
    obs = np.array([t_days, x])
    model = LPPLS(observations=obs)
    tc, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=25)
    return dict(tc=tc, m=m, w=w)


def run(n_scen, noise, plnn_models, mlnn_epochs):
    X, y_true = make_scenarios(n_scen, noise, seed=100 if noise == "white" else 200)
    t = np.linspace(0.0, 1.0, synthetic.N_POINTS)
    t_days = np.arange(synthetic.N_POINTS, dtype=float)
    rows, timings = [], {}

    # warm-up: exclude one-time JIT compilation (numba / XLA) from the
    # wall-clock statistics, as the paper times steady-state calibrations
    lm.fit_lm(t, X[0], max_searches=2, seed=0)
    fit_boulder(t_days, X[0])
    mlnn.fit_mlnn(X[0], epochs=2, seed=0)
    for params in plnn_models.values():
        plnn.predict(params, X[0].astype(np.float32))

    # --- LM (paper) ---
    times = []
    for i in range(n_scen):
        t0 = time.perf_counter()
        r = lm.fit_lm(t, X[i], seed=i)
        times.append(time.perf_counter() - t0)
        rows.append(dict(scen=i, noise=noise, method="LM", tc=r["tc"], m=r["m"], w=r["w"]))
    timings["LM"] = times
    print(f"  LM done ({np.mean(times):.3f}s avg)", flush=True)

    # --- Boulder lppls package (Nelder-Mead) ---
    times = []
    for i in range(n_scen):
        t0 = time.perf_counter()
        r = fit_boulder(t_days, X[i])
        times.append(time.perf_counter() - t0)
        # convert day-unit tc back to normalised time (m, w invariant)
        tc_norm = 1.0 + (r["tc"] - t_days[-1]) * DAY if r["tc"] != 0 else np.nan
        rows.append(dict(scen=i, noise=noise, method="lppls-repo (NM)", tc=tc_norm, m=r["m"] or np.nan, w=r["w"] or np.nan))
    timings["lppls-repo (NM)"] = times
    print(f"  Boulder repo done ({np.mean(times):.3f}s avg)", flush=True)

    # --- M-LNN ---
    times = []
    for i in range(n_scen):
        t0 = time.perf_counter()
        r = mlnn.fit_mlnn(X[i], epochs=mlnn_epochs, seed=i)
        times.append(time.perf_counter() - t0)
        rows.append(dict(scen=i, noise=noise, method="M-LNN", tc=r["tc"], m=r["m"], w=r["w"]))
    timings["M-LNN"] = times
    print(f"  M-LNN done ({np.mean(times):.3f}s avg)", flush=True)

    # --- P-LNN variants (batch inference, but time per-series singly) ---
    for name, params in plnn_models.items():
        times = []
        preds = []
        for i in range(n_scen):
            t0 = time.perf_counter()
            p = plnn.predict(params, X[i].astype(np.float32))[0]
            times.append(time.perf_counter() - t0)
            preds.append(p)
        for i, p in enumerate(preds):
            rows.append(dict(scen=i, noise=noise, method=name, tc=float(p[0]), m=float(p[1]), w=float(p[2])))
        timings[name] = times
        print(f"  {name} done ({np.mean(times) * 1000:.2f}ms avg)", flush=True)

    df = pd.DataFrame(rows)
    truth = pd.DataFrame(dict(scen=range(n_scen), tc_true=y_true[:, 0], m_true=y_true[:, 1], w_true=y_true[:, 2]))
    df = df.merge(truth, on="scen")
    # fit MSE of the calibrated LPPLS vs the *observed noisy* series
    mses = []
    for _, r in df.iterrows():
        if np.isnan(r["tc"]):
            mses.append(np.nan)
        else:
            mses.append(f1_loss(t, X[int(r["scen"])], r["tc"], r["m"], r["w"]))
    df["fit_mse"] = mses
    return df, timings


COLORS = {
    "LM": "tab:blue",
    "lppls-repo (NM)": "tab:red",
    "M-LNN": "tab:orange",
    "P-LNN-100K": "tab:purple",
    "P-LNN-100K-AR1": "tab:green",
    "P-LNN-100K-BOTH": "tab:brown",
}


def plot_cdfs(df, noise, out):
    metrics = [
        ("|tc error| (days)", lambda d: np.abs(d.tc - d.tc_true) / DAY, (0, 50)),
        ("|m error|", lambda d: np.abs(d.m - d.m_true), (0, 1.0)),
        ("|w error|", lambda d: np.abs(d.w - d.w_true), (0, 7)),
        ("fit MSE", lambda d: d.fit_mse, None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, (label, fn, xlim) in zip(axes, metrics):
        for method, g in df.groupby("method"):
            e = np.sort(fn(g).dropna().values)
            if len(e) == 0:
                continue
            cdf = np.arange(1, len(e) + 1) / len(e)
            ax.plot(e, cdf, label=method, color=COLORS.get(method), lw=1.8)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        if xlim:
            ax.set_xlim(*xlim)
        if label == "fit MSE":
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"CDF of absolute estimation error, {noise} noise test set (paper Fig. 3 analogue)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--mlnn-epochs", type=int, default=mlnn.EPOCHS)
    args = ap.parse_args()

    plnn_models = {}
    for name in ["P-LNN-100K", "P-LNN-100K-AR1", "P-LNN-100K-BOTH"]:
        path = ROOT / "models" / f"{name}.npz"
        if path.exists():
            plnn_models[name] = plnn.load_params(path)
    print(f"loaded P-LNN models: {list(plnn_models)}", flush=True)

    all_df, timing_rows = [], []
    for noise in ["white", "ar1"]:
        print(f"=== noise: {noise} ===", flush=True)
        df, timings = run(args.n, noise, plnn_models, args.mlnn_epochs)
        all_df.append(df)
        plot_cdfs(df, noise, ROOT / "results" / f"fig_synth_cdf_{noise}.png")
        for method, ts in timings.items():
            timing_rows.append(dict(noise=noise, method=method, mean_s=np.mean(ts), std_s=np.std(ts), n=len(ts)))

    pd.concat(all_df).to_csv(ROOT / "results" / "synthetic_estimates.csv", index=False)
    tdf = pd.DataFrame(timing_rows)
    tdf.to_csv(ROOT / "results" / "timing_synthetic.csv", index=False)
    print(tdf.to_string(index=False))

    # summary error table
    df = pd.concat(all_df)
    df["tc_err_days"] = np.abs(df.tc - df.tc_true) / DAY
    df["m_err"] = np.abs(df.m - df.m_true)
    df["w_err"] = np.abs(df.w - df.w_true)
    summary = df.groupby(["noise", "method"])[["tc_err_days", "m_err", "w_err", "fit_mse"]].median().round(4)
    summary.to_csv(ROOT / "results" / "synthetic_error_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
