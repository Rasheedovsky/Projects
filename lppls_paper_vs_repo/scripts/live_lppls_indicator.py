"""Live LPPLS bubble indicator with probability densities.

Standing at any date t2 (using ONLY data up to t2, so the indicator is
deployable in real time), fit an ensemble of calibration windows, keep the
fits that pass the reference repo's qualification filters, and render:

  * top strip   - probability density of the critical time t_c (over time)
  * right strip - probability density of the critical price p_c = exp(A)
                  (over price)
  * main panel  - price history, the qualified LPPLS fits, and the ensemble
                  confidence (share of qualified fits)

Two ensembles are shown: the classical Nelder-Mead multistart (red, the
reference repo's fit - ~1.5 s per snapshot) and the paper's pre-trained
P-LNN-100K (purple, ~50 ms per snapshot) - each across ~24 window lengths.

Usage:
  snapshot: python scripts/live_lppls_indicator.py --data tasi --date 2022-04-08
  GIF:      python scripts/live_lppls_indicator.py --data tasi \
                --animate 2021-06-01 2022-05-06 --step 5
"""

import argparse
import sys
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

from deep_lppls import plnn
from deep_lppls.core import minmax_scale, solve_linear
from lppls.lppls import LPPLS
from spy_bubble_scan import qualify

N = 252
WINDOW_LENGTHS = list(range(60, 420, 15))  # 24 ensemble members
HIST_SHOW = 500                            # trading days of history to draw
FUTURE_SHOW = 220                          # trading days of future axis space

DATASETS = {
    "tasi": dict(csv="TASI_daily_2020_2024.csv", label="TASI"),
    "spy": dict(csv="SPY_daily_1998_2021.csv", label="SPY"),
}


def load(key):
    df = pd.read_csv(ROOT / "data" / DATASETS[key]["csv"], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    return df


def fit_ensembles(df, t2_idx, plnn_params):
    """Ensemble fits at t2 over WINDOW_LENGTHS. Returns dict per ensemble:
    list of dicts (tc_idx, price_c, qualified, params...)."""
    t_all = np.arange(len(df), dtype=float)
    logp = df["logp"].values
    model = LPPLS(observations=np.array([t_all, logp]))
    out = {"NM": [], "P-LNN": []}
    for L in WINDOW_LENGTHS:
        t1_idx = t2_idx - L + 1
        if t1_idx < 0:
            continue
        # --- Nelder-Mead (reference repo) on the raw log-price window ---
        obs = np.array([t_all[t1_idx : t2_idx + 1], logp[t1_idx : t2_idx + 1]])
        tc, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=25, obs=obs)
        if tc != 0:
            ok = qualify(tc, m, w, b, c, O, D, float(t1_idx), float(t2_idx))
            out["NM"].append(dict(tc_idx=tc, price_c=float(np.exp(a)), qualified=ok,
                                  m=m, w=w, b=b, c1=c1, c2=c2, a=a, t1_idx=t1_idx, L=L))
        # --- P-LNN-100K on the resampled min-max-scaled window ---
        win = logp[t1_idx : t2_idx + 1]
        x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(win)), win)
        x_scaled, (mn, rng) = minmax_scale(x)
        tcn, mn_, wn = (float(v) for v in plnn.predict(plnn_params, x_scaled.astype(np.float32))[0])
        t_norm = np.linspace(0.0, 1.0, N)
        a4, b4, c14, c24 = solve_linear(t_norm, x_scaled, tcn, mn_, wn)
        tc_idx = t2_idx + (tcn - 1.0) * (L - 1)
        # qualification: same filters, in normalised units (O, D from fit)
        denom = tcn - 1.0
        O_p = (wn / (2 * np.pi)) * np.log(tcn / denom) if denom > 0 else np.inf
        c_abs = float(np.hypot(c14, c24))
        D_p = (mn_ * abs(b4)) / (wn * c_abs) if c_abs > 0 else np.inf
        ok = (
            (t2_idx - min(60, 0.5 * (L - 1))) < tc_idx < (t2_idx + min(252, 0.5 * (L - 1)))
            and 0 < mn_ < 1 and 2 < wn < 15 and O_p > 2.5 and D_p > 0.5 and b4 < 0
        )
        out["P-LNN"].append(dict(tc_idx=float(tc_idx), price_c=float(np.exp(a4 * rng + mn)),
                                 qualified=bool(ok), m=mn_, w=wn, L=L))
    return out


def idx_to_date(df, fidx):
    """Fractional index -> date; extrapolates past the data edge at the
    median trading-day pace so future tc densities keep a time axis."""
    if fidx <= len(df) - 1:
        lo = int(np.floor(fidx))
        hi = min(lo + 1, len(df) - 1)
        return df.loc[lo, "Date"] + (df.loc[hi, "Date"] - df.loc[lo, "Date"]) * (fidx - lo)
    step = (df["Date"].iloc[-1] - df["Date"].iloc[0]) / (len(df) - 1)
    return df["Date"].iloc[-1] + step * (fidx - (len(df) - 1))


def render(df, t2_idx, ensembles, key, ax_cache=None):
    """Render one dashboard frame; returns the matplotlib figure."""
    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 3.4],
                          hspace=0.04, wspace=0.03)
    ax_t = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_p = fig.add_subplot(gs[1, 1], sharey=ax)

    lo_idx = max(t2_idx - HIST_SHOW, 0)
    seg = df.iloc[lo_idx : t2_idx + 1]
    x_lo = df.loc[lo_idx, "Date"]
    x_hi = idx_to_date(df, t2_idx + FUTURE_SHOW)
    ax.plot(seg["Date"], seg["Close"], color="black", lw=1.0)
    ax.axvline(df.loc[t2_idx, "Date"], color="red", ls="-.", lw=1.4)
    ax.axvspan(df.loc[t2_idx, "Date"], x_hi, color="grey", alpha=0.08)

    price_lo = seg["Close"].min() * 0.92
    price_hi = seg["Close"].max() * 1.25
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(price_lo, price_hi)

    colors = {"NM": "tab:red", "P-LNN": "tab:purple"}
    labels = {"NM": "lppls-repo NM ensemble", "P-LNN": "P-LNN-100K ensemble"}
    conf_txt = []
    day_grid = np.linspace(t2_idx - 30, t2_idx + FUTURE_SHOW, 300)
    date_grid = [idx_to_date(df, v) for v in day_grid]
    price_grid = np.linspace(price_lo, price_hi, 300)

    for name, fits in ensembles.items():
        q = [f for f in fits if f["qualified"]]
        conf = len(q) / len(fits) if fits else 0.0
        conf_txt.append(f"{labels[name]}: conf {conf:.0%} ({len(q)}/{len(fits)})")
        # draw qualified NM fits faintly in the main panel
        if name == "NM":
            t_all = np.arange(len(df), dtype=float)
            for f in q:
                tt = np.linspace(f["t1_idx"], min(f["tc_idx"] - 0.5, t2_idx + FUTURE_SHOW * 0.7), 250)
                dt = np.abs(f["tc_idx"] - tt) + 1e-8
                yhat = f["a"] + dt ** f["m"] * (f["b"] + f["c1"] * np.cos(f["w"] * np.log(dt))
                                                + f["c2"] * np.sin(f["w"] * np.log(dt)))
                ax.plot([idx_to_date(df, v) for v in tt], np.exp(yhat),
                        color=colors[name], lw=0.7, alpha=0.25)
        if len(q) >= 3:
            tcs = np.array([f["tc_idx"] for f in q])
            pcs = np.array([f["price_c"] for f in q])
            tcs = tcs[(tcs > t2_idx - 30) & (tcs < t2_idx + FUTURE_SHOW)]
            pcs = pcs[(pcs > price_lo) & (pcs < price_hi)]
            if len(np.unique(tcs)) >= 3 and np.std(tcs) > 1e-9:
                kde = gaussian_kde(tcs)
                pdf = kde(day_grid)
                ax_t.plot(date_grid, pdf, color=colors[name], lw=1.6)
                ax_t.fill_between(date_grid, 0, pdf, color=colors[name], alpha=0.3)
                med = idx_to_date(df, float(np.median(tcs)))
                ax_t.axvline(med, color=colors[name], ls=":", lw=1.2)
            if len(np.unique(pcs)) >= 3 and np.std(pcs) > 1e-9:
                kdep = gaussian_kde(pcs)
                pdfp = kdep(price_grid)
                ax_p.plot(pdfp, price_grid, color=colors[name], lw=1.6)
                ax_p.fill_betweenx(price_grid, 0, pdfp, color=colors[name], alpha=0.3)
                ax_p.axhline(float(np.median(pcs)), color=colors[name], ls=":", lw=1.2)

    ax_t.axvline(df.loc[t2_idx, "Date"], color="red", ls="-.", lw=1.2)
    ax_t.set_ylabel("PDF($t_c$)")
    ax_t.set_yticks([])
    ax_t.set_title(f"{DATASETS[key]['label']} live LPPLS indicator @ {df.loc[t2_idx, 'Date'].date()}   |   "
                   + "   |   ".join(conf_txt), fontsize=11)
    ax_p.set_xlabel("PDF(price$_c$)")
    ax_p.set_xticks([])
    plt.setp(ax_p.get_yticklabels(), visible=False)
    plt.setp(ax_t.get_xticklabels(), visible=False)
    ax.set_ylabel(f"{DATASETS[key]['label']} close")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    handles = [plt.Line2D([], [], color=c, lw=2) for c in colors.values()]
    ax.legend(handles, labels.values(), loc="upper left", fontsize=9)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="tasi")
    ap.add_argument("--date", help="snapshot date YYYY-MM-DD")
    ap.add_argument("--animate", nargs=2, metavar=("START", "END"), help="GIF over date range")
    ap.add_argument("--step", type=int, default=5, help="trading days between frames")
    args = ap.parse_args()

    df = load(args.data)
    plnn_params = plnn.load_params(ROOT / "models" / "P-LNN-100K.npz")

    def nearest_idx(datestr):
        return int((df["Date"] - pd.Timestamp(datestr)).abs().idxmin())

    if args.animate:
        from PIL import Image

        i0, i1 = nearest_idx(args.animate[0]), nearest_idx(args.animate[1])
        frames = []
        for k, t2_idx in enumerate(range(i0, i1 + 1, args.step)):
            ens = fit_ensembles(df, t2_idx, plnn_params)
            fig = render(df, t2_idx, ens, args.data)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
            if (k + 1) % 10 == 0:
                print(f"  frame {k + 1}", flush=True)
        out = ROOT / "results" / f"live_indicator_{args.data}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=350, loop=0)
        print(f"saved {out} ({len(frames)} frames)")
    else:
        t2_idx = nearest_idx(args.date)
        ens = fit_ensembles(df, t2_idx, plnn_params)
        fig = render(df, t2_idx, ens, args.data)
        out = ROOT / "results" / f"fig_live_indicator_{args.data}_{args.date}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
