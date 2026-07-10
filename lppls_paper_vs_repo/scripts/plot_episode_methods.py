"""Visualise the per-episode predictions of every method (LM, M-LNN,
M-LNN-KAN, P-LNN-100K) recorded by spy_bubble_scan.py: one panel per flagged
positive episode, each method's (t_c, price_c) prediction as a marker,
realised peak as a black star.

Usage: python scripts/plot_episode_methods.py [--data spy|nasdaq]
Reads results/<key>_episode_methods.csv; writes results/fig_<key>_episode_methods.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spy_bubble_scan import DATASETS, load_prices  # noqa: E402  (same folder)

COLORS = {
    "LM": "tab:blue",
    "M-LNN": "tab:orange",
    "M-LNN-KAN": "tab:cyan",
    "P-LNN-100K": "tab:purple",
}
MARKERS = {"LM": "o", "M-LNN": "s", "M-LNN-KAN": "D", "P-LNN-100K": "^"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="spy")
    args = ap.parse_args()
    key = args.data

    df = load_prices(key)
    mdf = pd.read_csv(ROOT / "results" / f"{key}_episode_methods.csv",
                      parse_dates=["t2", "pred_tc", "realised_peak"])
    episodes = list(mdf["episode"].unique())
    n = len(episodes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, ep in zip(axes, episodes):
        g = mdf[mdf.episode == ep]
        t2 = g["t2"].iloc[0]
        peak_date = g["realised_peak"].iloc[0]
        peak_price = g["realised_price"].iloc[0]
        lo = t2 - pd.Timedelta(days=380)
        hi = max(peak_date, g["pred_tc"].max()) + pd.Timedelta(days=90)
        seg = df[(df.Date >= lo) & (df.Date <= hi)]
        ax.plot(seg["Date"], seg["Close"], color="black", lw=0.9)
        ax.axvline(t2, color="red", ls="-.", lw=1.2)
        ax.plot(peak_date, peak_price, marker="*", color="black", ms=17, zorder=6,
                label="realised peak")
        for _, r in g.iterrows():
            ax.plot(r["pred_tc"], r["pred_price_c"], marker=MARKERS[r["method"]],
                    color=COLORS[r["method"]], ms=9, mec="black", mew=0.5, zorder=7,
                    label=r["method"])
        ax.set_title(f"flag {ep}\n$t_2$={t2.date()}  peak {peak_date.date()} @ {peak_price:.1f}",
                     fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), loc="lower right", ncols=5, fontsize=10)
    fig.suptitle(f"{DATASETS[key]['label']}: each method's predicted ($t_c$, price$_c$) per flagged episode "
                 "(red dash-dot = prediction date $t_2$; star = realised peak)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = ROOT / "results" / f"fig_{key}_episode_methods.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
