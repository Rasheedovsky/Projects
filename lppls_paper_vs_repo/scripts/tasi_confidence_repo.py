"""Bonus: run the reference repo's flagship workflow - the LPPLS confidence
indicator (mp_compute_nested_fits + compute_indicators) - on TASI, so the
comparison also covers what the Boulder package is typically used for.

Output: results/fig_tasi_confidence_repo.png, results/tasi_confidence.csv
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

from lppls.lppls import LPPLS


def main():
    df = pd.read_csv(ROOT / "data" / "TASI_daily_2020_2024.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # focus on the bubble build-up and burst: 2020-10 .. 2022-12
    df = df[(df.Date >= "2020-10-01") & (df.Date <= "2022-12-31")].reset_index(drop=True)
    t = np.arange(len(df), dtype=float)  # trading-day index
    logp = np.log(df["Close"].values)
    obs = np.array([t, logp])

    model = LPPLS(observations=obs)
    t0 = time.time()
    res = model.mp_compute_nested_fits(
        workers=4,
        window_size=120,
        smallest_window_size=30,
        outer_increment=5,
        inner_increment=5,
        max_searches=25,
    )
    elapsed = time.time() - t0
    print(f"nested fits done in {elapsed:.0f}s", flush=True)

    ind = model.compute_indicators(res)
    ind["date"] = [df.loc[int(i), "Date"] for i in ind["time"]]
    ind[["date", "price", "pos_conf", "neg_conf"]].to_csv(ROOT / "results" / "tasi_confidence.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Date"], logp, color="black", lw=1.0, label="TASI ln(price)")
    peak_i = int(np.argmax(logp))
    ax.axvline(df.loc[peak_i, "Date"], color="red", ls="--", lw=1.2, label="realised peak")
    ax2 = ax.twinx()
    ax2.plot(ind["date"], ind["pos_conf"], color="tab:red", alpha=0.7, lw=1.4, label="pos bubble confidence")
    ax2.set_ylabel("LPPLS confidence")
    ax2.set_ylim(0, 1)
    ax.set_ylabel("ln(TASI close)")
    ax.set_title(f"Boulder lppls package: positive-bubble confidence indicator on TASI "
                 f"(nested fits, {elapsed:.0f}s wall-clock)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_tasi_confidence_repo.png", dpi=140)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
