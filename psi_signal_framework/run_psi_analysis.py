"""
Run the psi-signal analysis on real 1-minute OHLCV data (e.g. SPY).

Usage:
    python3 run_psi_analysis.py path/to/spy_1min.csv [--window 60] [--bins 12]
                                [--baseline-days 5] [--outdir output]

Produces, in --outdir:
    psi_results_<measure>.csv   timestamp, psi, z, signal for each measure
    psi_overview.png            price + z-scores over the full sample
    psi_flashcrash.png          zoom on 2010-05-06 (only if present in data)
    flagged_windows.csv         all windows with |z| >= 2, both measures

The CSV loader accepts flexible column names (datetime/timestamp or
date + time; open/high/low/close/volume, case-insensitive).  Timestamps are
assumed to be US/Eastern regular trading hours; bars outside 09:30-16:00
are dropped.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psi_signal import analyze, load_minute_bars

MEASURES = ("dollar_volume", "price")
FLASH_CRASH_DAY = pd.Timestamp("2010-05-06")


def main():
    ap = argparse.ArgumentParser(description="Psi-signal analysis of 1-min bars")
    ap.add_argument("csv", help="path to 1-minute OHLCV CSV")
    ap.add_argument("--window", type=int, default=60, help="window length in minutes")
    ap.add_argument("--bins", type=int, default=12, help="histogram bins for KL")
    ap.add_argument("--baseline-days", type=int, default=5,
                    help="trailing baseline length in trading days")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bars = load_minute_bars(args.csv)
    print(f"Loaded {len(bars):,} minute bars: {bars.index[0]} .. {bars.index[-1]}")

    baseline = args.baseline_days * 390
    results = {}
    for measure in MEASURES:
        res = analyze(bars, measure=measure, window=args.window, bins=args.bins,
                      baseline=baseline, min_periods=max(390, baseline // 3))
        res.to_csv(os.path.join(args.outdir, f"psi_results_{measure}.csv"))
        results[measure] = res
        n_alert = int((res["z"].abs() >= 2).sum())
        print(f"[{measure}] windows: {len(res):,}  alerts (|z|>=2): {n_alert:,}  "
              f"max |z|: {res['z'].abs().max():.1f}")

    # hourly mode: each clock hour as one population of ~60 minute-incomes
    hourly = analyze(bars, measure="dollar_volume", mode="hourly",
                     baseline=7 * args.baseline_days,
                     min_periods=max(7, 7 * args.baseline_days // 3))
    hourly.to_csv(os.path.join(args.outdir, "psi_results_dollar_volume_hourly.csv"))

    flagged = pd.concat(
        {m: r.loc[r["z"].abs() >= 2, ["psi", "z"]] for m, r in results.items()},
        names=["measure", "timestamp"],
    ).sort_values("z", key=lambda s: s.abs(), ascending=False)
    flagged.to_csv(os.path.join(args.outdir, "flagged_windows.csv"))
    print(f"\nTop 10 most dislocated windows:\n{flagged.head(10)}")

    plot_overview(bars, results, os.path.join(args.outdir, "psi_overview.png"))

    days = pd.DatetimeIndex(bars.index.normalize().unique())
    if FLASH_CRASH_DAY in days:
        plot_day(bars, results, hourly, FLASH_CRASH_DAY,
                 os.path.join(args.outdir, "psi_flashcrash.png"))
        report_flash_crash(results, hourly)
    print(f"\nResults written to {args.outdir}/")


def plot_overview(bars, results, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(bars.index, bars["close"], lw=0.4)
    axes[0].set_title("Close price")
    for ax, m in zip(axes[1:], MEASURES):
        z = results[m]["z"]
        ax.plot(z.index, z, lw=0.4, color="#444444")
        ax.axhline(2, color="red", ls="--", lw=0.8)
        ax.axhline(-2, color="red", ls="--", lw=0.8)
        ax.set_title(f"psi z-score — {m}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_day(bars, results, hourly, day, path):
    lo, hi = day + pd.Timedelta("9h30min"), day + pd.Timedelta("16h")
    sel = slice(lo, hi)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(bars.loc[sel].index, bars.loc[sel, "close"], lw=0.9)
    axes[0].set_title(f"{day.date()} — close price")

    for m, color in (("dollar_volume", "#d62728"), ("price", "#2ca02c")):
        z = results[m]["z"].loc[sel]
        axes[1].plot(z.index, z, lw=1.0, color=color, label=m)
    axes[1].axhline(2, color="red", ls="--", lw=0.8)
    axes[1].legend()
    axes[1].set_title("psi z-score, rolling 60-min windows")

    zh = hourly["z"].loc[sel]
    axes[2].bar(zh.index, zh, width=pd.Timedelta("50min"), align="edge",
                color=np.where(zh.abs() >= 2, "#d62728", "#999999"))
    axes[2].axhline(2, color="red", ls="--", lw=0.8)
    axes[2].set_title("psi z-score, non-overlapping hours (dollar-volume incomes)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report_flash_crash(results, hourly):
    start = FLASH_CRASH_DAY + pd.Timedelta("14h30min")
    end = FLASH_CRASH_DAY + pd.Timedelta("15h30min")
    print("\n=== Flash crash day (2010-05-06) ===")
    for m, res in results.items():
        win = res.loc[start:end, "z"].dropna()
        if len(win):
            print(f"[{m}] peak |z| 14:30-15:30: {win.abs().max():.1f}  "
                  f"first |z|>=2: {next(iter(win[win.abs() >= 2].index), 'none')}")
    ch = hourly.loc[hourly.index == FLASH_CRASH_DAY + pd.Timedelta("14h"), "z"]
    if len(ch):
        print(f"[hourly dollar-volume] z of the 14:00-15:00 hour: {float(ch.iloc[0]):.1f}")


if __name__ == "__main__":
    main()
