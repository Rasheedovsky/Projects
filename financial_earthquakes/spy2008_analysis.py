"""
Real-data run: SPY 5-minute bars, 22 Jan - 8 Dec 2008 (the Global Financial
Crisis), from the user's spy_5min_2008_2021_cleaned.csv (Google Drive; the
Drive connector's size limit truncates the 13-year file to its first ~1 MB,
which happens to be almost exactly the 2008 calendar year - stated plainly
everywhere results are reported).

1 period = one 5-minute bar (~81 bars per trading day).  Events = 5-minute
returns beyond the 99% tail quantile (~182 per tail over 224 days).
Everything else (models, tests, walk-forward) is the same machinery as the
AA analysis: see etas.py / overnight.py / kan_pin.py.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_analysis as RA
from etas import ETASModel, extract_events
from overnight import (OvernightETAS, extract_gap_stream,
                       split_overnight_intraday)
from run_analysis import C, HERE, set_ticker


def load_spy(csv_path: str = "SPY2008_5m.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    r_intra, gaps_all = split_overnight_intraday(df)
    falls = extract_events(r_intra, "fall", 0.99)
    runs = extract_events(r_intra, "run", 0.99)
    src = extract_gap_stream(gaps_all, 0.75, "abs")
    return df, r_intra, gaps_all, falls, runs, src


def figure_overview(df, r_intra, falls, runs, src):
    """Price with extreme events + model-free gap-trigger evidence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    px = df["Close"].values
    ax1.plot(np.arange(len(px)), px, color=C["ink2"], lw=1.0)
    fi = (falls.times - 1).astype(int)
    ri = (runs.times - 1).astype(int)
    ax1.scatter(fi, px[fi], marker="v", s=16, color=C["fall"], zorder=3,
                label=f"extreme 5-min falls (n={falls.n})")
    ax1.scatter(ri, px[ri], marker="^", s=16, color=C["run"], zorder=3,
                label=f"extreme 5-min runs (n={runs.n})")
    ticks = np.linspace(0, len(px) - 1, 7).astype(int)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([df.index[i].strftime("%b %y") for i in ticks])
    ax1.set_ylabel("SPY ($)")
    ax1.set_title("SPY 2008 (5-minute bars) and its extreme-return events")
    ax1.legend(fontsize=8, loc="lower left")

    # events per day by hour-of-day, big-gap days vs quiet days
    day_of = pd.Series(r_intra.index.date)
    barpos = day_of.groupby(day_of.values).cumcount() + 1
    hour = ((barpos - 1) // 12) + 1              # 12 five-min bars per hour
    is_event = np.zeros(len(r_intra), bool)
    is_event[(falls.times - 1).astype(int)] = True
    is_event[(runs.times - 1).astype(int)] = True
    gap_days = set(pd.DatetimeIndex(src.dates).date)
    on_trig = day_of.isin(gap_days).values
    n_t, n_q = len(gap_days), len(set(day_of)) - len(gap_days)
    hours = np.arange(1, 8)
    f_t = [np.sum(is_event & on_trig & (hour == h)) / n_t for h in hours]
    f_q = [np.sum(is_event & ~on_trig & (hour == h)) / n_q for h in hours]
    w = 0.38
    ax2.bar(hours - w / 2, f_t, w, color=C["fall"],
            label=f"big-gap days (n={n_t})")
    ax2.bar(hours + w / 2, f_q, w, color=C["muted"],
            label=f"quiet-gap days (n={n_q})")
    ax2.set_xlabel("hour of the trading day")
    ax2.set_ylabel("extreme 5-min moves per day")
    ax2.set_title("Extremes cluster after big overnight gaps\n(model-free)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RA.FIGS / "fig_spy_overview.png", dpi=150)
    plt.close(fig)
    print(f"saved {RA.FIGS.name}/fig_spy_overview.png")


def conclusion_spy(wf: pd.DataFrame, df: pd.DataFrame) -> dict:
    tot = {m: float((wf[f"oos_{m}"] - wf["oos_poisson"]).sum())
           for m in ("full", "selfonly", "kanpin")}
    wins_self = int((wf["oos_selfonly"] > wf["oos_poisson"]).sum())
    wins_full = int((wf["oos_full"] > wf["oos_poisson"]).sum())
    # crisis window: Sep 15 (Lehman) onward
    lehman = wf["date"] >= pd.Timestamp("2008-09-15")
    tot_crisis = {m: float((wf.loc[lehman, f"oos_{m}"]
                            - wf.loc[lehman, "oos_poisson"]).sum())
                  for m in ("full", "selfonly", "kanpin")}
    out = {"days": int(len(wf)), "oos_total_vs_poisson": tot,
           "oos_since_lehman": tot_crisis,
           "wins_full": wins_full, "wins_selfonly": wins_self}
    print("\n" + "=" * 72)
    print("SPY 2008 CONCLUSION (daily-readjusted models, 5-minute data)")
    print("=" * 72)
    print(f"{len(wf)} daily OOS days; cumulative log-score vs Poisson:")
    print(f"  overnight-trigger MLE {tot['full']:+.1f} | self-only ETAS "
          f"{tot['selfonly']:+.1f} | KAN-PIN {tot['kanpin']:+.1f}")
    print(f"  since Lehman (15 Sep): full {tot_crisis['full']:+.1f} | "
          f"self {tot_crisis['selfonly']:+.1f} | KP {tot_crisis['kanpin']:+.1f}")
    return out


def main():
    set_ticker("SPY2008")
    df, r_intra, gaps_all, falls, runs, src = load_spy()
    figure_overview(df, r_intra, falls, runs, src)
    wf = pd.read_pickle(HERE / "wf_daily_SPY2008.pkl")
    from run_daily_walkforward import figure_daily
    figure_daily(wf, df["Close"])
    out = conclusion_spy(wf, df)
    (HERE / "results_SPY2008.json").write_text(
        json.dumps(out, indent=2, default=float))
    return wf, out


if __name__ == "__main__":
    main()
