"""
Detect flash crashes / informed-trading episodes in SPY 1-minute data,
2008-2021, using the psi-signal framework with the mixture extension.

Per sliding 90-minute window (step 1 minute, never crossing a session):
  psi       = KL(empirical || max-entropy lognormal)  -- fairness deviation
  n_classes = BIC-optimal number of lognormal mixture components fitted to
              the window's log-standardized incomes -- "classes of society"
  sigma     = dispersion (sd of log incomes)

Signal: z = trailing z-score of psi (5-day baseline).  Alert windows
(z >= 3) are clustered into intraday episodes; each episode is labeled
  dispersed  -- sigma above trailing norm: crash/panic-type dislocation
  compressed -- sigma below trailing norm: pinned/suppressed-vol regime
Flash-crash candidates are the dispersed episodes, ranked by peak z.

Run:  python3 detect_flashcrashes.py ../data/spy_1min_2008_2021_cleaned.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psi_signal import income_series, load_minute_bars, rolling_panel, trailing_zscore

STEP = 1
ALERT_Z = 3.0
GAP_TOL = pd.Timedelta(minutes=5)
BASELINE_DAYS = 5

KNOWN_EVENTS = {
    "2008-09-15": "Lehman Brothers bankruptcy",
    "2008-09-18": "TARP rumor rally (+4% in final hour)",
    "2008-09-29": "TARP vote fails in House (-8.8% day)",
    "2008-10-10": "GFC panic week climax",
    "2008-10-13": "+11.6% rally (bank recapitalization)",
    "2008-10-15": "-9% day",
    "2008-10-24": "global crash morning",
    "2008-10-28": "+10.8% rally day",
    "2008-11-13": "intraday reversal day (+6.9%)",
    "2008-11-20": "GFC closing-hour plunge to cycle low",
    "2008-12-16": "FOMC cuts to zero (ZIRP)",
    "2009-03-09": "bear market bottom",
    "2010-04-27": "Greece/Portugal downgrades",
    "2010-05-06": "FLASH CRASH",
    "2010-05-20": "Euro-crisis selloff",
    "2010-06-04": "payrolls miss",
    "2011-08-04": "debt-ceiling selloff (-4.8%)",
    "2011-08-08": "S&P downgrade aftermath (-6.7%)",
    "2011-08-09": "FOMC reversal rally",
    "2011-08-10": "downgrade-week volatility",
    "2011-08-18": "-4.5% day",
    "2011-10-04": "final-hour melt-up (+4% in 45 min)",
    "2013-04-23": "AP TWITTER-HACK FLASH CRASH (13:07)",
    "2013-06-20": "taper tantrum",
    "2014-10-15": "Treasury flash rally / equity swing",
    "2015-08-24": "ETF FLASH CRASH at the open",
    "2015-08-25": "failed rebound, late plunge",
    "2016-06-24": "Brexit",
    "2018-02-02": "volmageddon eve (-2.1%)",
    "2018-02-05": "VOLMAGEDDON (XIV collapse, 15:00)",
    "2018-02-06": "volmageddon aftermath",
    "2018-02-08": "-3.8% day",
    "2018-10-10": "October 2018 break (-3.3%)",
    "2018-12-24": "Christmas Eve low",
    "2019-08-05": "yuan devaluation",
    "2019-08-14": "yield-curve inversion (-2.9%)",
    "2020-02-24": "COVID selloff begins",
    "2020-02-27": "COVID -4.4% day",
    "2020-02-28": "COVID week-1 climax",
    "2020-03-09": "COVID CIRCUIT BREAKER #1 (open)",
    "2020-03-12": "COVID CIRCUIT BREAKER #2 (-9.5% day)",
    "2020-03-16": "COVID CIRCUIT BREAKER #3 (-12% day)",
    "2020-03-18": "COVID intraday halt (~13:00)",
    "2020-03-23": "COVID bottom (Fed unlimited QE)",
    "2020-06-11": "-5.9% second-wave scare",
    "2020-09-03": "tech-led -3.5% day",
    "2021-01-27": "GameStop squeeze / market -2.6%",
}

FLASH_CRASH_STRICT = ["2010-05-06", "2013-04-23", "2015-08-24", "2018-02-05",
                      "2014-10-15", "2020-03-09", "2020-03-12", "2020-03-16",
                      "2020-03-18"]


def episodes_from_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster alert windows (z >= ALERT_Z) into intraday episodes."""
    alerts = df[df["z"] >= ALERT_Z]
    rows = []
    for day, g in alerts.groupby(alerts.index.normalize()):
        start = prev = None
        members = []
        chunks = []
        for t in g.index:
            if prev is not None and t - prev > GAP_TOL:
                chunks.append(members)
                members = []
            members.append(t)
            prev = t
        chunks.append(members)
        for member_times in chunks:
            sub = df.loc[member_times]
            peak_t = sub["z"].idxmax()
            rows.append({
                "date": day.date(),
                "start": member_times[0].time(),
                "end": member_times[-1].time(),
                "n_alert_windows": len(sub),
                "peak_z": sub["z"].max(),
                "peak_time": peak_t.time(),
                "peak_classes": int(df.loc[peak_t, "n_classes"]),
                "share_multiclass": (sub["n_classes"] >= 2).mean(),
                "sigma_ratio": df.loc[peak_t, "sigma"] / df.loc[peak_t, "sigma_med"],
                "event": KNOWN_EVENTS.get(str(day.date()), ""),
            })
    ep = pd.DataFrame(rows)
    ep["type"] = np.where(ep["sigma_ratio"] >= 1.5, "dispersed",
                          np.where(ep["sigma_ratio"] <= 0.67, "compressed", "mixed"))
    return ep.sort_values("peak_z", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--measure", default="gross_return")
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--outdir", default="results_full")
    args = ap.parse_args()
    per_day = 390 - args.window + 1
    os.makedirs(args.outdir, exist_ok=True)

    bars = load_minute_bars(args.csv)
    print(f"Loaded {len(bars):,} bars: {bars.index[0].date()} .. {bars.index[-1].date()}")

    income = income_series(bars, args.measure)
    panel = rolling_panel(income, window=args.window, step=STEP)
    print(f"panel: {len(panel):,} windows "
          f"({len(panel) / panel.index.normalize().nunique():.0f}/day)")

    panel["z"] = trailing_zscore(panel["psi"], baseline=BASELINE_DAYS * per_day,
                                 min_periods=2 * per_day)
    panel["sigma_med"] = (panel["sigma"].shift(1)
                          .rolling(BASELINE_DAYS * per_day, min_periods=2 * per_day)
                          .median())
    panel.to_csv(os.path.join(args.outdir, f"panel_{args.measure}.csv.gz"),
                 compression="gzip")

    # class statistics
    scored = panel.dropna(subset=["z"])
    frac_multi = (scored["n_classes"] >= 2).mean()
    frac_multi_alert = (scored.loc[scored["z"] >= ALERT_Z, "n_classes"] >= 2).mean()
    frac_multi_calm = (scored.loc[scored["z"].abs() < 1, "n_classes"] >= 2).mean()
    print(f"\nn_classes >= 2: overall {frac_multi:.1%}, "
          f"alert windows {frac_multi_alert:.1%}, calm windows {frac_multi_calm:.1%}")
    print("n_classes distribution:",
          dict(scored["n_classes"].value_counts(normalize=True).round(3)))

    ep = episodes_from_alerts(scored)
    ep.to_csv(os.path.join(args.outdir, f"episodes_{args.measure}.csv"), index=False)

    disp = ep[ep["type"] == "dispersed"]
    print(f"\nepisodes: {len(ep)} total, {len(disp)} dispersed (crash-type)")
    print("\n=== TOP 30 DISPERSED (flash-crash-type) EPISODES, 2008-2021 ===")
    cols = ["date", "start", "peak_time", "peak_z", "n_alert_windows",
            "peak_classes", "share_multiclass", "event"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(disp.head(30)[cols].round(2).to_string(index=False))

    hits = {d: (disp["date"].astype(str) == d).any() for d in FLASH_CRASH_STRICT}
    print("\nstrict flash-crash catalog coverage (dispersed episode on the day):")
    for d, h in hits.items():
        print(f"  {d} {KNOWN_EVENTS[d]:45s} {'DETECTED' if h else 'missed'}")

    annotated = disp.head(40)["event"].ne("").mean()
    print(f"\nshare of top-40 dispersed episodes on known-event days: {annotated:.0%}")

    plot_timeline(scored, os.path.join(args.outdir, "timeline_2008_2021.png"))
    plot_zooms(bars, scored, os.path.join(args.outdir, "flashcrash_zooms.png"))
    print(f"\nresults in {args.outdir}/")


def plot_timeline(panel, path):
    daily = panel.groupby(panel.index.normalize()).agg(
        z=("z", "max"), multi=("n_classes", lambda s: (s >= 2).mean()))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily["z"], lw=0.5, color="#444444")
    ax.axhline(ALERT_Z, color="red", ls="--", lw=0.8)
    for d in FLASH_CRASH_STRICT:
        ts = pd.Timestamp(d)
        if ts in daily.index:
            ax.plot(ts, daily.loc[ts, "z"], "v", color="red", ms=7)
    ax.set_title("Daily max psi z-score (gross-return incomes, 90-min windows) — "
                 "red markers: canonical flash-crash days")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_zooms(bars, panel, path):
    days = ["2010-05-06", "2013-04-23", "2015-08-24", "2018-02-05"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, d in zip(axes.ravel(), days):
        day = pd.Timestamp(d)
        px = bars.loc[d, "close"]
        sub = panel.loc[d]
        ax.plot(px.index, px, lw=0.9, color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(sub.index, sub["z"], lw=0.9, color="#d62728")
        ax2.axhline(ALERT_Z, color="red", ls="--", lw=0.7)
        both = sub[(sub["z"] >= ALERT_Z) & (sub["n_classes"] >= 2)]
        for t in both.index:
            ax2.axvspan(t - pd.Timedelta("30s"), t + pd.Timedelta("30s"),
                        color="orange", alpha=0.25, lw=0)
        ax.set_title(f"{d} — {KNOWN_EVENTS.get(d, '')}", fontsize=10)
        ax.set_ylabel("price", color="#1f77b4")
        ax2.set_ylabel("psi z", color="#d62728")
    fig.suptitle("price (blue), psi z (red), alert windows with ≥2 classes (orange)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
