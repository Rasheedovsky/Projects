"""
DAILY walk-forward for the overnight-trigger model + the directional
conclusion ("is it going up or down lately, per the model?").

Re-estimates every trading day after a 30-day burn-in (warm-started),
scores each day out of sample, and turns the two fitted tails (falls and
runs) into a daily directional gauge computed AT EACH OPEN, when the
overnight gap is known.

Run:  python3 run_daily_walkforward.py        (uses cached wf_daily.pkl if
                                               present; delete it to refit)
Produces figures/fig10_daily_wf.png and a printed conclusion; key numbers
are appended to results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etas import ETASModel, extract_events, load_yfinance_csv
from overnight import (extract_gap_stream, split_overnight_intraday,
                       walk_forward_daily)
import run_analysis as RA
from run_analysis import C, HERE
import matplotlib.pyplot as plt

def _cache_for(csv_path: str) -> Path:
    stem = Path(csv_path).stem.split("_")[0].upper()
    return HERE / ("wf_daily.pkl" if stem == "AA" else f"wf_daily_{stem}.pkl")


CACHE = HERE / "wf_daily.pkl"


def get_daily_wf(csv_path: str = "AA_h.csv", use_cache: bool = True):
    global CACHE
    CACHE = _cache_for(csv_path)
    df = load_yfinance_csv(csv_path)
    r_intra, gaps_all = split_overnight_intraday(df)
    falls = extract_events(r_intra, "fall", 0.95)
    runs = extract_events(r_intra, "run", 0.95)
    src = extract_gap_stream(gaps_all, 0.75, "abs")
    if use_cache and CACHE.exists():
        wf = pd.read_pickle(CACHE)
    else:
        wf = walk_forward_daily(falls, runs, src, gaps_all, start_days=30)
        wf.to_pickle(CACHE)
    return df, r_intra, gaps_all, falls, runs, src, wf


def figure_daily(wf: pd.DataFrame, close_h: pd.Series):
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2)
    x = np.arange(len(wf))
    dates = wf["date"].dt.strftime("%d %b %y")
    ticks = np.linspace(0, len(wf) - 1, 6).astype(int)

    # (a) cumulative OOS advantage over Poisson
    ax = fig.add_subplot(gs[0, 0])
    for col, name, colr in (("oos_full", "overnight-trigger MLE", C["fall"]),
                            ("oos_selfonly", "self-only ETAS", C["yellow"]),
                            ("oos_kanpin", "overnight KAN-PIN", C["run"])):
        ax.plot(x, (wf[col] - wf["oos_poisson"]).cumsum(), color=colr,
                lw=1.8, label=name)
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xticks(ticks); ax.set_xticklabels(dates.iloc[ticks], fontsize=8)
    ax.set_ylabel("cumulative OOS log-score minus Poisson")
    ax.set_title("Daily re-estimation: cumulative out-of-sample skill")
    ax.legend(fontsize=8)

    # (b) daily event probabilities at the open (gap known)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, wf["p_fall"], color=C["fall"], lw=1.3,
            label="P(extreme fall today), at the open")
    ax.plot(x, wf["p_run"], color=C["run"], lw=1.3,
            label="P(extreme run today), at the open")
    rf = np.flatnonzero(wf["realized_fall"].values)
    rr = np.flatnonzero(wf["realized_run"].values)
    ax.scatter(rf, np.full(len(rf), 1.00), marker="v", s=14, color=C["fall"],
               clip_on=False)
    ax.scatter(rr, np.full(len(rr), 1.05), marker="^", s=14, color=C["run"],
               clip_on=False)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(ticks); ax.set_xticklabels(dates.iloc[ticks], fontsize=8)
    ax.set_ylabel("probability")
    ax.set_title("Daily nowcasts (markers = realized extreme days)")
    ax.legend(fontsize=8, loc="center left")

    # (c) parameter paths under daily refits
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, wf["mle_K_o"], color=C["fall"], lw=1.6,
            label="$K_o$ gap trigger (falls), MLE")
    ax.plot(x, wf["kp_K_o"], color=C["run"], lw=1.6, label="$K_o$, KAN-PIN")
    ax.plot(x, wf["mle_K_s"], color=C["muted"], lw=1.3, ls="--",
            label="$K_s$ intraday self, MLE")
    ax.set_xticks(ticks); ax.set_xticklabels(dates.iloc[ticks], fontsize=8)
    ax.set_title("Daily-readjusted parameter paths")
    ax.legend(fontsize=8)

    # (d) the directional gauge: E[runs] - E[falls] per day
    ax = fig.add_subplot(gs[1, 1])
    gauge = wf["e_run"] - wf["e_fall"]
    colors = [C["run"] if v >= 0 else C["fall"] for v in gauge]
    ax.bar(x, gauge, width=1.0, color=colors)
    roll = gauge.rolling(10, min_periods=3).mean()
    ax.plot(x, roll, color=C["ink"], lw=1.6, label="10-day average")
    ax.axhline(0, color=C["ink2"], lw=1)
    # robust limits: a single exploding forecast (exponential gap-leverage
    # extrapolating on an unprecedented gap) must not flatten the panel
    lo, hi = np.quantile(gauge, [0.01, 0.99])
    pad = 0.2 * max(abs(lo), abs(hi), 0.1)
    ax.set_ylim(min(lo, -pad) - pad, max(hi, pad) + pad)
    ax.set_xticks(ticks); ax.set_xticklabels(dates.iloc[ticks], fontsize=8)
    ax.set_ylabel("E[extreme runs] − E[extreme falls], per day")
    ax.set_title("Directional tilt of the model (up-tail minus down-tail)")
    ax.legend(fontsize=8)
    fig.suptitle("Daily walk-forward: overnight-trigger model, re-estimated "
                 "every trading day", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RA.FIGS / "fig10_daily_wf.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig10_daily_wf.png")


def conclusion(wf: pd.DataFrame, close_h: pd.Series) -> dict:
    px = close_h.iloc[-1]
    px_1m = close_h.iloc[-147] if len(close_h) > 147 else close_h.iloc[0]
    px_1w = close_h.iloc[-35] if len(close_h) > 35 else close_h.iloc[0]
    hi_52w = close_h.max()
    tot = {m: float((wf[f"oos_{m}"] - wf["oos_poisson"]).sum())
           for m in ("full", "selfonly", "kanpin")}
    wins = int(((wf["oos_full"] > wf["oos_poisson"])).sum())
    last = wf.iloc[-1]
    tail10 = wf.tail(10)
    gauge10 = float((tail10["e_run"] - tail10["e_fall"]).mean())
    out = {
        "last_close": float(px),
        "chg_1m_pct": float(100 * (px / px_1m - 1)),
        "chg_1w_pct": float(100 * (px / px_1w - 1)),
        "off_high_pct": float(100 * (px / hi_52w - 1)),
        "p_fall_today": float(last["p_fall"]),
        "p_run_today": float(last["p_run"]),
        "gauge_10d": gauge10,
        "oos_total_vs_poisson": tot,
        "days": int(len(wf)), "wins_vs_poisson": wins,
    }
    print("\n" + "=" * 72)
    print("CONCLUSION (daily-readjusted overnight-trigger model)")
    print("=" * 72)
    print(f"Validation: over {len(wf)} daily out-of-sample days, cumulative "
          f"log-score vs Poisson:\n  overnight-MLE {tot['full']:+.1f} | "
          f"self-only {tot['selfonly']:+.1f} | KAN-PIN {tot['kanpin']:+.1f}")
    print(f"\nPrice state: last close ${px:.2f}; {out['chg_1m_pct']:+.1f}% "
          f"over the last month, {out['chg_1w_pct']:+.1f}% over the last "
          f"week, {out['off_high_pct']:.1f}% from the 52-week high.")
    print(f"Model state at the last open: P(extreme fall today) = "
          f"{last['p_fall']:.2f}, P(extreme run today) = {last['p_run']:.2f}.")
    print(f"Directional tilt, last 10 days: E[runs]-E[falls] = "
          f"{gauge10:+.3f} events/day (≈ 0 = no directional edge).")
    return out


def main(csv_path: str = "AA_h.csv"):
    df, r_intra, gaps_all, falls, runs, src, wf = get_daily_wf(csv_path)
    close_h = df["Close"]
    figure_daily(wf, close_h)
    out = conclusion(wf, close_h)
    res_path = HERE / ("results.json" if RA.TICKER == "AA"
                       else f"results_{RA.TICKER}.json")
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["daily_walkforward"] = out
    res_path.write_text(json.dumps(res, indent=2, default=float))
    return wf, out


if __name__ == "__main__":
    import sys
    from run_analysis import set_ticker
    path = sys.argv[1] if len(sys.argv) > 1 else "AA_h.csv"
    stem = Path(path).stem
    set_ticker(stem.split("_")[0].upper() if "_" in stem else stem.upper())
    main(path)
