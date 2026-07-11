"""
Validation of the psi-signal detector on synthetic 1-minute data with an
injected flash crash, mimicking SPY on 2010-05-06 (~9% plunge in ~13 minutes
starting 14:32 ET, 20-40x volume spike, partial recovery within ~25 minutes).

Success criteria:
  1. psi z-score reaches |z| >= 2 during the crash window for the
     dollar-volume (price*volume) measure and/or the price measure.
  2. False-alarm rate outside the crash day stays low (< ~5% of windows;
     a two-sided 2-sigma rule under a normal null allows 4.6%).

Run:  python3 synthetic_flashcrash_test.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psi_signal import analyze

RNG = np.random.default_rng(7)

N_DAYS = 25
MIN_PER_DAY = 390            # 09:30-16:00
CRASH_DAY = 20               # 0-indexed; leaves a clean 5-day post period
CRASH_START_MIN = 302        # 14:32 ET
CRASH_DOWN_LEN = 13          # minutes of plunge
CRASH_REBOUND_LEN = 25       # minutes of recovery
ALERT_Z = 2.0


def u_shape(n=MIN_PER_DAY, amp=1.6):
    """Intraday U-shape multiplier for vol and volume (high at open/close)."""
    t = np.linspace(0, 1, n)
    return 1.0 + amp * ((t - 0.45) ** 2) / 0.25


def make_synthetic():
    dates = pd.bdate_range("2010-04-05", periods=N_DAYS)
    idx, close, volume = [], [], []
    price = 115.0                          # SPY-ish level in spring 2010
    base_sigma = 0.010 / np.sqrt(MIN_PER_DAY)   # ~1% daily vol
    base_vol = 220_000                     # shares per minute

    for d, day in enumerate(dates):
        minutes = pd.date_range(day + pd.Timedelta("9h30min"),
                                periods=MIN_PER_DAY, freq="1min")
        shape = u_shape()
        rets = RNG.normal(0.0, base_sigma * np.sqrt(shape))
        vols = base_vol * shape * RNG.lognormal(0.0, 0.45, MIN_PER_DAY)

        if d == CRASH_DAY:
            s, e = CRASH_START_MIN, CRASH_START_MIN + CRASH_DOWN_LEN
            r = e + CRASH_REBOUND_LEN
            # plunge: -9% total, accelerating, huge two-sided noise
            plunge = -0.09 * np.diff(np.linspace(0, 1, CRASH_DOWN_LEN + 1) ** 1.5)
            rets[s:e] = plunge + RNG.normal(0, 8 * base_sigma, CRASH_DOWN_LEN)
            # rebound: +6% back, still frantic
            bounce = 0.06 * np.diff(np.linspace(0, 1, CRASH_REBOUND_LEN + 1) ** 0.7)
            rets[e:r] = bounce + RNG.normal(0, 5 * base_sigma, CRASH_REBOUND_LEN)
            # volume: 20-40x during plunge, decaying ~8x -> 2x during rebound
            vols[s:e] *= RNG.uniform(20, 40, CRASH_DOWN_LEN)
            vols[e:r] *= np.linspace(8, 2, CRASH_REBOUND_LEN)

        prices = price * np.exp(np.cumsum(rets))
        price = prices[-1]
        idx.append(minutes)
        close.append(prices)
        volume.append(vols)

    bars = pd.DataFrame({
        "close": np.concatenate(close),
        "volume": np.concatenate(volume).round(0),
    }, index=pd.DatetimeIndex(np.concatenate([m.values for m in idx])))
    return bars, dates[CRASH_DAY]


def evaluate(res: pd.DataFrame, crash_day: pd.Timestamp, label: str):
    crash_start = crash_day + pd.Timedelta("14h32min")
    crash_end = crash_start + pd.Timedelta(minutes=CRASH_DOWN_LEN
                                           + CRASH_REBOUND_LEN + 60)
    in_crash = (res.index >= crash_start) & (res.index <= crash_end)
    scored = res["z"].notna()

    z_crash = res.loc[in_crash & scored, "z"]
    hit = bool((z_crash.abs() >= ALERT_Z).any())
    peak = z_crash.abs().max() if len(z_crash) else np.nan

    off = res.loc[scored & (res.index.normalize() != crash_day.normalize())]
    false_rate = (off["z"].abs() >= ALERT_Z).mean() if len(off) else np.nan

    first_alert = None
    alerts = res.loc[in_crash & scored & (res["z"].abs() >= ALERT_Z)]
    if len(alerts):
        first_alert = alerts.index[0]

    print(f"[{label}]")
    print(f"  crash detected (|z|>={ALERT_Z:.0f} in crash window): {hit}")
    print(f"  peak |z| in crash window: {peak:.1f}")
    print(f"  first alert: {first_alert}  (crash begins {crash_start})")
    print(f"  false-alarm rate on non-crash days: {false_rate:.2%}")
    return hit, peak, first_alert, false_rate


def main():
    bars, crash_day = make_synthetic()
    print(f"Synthetic data: {len(bars)} minute bars, "
          f"crash injected on {crash_day.date()} at 14:32\n")

    results = {}
    for measure in ("dollar_volume", "price"):
        res = analyze(bars, measure=measure, window=60, step=1, bins=12,
                      baseline=1950, min_periods=780)
        results[measure] = res
        evaluate(res, crash_day, measure)
        print()

    # hourly mode: the hour as one population of 60 minute-incomes
    res_h = analyze(bars, measure="dollar_volume", mode="hourly",
                    baseline=35, min_periods=21)
    results["dollar_volume_hourly"] = res_h
    crash_hour = crash_day + pd.Timedelta("14h")
    zh = res_h.loc[res_h.index == crash_hour, "z"]
    print(f"[dollar_volume, hourly mode] z of the 14:00-15:00 crash hour: "
          f"{float(zh.iloc[0]):.1f}" if len(zh) else "crash hour missing")

    plot(bars, results, crash_day)
    print("\nfigure saved: synthetic_flashcrash_result.png")


def plot(bars, results, crash_day):
    day0 = crash_day + pd.Timedelta("9h30min")
    day1 = crash_day + pd.Timedelta("16h")
    crash_start = crash_day + pd.Timedelta("14h32min")
    crash_end = crash_start + pd.Timedelta(minutes=CRASH_DOWN_LEN + CRASH_REBOUND_LEN)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)

    ax = axes[0]
    ax.plot(bars.index, bars["close"], lw=0.4, color="#1f77b4")
    ax.axvspan(crash_start, crash_end, color="red", alpha=0.25)
    ax.set_title("Synthetic SPY 1-min close (full sample, crash day shaded)")

    ax = axes[1]
    z = results["dollar_volume"]["z"]
    ax.plot(z.index, z, lw=0.5, color="#444444")
    ax.axhline(2, color="red", ls="--", lw=0.8)
    ax.axhline(-2, color="red", ls="--", lw=0.8)
    ax.axvspan(crash_start, crash_end, color="red", alpha=0.25)
    ax.set_title("psi z-score — dollar-volume incomes (rolling 60-min, full sample)")

    day = slice(day0, day1)
    ax = axes[2]
    ax.plot(bars.loc[day].index, bars.loc[day, "close"], lw=0.8, color="#1f77b4")
    ax.axvspan(crash_start, crash_end, color="red", alpha=0.25)
    ax.set_title(f"Crash day {crash_day.date()} — price")

    ax = axes[3]
    for m, color in (("dollar_volume", "#d62728"), ("price", "#2ca02c")):
        zz = results[m]["z"].loc[day]
        ax.plot(zz.index, zz, lw=1.0, color=color, label=m)
    ax.axhline(2, color="red", ls="--", lw=0.8)
    ax.axvspan(crash_start, crash_end, color="red", alpha=0.25)
    ax.legend()
    ax.set_title("Crash day — psi z-scores (rolling 60-min windows)")

    fig.tight_layout()
    fig.savefig("synthetic_flashcrash_result.png", dpi=130)


if __name__ == "__main__":
    main()
