"""Phase-0 audit checks. Every check returns a plain dict/DataFrame so the
report renderer and the tests consume identical objects.

Checks operate on ``days``: dict[iso_date_str, per-day RTH frame in ET].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spydt.data.sessions import session_stats

# Approximate SPY average daily share volume by year (order-of-magnitude
# reference for unit determination only — the two candidate multipliers are
# two orders of magnitude apart, so rough constants are sufficient).
SPY_REF_DAILY_SHARES: dict[int, float] = {
    2008: 280e6, 2009: 240e6, 2010: 190e6, 2011: 180e6, 2012: 130e6,
    2013: 110e6, 2014: 100e6, 2015: 110e6, 2016: 90e6, 2017: 65e6,
    2018: 85e6, 2019: 60e6, 2020: 85e6, 2021: 65e6,
}

# SPY distributions go ex-dividend on (approximately) the third Friday of
# Mar/Jun/Sep/Dec throughout 2008-2021.
_EXDIV_MONTHS = (3, 6, 9, 12)


def third_fridays(years: range) -> set[str]:
    out: set[str] = set()
    for y in years:
        for m in _EXDIV_MONTHS:
            d = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
            fridays = d[(d.weekday == 4) & (d.month == m)]
            out.add(fridays[2].strftime("%Y-%m-%d"))
    return out


def classify_days(
    days: dict[str, pd.DataFrame],
    half_days: set[str],
    *,
    max_missing_frac_morning: float = 0.10,
) -> pd.DataFrame:
    """Apply the ex-ante exclusion rules; one row per day with reason."""
    rows = []
    for date, frame in sorted(days.items()):
        if date in half_days:
            rows.append((date, False, "half_day"))
            continue
        st = session_stats(frame)
        if st["missing_frac_morning"] > max_missing_frac_morning:
            rows.append((date, False, "morning_gaps"))
        elif not st["decision_leg_intact"]:
            rows.append((date, False, "decision_leg"))
        else:
            rows.append((date, True, ""))
    return pd.DataFrame(rows, columns=["date", "included", "reason"]).set_index("date")


def missing_minute_census(days: dict[str, pd.DataFrame], half_days: set[str]) -> pd.DataFrame:
    """Missing minutes per year over full (non-half) days."""
    rows = []
    for date, frame in days.items():
        if date in half_days:
            continue
        st = session_stats(frame)
        rows.append((int(date[:4]), st["missing_minutes"]))
    df = pd.DataFrame(rows, columns=["year", "missing"])
    return df.groupby("year").agg(
        days=("missing", "size"),
        total_missing=("missing", "sum"),
        max_missing=("missing", "max"),
        days_with_gaps=("missing", lambda s: int((s > 0).sum())),
    )


def check_outliers(days: dict[str, pd.DataFrame], *, sigma: float = 20.0) -> dict:
    """Price prints > sigma robust-z on 1-min log returns; volume sanity."""
    n_price = 0
    n_neg = 0
    n_zero = 0
    worst: list[tuple[str, float]] = []
    for date, frame in days.items():
        r = np.log(frame["close"]).diff().dropna()
        mad = (r - r.median()).abs().median()
        scale = max(1.4826 * mad, 1e-8)
        z = (r - r.median()).abs() / scale
        n = int((z > sigma).sum())
        n_price += n
        if n:
            worst.append((date, float(z.max())))
        n_neg += int((frame["volume"] < 0).sum())
        n_zero += int((frame["volume"] == 0).sum())
    worst.sort(key=lambda t: -t[1])
    return {
        "n_price_outliers": n_price,
        "n_negative_volume": n_neg,
        "n_zero_volume": n_zero,
        "worst_days": worst[:10],
    }


def check_adjustment(days: dict[str, pd.DataFrame]) -> dict:
    """Unadjusted prices show systematic overnight drops on ex-div dates.

    Compares overnight log return (prev close -> open) on candidate ex-div
    dates (third Fridays of Mar/Jun/Sep/Dec, shifted to the prior present
    session on holiday collisions, e.g. Good Friday 2008) vs all other days.
    Supporting evidence only — ``check_price_level_anchor`` is decisive.
    """
    dates = sorted(days)
    years = range(int(dates[0][:4]), int(dates[-1][:4]) + 1)
    exdiv_raw = third_fridays(years)
    present = set(dates)
    exdiv: set[str] = set()
    for d in exdiv_raw:
        if d in present:
            exdiv.add(d)
        else:  # holiday collision: ex-div shifts to the prior session
            before = [x for x in dates if x < d]
            if before:
                exdiv.add(before[-1])
    on_ex, on_other = [], []
    for prev, cur in zip(dates[:-1], dates[1:]):
        r_on = float(np.log(days[cur]["open"].iloc[0] / days[prev]["close"].iloc[-1]))
        (on_ex if cur in exdiv else on_other).append(r_on)
    if not on_ex:
        return {"verdict": "no_exdiv_dates_in_sample", "exdiv_minus_other_bp": 0.0}
    diff_bp = (np.mean(on_ex) - np.mean(on_other)) * 1e4
    med_diff_bp = (np.median(on_ex) - np.median(on_other)) * 1e4
    verdict = "unadjusted" if diff_bp < -20 else "adjusted_or_no_signal"
    return {
        "verdict": verdict,
        "exdiv_minus_other_bp": float(diff_bp),
        "exdiv_minus_other_median_bp": float(med_diff_bp),
        "n_exdiv": len(on_ex),
        "mean_exdiv_on_bp": float(np.mean(on_ex) * 1e4),
        "mean_other_on_bp": float(np.mean(on_other) * 1e4),
    }


# SPY's actual (unadjusted) close hovered in ~120-148 through Jan-Feb 2008. A
# series back-adjusted for 2008-2021 dividends would sit ~20-25% lower
# (~95-115). The two hypotheses are separated by far more than price noise.
_ANCHOR_WINDOW_END = "2008-03-01"
_UNADJUSTED_MIN = 120.0
_ADJUSTED_MAX = 118.0


def check_price_level_anchor(days: dict[str, pd.DataFrame]) -> dict:
    """Decisive adjustment classification from early-2008 price levels."""
    early = [float(f["close"].median()) for d, f in days.items() if d < _ANCHOR_WINDOW_END]
    if not early:
        return {"verdict": "no_anchor_window", "median_close": float("nan")}
    m = float(np.median(early))
    if m > _UNADJUSTED_MIN:
        verdict = "unadjusted"
    elif m < _ADJUSTED_MAX:
        verdict = "dividend_adjusted"
    else:
        verdict = "ambiguous"
    return {"verdict": verdict, "median_close": m, "n_days": len(early)}


def check_volume_units(
    days: dict[str, pd.DataFrame],
    *,
    reference_daily_shares: dict[int, float] | None = None,
) -> dict:
    """Decide multiplier in {1, 100} by comparing daily sums to references."""
    ref = reference_daily_shares or SPY_REF_DAILY_SHARES
    ratios: dict[int, list[float]] = {}
    for date, frame in days.items():
        y = int(date[:4])
        if y in ref:
            ratios.setdefault(y, []).append(frame["volume"].sum() / ref[y])
    med = float(np.median([np.median(v) for v in ratios.values()]))
    multiplier = 100 if med < 0.1 else 1
    return {"multiplier": multiplier, "median_ratio_to_reference": med}


def label_balance(days: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """y = sign(ln(close_1600 / close_1530)) per year (economics use the
    official close; here the 15:59 bar close proxies the 16:00 print)."""
    rows = []
    for date, frame in days.items():
        t = frame.index.strftime("%H:%M")
        c1530 = frame.loc[t == "15:30", "close"]
        c_last = frame["close"].iloc[-1]
        if c1530.empty:
            continue
        r = float(np.log(c_last / c1530.iloc[0]))
        rows.append((int(date[:4]), r))
    df = pd.DataFrame(rows, columns=["year", "r"])
    return df.groupby("year").agg(
        n=("r", "size"),
        n_up=("r", lambda s: int((s > 0).sum())),
        n_down=("r", lambda s: int((s < 0).sum())),
        n_flat=("r", lambda s: int((s == 0).sum())),
        mean_bp=("r", lambda s: float(s.mean() * 1e4)),
        std_bp=("r", lambda s: float(s.std() * 1e4)),
    )
