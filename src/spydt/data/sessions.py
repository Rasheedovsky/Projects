"""Sessionizing tz-aware ET minute bars and per-day completeness stats."""
from __future__ import annotations

import pandas as pd

ET = "America/New_York"

RTH_START = "09:30"
RTH_END_EXCL = "16:00"        # last RTH 1-min bar is labeled 15:59
MORNING_END_EXCL = "11:00"
DECISION_LEG = ("15:25", "16:00")  # must be fully intact (bars 15:25..15:59)


def rth_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars labeled [09:30, 16:00) ET."""
    t = df.index.strftime("%H:%M")
    return df[(t >= RTH_START) & (t < RTH_END_EXCL)]


def sessionize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an RTH frame into per-day frames keyed by ISO date string."""
    return {str(d): g for d, g in rth_slice(df).groupby(df.index.date)}


def _expected_index(date: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{date} {RTH_START}", periods=390, freq="1min", tz=ET)


def session_stats(day: pd.DataFrame) -> dict:
    """Completeness stats for one session frame (full-day expectation).

    Keys: n_minutes, missing_minutes, missing_frac_morning,
    decision_leg_intact, first_bar, last_bar.
    """
    date = day.index[0].strftime("%Y-%m-%d")
    expected = _expected_index(date)
    present = day.index
    missing = expected.difference(present)

    morning = expected[expected.strftime("%H:%M") < MORNING_END_EXCL]
    missing_morning = morning.difference(present)

    leg = expected[
        (expected.strftime("%H:%M") >= DECISION_LEG[0])
        & (expected.strftime("%H:%M") < DECISION_LEG[1])
    ]
    leg_intact = len(leg.difference(present)) == 0

    return {
        "n_minutes": len(day),
        "missing_minutes": len(missing),
        "missing_frac_morning": len(missing_morning) / len(morning),
        "decision_leg_intact": bool(leg_intact),
        "first_bar": day.index[0].strftime("%H:%M"),
        "last_bar": day.index[-1].strftime("%H:%M"),
    }
