"""NYSE trading calendar wrapper (sessions, closes, early closes) in ET."""
from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

ET = "America/New_York"


@lru_cache(maxsize=8)
def _xnys():
    return xcals.get_calendar("XNYS", start="2007-01-01", end="2022-12-31")


def nyse_sessions(start: str, end: str) -> pd.DataFrame:
    """Sessions between start and end (inclusive), indexed by session date.

    Columns:
        close_et       tz-aware ET close timestamp
        is_early_close True where the session closes before 16:00 ET
    """
    cal = _xnys()
    sessions = cal.sessions_in_range(start, end)
    closes = cal.closes.loc[sessions].dt.tz_convert(ET)
    closes.index = pd.DatetimeIndex(sessions, name="session")
    is_early = closes.dt.strftime("%H:%M") != "16:00"
    return pd.DataFrame({"close_et": closes, "is_early_close": is_early})
