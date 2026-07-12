"""Sessionizing, RTH slicing, per-day completeness."""
import pandas as pd

from spydt.data.sessions import rth_slice, session_stats, sessionize
from tests.conftest import ET, make_multi_day, make_rth_day


def test_rth_slice_removes_extended_hours():
    day = make_rth_day("2015-06-15")
    pre = day.iloc[:330].copy()  # premarket 04:00-09:29, no RTH overlap
    pre.index = pd.date_range("2015-06-15 04:00", periods=330, freq="1min", tz=ET)
    both = pd.concat([pre, day]).sort_index()
    out = rth_slice(both)
    assert len(out) == 390
    assert out.index[0].strftime("%H:%M") == "09:30"
    assert out.index[-1].strftime("%H:%M") == "15:59"


def test_sessionize_groups_by_et_date(clean_week):
    days = sessionize(clean_week)
    assert len(days) == 5
    assert all(len(v) == 390 for v in days.values())


def test_session_stats_complete_day(clean_day):
    st = session_stats(clean_day)
    assert st["n_minutes"] == 390
    assert st["missing_minutes"] == 0
    assert st["missing_frac_morning"] == 0.0
    assert st["decision_leg_intact"] is True


def test_session_stats_missing_morning_minutes():
    day = make_rth_day("2015-06-15")
    # drop 12 minutes inside 09:30-11:00 (12/90 > 10%)
    drop = day.index[5:17]
    st = session_stats(day.drop(drop))
    assert st["missing_minutes"] == 12
    assert st["missing_frac_morning"] > 0.10


def test_session_stats_broken_decision_leg():
    day = make_rth_day("2015-06-15")
    t = pd.Timestamp("2015-06-15 15:40", tz=ET)
    st = session_stats(day.drop(t))
    assert st["decision_leg_intact"] is False
