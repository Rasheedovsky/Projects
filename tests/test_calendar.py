"""NYSE calendar wrapper: sessions, early closes, in the 2008-2021 window."""
import pandas as pd

from spydt.data.calendar import nyse_sessions


def test_session_count_2008_2021():
    s = nyse_sessions("2008-01-01", "2021-12-31")
    # 14 years of ~252 sessions
    assert 3480 <= len(s) <= 3560


def test_known_early_closes_detected():
    s = nyse_sessions("2008-01-01", "2021-12-31")
    early = set(s.index[s["is_early_close"]].strftime("%Y-%m-%d"))
    for d in ["2008-07-03", "2009-11-27", "2015-12-24", "2019-07-03", "2020-11-27"]:
        assert d in early, f"{d} should be an early close"
    for d in ["2010-05-06", "2020-03-16", "2015-06-15"]:
        assert d not in early, f"{d} is a full session"


def test_full_sessions_have_1600_close():
    s = nyse_sessions("2015-01-01", "2015-12-31")
    full = s[~s["is_early_close"]]
    assert (full["close_et"].dt.strftime("%H:%M") == "16:00").all()


def test_early_close_is_1300():
    s = nyse_sessions("2019-01-01", "2019-12-31")
    early = s[s["is_early_close"]]
    assert len(early) > 0
    assert (early["close_et"].dt.strftime("%H:%M") == "13:00").all()


def test_holidays_absent():
    s = nyse_sessions("2020-01-01", "2020-12-31")
    dates = set(s.index.strftime("%Y-%m-%d"))
    for d in ["2020-01-01", "2020-07-03", "2020-12-25"]:  # New Year, Jul-4 observed, Christmas
        assert d not in dates
