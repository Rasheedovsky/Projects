"""Audit checks tested on synthetic corrupted fixtures with known ground truth."""
import numpy as np
import pandas as pd

from spydt.data.audit import (
    check_adjustment,
    check_outliers,
    check_price_level_anchor,
    check_volume_units,
    classify_days,
    label_balance,
)
from tests.conftest import ET, make_multi_day, make_rth_day


def test_outlier_spike_flagged(clean_day):
    day = clean_day.copy()
    t = day.index[100]
    day.loc[t, ["high", "close"]] = day["close"].median() * 1.5  # absurd print
    res = check_outliers({"2015-06-15": day}, sigma=20.0)
    assert res["n_price_outliers"] >= 1


def test_negative_and_zero_volume_flagged(clean_day):
    day = clean_day.copy()
    day.iloc[10, day.columns.get_loc("volume")] = -5.0
    day.iloc[11, day.columns.get_loc("volume")] = 0.0
    res = check_outliers({"2015-06-15": day}, sigma=20.0)
    assert res["n_negative_volume"] == 1
    assert res["n_zero_volume"] == 1


def test_clean_day_no_outliers(clean_day):
    res = check_outliers({"2015-06-15": clean_day}, sigma=20.0)
    assert res["n_price_outliers"] == 0
    assert res["n_negative_volume"] == 0


def test_classify_days_exclusions():
    good = make_rth_day("2015-06-15")
    broken_morning = make_rth_day("2015-06-16", seed=1).drop(
        make_rth_day("2015-06-16", seed=1).index[5:17]
    )  # >10% morning missing
    broken_leg = make_rth_day("2015-06-17", seed=2).drop(
        pd.Timestamp("2015-06-17 15:40", tz=ET)
    )
    days = {"2015-06-15": good, "2015-06-16": broken_morning, "2015-06-17": broken_leg}
    half_days = set()
    cls = classify_days(days, half_days, max_missing_frac_morning=0.10)
    assert cls.loc["2015-06-15", "included"]
    assert not cls.loc["2015-06-16", "included"]
    assert cls.loc["2015-06-16", "reason"] == "morning_gaps"
    assert not cls.loc["2015-06-17", "included"]
    assert cls.loc["2015-06-17", "reason"] == "decision_leg"


def test_classify_days_half_day_dropped():
    hd = make_rth_day("2015-12-24", n_minutes=210)  # 09:30-13:00
    cls = classify_days({"2015-12-24": hd}, {"2015-12-24"}, max_missing_frac_morning=0.10)
    assert not cls.loc["2015-12-24", "included"]
    assert cls.loc["2015-12-24", "reason"] == "half_day"


def test_label_balance_signs():
    up = make_rth_day("2015-06-15")
    up.loc[up.index[-1], "close"] = float(up["close"].iloc[-31]) * 1.01  # last-30m up
    down = make_rth_day("2015-06-16", seed=1)
    down.loc[down.index[-1], "close"] = float(down["close"].iloc[-31]) * 0.99
    lb = label_balance({"2015-06-15": up, "2015-06-16": down})
    assert lb.loc[2015, "n_up"] == 1
    assert lb.loc[2015, "n_down"] == 1


def _chained_days(exdiv_drop: float) -> dict:
    """Days whose opens chain exactly to the prior close, so overnight
    returns are 0 except a controlled drop on scheduled ex-div dates."""
    dates = pd.bdate_range("2015-01-05", "2015-12-31").strftime("%Y-%m-%d").tolist()
    exdiv = {"2015-03-20", "2015-06-19", "2015-09-18", "2015-12-18"}  # 3rd Fridays
    days, price = {}, 100.0
    for i, d in enumerate(dates):
        if d in exdiv:
            price *= 1.0 - exdiv_drop
        day = make_rth_day(d, base_price=price, seed=i, open_spike=False)
        days[d] = day
        price = float(day["close"].iloc[-1])
    return days


def test_adjustment_detects_exdiv_drops():
    res = check_adjustment(_chained_days(exdiv_drop=0.006))
    assert res["verdict"] == "unadjusted"
    assert res["exdiv_minus_other_bp"] < -30


def test_adjustment_no_drops_means_adjusted():
    res = check_adjustment(_chained_days(exdiv_drop=0.0))
    assert res["verdict"] == "adjusted_or_no_signal"


def test_price_anchor_separates_hypotheses():
    raw_level = {"2008-01-22": make_rth_day("2008-01-22", base_price=131.0)}
    adj_level = {"2008-01-22": make_rth_day("2008-01-22", base_price=104.0)}
    assert check_price_level_anchor(raw_level)["verdict"] == "unadjusted"
    assert check_price_level_anchor(adj_level)["verdict"] == "dividend_adjusted"


def test_volume_units_lots_vs_shares():
    day = make_rth_day("2015-06-15")
    # reference ~120M shares/day in 2015; synthetic day sums to ~1.2M "units"
    scale = 1.2e6 / day["volume"].sum()
    day["volume"] *= scale
    res = check_volume_units({"2015-06-15": day}, reference_daily_shares={2015: 120e6})
    assert res["multiplier"] == 100
    day2 = day.copy()
    day2["volume"] *= 100
    res2 = check_volume_units({"2015-06-15": day2}, reference_daily_shares={2015: 120e6})
    assert res2["multiplier"] == 1
