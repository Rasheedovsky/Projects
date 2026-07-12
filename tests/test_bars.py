"""Information-driven bars: tick-rule, thresholds, floor/cap, determinism."""
import numpy as np
import pandas as pd
import pytest

from spydt.bars import (
    build_bars_for_window,
    calibrate_threshold,
    tick_rule_signs,
)


def _window(prices, volumes):
    return pd.DataFrame({"close": prices, "volume": volumes})


def test_tick_rule_signs_carry_forward():
    p = np.array([100.0, 100.5, 100.5, 100.2, 100.2, 100.9])
    s = tick_rule_signs(p)
    assert list(s) == [1, 1, 1, -1, -1, 1]  # zero-change carries prior sign


def test_dollar_bars_boundary_exact():
    # 6 buckets of exactly 100 dollars each, threshold 200 -> bars of 2 buckets
    w = _window([1.0] * 6, [100.0] * 6)
    bars = build_bars_for_window(w, kind="dollar", threshold=200.0, floor_buckets=1)
    assert len(bars) == 3
    assert list(bars["n_buckets"]) == [2, 2, 2]


def test_dollar_bars_floor_binds():
    w = _window([1.0] * 8, [1000.0] * 8)  # each bucket alone crosses threshold
    bars = build_bars_for_window(w, kind="dollar", threshold=500.0, floor_buckets=2)
    assert (bars["n_buckets"] >= 2).all()
    assert bars["floor_triggered"].all()


def test_dollar_bars_cap_binds():
    w = _window([1.0] * 12, [1.0] * 12)  # never crosses threshold
    bars = build_bars_for_window(w, kind="dollar", threshold=1e9, floor_buckets=1,
                                 cap_buckets=4)
    assert len(bars) == 3
    assert bars["cap_triggered"].all()


def test_imbalance_bars_sign_dependence():
    # alternating price moves -> signed flow cancels -> slower bar completion
    up = _window([100 + 0.1 * i for i in range(10)], [100.0] * 10)
    alt = _window([100 + 0.1 * (i % 2) for i in range(10)], [100.0] * 10)
    t = 25_000.0  # > one bucket's flow, so accumulation/cancellation matters
    b_up = build_bars_for_window(up, kind="dollar_imbalance", threshold=t, floor_buckets=1)
    b_alt = build_bars_for_window(alt, kind="dollar_imbalance", threshold=t, floor_buckets=1)
    assert len(b_up) > len(b_alt)


def test_bars_deterministic():
    rng = np.random.default_rng(7)
    w = _window(100 + np.cumsum(rng.normal(0, 0.1, 90)), rng.integers(1e3, 1e5, 90))
    a = build_bars_for_window(w, kind="dollar", threshold=2e6, floor_buckets=2)
    b = build_bars_for_window(w, kind="dollar", threshold=2e6, floor_buckets=2)
    pd.testing.assert_frame_equal(a, b)


def test_calibrate_threshold_median_day_yields_k():
    rng = np.random.default_rng(3)
    windows = [
        _window(100 + np.cumsum(rng.normal(0, 0.1, 90)),
                rng.integers(10_000, 50_000, 90).astype(float))
        for _ in range(41)
    ]
    K = 16
    t = calibrate_threshold(windows, kind="dollar", k_target=K, floor_buckets=1)
    counts = [len(build_bars_for_window(w, kind="dollar", threshold=t, floor_buckets=1))
              for w in windows]
    assert abs(np.median(counts) - K) <= 1


def test_first_k_extraction_pads_short_days():
    from spydt.bars import first_k_bar_series

    w = _window([100.0 + i * 0.1 for i in range(90)], [100.0] * 90)
    bars = build_bars_for_window(w, kind="dollar", threshold=1e9, floor_buckets=1,
                                 cap_buckets=30)  # only 3 bars possible
    series, mask, clockspan = first_k_bar_series(bars, w, k=16)
    assert len(series) == 16 and len(mask) == 16
    assert mask[:3].all() and not mask[3:].any()
    assert 0 < clockspan <= 90
