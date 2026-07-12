"""Axis bundle: train-only fitting, padding, timestamp bounds, label math."""
import numpy as np
import pytest

from spydt.axes import DayData, axis_similarity, build_bundle

RNG = np.random.default_rng(0)


def _mk_day(date: str, seed: int, base: float = 100.0) -> DayData:
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0, 3e-4, 390)))
    # opens differ from prior closes by micro-noise, as in real prints
    open_ = np.concatenate([[base], close[:-1]]) * np.exp(rng.normal(0, 5e-5, 390))
    vol = rng.integers(50_000, 500_000, 390).astype(float)
    return DayData(date, close, open_, vol, prev_close=base * 0.999)


@pytest.fixture(scope="module")
def bundle():
    days = [_mk_day(f"2015-{1 + i // 21:02d}-{1 + i % 21:02d}", seed=i) for i in range(60)]
    train_pos = np.arange(40)
    return build_bundle(days, train_pos, bar_kind="dollar", K=16, L=32,
                        ffd_grid_step=0.5)


def test_shapes_and_scaling(bundle):
    n = len(bundle.dates)
    assert bundle.clock.shape == (n, 32) and bundle.info.shape == (n, 32)
    ok = ~np.isnan(bundle.clock).any(axis=1)
    assert ok.sum() > n * 0.8
    assert np.nanmax(bundle.clock) <= 1.0 and np.nanmin(bundle.clock) >= -1.0
    assert np.nanmax(bundle.info) <= 1.0 and np.nanmin(bundle.info) >= -1.0


def test_labels_and_exec_returns(bundle):
    assert set(np.unique(bundle.y)) <= {-1.0, 1.0}
    # executable return differs from label return (entry at 15:31 open)
    assert not np.allclose(bundle.r_label, bundle.r_exec)
    assert np.abs(bundle.w.mean() - 1.0) < 0.5


def test_features_use_only_morning_and_overnight(bundle):
    # realized_vol and clockspan derive from the morning window; sanity: no
    # correlation with the afternoon label beyond chance on synthetic noise
    ok = ~np.isnan(bundle.clock).any(axis=1)
    c = abs(np.corrcoef(bundle.tabular[ok, 1], bundle.r_label[ok])[0, 1])
    assert c < 0.5


def test_diag_keys(bundle):
    for k in ["threshold", "median_morning_bars", "pad_rate", "floor_rate",
              "cap_rate", "d_clock", "d_info"]:
        assert k in bundle.diag


def test_axis_similarity_range(bundle):
    sim = axis_similarity(bundle)
    assert 0 <= sim["median_abs_corr"] <= 1
    assert sim["n"] > 0
