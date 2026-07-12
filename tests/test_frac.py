"""FFD weights vs closed form; stationarity selection on synthetic series."""
import numpy as np

from spydt.frac import apply_ffd, ffd_weights, select_dstar


def test_ffd_weights_closed_form():
    # w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k
    d = 0.5
    w = ffd_weights(d, tol=1e-4)
    assert w[0] == 1.0
    assert np.isclose(w[1], -d)
    assert np.isclose(w[2], -w[1] * (d - 1) / 2)
    assert abs(w[-1]) >= 1e-4 and len(w) > 10


def test_ffd_d0_is_identity_d1_is_diff():
    x = np.cumsum(np.random.default_rng(0).normal(0, 1, 500))
    w0 = ffd_weights(0.0, tol=1e-4)
    assert len(w0) == 1 and w0[0] == 1.0
    y1 = apply_ffd(x, ffd_weights(1.0, tol=1e-4))
    np.testing.assert_allclose(y1[1:], np.diff(x), atol=1e-10)


def test_apply_ffd_burn_in_is_nan():
    x = np.arange(100.0)
    w = ffd_weights(0.5, tol=1e-2)
    y = apply_ffd(x, w)
    assert np.isnan(y[: len(w) - 1]).all()
    assert not np.isnan(y[len(w) - 1 :]).any()


def test_select_dstar_random_walk_needs_differencing():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 4000))  # unit root: d* > 0
    d = select_dstar(x, grid_step=0.1, max_points=2000)
    assert d > 0.0


def test_select_dstar_stationary_is_zero():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 4000)  # already stationary
    assert select_dstar(x, grid_step=0.1, max_points=2000) == 0.0
