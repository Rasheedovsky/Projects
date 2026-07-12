"""Encodings: closed forms, cross-field properties, construction invariants."""
import numpy as np
import pytest

from spydt.encode import (
    Scaler,
    build_va,
    build_vb,
    build_vc,
    build_single_axis,
    cross_gasf,
    gadf,
    gasf,
    interp_to,
    paa,
    vb_as_2d,
)


def test_paa_exact_integer_ratio():
    x = np.arange(8.0)
    np.testing.assert_allclose(paa(x, 4), [0.5, 2.5, 4.5, 6.5])


def test_paa_fractional_preserves_mean():
    x = np.random.default_rng(0).normal(0, 1, 90)
    np.testing.assert_allclose(paa(x, 32).mean(), x.mean(), atol=1e-10)


def test_scaler_train_fit_and_clip():
    train = np.linspace(-1, 1, 1001)
    s = Scaler.fit(train)
    out = s(np.array([-5.0, 0.0, 5.0]))
    assert out[0] >= -1 and out[2] <= 1  # clipped
    assert abs(out[1]) < 0.01


def test_gasf_closed_form():
    x = np.array([1.0, 0.0, -1.0]) * (1 - 1e-6)
    g = gasf(x)
    # phi = [0, pi/2, pi]; GASF_ij = cos(phi_i + phi_j)
    expected = np.cos(np.add.outer([0, np.pi / 2, np.pi], [0, np.pi / 2, np.pi]))
    # arccos has sqrt(eps) sensitivity at the domain edges
    np.testing.assert_allclose(g, expected, atol=5e-3)


def test_gadf_antisymmetric():
    x = np.random.default_rng(1).uniform(-0.9, 0.9, 16)
    g = gadf(x)
    np.testing.assert_allclose(g, -g.T, atol=1e-12)


def test_cross_gasf_reduces_to_gasf_when_axes_identical():
    x = np.random.default_rng(2).uniform(-0.9, 0.9, 16)
    np.testing.assert_allclose(cross_gasf(x, x), gasf(x), atol=1e-12)


def test_cross_gasf_is_rectangular_in_general():
    x = np.random.default_rng(3).uniform(-0.9, 0.9, 8)
    z = np.random.default_rng(4).uniform(-0.9, 0.9, 8)
    c = cross_gasf(x, z)
    assert c.shape == (8, 8)
    assert not np.allclose(c, c.T)  # generally not symmetric


def test_vb_shape_and_depth_order():
    clock = np.random.default_rng(5).uniform(-0.9, 0.9, 32)
    info = np.random.default_rng(6).uniform(-0.9, 0.9, 32)
    mask = np.ones(32, dtype=bool)
    vb = build_vb(clock, info, mask, l_sub=16)
    assert vb.shape == (3, 10, 16, 16)
    # depth 0..4 from clock only: changing info must not affect them
    vb2 = build_vb(clock, np.zeros(32), mask, l_sub=16)
    np.testing.assert_allclose(vb[:2, :5], vb2[:2, :5])
    assert not np.allclose(vb[:2, 5:], vb2[:2, 5:])


def test_vb_mask_channel_carries_padding():
    clock = np.random.default_rng(7).uniform(-0.9, 0.9, 32)
    info = np.random.default_rng(8).uniform(-0.9, 0.9, 32)
    mask = np.zeros(32, dtype=bool)
    mask[:8] = True  # only first quarter of info axis is real
    vb = build_vb(clock, info, mask, l_sub=16)
    assert vb[2, :5].min() == 1.0          # clock slices fully valid
    assert vb[2, 9].max() < 0.5            # last info slice mostly padding


def test_vc_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal axis lengths"):
        build_vc(np.zeros(32), np.zeros(16), np.ones(16, dtype=bool))


def test_vc_shape_and_va_shape():
    x = np.random.default_rng(9).uniform(-0.9, 0.9, 32)
    z = np.random.default_rng(10).uniform(-0.9, 0.9, 32)
    m = np.ones(32, dtype=bool)
    assert build_vc(x, z, m).shape == (3, 3, 32, 32)
    assert build_va(x, z, m).shape == (3, 2, 32, 32)
    assert build_single_axis(x, 16).shape == (3, 5, 16, 16)


def test_2d_control_has_identical_content():
    x = np.random.default_rng(11).uniform(-0.9, 0.9, 32)
    z = np.random.default_rng(12).uniform(-0.9, 0.9, 32)
    vb = build_vb(x, z, np.ones(32, dtype=bool), l_sub=16)
    flat = vb_as_2d(vb)
    assert flat.shape == (30, 16, 16)
    np.testing.assert_allclose(flat.sum(), vb.sum())


def test_interp_to_endpoints():
    x = np.array([0.0, 1.0, 4.0])
    y = interp_to(x, 5)
    assert y[0] == 0.0 and y[-1] == 4.0 and len(y) == 5
