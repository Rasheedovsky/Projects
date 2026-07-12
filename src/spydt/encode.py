"""Series -> image encodings and dual-time volume constructions.

Scaling contract (leakage-critical): rescaling to [-1, 1] uses TRAIN-FOLD
quantiles, frozen, applied with clipping — never per-sample min-max, never
full-sample statistics. arccos inputs are clipped to [-1+eps, 1-eps].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def paa(x: np.ndarray, length: int) -> np.ndarray:
    """Piecewise aggregate approximation to `length` points (fractional bins)."""
    n = len(x)
    if n == length:
        return x.astype(float)
    edges = np.linspace(0, n, length + 1)
    idx = np.arange(n)
    out = np.empty(length)
    for i in range(length):
        lo, hi = edges[i], edges[i + 1]
        w = np.clip(np.minimum(idx + 1, hi) - np.maximum(idx, lo), 0, 1)
        out[i] = np.sum(w * x) / (hi - lo)
    return out


def interp_to(x: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation length-normalization (used for the bar axis)."""
    if len(x) == length:
        return x.astype(float)
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(x)), x)


@dataclass(frozen=True)
class Scaler:
    """Train-fold robust scaler mapping [q_lo, q_hi] -> [-1, 1], clipped."""

    lo: float
    hi: float

    @classmethod
    def fit(cls, train_values: np.ndarray, q: float = 0.005) -> "Scaler":
        v = train_values[~np.isnan(train_values)]
        lo, hi = float(np.quantile(v, q)), float(np.quantile(v, 1 - q))
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        return cls(lo, hi)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0
        return np.clip(y, -1.0 + EPS, 1.0 - EPS)


def gasf(x: np.ndarray) -> np.ndarray:
    """cos(phi_i + phi_j) for x in [-1, 1]."""
    phi = np.arccos(np.clip(x, -1 + EPS, 1 - EPS))
    return np.cos(phi[:, None] + phi[None, :])


def gadf(x: np.ndarray) -> np.ndarray:
    """sin(phi_i - phi_j)."""
    phi = np.arccos(np.clip(x, -1 + EPS, 1 - EPS))
    return np.sin(phi[:, None] - phi[None, :])


def cross_gasf(x_clock: np.ndarray, z_info: np.ndarray) -> np.ndarray:
    """Cross-Time GASF: C_ij = cos(phi_i + psi_j); rows clock, cols info."""
    phi = np.arccos(np.clip(x_clock, -1 + EPS, 1 - EPS))
    psi = np.arccos(np.clip(z_info, -1 + EPS, 1 - EPS))
    return np.cos(phi[:, None] + psi[None, :])


def cross_gadf(x_clock: np.ndarray, z_info: np.ndarray) -> np.ndarray:
    """Cross-Time GADF analogue: sin(phi_i - psi_j)."""
    phi = np.arccos(np.clip(x_clock, -1 + EPS, 1 - EPS))
    psi = np.arccos(np.clip(z_info, -1 + EPS, 1 - EPS))
    return np.sin(phi[:, None] - psi[None, :])


# ---------------------------------------------------------------------------
# Volume constructions. All take the two length-L scaled axis series (plus the
# info-axis validity mask) and return (C, D, H, W) float32 volumes.
# Sub-windows (V-B): 5 overlapping spans of the length-L series, each
# re-normalized to length L_SUB and encoded.
# ---------------------------------------------------------------------------

N_SUB = 5


def _subwindows(x: np.ndarray, l_sub: int) -> list[np.ndarray]:
    L = len(x)
    width = max(2, L // 3)
    starts = np.linspace(0, L - width, N_SUB).round().astype(int)
    return [interp_to(x[s : s + width], l_sub) for s in starts]


def _sub_masks(mask: np.ndarray) -> np.ndarray:
    """Fraction of valid (non-padded) content per sub-window."""
    L = len(mask)
    width = max(2, L // 3)
    starts = np.linspace(0, L - width, N_SUB).round().astype(int)
    return np.array([mask[s : s + width].mean() for s in starts], dtype=np.float32)


def build_vb(clock: np.ndarray, info: np.ndarray, info_mask: np.ndarray,
             l_sub: int) -> np.ndarray:
    """V-B deep dual-time volume: depth [clock_1..5, info_1..5].

    Channels: 0=GASF, 1=GADF, 2=validity (broadcast per depth slice).
    """
    slices = _subwindows(clock, l_sub) + _subwindows(info, l_sub)
    valid = np.concatenate([np.ones(N_SUB, dtype=np.float32), _sub_masks(info_mask)])
    vol = np.empty((3, 2 * N_SUB, l_sub, l_sub), dtype=np.float32)
    for d, s in enumerate(slices):
        vol[0, d] = gasf(s)
        vol[1, d] = gadf(s)
        vol[2, d] = valid[d]
    return vol


def build_va(clock: np.ndarray, info: np.ndarray, info_mask: np.ndarray) -> np.ndarray:
    """V-A direct dual-time stack: depth [clock, info], full-window images."""
    L = len(clock)
    vol = np.empty((3, 2, L, L), dtype=np.float32)
    vol[0, 0], vol[1, 0], vol[2, 0] = gasf(clock), gadf(clock), 1.0
    vol[0, 1], vol[1, 1], vol[2, 1] = gasf(info), gadf(info), float(info_mask.mean())
    return vol


def build_vc(clock: np.ndarray, info: np.ndarray, info_mask: np.ndarray) -> np.ndarray:
    """V-C Cross-Time Gramian volume: depth [GASF_clock, C_cross, GASF_info].

    Requires len(clock) == len(info) (both axes length-normalized to L).
    """
    if len(clock) != len(info):
        raise ValueError("V-C requires equal axis lengths (L = K normalization)")
    L = len(clock)
    vol = np.empty((3, 3, L, L), dtype=np.float32)
    vol[0, 0], vol[1, 0], vol[2, 0] = gasf(clock), gadf(clock), 1.0
    vol[0, 1] = cross_gasf(clock, info)
    vol[1, 1] = cross_gadf(clock, info)
    vol[2, 1] = float(info_mask.mean())
    vol[0, 2], vol[1, 2], vol[2, 2] = gasf(info), gadf(info), float(info_mask.mean())
    return vol


def build_single_axis(x: np.ndarray, l_sub: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Clock-only / info-only control: 5 sub-window slices of one axis."""
    slices = _subwindows(x, l_sub)
    v = np.ones(N_SUB, dtype=np.float32) if valid is None else _sub_masks(valid)
    vol = np.empty((3, N_SUB, l_sub, l_sub), dtype=np.float32)
    for d, s in enumerate(slices):
        vol[0, d] = gasf(s)
        vol[1, d] = gadf(s)
        vol[2, d] = v[d]
    return vol


def vb_as_2d(vb: np.ndarray) -> np.ndarray:
    """Identical V-B content re-laid-out for the 2D control: (C*D, H, W)."""
    c, d, h, w = vb.shape
    return vb.reshape(c * d, h, w)
