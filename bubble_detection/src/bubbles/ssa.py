"""Singular Spectrum Analysis (SSA) denoising — causal, trailing-window.

SSA (Broomhead & King 1986; Golyandina et al. 2001) embeds a window of the
series into a Hankel trajectory matrix, takes its SVD, keeps the leading
components (trend + dominant oscillations), and reconstructs by diagonal
averaging.  The discarded tail components are treated as noise.

Causality: for each bar t we decompose ONLY the trailing window ending at t
and keep the *last point* of the reconstruction as the cleaned value at t.
No future observation ever enters the estimate.

The cleaned series feeds the structural-stability tests (CUSUM, Chow, QLR,
Bai-Perron), whose small-sample behaviour suffers most from high-frequency
noise.  The PSY unit-root tests keep the RAW series — their critical values
are simulated under a pure random-walk null, and pre-filtering would change
that null distribution.  The removed component's local scale is itself a
feature (``ssa_noise_vol``): a microstructure-noise gauge.
"""
from __future__ import annotations

import numpy as np

try:                                       # fast truncated SVD for big windows
    from sklearn.utils.extmath import randomized_svd
    _HAS_RSVD = True
except ImportError:                        # pragma: no cover
    _HAS_RSVD = False


def _ssa_endpoint(x: np.ndarray, L: int, max_rank: int, var_frac: float,
                  lag: int) -> float:
    """Diagonal-averaged SSA reconstruction at position W-1-lag of window x.

    The very last anti-diagonal of the trajectory matrix has a single cell,
    so the endpoint estimate inherits the raw noise; ``lag`` bars back the
    anti-diagonal pools lag+1 cells, cutting the estimate's variance by
    ~(lag+1)x at the cost of a fixed, uniform group delay — still causal.
    """
    W = x.size
    K = W - L + 1
    mu = x.mean()                     # center so the level component doesn't
    x = x - mu                        # swamp the rank-selection energy rule
    idx = np.arange(L)[:, None] + np.arange(K)[None, :]
    X = x[idx]                                            # Hankel (L x K)
    r_max = min(max_rank, L, K)
    if _HAS_RSVD and min(L, K) > 3 * r_max:
        U, s, Vt = randomized_svd(X, n_components=r_max, random_state=0)
    else:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        U, s, Vt = U[:, :r_max], s[:r_max], Vt[:r_max]
    # keep the smallest rank capturing var_frac of (truncated) spectrum energy
    e = s ** 2
    r = int(np.searchsorted(np.cumsum(e) / e.sum(), var_frac) + 1)
    r = max(1, min(r, s.size))
    # anti-diagonal j = W-1-lag: cells (l, j-l)
    j = W - 1 - lag
    l = np.arange(max(0, j - K + 1), min(L - 1, j) + 1)
    cells = np.einsum("lr,r,rl->l", U[l, :r], s[:r], Vt[:r, j - l])
    return float(cells.mean() + mu)


def rolling_ssa_denoise(y: np.ndarray, window: int = 168, L: int | None = None,
                        max_rank: int = 8, var_frac: float = 0.90,
                        stride: int = 1, endpoint_lag: int = 8) -> np.ndarray:
    """Causal SSA-cleaned series: cleaned[t] estimates the smooth component
    at calendar time t - endpoint_lag, from the window ending at t.

    With ``stride`` > 1 the SSA smoothing is computed every ``stride`` bars
    and the *noise level* (y - clean) is carried forward in between — the
    smooth component moves slowly, so this is a faithful approximation and
    keeps large intraday datasets tractable.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    L = L or max(8, window // 3)
    lag = min(endpoint_lag, window // 4)
    clean = np.full(n, np.nan)
    last_noise = np.nan
    for t in range(window - 1, n):
        if (t - (window - 1)) % stride == 0:
            v = _ssa_endpoint(y[t - window + 1: t + 1], L, max_rank, var_frac, lag)
            clean[t] = v
            last_noise = y[t - lag] - v
        else:
            clean[t] = y[t - lag] - last_noise
    return clean
