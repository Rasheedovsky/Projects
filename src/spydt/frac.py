"""Fractional differentiation, fixed-width window (FFD), AFML ch. 5."""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller

MAX_WIDTH = 500


def ffd_weights(d: float, *, tol: float = 1e-4) -> np.ndarray:
    """w_0 = 1; w_k = -w_{k-1} (d - k + 1) / k; truncated at |w| < tol."""
    w = [1.0]
    k = 1
    while k < MAX_WIDTH:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < tol:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w)


def apply_ffd(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Convolve the fixed-width weight window over x; burn-in is NaN."""
    n_w = len(weights)
    if n_w == 1:
        return x.astype(float)
    y = np.convolve(x, weights, mode="full")[: len(x)]
    y[: n_w - 1] = np.nan
    return y


def select_dstar(
    x: np.ndarray,
    *,
    grid_step: float = 0.1,
    alpha: float = 0.05,
    tol: float = 1e-4,
    max_points: int = 5000,
) -> float:
    """Smallest d in [0, 1] whose FFD series passes ADF at level alpha.

    Estimated on TRAINING data only (caller's responsibility). ADF runs on an
    evenly-spaced subsample of at most max_points for tractability.
    """
    for d in np.arange(0.0, 1.0 + 1e-9, grid_step):
        y = apply_ffd(x, ffd_weights(float(d), tol=tol))
        y = y[~np.isnan(y)]
        if len(y) < 100:
            continue
        step = max(1, len(y) // max_points)
        sub = y[::step]
        try:
            pval = adfuller(sub, maxlag=10, autolag=None)[1]
        except Exception:
            continue
        if pval < alpha:
            return float(round(d, 3))
    return 1.0
