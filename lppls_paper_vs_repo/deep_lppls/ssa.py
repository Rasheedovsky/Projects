"""Singular Spectrum Analysis (SSA) denoising for price series.

Basic SSA (Broomhead-King / Golyandina et al.): embed the series x (length n)
into the trajectory (Hankel) matrix with window length W, take the SVD, keep
the leading components, and reconstruct by anti-diagonal (Hankel) averaging.

Used here as an optional cleaning stage BEFORE LPPLS calibration. Component
count is adaptive: keep singular values >= SV_FRACTION of the largest one
(bounded to [RANK_MIN, RANK_MAX]) so that heterogeneous windows keep the
trend + the strongest oscillatory pairs while dropping the noise floor.
Everything operates strictly inside the calibration window, so the cleaning
is causal at the window level (no data beyond t2 is used).
"""

import numpy as np

SV_FRACTION = 0.02
RANK_MIN = 4
RANK_MAX = 20


def ssa_clean(x, window=None, rank=None):
    """Denoise 1-D series x. window defaults to n // 3. If rank is None the
    adaptive singular-value rule is used. Returns the reconstructed series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    W = window or max(20, n // 3)
    W = min(W, n - 1)
    K = n - W + 1
    # trajectory matrix (W x K), columns are lagged windows
    idx = np.arange(W)[:, None] + np.arange(K)[None, :]
    X = x[idx]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        rank = int(np.clip(np.sum(s >= SV_FRACTION * s[0]), RANK_MIN, RANK_MAX))
    Xr = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    # Hankelisation: average anti-diagonals back into a series
    out = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(out, idx.ravel(), Xr.ravel())
    np.add.at(counts, idx.ravel(), 1.0)
    return out / counts
