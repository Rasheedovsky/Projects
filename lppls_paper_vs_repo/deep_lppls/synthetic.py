"""Synthetic LPPLS series generation (paper Sec. 2.2.2 and Table 1).

Table 1 parameter ranges, for windows of n = 252 daily points normalised to
[0, 1] (one day = 1/(n-1)):

    tc     ~ U(t2, t2 + 50 days),  t2 = 1
    m      ~ U(0.1, 0.9)
    w      ~ U(6, 13)
    white noise amplitude  alpha ~ U(0.01, 0.15)  (fraction of range = 1)
    AR(1) noise amplitude  alpha ~ U(0.01, 0.05), phi = 0.9

The paper does not specify how A, B, C, phi were sampled. Since each series
is min-max rescaled to [0, 1] before use, A and |B| have no effect on the
network input; only the sign of B (bubble direction) and the relative
oscillation amplitude |C/B| shape the curve. We therefore fix A = 0,
B = -1 (positive bubble: price rises into tc) and sample
|C| ~ U(0.05, 0.3), phase phi ~ U(0, 2*pi). This choice is documented as an
implementation decision in the report.
"""

import numpy as np

from .core import lppls, minmax_scale

N_POINTS = 252
TC_MAX_DAYS = 50.0


def sample_params(rng, n):
    """Sample n sets of (tc, m, w, c1, c2) per Table 1 (normalised time)."""
    day = 1.0 / (N_POINTS - 1)
    tc = 1.0 + rng.uniform(0.0, TC_MAX_DAYS * day, n)
    m = rng.uniform(0.1, 0.9, n)
    w = rng.uniform(6.0, 13.0, n)
    c = rng.uniform(0.05, 0.3, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return tc, m, w, c * np.cos(phi), c * np.sin(phi)


def clean_series(tc, m, w, c1, c2, n_points=N_POINTS):
    """Noise-free LPPLS series on [0, 1], min-max rescaled to [0, 1]."""
    t = np.linspace(0.0, 1.0, n_points)
    x = lppls(t, tc, m, w, a=0.0, b=-1.0, c1=c1, c2=c2)
    x, _ = minmax_scale(x)
    return x


def white_noise(rng, shape, lo=0.01, hi=0.15):
    amp = rng.uniform(lo, hi, (shape[0], 1))
    return amp * rng.standard_normal(shape)


def ar1_noise(rng, shape, lo=0.01, hi=0.05, phi=0.9):
    """AR(1) noise with stationary std sampled in [lo, hi] (Sec. 2.2.2:
    sigma_eta^2 = sigma^2 / (1 - phi^2), so eps std = eta_std*sqrt(1-phi^2))."""
    n_series, n_points = shape
    eta_std = rng.uniform(lo, hi, (n_series, 1))
    eps_std = eta_std * np.sqrt(1.0 - phi**2)
    eps = eps_std * rng.standard_normal(shape)
    out = np.empty(shape)
    out[:, 0] = eps[:, 0]
    for i in range(1, n_points):
        out[:, i] = phi * out[:, i - 1] + eps[:, i]
    return out


def make_dataset(n_series, noise="white", seed=0, n_points=N_POINTS):
    """Generate (X, y) with X noisy min-max-scaled LPPLS series and
    y = (tc, m, w). noise in {'white', 'ar1', 'both'} ('both' = 50/50 mix)."""
    rng = np.random.default_rng(seed)
    tc, m, w, c1, c2 = sample_params(rng, n_series)
    X = np.empty((n_series, n_points), dtype=np.float32)
    for i in range(n_series):
        X[i] = clean_series(tc[i], m[i], w[i], c1[i], c2[i], n_points)
    if noise == "white":
        X += white_noise(rng, X.shape).astype(np.float32)
    elif noise == "ar1":
        X += ar1_noise(rng, X.shape).astype(np.float32)
    elif noise == "both":
        half = n_series // 2
        X[:half] += white_noise(rng, X[:half].shape).astype(np.float32)
        X[half:] += ar1_noise(rng, X[half:].shape).astype(np.float32)
    else:
        raise ValueError(noise)
    y = np.column_stack([tc, m, w]).astype(np.float32)
    return X, y
