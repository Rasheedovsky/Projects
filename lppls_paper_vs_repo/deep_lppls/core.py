"""Core LPPLS math shared by all calibration methods.

Follows Appendix A.1 of the paper: the model is reformulated with two linear
parameters C1 = C cos(phi), C2 = C sin(phi) (Eq. 4-5), the four linear
parameters {A, B, C1, C2} are obtained analytically from the normal
equations (Eq. 8), which leaves the reduced loss F1(tc, m, w) (Eq. 9) over
the three nonlinear parameters.

All functions operate in *normalised time*: the observation window [t1, t2]
is mapped to [0, 1] (paper Sec. 2.1.1), so t2 = 1 and tc is expressed in
units of the window length.
"""

import numpy as np

EPS = 1e-8


def lppls(t, tc, m, w, a, b, c1, c2):
    """LPPLS model value (Eq. 4-5). dt is clamped like the reference repo."""
    dt = np.abs(tc - t) + EPS
    return a + np.power(dt, m) * (b + c1 * np.cos(w * np.log(dt)) + c2 * np.sin(w * np.log(dt)))


def design_matrix(t, tc, m, w):
    """Columns [1, f, g, h] of Eq. 4 evaluated at times t."""
    dt = np.abs(tc - t) + EPS
    logdt = np.log(dt)
    f = np.power(dt, m)
    g = f * np.cos(w * logdt)
    h = f * np.sin(w * logdt)
    return np.column_stack([np.ones_like(t), f, g, h])


def solve_linear(t, x, tc, m, w):
    """Least-squares solve of Eq. 8 for (A, B, C1, C2), with tiny ridge for
    numerical stability (same 1e-8 regularisation the reference repo uses)."""
    M = design_matrix(t, tc, m, w)
    A = M.T @ M + EPS * np.eye(4)
    b = M.T @ x
    return np.linalg.solve(A, b)


def f1_loss(t, x, tc, m, w):
    """Reduced loss F1(tc, m, w) of Eq. 9 (mean squared error, Eq. 6)."""
    beta = solve_linear(t, x, tc, m, w)
    resid = x - design_matrix(t, tc, m, w) @ beta
    return float(np.mean(resid**2))


def minmax_scale(x):
    """Min-max scale to [0, 1]; returns scaled array and (min, range)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    rng = hi - lo if hi > lo else 1.0
    return (x - lo) / rng, (lo, rng)
