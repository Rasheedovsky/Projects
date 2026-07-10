"""Paper's benchmark calibration (Appendix A.1): Levenberg-Marquardt on the
reduced residuals of Eq. 9, with the same random multistart protocol the
reference repo uses for its Nelder-Mead search (25 random initialisations of
(tc, m, w) inside tc in [t2 - 0.2*dt, t2 + 0.2*dt], m in [0.1, 1], w in [6, 13]).

The paper itself does not state the restart count for LM; we mirror the
reference repo's max_searches=25 so both classical methods get the same
search budget.
"""

import numpy as np
from scipy.optimize import least_squares

from .core import design_matrix, solve_linear

INIT_BOUNDS = dict(m=(0.1, 1.0), w=(6.0, 13.0))


def _residuals(theta, t, x):
    tc, m, w = theta
    beta = solve_linear(t, x, tc, m, w)
    return design_matrix(t, tc, m, w) @ beta - x


def fit_lm(t, x, max_searches=25, seed=None):
    """Multistart LM fit. t is normalised time on [0, 1], x the (scaled)
    observable. Returns dict with tc, m, w, linear params and SSE."""
    rng = np.random.default_rng(seed)
    t1, t2 = float(t[0]), float(t[-1])
    dt_win = t2 - t1
    best = None
    for _ in range(max_searches):
        x0 = np.array(
            [
                rng.uniform(t2 - 0.2 * dt_win, t2 + 0.2 * dt_win),
                rng.uniform(*INIT_BOUNDS["m"]),
                rng.uniform(*INIT_BOUNDS["w"]),
            ]
        )
        try:
            sol = least_squares(_residuals, x0, method="lm", args=(t, x), max_nfev=2000)
        except Exception:
            continue
        sse = float(np.sum(sol.fun**2))
        if best is None or sse < best["sse"]:
            tc, m, w = sol.x
            a, b, c1, c2 = solve_linear(t, x, tc, m, w)
            best = dict(tc=float(tc), m=float(m), w=float(w), a=float(a), b=float(b), c1=float(c1), c2=float(c2), sse=sse)
    if best is None:
        best = dict(tc=t2, m=0.5, w=9.5, a=float(np.mean(x)), b=0.0, c1=0.0, c2=0.0, sse=float(np.sum((x - np.mean(x)) ** 2)))
    return best
