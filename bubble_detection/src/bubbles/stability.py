"""Parameter-stability tests as rolling features: CUSUM, Chow, QLR (sup-F).

All tests are applied to the AR(1) levels model  y[t] = a + b*y[t-1] + e
on log prices over a trailing window ending at each bar.  A bubble phase
is a *structural change* in this regression (drift/persistence shifts
toward the explosive region), so instability statistics are informative
signals for bubble inception and collapse.

* CUSUM  (Brown, Durbin & Evans 1975): cumulative sum of standardized
  recursive residuals; drifts outside its confidence boundary when the
  regression parameters change.  Feature = max_r |W_r| / boundary(r)
  (values > 1 mean 5% instability), plus the CUSUM-of-squares deviation
  (sensitive to variance shifts).
* Chow (1960): F-test for a single break at a *known* date; we use the
  window midpoint as the conventional candidate.
* QLR / sup-F (Quandt 1960; Andrews 1993): sup of Chow F over all
  candidate break dates in the central 70% of the window — the correct
  test when the break date is unknown.  Features: the sup-F value and the
  relative location of the most likely break.

Everything is computed with prefix-sum sufficient statistics, vectorised
across (window x candidate) pairs; each bar's value uses only data up to
that bar.
"""
from __future__ import annotations

import numpy as np

from .prefix_ols import make_ar1_engine

K_AR1 = 2  # parameters per regime in the AR(1) model (intercept + slope)


# ----------------------------------------------------------------------
# CUSUM of recursive residuals (Brown-Durbin-Evans) — fully vectorised.
# ----------------------------------------------------------------------
def rolling_cusum(y: np.ndarray, window: int = 336, warm: int = 6):
    """Rolling BDE CUSUM and CUSUM-of-squares statistics.

    For each window we compute recursive residuals w_j of the AR(1)
    regression via closed-form prefix sums (an O(1) recursive-least-squares
    prediction per observation), then:

      cusum_stat = max_r |sum_{j<=r} w_j / sigma_w| / boundary_5pct(r)
      cusumsq_stat = sqrt(m) * max_r |S_r - E[S_r]|,  S_r = cumsum(w^2)/sum(w^2)

    ``warm`` residuals at the start of each window are discarded (the
    textbook test starts at k+1; a slightly longer warm-up avoids
    near-singular startup fits and is standard numerical practice).
    Returns (cusum_stat, cusumsq_stat) arrays aligned to bar index.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    n = y.size
    x = np.concatenate([[0.0], y[:-1]])              # y[t-1], t>=1

    # prefix sums over t (obs t valid for t >= 1)
    def pref(a):
        p = np.zeros(n + 1)
        p[1:] = np.cumsum(a)
        return p
    valid = np.ones(n); valid[0] = 0.0
    xm = np.where(np.arange(n) >= 1, x, 0.0)
    ym = np.where(np.arange(n) >= 1, y, 0.0)
    P1, Px, Py = pref(valid), pref(xm), pref(ym)
    Pxx, Pxy = pref(xm * xm), pref(xm * ym)

    ends = np.arange(window - 1, n, dtype=np.int64)
    n_win = ends.size
    starts = ends - window + 1                        # price-window start s
    first_obs = starts + 1                            # first regression obs t
    m_obs = window - 1                                # obs per window

    # J[w, c] = absolute obs index of the c-th observation in window w
    offs = np.arange(m_obs)
    J = first_obs[:, None] + offs[None, :]            # (n_win, m_obs)

    # Sufficient stats of the fit on obs [first_obs, J-1] (all obs before J)
    a, b = first_obs[:, None], J                      # sum over [a, b-1] = P[b]-P[a]
    n0 = P1[b] - P1[a]
    Sx = Px[b] - Px[a]
    Sy = Py[b] - Py[a]
    Sxx = Pxx[b] - Pxx[a]
    Sxy = Pxy[b] - Pxy[a]
    D = n0 * Sxx - Sx * Sx
    xj, yj = x[J], y[J]
    with np.errstate(divide="ignore", invalid="ignore"):
        bhat = (n0 * Sxy - Sx * Sy) / D
        ahat = (Sy - bhat * Sx) / n0
        h = (Sxx - 2.0 * xj * Sx + n0 * xj * xj) / D
        w = (yj - ahat - bhat * xj) / np.sqrt(1.0 + h)

    keep = offs >= (K_AR1 + warm)                     # discard warm-up residuals
    w = np.where(keep[None, :], w, np.nan)
    w = np.where(np.isfinite(w), w, np.nan)
    m = np.sum(np.isfinite(w), axis=1).astype(np.float64)   # residuals per window

    wbar = np.nanmean(w, axis=1, keepdims=True)
    sig = np.sqrt(np.nansum((w - wbar) ** 2, axis=1) / np.maximum(m - 1.0, 1.0))
    w0 = np.nan_to_num(w, nan=0.0)

    # BDE CUSUM against the 5% boundary 0.948*(sqrt(m) + 2 r/sqrt(m))
    W = np.cumsum(w0, axis=1) / np.maximum(sig, 1e-12)[:, None]
    r = np.cumsum(np.isfinite(w), axis=1).astype(np.float64)  # residual rank
    bound = 0.948 * (np.sqrt(m)[:, None] + 2.0 * r / np.maximum(np.sqrt(m), 1e-12)[:, None])
    ratio = np.where(r > 0, np.abs(W) / np.maximum(bound, 1e-12), 0.0)
    cusum_stat = ratio.max(axis=1)

    # CUSUM of squares deviation from its null expectation r/m
    cw2 = np.cumsum(w0 * w0, axis=1)
    tot = np.maximum(cw2[:, -1], 1e-12)[:, None]
    S = cw2 / tot
    dev = np.where(r > 0, np.abs(S - r / np.maximum(m, 1.0)[:, None]), 0.0)
    cusumsq_stat = np.sqrt(np.maximum(m, 1.0)) * dev.max(axis=1)

    out1 = np.full(n, np.nan); out2 = np.full(n, np.nan)
    out1[ends] = cusum_stat
    out2[ends] = cusumsq_stat
    return out1, out2


# ----------------------------------------------------------------------
# Chow and QLR (sup-F) — vectorised over (window, candidate break) pairs.
# ----------------------------------------------------------------------
def _chow_f(eng, s_obs, tau, e_obs):
    """Chow F for a break after obs ``tau`` on obs range [s_obs, e_obs]."""
    ssr_full = eng.ssr(s_obs, e_obs)
    ssr1 = eng.ssr(s_obs, tau)
    ssr2 = eng.ssr(tau + 1, e_obs)
    n0 = (e_obs - s_obs + 1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = ((ssr_full - ssr1 - ssr2) / K_AR1) / \
            np.maximum((ssr1 + ssr2) / np.maximum(n0 - 2 * K_AR1, 1.0), 1e-14)
    return np.where(np.isfinite(f), np.maximum(f, 0.0), np.nan)


def rolling_chow(y: np.ndarray, window: int = 336) -> np.ndarray:
    """Chow F statistic for a break at the window midpoint, per bar."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    eng = make_ar1_engine(y)
    ends = np.arange(window - 1, n, dtype=np.int64)
    starts = ends - window + 1
    s_obs = starts + 1
    tau = (s_obs + ends) // 2
    f = _chow_f(eng, s_obs, tau, ends)
    out = np.full(n, np.nan)
    out[ends] = f
    return out


def rolling_qlr(y: np.ndarray, window: int = 336, trim: float = 0.15,
                cand_step: int = 2):
    """QLR (sup-F) statistic and argmax break location per bar.

    Candidate breaks span the central (1 - 2*trim) fraction of the window
    (Andrews' 15% trimming), evaluated every ``cand_step`` observations.
    Returns (supf, loc) with loc in (0, 1) = relative position of the most
    likely break inside the window (near 1 => very recent break).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    eng = make_ar1_engine(y)
    ends = np.arange(window - 1, n, dtype=np.int64)
    starts = ends - window + 1
    s_obs = starts + 1
    m_obs = window - 1
    lo = int(np.ceil(trim * m_obs))
    hi = int(np.floor((1.0 - trim) * m_obs))
    rel = np.arange(lo, hi, cand_step, dtype=np.int64)         # relative break offsets

    # Broadcast windows x candidates, flatten, one batched Chow evaluation.
    S = np.repeat(s_obs, rel.size)
    E = np.repeat(ends, rel.size)
    T = (s_obs[:, None] + rel[None, :]).ravel()
    f = _chow_f(eng, S, T, E).reshape(ends.size, rel.size)

    supf = np.nanmax(f, axis=1)
    arg = np.nanargmax(np.nan_to_num(f, nan=-np.inf), axis=1)
    loc = rel[arg] / float(m_obs)
    out_f = np.full(n, np.nan); out_l = np.full(n, np.nan)
    out_f[ends] = supf
    out_l[ends] = loc
    return out_f, out_l
