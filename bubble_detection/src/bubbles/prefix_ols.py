"""Prefix-sum sufficient-statistic engine for windowed OLS regressions.

Every statistical test in this project (right-tailed ADF, BSADF/GSADF,
Chow, QLR, Bai-Perron, CUSUM) reduces to OLS on many overlapping windows.
Instead of refitting each window (O(W) per window), we precompute prefix
sums of all regressor/dependent cross-products once, so the sufficient
statistics (X'X, X'y, y'y) of *any* window are O(1) gathers.  t-stats and
SSRs for hundreds of thousands of (start, end) pairs are then computed in
a handful of vectorised numpy operations.

Conditioning note: regressions are performed on within-window *centered*
moments (equivalent to including an intercept), and the input series is
globally demeaned on entry, which keeps the centered cross-moment
subtraction well away from catastrophic cancellation.
"""
from __future__ import annotations

import numpy as np


class WindowedOLS:
    """Windowed OLS with intercept: dep[t] ~ 1 + vars[t, :].

    Parameters
    ----------
    variables : (n, q) array; regressors indexed by time t (NaN where undefined).
    dep       : (n,) array; dependent variable (NaN where undefined).
    first_valid : first time index t at which all variables and dep are defined.

    A "window" (s, e) means: regression observations are all t in
    [max(s, first_valid_offset applied by caller), e].  Callers pass the
    first regression observation index directly as `s_obs`.
    """

    def __init__(self, variables: np.ndarray, dep: np.ndarray, first_valid: int):
        variables = np.asarray(variables, dtype=np.float64)
        if variables.ndim == 1:
            variables = variables[:, None]
        dep = np.asarray(dep, dtype=np.float64)
        n, q = variables.shape
        self.n, self.q = n, q
        self.first_valid = int(first_valid)

        # Stack [vars, dep] -> (n, q+1); zero-out invalid rows so cumsums ignore them.
        Z = np.column_stack([variables, dep])
        Z[: self.first_valid] = 0.0
        Z = np.nan_to_num(Z, nan=0.0)

        # Prefix sums of values and of pairwise products.
        self._P1 = np.zeros((n + 1, q + 1))
        self._P1[1:] = np.cumsum(Z, axis=0)
        outer = Z[:, :, None] * Z[:, None, :]                     # (n, q+1, q+1)
        self._P2 = np.zeros((n + 1, q + 1, q + 1))
        self._P2[1:] = np.cumsum(outer, axis=0)

    # ------------------------------------------------------------------
    def _moments(self, s_obs: np.ndarray, e_obs: np.ndarray):
        """Centered cross-moment matrices for obs ranges [s_obs, e_obs] (inclusive)."""
        s_obs = np.maximum(np.asarray(s_obs, dtype=np.int64), self.first_valid)
        e_obs = np.asarray(e_obs, dtype=np.int64)
        n0 = (e_obs - s_obs + 1).astype(np.float64)
        S1 = self._P1[e_obs + 1] - self._P1[s_obs]                # (m, q+1)
        S2 = self._P2[e_obs + 1] - self._P2[s_obs]                # (m, q+1, q+1)
        C = S2 - S1[:, :, None] * S1[:, None, :] / n0[:, None, None]
        return n0, C

    def stats(self, s_obs: np.ndarray, e_obs: np.ndarray):
        """Batched OLS over observation ranges.

        Returns dict with:
          beta  : (m, q) slope estimates (intercept concentrated out)
          tstat0: (m,) t-statistic of the first regressor's slope
          ssr   : (m,) residual sum of squares
          nobs  : (m,) observations per window
        """
        n0, C = self._moments(s_obs, e_obs)
        q = self.q
        A = C[:, :q, :q]                                          # (m, q, q)
        b = C[:, :q, -1]                                          # (m, q)
        syy = C[:, -1, -1]                                        # (m,)

        # Guard against singular windows (too few obs / zero variance).
        ok = n0 >= q + 2
        eye = np.eye(q)
        A_safe = np.where(ok[:, None, None], A, eye)
        Ainv = np.linalg.inv(A_safe + 1e-12 * eye)
        beta = np.einsum("mij,mj->mi", Ainv, b)
        ssr = np.maximum(syy - np.einsum("mi,mi->m", beta, b), 0.0)
        dof = n0 - q - 1.0                                        # -1 for intercept
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma2 = ssr / np.maximum(dof, 1.0)
            se0 = np.sqrt(sigma2 * Ainv[:, 0, 0])
            tstat0 = beta[:, 0] / se0
        bad = ~ok | (dof < 1) | ~np.isfinite(tstat0)
        tstat0 = np.where(bad, np.nan, tstat0)
        ssr = np.where(ok, ssr, np.nan)
        return {"beta": beta, "tstat0": tstat0, "ssr": ssr, "nobs": n0}

    def ssr(self, s_obs: np.ndarray, e_obs: np.ndarray) -> np.ndarray:
        return self.stats(s_obs, e_obs)["ssr"]


# ----------------------------------------------------------------------
def make_adf_engine(y: np.ndarray, lags: int = 1) -> WindowedOLS:
    """Engine for the ADF regression  dy[t] ~ 1 + y[t-1] + dy[t-1..t-lags].

    The t-statistic of y[t-1] (``tstat0``) is the ADF statistic; in the
    right-tailed (explosive-root) variant large *positive* values reject
    the unit-root null in favour of explosiveness.
    Observation t is valid for t >= lags + 1.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - np.nanmean(y)                       # conditioning only; ADF t is invariant
    n = y.size
    dy = np.full(n, np.nan)
    dy[1:] = y[1:] - y[:-1]
    cols = [np.concatenate([[np.nan], y[:-1]])]                   # y[t-1]
    for i in range(1, lags + 1):
        li = np.full(n, np.nan)
        li[i:] = dy[: n - i] if i > 0 else dy
        cols.append(li)
    variables = np.column_stack(cols)
    return WindowedOLS(variables, dy, first_valid=lags + 1)


def make_ar1_engine(y: np.ndarray) -> WindowedOLS:
    """Engine for the AR(1) levels regression  y[t] ~ 1 + y[t-1].

    Used by the structural-stability tests (Chow, QLR, Bai-Perron, CUSUM):
    a bubble regime shows up as a shift of the AR(1) persistence/drift.
    Observation t is valid for t >= 1.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - np.nanmean(y)
    ylag = np.concatenate([[np.nan], y[:-1]])
    return WindowedOLS(ylag, y, first_valid=1)
