"""Change-point features from the ``ruptures`` package (Truong et al. 2020).

We run PELT (Killick, Fearnhead & Eckley 2012) — exact penalised
change-point detection in linear time — with the Gaussian ("normal") cost,
which reacts to shifts in both the mean and the variance of log returns,
over a trailing window ending at each bar.  Volatility-regime changes are
a canonical companion signal of bubble inception (quiet grind-up) and
collapse (variance explosion).

Features per bar:
  rup_ncp   : number of change points detected in the trailing window
  rup_since : bars since the most recent change point (window length if none)
  rup_lvr   : log variance ratio, last segment vs. previous segment (0 if < 2 segments)
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt


def rolling_pelt(returns: np.ndarray, window: int = 336, pen_mult: float = 8.0,
                 min_size: int = 15, jump: int = 2, stride: int = 5):
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    ncp = np.full(n, np.nan)
    since = np.full(n, np.nan)
    lvr = np.full(n, np.nan)

    for e in range(window - 1, n, stride):
        seg = r[e - window + 1: e + 1]
        sd = seg.std()
        z = (seg - seg.mean()) / (sd if sd > 0 else 1.0)
        algo = rpt.Pelt(model="normal", min_size=min_size, jump=jump).fit(z[:, None])
        # penalty: BIC-flavoured, scaled by pen_mult (per-parameter cost 2)
        bkps = algo.predict(pen=pen_mult * np.log(window))
        internal = [b for b in bkps if b < window]
        ncp[e] = float(len(internal))
        if internal:
            last = internal[-1]
            since[e] = float(window - last)
            prev = internal[-2] if len(internal) >= 2 else 0
            v2 = np.var(z[last:]) if window - last > 1 else np.nan
            v1 = np.var(z[prev:last]) if last - prev > 1 else np.nan
            lvr[e] = np.log(max(v2, 1e-12) / max(v1, 1e-12)) if np.isfinite(v1) and np.isfinite(v2) else 0.0
        else:
            since[e] = float(window)
            lvr[e] = 0.0

    # forward-fill stride gaps (causal)
    for arr in (ncp, since, lvr):
        idx = np.where(~np.isnan(arr))[0]
        if idx.size:
            pos = np.searchsorted(idx, np.arange(n), side="right") - 1
            ok = pos >= 0
            filled = np.full(n, np.nan)
            filled[ok] = arr[idx[pos[ok]]]
            arr[:] = np.where(np.arange(n) >= idx[0], filled, np.nan)
    return ncp, since, lvr
