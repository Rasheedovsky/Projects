"""Phillips-Shi-Yu style explosiveness tests.

Implements, on log prices:

* ``rolling_rtadf``  – right-tailed ADF t-stat on a fixed trailing window.
  (Right-tailed ADF: H0 unit root vs H1 explosive root, Phillips/Wu/Yu 2011.)
* ``bsadf``          – Backward Sup ADF sequence of Phillips, Shi & Yu (2015):
  at each bar t, the supremum of ADF statistics over all windows *ending*
  at t with length in [min_window, max_window].  This is the real-time
  date-stamping statistic; its full-sample supremum is the GSADF statistic.
* ``mc_critical_values`` – Monte-Carlo critical values of the BSADF sequence
  under the random-walk null, matched to the exact window configuration.

All statistics at bar t use only data up to t, so the resulting features
are walk-forward safe by construction.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .prefix_ols import make_adf_engine


def bsadf(y: np.ndarray, min_window: int = 40, max_window: int = 400,
          lags: int = 1, start_step: int = 1, block: int = 2000) -> np.ndarray:
    """Backward Sup ADF statistic for each bar (NaN before min_window).

    Processes end bars in blocks so memory stays bounded on large intraday
    samples (pairs per block <= block * max_window / start_step).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    eng = make_adf_engine(y, lags=lags)
    out = np.full(n, np.nan)
    for e0 in range(0, n, block):
        e1 = min(e0 + block, n)
        starts_l, ends_l = [], []
        for e in range(e0, e1):
            lo = max(0, e - max_window + 1)
            hi = e - min_window + 1
            if hi < lo:
                continue
            s = np.arange(lo, hi + 1, start_step, dtype=np.int64)
            if s.size:
                starts_l.append(s)
                ends_l.append(np.full(s.size, e, dtype=np.int64))
        if not starts_l:
            continue
        starts = np.concatenate(starts_l)
        ends = np.concatenate(ends_l)
        t = eng.stats(starts + lags + 1, ends)["tstat0"]
        red = np.full(n, -np.inf)
        ok = np.isfinite(t)
        np.maximum.at(red, ends[ok], t[ok])
        upd = red[e0:e1] > -np.inf
        seg = out[e0:e1]
        seg[upd] = red[e0:e1][upd]
        out[e0:e1] = seg
    return out


def rolling_rtadf(y: np.ndarray, window: int = 168, lags: int = 1) -> np.ndarray:
    """Right-tailed ADF t-stat over a fixed trailing window ending at each bar."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    eng = make_adf_engine(y, lags=lags)
    ends = np.arange(window - 1, n, dtype=np.int64)
    starts = ends - window + 1
    t = eng.stats(starts + lags + 1, ends)["tstat0"]
    out = np.full(n, np.nan)
    out[ends] = t
    return out


def gsadf_stat(bsadf_seq: np.ndarray) -> float:
    """Full-sample GSADF statistic = sup over t of BSADF(t)."""
    return float(np.nanmax(bsadf_seq))


def mc_critical_values(n: int, min_window: int, max_window: int, lags: int = 1,
                       start_step: int = 1, n_sims: int = 200,
                       quantiles=(0.90, 0.95, 0.99), seed: int = 42,
                       cache_path: str | None = None):
    """Monte-Carlo critical values for the BSADF sequence and GSADF stat.

    Simulates pure random walks (the PSY null; the ADF t-stat is invariant
    to the innovation scale) of the same length and window configuration,
    computes the BSADF sequence of each, and returns per-bar quantiles plus
    the distribution of the full-sample GSADF statistic.

    Stationary-tail optimisation: with a capped backward window, the null
    distribution of BSADF(t) is identical for every t >= max_window, so for
    long samples we simulate only ``max_window + 264`` bars and extend the
    final quantile row — exact for the per-bar critical values.  (The GSADF
    sup-statistic critical value is then a mild lower bound for n much
    larger than the simulated length; the real-time flag is unaffected.)
    """
    eff_n = min(n, max_window + 264)
    key = f"n{eff_n}_mw{min_window}_Mw{max_window}_l{lags}_ss{start_step}_s{n_sims}_{seed}_v2"
    def _extend(cv: np.ndarray) -> np.ndarray:
        if n <= eff_n:
            return cv[:n]
        return np.concatenate([cv, np.full(n - eff_n, cv[-1])])

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            blob = json.load(fh)
        if blob.get("key") == key:
            return {
                "bsadf_cv": {float(q): _extend(np.asarray(v)) for q, v in blob["bsadf_cv"].items()},
                "gsadf_cv": {float(q): v for q, v in blob["gsadf_cv"].items()},
            }

    rng = np.random.default_rng(seed)
    stats = np.full((n_sims, eff_n), np.nan)
    for i in range(n_sims):
        stats[i] = bsadf(np.cumsum(rng.standard_normal(eff_n)),
                         min_window, max_window, lags, start_step)

    bsadf_cv = {float(q): np.nanquantile(stats, q, axis=0) for q in quantiles}
    # For t >= max_window the BSADF null distribution is stationary; pool the
    # per-bar quantile estimates over that region to cut MC noise.
    if eff_n > max_window:
        for q in bsadf_cv:
            bsadf_cv[q][max_window:] = np.nanmean(bsadf_cv[q][max_window:])
    gmax = np.nanmax(stats, axis=1)
    gsadf_cv = {float(q): float(np.quantile(gmax, q)) for q in quantiles}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({
                "key": key,
                "bsadf_cv": {str(q): v.tolist() for q, v in bsadf_cv.items()},
                "gsadf_cv": {str(q): v for q, v in gsadf_cv.items()},
            }, fh)
    return {"bsadf_cv": {q: _extend(v) for q, v in bsadf_cv.items()},
            "gsadf_cv": gsadf_cv}


def date_stamp_episodes(flag: np.ndarray, min_duration: int = 5):
    """PSY date-stamping: contiguous runs of the exceedance flag lasting at
    least ``min_duration`` bars.  Returns list of (start, end) index pairs."""
    episodes = []
    in_ep, start = False, 0
    f = np.asarray(flag).astype(bool)
    for i, v in enumerate(f):
        if v and not in_ep:
            in_ep, start = True, i
        elif not v and in_ep:
            in_ep = False
            if i - start >= min_duration:
                episodes.append((start, i - 1))
    if in_ep and len(f) - start >= min_duration:
        episodes.append((start, len(f) - 1))
    return episodes
