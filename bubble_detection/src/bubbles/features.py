"""Assemble every statistical test into a causal (walk-forward safe) feature
matrix.

Every column's value at bar t is a deterministic function of data up to and
including bar t only:
  * the windowed test statistics use trailing windows ending at t;
  * the HMM is refit walk-forward and filtered (never smoothed);
  * BSADF critical values come from Monte Carlo under the null (no data);
  * stride-computed features are forward-filled (a value computed at an
    earlier bar is still known at t).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import psy, stability, bai_perron, ruptures_feats, hmm_feats


@dataclass
class FeatureConfig:
    # PSY explosiveness (hourly bars: ~7/day)
    bsadf_min_window: int = 40          # ~1.2 weeks
    bsadf_max_window: int = 400         # ~2.7 months
    adf_lags: int = 1
    bsadf_start_step: int = 1
    rtadf_window: int = 168             # ~5 weeks
    cv_quantile: float = 0.95
    mc_sims: int = 200
    # stability / breaks
    stab_window: int = 336              # ~2.3 months
    qlr_trim: float = 0.15
    bp_max_breaks: int = 3
    bp_min_seg: int = 30
    bp_grid_step: int = 5
    bp_stride: int = 8
    rup_stride: int = 5
    # HMM
    hmm_states: int = 3
    hmm_min_history: int = 300
    hmm_refit_every: int = 50
    # controls
    mom_windows: tuple = (7, 35)
    vol_window: int = 35
    seed: int = 42
    cache_path: str | None = None


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, dict]:
    y = df["log_close"].to_numpy()
    r = df["ret"].to_numpy()
    n = y.size
    out = pd.DataFrame(index=df.index)

    # ---- PSY explosiveness ------------------------------------------------
    out["rtadf"] = psy.rolling_rtadf(y, window=cfg.rtadf_window, lags=cfg.adf_lags)
    bs = psy.bsadf(y, cfg.bsadf_min_window, cfg.bsadf_max_window,
                   cfg.adf_lags, cfg.bsadf_start_step)
    cvs = psy.mc_critical_values(n, cfg.bsadf_min_window, cfg.bsadf_max_window,
                                 cfg.adf_lags, cfg.bsadf_start_step,
                                 n_sims=cfg.mc_sims, seed=cfg.seed,
                                 cache_path=cfg.cache_path)
    cv = cvs["bsadf_cv"][cfg.cv_quantile]
    out["bsadf"] = bs
    out["bsadf_gap"] = bs - cv
    flag = (bs > cv).astype(float)
    flag[~np.isfinite(bs)] = np.nan
    out["bubble_flag"] = flag
    # consecutive bars the explosiveness flag has been on
    age = np.zeros(n)
    for t in range(1, n):
        age[t] = age[t - 1] + 1 if flag[t] == 1.0 else 0.0
    out["bubble_age"] = age

    # ---- stability tests ---------------------------------------------------
    cusum, cusumsq = stability.rolling_cusum(y, window=cfg.stab_window)
    out["cusum"] = cusum
    out["cusum_sq"] = cusumsq
    out["chow_f"] = stability.rolling_chow(y, window=cfg.stab_window)
    qlr_f, qlr_loc = stability.rolling_qlr(y, window=cfg.stab_window, trim=cfg.qlr_trim)
    out["qlr_f"] = qlr_f
    out["qlr_loc"] = qlr_loc

    # ---- multiple breaks ----------------------------------------------------
    bp_n, bp_since, bp_dmean = bai_perron.rolling_bai_perron(
        y, window=cfg.stab_window, max_breaks=cfg.bp_max_breaks,
        min_seg=cfg.bp_min_seg, grid_step=cfg.bp_grid_step, stride=cfg.bp_stride)
    out["bp_nbreaks"] = bp_n
    out["bp_since"] = bp_since
    out["bp_dmean"] = bp_dmean

    rp_n, rp_since, rp_lvr = ruptures_feats.rolling_pelt(
        np.nan_to_num(r, nan=0.0), window=cfg.stab_window, stride=cfg.rup_stride)
    out["rup_ncp"] = rp_n
    out["rup_since"] = rp_since
    out["rup_lvr"] = rp_lvr

    # ---- HMM regimes ---------------------------------------------------------
    hmm_cols = hmm_feats.walkforward_hmm(np.nan_to_num(r, nan=0.0),
                                         n_states=cfg.hmm_states,
                                         min_history=cfg.hmm_min_history,
                                         refit_every=cfg.hmm_refit_every,
                                         seed=cfg.seed)
    for k, v in hmm_cols.items():
        out[k] = v

    # ---- simple controls ------------------------------------------------------
    lc = df["log_close"]
    for w in cfg.mom_windows:
        out[f"mom_{w}"] = lc.diff(w)
    out[f"vol_{cfg.vol_window}"] = df["ret"].rolling(cfg.vol_window).std()

    meta = {
        "gsadf_stat": psy.gsadf_stat(bs),
        "gsadf_cv": cvs["gsadf_cv"],
        "bsadf_cv_series": cv,
    }
    return out, meta
