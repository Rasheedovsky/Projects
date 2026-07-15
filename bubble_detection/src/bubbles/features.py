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
from .ssa import rolling_ssa_denoise


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
    hmm_max_history: int | None = None
    hmm_filter_warm: int = 2000
    # SSA denoising (feeds the structural-stability tests only)
    use_ssa: bool = False
    ssa_window: int = 168
    ssa_max_rank: int = 8
    ssa_var_frac: float = 0.90
    ssa_stride: int = 1
    # controls
    mom_windows: tuple = (7, 35)
    vol_window: int = 35
    seed: int = 42
    cache_path: str | None = None

    def scale_for_length(self, n: int) -> "FeatureConfig":
        """Adapt stride/refit parameters to the sample size so large
        intraday datasets stay tractable (feature definitions unchanged);
        for very short series, shrink the test windows so features exist
        over a useful fraction of the sample."""
        if n < 600:
            self.bsadf_min_window, self.bsadf_max_window = 15, 120
            self.rtadf_window = 40
            self.stab_window = 60
            self.bp_min_seg, self.bp_grid_step, self.bp_max_breaks = 12, 2, 2
            self.bp_stride = self.rup_stride = 1
            self.hmm_states, self.hmm_min_history, self.hmm_refit_every = 2, 50, 10
            self.ssa_window = 40
            self.mom_windows, self.vol_window = (5, 21), 21
        elif n > 150000:
            self.rup_stride, self.bp_stride, self.ssa_stride = 30, 32, 8
            self.hmm_refit_every, self.hmm_max_history = 1500, 15000
        elif n > 60000:
            self.rup_stride, self.bp_stride, self.ssa_stride = 20, 24, 6
            self.hmm_refit_every, self.hmm_max_history = 500, 20000
        elif n > 15000:
            self.rup_stride, self.bp_stride, self.ssa_stride = 10, 12, 2
            self.hmm_refit_every, self.hmm_max_history = 150, 10000
        return self


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

    # ---- optional SSA cleaning for the structural tests ---------------------
    # PSY tests stay on the raw series (their MC critical values assume an
    # unfiltered random-walk null); the stability/break tests run on the
    # SSA-cleaned level, where high-frequency noise costs them the most power.
    y_struct = y
    off = 0
    if cfg.use_ssa:
        y_clean = rolling_ssa_denoise(y, window=cfg.ssa_window,
                                      max_rank=cfg.ssa_max_rank,
                                      var_frac=cfg.ssa_var_frac,
                                      stride=cfg.ssa_stride)
        out["ssa_noise_vol"] = (pd.Series(y - y_clean, index=df.index)
                                .rolling(cfg.vol_window).std())
        off = cfg.ssa_window - 1                    # NaN head of the cleaned series
        y_struct = y_clean[off:]

    def _shift(a):
        return np.concatenate([np.full(off, np.nan), a]) if off else a

    # ---- stability tests ---------------------------------------------------
    cusum, cusumsq = stability.rolling_cusum(y_struct, window=cfg.stab_window)
    out["cusum"] = _shift(cusum)
    out["cusum_sq"] = _shift(cusumsq)
    out["chow_f"] = _shift(stability.rolling_chow(y_struct, window=cfg.stab_window))
    qlr_f, qlr_loc = stability.rolling_qlr(y_struct, window=cfg.stab_window, trim=cfg.qlr_trim)
    out["qlr_f"] = _shift(qlr_f)
    out["qlr_loc"] = _shift(qlr_loc)

    # ---- multiple breaks ----------------------------------------------------
    bp_n, bp_since, bp_dmean = bai_perron.rolling_bai_perron(
        y_struct, window=cfg.stab_window, max_breaks=cfg.bp_max_breaks,
        min_seg=cfg.bp_min_seg, grid_step=cfg.bp_grid_step, stride=cfg.bp_stride)
    bp_n, bp_since, bp_dmean = _shift(bp_n), _shift(bp_since), _shift(bp_dmean)
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
                                         seed=cfg.seed,
                                         max_history=cfg.hmm_max_history,
                                         filter_warm=cfg.hmm_filter_warm)
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
