"""Per-day dual-time axes: clock series, information-bar series, tabular
features, labels. All statistics (bar thresholds, FFD d*, scalers) are fit on
train-fold days only and applied causally everywhere else.

The information clock is a CONTINUOUS bar stream over MORNING windows only
(09:30-11:00), concatenated across days; the threshold is calibrated so the
median TRAIN morning yields K bars. Nothing from any day's post-11:00 tape
can touch a feature — enforced by leakage gate (b), which corrupts all
post-11:00 non-label minutes and asserts bit-identical features. FFD runs
causally over each axis's continuous stream, so burn-in only affects the
stream start, never individual windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spydt.bars import build_bars_for_window, calibrate_threshold
from spydt.encode import Scaler, interp_to, paa
from spydt.frac import apply_ffd, ffd_weights, select_dstar

MORNING_MINUTES = 90          # 09:30–11:00
DECISION_MINUTE = 360         # bar index of 15:30 within the 390-bar session
LAST_MINUTE = 389             # 15:59 bar (close proxies the official close)
ENTRY_MINUTE = 361            # 15:31 bar (entry at its open)


@dataclass
class DayData:
    """One session's raw arrays (all timestamps < 16:00 ET by construction)."""

    date: str
    close: np.ndarray           # 390 closes
    open_: np.ndarray           # 390 opens
    volume: np.ndarray          # 390 share volumes
    prev_close: float | None    # prior session 15:59 close


@dataclass
class AxisBundle:
    """Everything the models consume for one day set, one configuration."""

    dates: list[str]
    clock: np.ndarray           # (n, L) scaled clock-axis series
    info: np.ndarray            # (n, L) scaled info-axis series
    info_mask: np.ndarray       # (n, L) validity (False = padded)
    tabular: np.ndarray         # (n, 5) standardized
    y: np.ndarray               # (n,) sign labels in {-1, +1}
    w: np.ndarray               # (n,) |return| sample weights (mean 1)
    r_label: np.ndarray         # (n,) 15:30 close -> 15:59 close log return
    r_exec: np.ndarray          # (n,) executable: 15:31 open -> 15:59 close
    diag: dict = field(default_factory=dict)


def load_daydata(parquet_path: str, cls_path: str, volume_multiplier: float) -> list[DayData]:
    """Included full sessions, chronological, with prior-session close."""
    rth = pd.read_parquet(parquet_path)
    cls = pd.read_parquet(cls_path)
    included = set(cls.index[cls["included"]])
    out: list[DayData] = []
    prev_close: float | None = None
    for date, g in rth.groupby(rth.index.date):
        d = str(date)
        if len(g) == 390:
            if d in included:
                out.append(
                    DayData(
                        d,
                        g["close"].to_numpy(float),
                        g["open"].to_numpy(float),
                        g["volume"].to_numpy(float) * volume_multiplier,
                        prev_close,
                    )
                )
            prev_close = float(g["close"].iloc[-1])
        elif len(g):
            prev_close = float(g["close"].iloc[-1])
    return out


def _session_frame(day: DayData) -> pd.DataFrame:
    return pd.DataFrame({"close": day.close, "volume": day.volume})


def _morning_frame(day: DayData) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": day.close[:MORNING_MINUTES], "volume": day.volume[:MORNING_MINUTES]}
    )


def calibrate_bar_threshold(
    days: list[DayData], train_pos: np.ndarray, *, kind: str,
    k_target: int, floor_buckets: int, n_calib: int = 300,
) -> float:
    """Threshold so the median TRAIN morning yields k_target bars."""
    rng = np.random.default_rng(0)
    pick = rng.choice(train_pos, size=min(n_calib, len(train_pos)), replace=False)
    windows = [_morning_frame(days[i]) for i in np.sort(pick)]
    return calibrate_threshold(
        windows, kind=kind, k_target=k_target, floor_buckets=floor_buckets
    )


def day_bars(day: DayData, *, kind: str, threshold: float,
             floor_buckets: int, cap_buckets: int) -> pd.DataFrame:
    """Morning-window bars only — features may not touch post-11:00 tape."""
    return build_bars_for_window(
        _morning_frame(day), kind=kind, threshold=threshold,
        floor_buckets=floor_buckets, cap_buckets=cap_buckets,
    )


def build_bundle(
    days: list[DayData],
    train_pos: np.ndarray,
    *,
    bar_kind: str,
    K: int,
    L: int,
    floor_buckets: int = 2,
    cap_mult: float = 3.0,
    ffd_grid_step: float = 0.1,
    input_mode: str = "ffd",   # "ffd" | "volscaled"
) -> AxisBundle:
    """Assemble the full axis bundle; every fitted statistic uses train_pos only."""
    n = len(days)
    train_set = set(int(i) for i in train_pos)
    cap_buckets = max(int(cap_mult * MORNING_MINUTES / K), floor_buckets + 1)

    threshold = calibrate_bar_threshold(
        days, train_pos, kind=bar_kind, k_target=K, floor_buckets=floor_buckets,
    )

    # --- continuous streams --------------------------------------------------
    clock_logp = np.concatenate([np.log(d.close[:MORNING_MINUTES]) for d in days])
    all_bars = [day_bars(d, kind=bar_kind, threshold=threshold,
                         floor_buckets=floor_buckets, cap_buckets=cap_buckets)
                for d in days]
    bar_logp = np.concatenate(
        [np.log(b["close"].to_numpy(float)) if len(b) else np.empty(0) for b in all_bars]
    )
    bar_day = np.concatenate(
        [np.full(len(b), i) for i, b in enumerate(all_bars)]
    ).astype(int) if len(bar_logp) else np.empty(0, int)

    # --- per-axis transformation (FFD on train-fold d*, or vol-scaled rets) --
    clock_train_mask = np.repeat(
        np.array([i in train_set for i in range(n)]), MORNING_MINUTES
    )
    if input_mode == "ffd":
        d_clock = select_dstar(clock_logp[clock_train_mask], grid_step=ffd_grid_step)
        d_info = select_dstar(bar_logp[np.isin(bar_day, list(train_set))],
                              grid_step=ffd_grid_step) if len(bar_logp) else 0.0
        clock_t = apply_ffd(clock_logp, ffd_weights(d_clock))
        info_t = apply_ffd(bar_logp, ffd_weights(d_info))
        diag_d = {"d_clock": d_clock, "d_info": d_info}
    else:  # vol-scaled 1-step returns (EWMA sigma fit causally, half-life 5 days)
        clock_r = np.diff(clock_logp, prepend=np.nan)
        info_r = np.diff(bar_logp, prepend=np.nan) if len(bar_logp) else bar_logp
        clock_t = _ewma_volscale(clock_r, halflife=5 * MORNING_MINUTES)
        info_t = _ewma_volscale(info_r, halflife=5 * K)
        diag_d = {"d_clock": None, "d_info": None}

    # --- slice per day, length-normalize to L --------------------------------
    clock_days = clock_t.reshape(n, MORNING_MINUTES)
    clock_L = np.full((n, L), np.nan)
    info_L = np.full((n, L), np.nan)
    mask_L = np.zeros((n, L), dtype=bool)
    clockspan = np.zeros(n)
    pad_flag = np.zeros(n)
    morning_count = np.zeros(n, dtype=int)
    floor_rate = np.zeros(n)
    cap_rate = np.zeros(n)

    for i, (d, bars) in enumerate(zip(days, all_bars)):
        row = clock_days[i]
        if np.isnan(row).any():          # FFD burn-in at stream start
            continue
        clock_L[i] = paa(row, L)

        vals = info_t[bar_day == i] if len(bar_logp) else np.empty(0)
        morning = bars[bars["end_idx"] < MORNING_MINUTES] if len(bars) else bars
        m = len(morning)
        morning_count[i] = m
        if len(bars):
            floor_rate[i] = float(morning["floor_triggered"].mean()) if m else 0.0
            cap_rate[i] = float(morning["cap_triggered"].mean()) if m else 0.0
        take = min(m, K)
        v = vals[:take]
        if np.isnan(v).any() or take < 2:
            pad_flag[i] = 1.0
            continue
        series_k = np.full(K, v[-1])
        series_k[:take] = v
        mask_k = np.zeros(K, dtype=bool)
        mask_k[:take] = True
        info_L[i] = interp_to(series_k, L)
        mask_L[i] = interp_to(mask_k.astype(float), L) > 0.5
        pad_flag[i] = float(take < K)
        clockspan[i] = float(morning["end_idx"].iloc[take - 1] + 1)

    # --- scaling (train-fold quantiles, frozen, clipped) ---------------------
    train_rows = np.array([i in train_set for i in range(n)])
    ok = ~np.isnan(clock_L).any(axis=1) & ~np.isnan(info_L).any(axis=1)
    sc_clock = Scaler.fit(clock_L[train_rows & ok].ravel())
    sc_info = Scaler.fit(info_L[train_rows & ok].ravel())
    clock_s = np.where(ok[:, None], sc_clock(np.nan_to_num(clock_L)), np.nan)
    info_s = np.where(ok[:, None], sc_info(np.nan_to_num(info_L)), np.nan)

    # --- tabular, labels, returns --------------------------------------------
    r_on = np.array(
        [np.log(d.open_[0] / d.prev_close) if d.prev_close else 0.0 for d in days]
    )
    rvol = np.array(
        [np.std(np.diff(np.log(d.close[:MORNING_MINUTES]))) * np.sqrt(390) for d in days]
    )
    dow = np.array([pd.Timestamp(d.date).dayofweek / 4.0 for d in days])
    tab_raw = np.column_stack([r_on, rvol, clockspan / MORNING_MINUTES, pad_flag, dow])
    mu = tab_raw[train_rows & ok].mean(axis=0)
    sd = tab_raw[train_rows & ok].std(axis=0) + 1e-9
    tabular = (tab_raw - mu) / sd

    r_label = np.array(
        [np.log(d.close[LAST_MINUTE] / d.close[DECISION_MINUTE]) for d in days]
    )
    r_exec = np.array(
        [np.log(d.close[LAST_MINUTE] / d.open_[ENTRY_MINUTE]) for d in days]
    )
    y = np.where(r_label >= 0, 1.0, -1.0)
    w = np.abs(r_label)
    w = np.clip(w / (w[train_rows & ok].mean() + 1e-12), 0.1, 3.0)

    diag = {
        **diag_d,
        "threshold": threshold,
        "k_target": K,
        "cap_buckets": cap_buckets,
        "median_bars_per_day": float(np.median([len(b) for b in all_bars])),
        "median_morning_bars": float(np.median(morning_count)),
        "pad_rate": float(pad_flag[ok].mean()) if ok.any() else 1.0,
        "floor_rate": float(np.mean(floor_rate[ok])) if ok.any() else 1.0,
        "cap_rate": float(np.mean(cap_rate[ok])) if ok.any() else 1.0,
        "n_ok": int(ok.sum()),
    }
    return AxisBundle(
        dates=[d.date for d in days],
        clock=clock_s, info=info_s, info_mask=mask_L, tabular=tabular,
        y=y, w=w, r_label=r_label, r_exec=r_exec, diag=diag,
    )


def _ewma_volscale(r: np.ndarray, *, halflife: float) -> np.ndarray:
    s = pd.Series(r)
    sigma = s.pow(2).ewm(halflife=halflife, min_periods=20).mean().pow(0.5).shift(1)
    return (s / (sigma + 1e-9)).to_numpy()


def axis_similarity(bundle: AxisBundle) -> dict:
    """Falsifiability diagnostic: per-day clock-vs-info similarity.

    If information time is indistinguishable from clock time, the dual-time
    hypothesis is untestable on this data — reported, and the pilot flags it.
    """
    cors = []
    for i in range(len(bundle.dates)):
        c, z = bundle.clock[i], bundle.info[i]
        if np.isnan(c).any() or np.isnan(z).any():
            continue
        if c.std() < 1e-9 or z.std() < 1e-9:
            continue
        cors.append(float(np.corrcoef(c, z)[0, 1]))
    cors = np.array(cors)
    return {
        "n": len(cors),
        "median_abs_corr": float(np.median(np.abs(cors))),
        "q90_abs_corr": float(np.quantile(np.abs(cors), 0.9)),
        "frac_above_0.95": float((np.abs(cors) > 0.95).mean()),
    }
