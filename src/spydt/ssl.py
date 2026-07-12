"""SSL corpus: rolling 90-min windows from the pretraining era ONLY.

Temporal firewall: the era must end strictly before the first supervised day
(asserted here and re-checked by the leakage gates). All statistics (bar
threshold, FFD d*, scalers) are fit inside the era.
"""
from __future__ import annotations

import numpy as np

from spydt.axes import DayData, MORNING_MINUTES, day_bars
from spydt.bars import calibrate_threshold
from spydt.encode import Scaler, interp_to, paa
from spydt.frac import apply_ffd, ffd_weights, select_dstar

SESSION_MINUTES = 390


def build_ssl_windows(
    days: list[DayData],
    *,
    K: int,
    L: int,
    stride_min: int,
    floor_buckets: int = 2,
    cap_mult: float = 3.0,
    ffd_grid_step: float = 0.1,
) -> dict:
    """Returns dict with clock (n,L), info (n,L), mask (n,L), provenance."""
    import pandas as pd

    n_days = len(days)
    bars_per_day_target = int(round(K * SESSION_MINUTES / MORNING_MINUTES))
    cap_buckets = max(int(cap_mult * SESSION_MINUTES / bars_per_day_target),
                      floor_buckets + 1)

    calib = [pd.DataFrame({"close": d.close, "volume": d.volume})
             for d in days[: min(300, n_days)]]
    threshold = calibrate_threshold(calib, kind="dollar",
                                    k_target=bars_per_day_target,
                                    floor_buckets=floor_buckets)

    clock_logp = np.concatenate([np.log(d.close) for d in days])
    all_bars = [day_bars(d, kind="dollar", threshold=threshold,
                         floor_buckets=floor_buckets, cap_buckets=cap_buckets)
                for d in days]
    bar_logp = np.concatenate(
        [np.log(b["close"].to_numpy(float)) if len(b) else np.empty(0)
         for b in all_bars]
    )
    bar_day = np.concatenate([np.full(len(b), i) for i, b in enumerate(all_bars)]).astype(int)

    d_clock = select_dstar(clock_logp, grid_step=ffd_grid_step)
    d_info = select_dstar(bar_logp, grid_step=ffd_grid_step)
    clock_t = apply_ffd(clock_logp, ffd_weights(d_clock)).reshape(n_days, SESSION_MINUTES)
    info_t = apply_ffd(bar_logp, ffd_weights(d_info))

    sc_clock = Scaler.fit(clock_t[~np.isnan(clock_t)])
    sc_info = Scaler.fit(info_t[~np.isnan(info_t)])

    starts = list(range(0, SESSION_MINUTES - MORNING_MINUTES + 1, stride_min))
    clocks, infos, masks, prov = [], [], [], []
    for i, (d, bars) in enumerate(zip(days, all_bars)):
        row = clock_t[i]
        vals = info_t[bar_day == i]
        ends = bars["end_idx"].to_numpy(int) if len(bars) else np.empty(0, int)
        for s in starts:
            seg = row[s : s + MORNING_MINUTES]
            if np.isnan(seg).any():
                continue
            in_win = (ends >= s) & (ends < s + MORNING_MINUTES)
            v = vals[in_win][:K]
            if len(v) < 2 or np.isnan(v).any():
                continue
            series_k = np.full(K, v[-1])
            series_k[: len(v)] = v
            mask_k = np.zeros(K, dtype=bool)
            mask_k[: len(v)] = True
            clocks.append(sc_clock(paa(seg, L)))
            infos.append(sc_info(interp_to(series_k, L)))
            masks.append(interp_to(mask_k.astype(float), L) > 0.5)
            prov.append((d.date, s))
    return {
        "clock": np.array(clocks, dtype=np.float32),
        "info": np.array(infos, dtype=np.float32),
        "mask": np.array(masks),
        "provenance": prov,
        "meta": {"d_clock": d_clock, "d_info": d_info, "threshold": threshold,
                 "n_windows": len(clocks), "first_day": days[0].date,
                 "last_day": days[-1].date},
    }
