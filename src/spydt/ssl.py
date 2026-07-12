"""SSL corpus: rolling 90-min windows from the pretraining era ONLY.

Temporal firewall: the era must end strictly before the first supervised day
(asserted by the caller and leakage gate (e)). All statistics (bar threshold,
FFD d*, scalers) are fit inside the era.

Construction mirrors the supervised pipeline semantics exactly: each 90-min
window is barred INDEPENDENTLY on its own buckets (as supervised mornings
are), with the threshold calibrated on era MORNING windows; FFD runs over
per-offset lanes (the stream of same-offset windows across days) so windows
never suffer per-window burn-in. Windows may span afternoon tape — legitimate
for unlabeled pretraining; the <15:30 contract binds supervised features only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spydt.axes import DayData, MORNING_MINUTES
from spydt.bars import build_bars_for_window, calibrate_threshold
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
    n_days = len(days)
    cap_buckets = max(int(cap_mult * MORNING_MINUTES / K), floor_buckets + 1)

    calib = [pd.DataFrame({"close": d.close[:MORNING_MINUTES],
                           "volume": d.volume[:MORNING_MINUTES]})
             for d in days[: min(300, n_days)]]
    threshold = calibrate_threshold(calib, kind="dollar", k_target=K,
                                    floor_buckets=floor_buckets)

    starts = list(range(0, SESSION_MINUTES - MORNING_MINUTES + 1, stride_min))

    # clock axis: one continuous full-RTH stream (causal FFD, then slice)
    clock_logp = np.concatenate([np.log(d.close) for d in days])
    d_clock = select_dstar(clock_logp, grid_step=ffd_grid_step)
    clock_t = apply_ffd(clock_logp, ffd_weights(d_clock)).reshape(
        n_days, SESSION_MINUTES)

    # info axis: bar each window independently; FFD per offset lane
    win_bars: dict[tuple[int, int], np.ndarray] = {}
    lane_logp: dict[int, list[np.ndarray]] = {s: [] for s in starts}
    lane_len: dict[int, list[int]] = {s: [] for s in starts}
    for i, d in enumerate(days):
        for s in starts:
            w = pd.DataFrame({"close": d.close[s : s + MORNING_MINUTES],
                              "volume": d.volume[s : s + MORNING_MINUTES]})
            bars = build_bars_for_window(w, kind="dollar", threshold=threshold,
                                         floor_buckets=floor_buckets,
                                         cap_buckets=cap_buckets)
            v = np.log(bars["close"].to_numpy(float)) if len(bars) else np.empty(0)
            lane_logp[s].append(v)
            lane_len[s].append(len(v))

    d_info = select_dstar(np.concatenate(lane_logp[0]), grid_step=ffd_grid_step)
    w_info = ffd_weights(d_info)
    for s in starts:
        stream = np.concatenate(lane_logp[s]) if lane_logp[s] else np.empty(0)
        vals = apply_ffd(stream, w_info)
        pos = 0
        for i, ln in enumerate(lane_len[s]):
            win_bars[(i, s)] = vals[pos : pos + ln]
            pos += ln

    sc_clock = Scaler.fit(clock_t[~np.isnan(clock_t)])
    all_info = np.concatenate([v for v in win_bars.values() if len(v)])
    sc_info = Scaler.fit(all_info[~np.isnan(all_info)])

    clocks, infos, masks, prov = [], [], [], []
    for i, d in enumerate(days):
        for s in starts:
            seg = clock_t[i][s : s + MORNING_MINUTES]
            v = win_bars[(i, s)][:K]
            if np.isnan(seg).any() or len(v) < 2 or np.isnan(v).any():
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
                 "last_day": days[-1].date,
                 "construction": "per-window bars, era-morning threshold, "
                                 "per-offset FFD lanes (v2, post-gate-fix)"},
    }
