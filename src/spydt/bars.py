"""Information-driven bars (AFML ch. 2.3) on 1-minute buckets.

The atomic unit is the 1-minute bucket (tick data unavailable; tick-rule
signing on 1-min closes is the standard construction). Bars are built
per morning window with a threshold calibrated on training folds only.

Pilot simplification (documented in the paper/report): thresholds are
bisection-calibrated constants per training fold rather than EWMA-adaptive;
adaptive expectations are deferred to the full run. Floor/cap rules guard
against degenerate one-bucket bars and runaway bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_KINDS = ("dollar", "dollar_imbalance", "volume", "tick")


def tick_rule_signs(close: np.ndarray) -> np.ndarray:
    """b_t = sign(delta p); zero change carries the prior sign; b_0 = +1."""
    d = np.diff(close, prepend=close[0])
    signs = np.sign(d)
    signs[0] = 1.0
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return signs


def _flow(window: pd.DataFrame, kind: str) -> np.ndarray:
    p = window["close"].to_numpy(float)
    v = window["volume"].to_numpy(float)
    if kind == "dollar":
        return p * v
    if kind == "dollar_imbalance":
        return tick_rule_signs(p) * p * v
    if kind == "volume":
        return v
    if kind == "tick":
        return np.ones_like(v)
    raise ValueError(f"unknown bar kind {kind!r}")


def build_bars_for_window(
    window: pd.DataFrame,
    *,
    kind: str,
    threshold: float,
    floor_buckets: int = 2,
    cap_buckets: int | None = None,
) -> pd.DataFrame:
    """Build bars over one morning window of 1-minute buckets.

    A bar closes when |accumulated flow| >= threshold (imbalance kinds use the
    signed accumulator), subject to a floor (min buckets per bar, guards the
    info clock from collapsing onto clock time) and a cap (max buckets per
    bar, guards against runaway bars). Only COMPLETED bars are returned.
    """
    flow = _flow(window, kind)
    close = window["close"].to_numpy(float)
    rows = []
    acc = 0.0
    start = 0
    for i in range(len(flow)):
        acc += flow[i]
        n = i - start + 1
        crossed = abs(acc) >= threshold and n >= floor_buckets
        capped = cap_buckets is not None and n >= cap_buckets
        if crossed or capped:
            rows.append(
                {
                    "start_idx": start,
                    "end_idx": i,
                    "n_buckets": n,
                    "close": close[i],
                    "abs_flow": abs(acc),
                    "floor_triggered": bool(abs(acc) >= threshold and n == floor_buckets
                                            and floor_buckets > 1),
                    "cap_triggered": bool(capped and not crossed),
                }
            )
            acc = 0.0
            start = i + 1
    return pd.DataFrame(
        rows,
        columns=["start_idx", "end_idx", "n_buckets", "close", "abs_flow",
                 "floor_triggered", "cap_triggered"],
    )


def calibrate_threshold(
    windows: list[pd.DataFrame],
    *,
    kind: str,
    k_target: int,
    floor_buckets: int = 2,
    n_iter: int = 40,
) -> float:
    """Bisect a constant threshold so the MEDIAN window yields k_target bars.

    Calibration data must come from training folds only — enforced by the
    caller (tensor builder passes train-fold windows exclusively).
    """
    totals = np.array([np.abs(_flow(w, kind)).sum() for w in windows])
    lo = float(np.median(totals) / (k_target * 20))
    hi = float(np.median(totals) * 2)

    def med_count(t: float) -> float:
        return float(
            np.median(
                [len(build_bars_for_window(w, kind=kind, threshold=t,
                                           floor_buckets=floor_buckets))
                 for w in windows]
            )
        )

    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)  # geometric bisection: counts ~ 1/threshold
        if med_count(mid) >= k_target:
            lo = mid
        else:
            hi = mid
    return float(lo)


def first_k_bar_series(
    bars: pd.DataFrame, window: pd.DataFrame, k: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """First k bar-close log prices, a validity mask, and the clock span.

    Days with fewer than k completed bars are padded (mask False) — padding
    value is the last valid log price so downstream returns are zero, and the
    mask channel carries the information that the tail is padding.
    Returns (log_prices[k], mask[k], clockspan_minutes).
    """
    logp = np.log(bars["close"].to_numpy(float))[:k]
    n = len(logp)
    mask = np.zeros(k, dtype=bool)
    mask[:n] = True
    if n == 0:
        pad_val = float(np.log(window["close"].iloc[0]))
        clockspan = 0.0
    else:
        pad_val = float(logp[-1])
        clockspan = float(bars["end_idx"].iloc[min(n, len(bars)) - 1] + 1)
    out = np.full(k, pad_val)
    out[:n] = logp
    return out, mask, clockspan
