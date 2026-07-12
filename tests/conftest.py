"""Synthetic fixtures for Phase-0 tests.

All fixtures build tz-naive "file-clock" frames (ET - 2h) or tz-aware ET frames
with known, controllable defects so audit functions can be tested against ground
truth rather than against the real (unverified) dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ET = "America/New_York"


def make_rth_day(
    date: str,
    *,
    tz_aware: bool = True,
    n_minutes: int = 390,
    base_price: float = 100.0,
    open_spike: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """One synthetic RTH session: 390 one-minute bars, 09:30..15:59 ET."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{date} 09:30", periods=n_minutes, freq="1min", tz=ET if tz_aware else None)
    rets = rng.normal(0, 3e-4, n_minutes)
    close = base_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1e-4, n_minutes)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1e-4, n_minutes)))
    volume = rng.integers(1_000, 5_000, n_minutes).astype(float)
    if open_spike:
        volume[0] = 80_000.0  # opening-minute volume spike, used by offset inference
        volume[-1] = 60_000.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_multi_day(dates: list[str], **kwargs) -> pd.DataFrame:
    frames = [make_rth_day(d, seed=i, **kwargs) for i, d in enumerate(dates)]
    return pd.concat(frames)


@pytest.fixture
def clean_day() -> pd.DataFrame:
    return make_rth_day("2015-06-15")


@pytest.fixture
def clean_week() -> pd.DataFrame:
    return make_multi_day(["2015-06-15", "2015-06-16", "2015-06-17", "2015-06-18", "2015-06-19"])
