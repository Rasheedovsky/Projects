"""Data loading for yfinance-style multi-header CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_yf_csv(path: str) -> pd.DataFrame:
    """Load a yfinance CSV with the (Price / Ticker / Datetime) triple header.

    Returns a tz-aware, sorted, de-duplicated OHLCV frame with a
    ``log_close`` and log-return ``ret`` column.
    """
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Close"])
    df["log_close"] = np.log(df["Close"].astype(float))
    df["ret"] = df["log_close"].diff()
    return df
