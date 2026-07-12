"""Data loading for yfinance-style multi-header CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_yf_csv(path: str) -> pd.DataFrame:
    """Load a price CSV — yfinance triple-header exports or plain
    (datetime, open, high, low, close, volume) files in any column casing.

    Returns a tz-aware, sorted, de-duplicated OHLCV frame with a
    ``log_close`` and log-return ``ret`` column.
    """
    head = pd.read_csv(path, nrows=3)
    if head.columns[0] == "Price":                       # yfinance triple header
        df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(path)
        dt_col = next((c for c in df.columns
                       if c.lower() in ("datetime", "date", "timestamp", "time", "dt")),
                      df.columns[0])
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, format="mixed")
        df = df.set_index(dt_col)
        df.columns = [c.strip().capitalize() for c in df.columns]
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Close"])
    df["log_close"] = np.log(df["Close"].astype(float))
    df["ret"] = df["log_close"].diff()
    return df
