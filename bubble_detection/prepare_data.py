"""Resample intraday (e.g. 1-minute) OHLCV data to a coarser bar size for the
bubble pipeline, and normalise the schema to what run_pipeline.py expects.

Works with plain CSVs (datetime,open,high,low,close,volume in any casing /
column order, datetime as a column or the index) and with yfinance
triple-header exports.

Examples:
    python prepare_data.py spy_1min_2008_2021_cleaned.csv data/SPY_h.csv --rule 1h
    python prepare_data.py spy_1min.csv data/SPY_5m.csv --rule 5min
"""
from __future__ import annotations

import argparse

import pandas as pd


def load_any_csv(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=3)
    if head.columns[0] == "Price":                       # yfinance triple header
        df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(path)
        dt_col = None
        for c in df.columns:
            if c.lower() in ("datetime", "date", "timestamp", "time", "dt"):
                dt_col = c
                break
        if dt_col is None:
            dt_col = df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, format="mixed")
        df = df.set_index(dt_col)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df.index.name = "datetime"
    return df[~df.index.duplicated(keep="first")].sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {}
    for col, how in (("Open", "first"), ("High", "max"), ("Low", "min"),
                     ("Close", "last"), ("Volume", "sum")):
        if col in df.columns:
            agg[col] = how
    out = df.resample(rule).agg(agg).dropna(subset=["Close"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rule", default="1h", help="pandas resample rule (1h, 5min, 30min, 1D)")
    args = ap.parse_args()
    df = load_any_csv(args.src)
    out = resample_ohlcv(df, args.rule)
    out.to_csv(args.dst)
    print(f"{args.src}: {len(df)} rows -> {args.dst}: {len(out)} bars "
          f"({out.index[0]} .. {out.index[-1]})")


if __name__ == "__main__":
    main()
