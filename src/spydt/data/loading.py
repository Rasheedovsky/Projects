"""Raw CSV parsing, file-clock offset inference, ET normalization.

The raw file (IB-style export) is unsorted, tz-naive, and on an unknown wall
clock. Loading is split into three explicit steps so each is testable:

1. ``parse_raw_csv``   — schema, dtypes, chronological sort, de-duplication.
2. ``infer_et_offset_hours`` — evidence-based offset estimate: NYSE opens at
   09:30 ET with a large opening-minute volume spike, so the modal spike minute
   pins the file clock's offset from ET.
3. ``normalize_to_et`` — apply a *claimed* offset (from config), optionally
   verifying it against the inferred one; hard-fails on contradiction.
"""
from __future__ import annotations

from typing import IO

import numpy as np
import pandas as pd

ET = "America/New_York"

_RAW_COLUMNS = {
    "date": "dt_file",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "barCount": "bar_count",
    "average": "average",
}


def parse_raw_csv(path_or_buf: str | IO[str]) -> pd.DataFrame:
    """Parse the raw export into a chronologically sorted, de-duplicated frame.

    Index: naive file-clock timestamps (``dt_file``). Columns: open, high,
    low, close, volume, bar_count, average (floats).
    """
    df = pd.read_csv(path_or_buf, index_col=0)
    missing = set(_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"raw csv missing columns: {sorted(missing)}")
    df = df.rename(columns=_RAW_COLUMNS)
    # format: "20090522  07:30:00" (double space)
    dt = pd.to_datetime(df["dt_file"].str.replace(r"\s+", " ", regex=True),
                        format="%Y%m%d %H:%M:%S")
    df = df.drop(columns=["dt_file"]).set_index(dt.rename("dt_file"))
    df = df[["open", "high", "low", "close", "volume", "bar_count", "average"]]
    df = df.astype(float).sort_index()

    dup_mask = df.index.duplicated(keep=False)
    n_conflicting = 0
    if dup_mask.any():
        # duplicates are only droppable if they are exact re-exports
        n_conflicting = int(
            (df[dup_mask].groupby(level=0).nunique().max(axis=1) > 1).sum()
        )
        if n_conflicting:
            raise ValueError(
                f"{n_conflicting} duplicated timestamps carry CONFLICTING values — "
                "refusing to silently pick one; resolve upstream"
            )
    n_raw = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df.attrs["n_raw_rows"] = n_raw
    df.attrs["n_duplicate_rows_dropped"] = n_raw - len(df)
    return df


_RTH_MINUTES = 390
_RTH_OPEN_MINUTE_OF_DAY = 9 * 60 + 30  # 09:30


def infer_et_offset_hours(df: pd.DataFrame) -> int:
    """Infer (file clock − ET) in whole hours from the volume day-profile.

    The 390-minute window with maximum TOTAL volume across the sample is the
    RTH block (extended-hours bars are both rarer and thinner); its start
    minute is the file-clock rendering of 09:30 ET. Total — not per-day-present
    mean — so that a heavy bar present on few days (e.g. a closing-auction
    print) cannot outweigh a normal bar present every day.
    """
    minute_of_day = df.index.hour * 60 + df.index.minute
    profile = np.zeros(24 * 60)
    sums = pd.Series(df["volume"].values).groupby(minute_of_day).sum()
    profile[sums.index] = sums.values
    window_mass = np.convolve(profile, np.ones(_RTH_MINUTES), mode="valid")
    start = int(np.argmax(window_mass))
    shift = start - _RTH_OPEN_MINUTE_OF_DAY
    if shift % 60 != 0:
        raise ValueError(
            f"RTH volume block starts at minute-of-day {start} — not a "
            "whole-hour offset from 09:30 ET; inspect the data manually"
        )
    return shift // 60


def normalize_to_et(
    df: pd.DataFrame, offset_hours: int, *, verify: bool = True
) -> pd.DataFrame:
    """Shift the naive file clock by ``-offset_hours`` and localize to ET.

    With ``verify=True`` (default) the claimed offset is checked against the
    evidence-based estimate and a contradiction raises ``ValueError`` — the
    pipeline must never run on a silently mis-zoned dataset.
    """
    if verify:
        inferred = infer_et_offset_hours(df)
        if inferred != offset_hours:
            raise ValueError(
                f"claimed offset {offset_hours}h contradicts inferred offset "
                f"{inferred}h (opening-spike evidence)"
            )
    out = df.copy()
    out.index = (df.index - pd.Timedelta(hours=offset_hours)).tz_localize(
        ET, nonexistent="NaT", ambiguous="NaT"
    )
    out = out[out.index.notna()]
    out.index.name = "dt_et"
    return out
