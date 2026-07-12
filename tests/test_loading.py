"""Raw CSV parsing, chronological ordering, offset inference, ET normalization."""
import io

import numpy as np
import pandas as pd
import pytest

from spydt.data.loading import (
    infer_et_offset_hours,
    normalize_to_et,
    parse_raw_csv,
)
from tests.conftest import make_multi_day, make_rth_day

RAW_SAMPLE = """,date,open,high,low,close,volume,barCount,average
0,20090522  07:30:00,89.45,89.46,89.37,89.37,7872,2102,89.424
1,20090522  07:31:00,89.38,89.53,89.37,89.5,5336,1938,89.468
2,20080122  05:30:00,126.45,127.18,126.0,126.78,83061,900,126.9
"""


def test_parse_raw_csv_schema_and_dtypes():
    df = parse_raw_csv(io.StringIO(RAW_SAMPLE))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "bar_count", "average"]
    assert df.index.name == "dt_file"
    assert df.index.tz is None  # file clock is naive until offset is verified
    assert df["volume"].dtype == float


def test_parse_raw_csv_sorts_chronologically():
    df = parse_raw_csv(io.StringIO(RAW_SAMPLE))
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2008-01-22 05:30:00")


def test_parse_raw_csv_drops_identical_duplicate_timestamps():
    dup = RAW_SAMPLE + "3,20090522  07:31:00,89.38,89.53,89.37,89.5,5336,1938,89.468\n"
    df = parse_raw_csv(io.StringIO(dup))
    assert df.index.is_unique
    assert df.attrs["n_duplicate_rows_dropped"] == 1


def test_parse_raw_csv_rejects_conflicting_duplicates():
    conflict = RAW_SAMPLE + "3,20090522  07:31:00,89.38,89.53,89.37,99.99,5336,1938,89.468\n"
    with pytest.raises(ValueError, match="CONFLICTING"):
        parse_raw_csv(io.StringIO(conflict))


def test_offset_inference_recovers_known_shift():
    # Build ET ground truth, then present it on a file clock shifted -2h.
    et = make_multi_day(["2015-06-15", "2015-06-16", "2015-06-17"])
    file_clock = et.copy()
    file_clock.index = et.index.tz_localize(None) - pd.Timedelta(hours=2)
    assert infer_et_offset_hours(file_clock) == -2


def test_offset_inference_zero_shift():
    et = make_multi_day(["2015-06-15", "2015-06-16", "2015-06-17"])
    file_clock = et.copy()
    file_clock.index = et.index.tz_localize(None)
    assert infer_et_offset_hours(file_clock) == 0


def test_normalize_to_et_first_bar_0930():
    et = make_rth_day("2015-06-15")
    file_clock = et.copy()
    file_clock.index = et.index.tz_localize(None) - pd.Timedelta(hours=2)
    out = normalize_to_et(file_clock, offset_hours=-2)
    assert str(out.index.tz) == "America/New_York"
    assert out.index[0].strftime("%H:%M") == "09:30"
    # prices untouched
    np.testing.assert_allclose(out["close"].values, et["close"].values)


def test_normalize_rejects_wrong_offset_evidence():
    et = make_multi_day(["2015-06-15", "2015-06-16", "2015-06-17"])
    file_clock = et.copy()
    file_clock.index = et.index.tz_localize(None) - pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="offset"):
        # claiming 0 when the data says -2 must hard-fail
        normalize_to_et(file_clock, offset_hours=0, verify=True)
