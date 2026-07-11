"""
Psi-Signal Framework: detecting market regimes via fairness deviations.

Implements the framework from Venkatasubramanian (2010, 2015, 2017, 2019) and
Kanbur & Venkatasubramanian (2020): the maximum-entropy (fairest) distribution
of positive quantities under log-moment constraints is the lognormal.  Each
window of intraday data is treated as a cross-sectional "income" distribution
(one income per minute), and psi is the KL divergence between the empirical
distribution and the ideal lognormal fitted from the window's own log-moments.

psi_k = KL(p_empirical || q_lognormal) >= 0,  psi_k = 0 iff perfectly "fair".

A window is flagged as dislocated when the z-score of psi against a trailing
baseline exceeds a threshold (default |z| >= 2).

Income measures for 1-minute OHLCV bars:
  - "price":         close price of each minute (price-only fairness)
  - "dollar_volume": close * volume of each minute (equity-wise, price-volume)
  - "abs_return":    |log return| of each minute (volatility shape)

Note: psi is invariant to rescaling the incomes (dividing by the window mean
shifts every log by a constant, which the fitted mu absorbs), so standardizing
each window to income shares — as in the income-inequality papers — leaves psi
unchanged.  The z-score step is what standardizes psi across time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

VALID_MEASURES = ("price", "dollar_volume", "abs_return")


# ---------------------------------------------------------------------------
# Core psi computation
# ---------------------------------------------------------------------------

def psi_score(x, bins: int = 12, min_obs: int = 20) -> float:
    """KL divergence between the empirical distribution of a positive sample
    and the maximum-entropy lognormal fitted from the sample's log-moments.

    Binning is done in log space, so this equals the binned KL between the
    empirical distribution of x and the fitted lognormal with matched bins.
    The ideal bin mass is renormalized over the observed support so psi
    measures *shape* deviation inside the data range.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < min_obs:
        return np.nan

    lx = np.log(x)
    mu = lx.mean()
    sigma = lx.std(ddof=0)
    if sigma < 1e-12:
        # Degenerate window (all incomes identical): perfectly equal, psi = 0.
        return 0.0

    edges = np.linspace(lx.min(), lx.max() + 1e-12, bins + 1)
    counts, _ = np.histogram(lx, bins=edges)
    p = counts / n

    q = np.diff(stats.norm.cdf(edges, loc=mu, scale=sigma))
    qsum = q.sum()
    if qsum <= 0:
        return np.nan
    q = np.maximum(q / qsum, 1e-12)

    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


# ---------------------------------------------------------------------------
# Income measures from 1-minute OHLCV bars
# ---------------------------------------------------------------------------

def income_series(bars: pd.DataFrame, measure: str) -> pd.Series:
    """Map 1-minute OHLCV bars to a positive 'income' per minute."""
    if measure not in VALID_MEASURES:
        raise ValueError(f"measure must be one of {VALID_MEASURES}, got {measure!r}")
    if measure == "price":
        return bars["close"].astype(float)
    if measure == "dollar_volume":
        return (bars["close"] * bars["volume"]).astype(float)
    # abs_return: zeros are dropped inside psi_score (x > 0 filter)
    return np.log(bars["close"].astype(float)).diff().abs()


# ---------------------------------------------------------------------------
# Rolling and hourly psi
# ---------------------------------------------------------------------------

def rolling_psi(income: pd.Series, window: int = 60, step: int = 1,
                bins: int = 12, min_obs: int = 30) -> pd.Series:
    """Psi for each rolling window of `window` bars, stamped at the window end.

    Windows never span session breaks: bars must already be restricted to
    regular trading hours, and any window containing a gap larger than
    `window` minutes of clock time is skipped.
    """
    values = income.to_numpy(dtype=float)
    times = income.index
    out_idx, out_val = [], []
    max_span = pd.Timedelta(minutes=2 * window)
    for end in range(window, len(values) + 1, step):
        start = end - window
        if times[end - 1] - times[start] > max_span:
            continue  # window straddles an overnight/weekend gap
        out_idx.append(times[end - 1])
        out_val.append(psi_score(values[start:end], bins=bins, min_obs=min_obs))
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx), name="psi")


def hourly_psi(income: pd.Series, bins: int = 12, min_obs: int = 30) -> pd.Series:
    """Psi for each clock hour: the hour as one population of ~60 minute-incomes."""
    grouped = income.groupby(income.index.floor("h"))
    out = grouped.apply(lambda g: psi_score(g.to_numpy(), bins=bins, min_obs=min_obs))
    out.name = "psi"
    return out


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def trailing_zscore(psi: pd.Series, baseline: int, min_periods: int) -> pd.Series:
    """Z-score of psi against the trailing `baseline` observations (excluding
    the current one, so a spike cannot inflate its own baseline)."""
    past = psi.shift(1)
    m = past.rolling(baseline, min_periods=min_periods).mean()
    s = past.rolling(baseline, min_periods=min_periods).std()
    z = (psi - m) / s
    z.name = "z"
    return z


def global_zscore(psi: pd.Series) -> pd.Series:
    """Full-sample z-score, as specified in the framework document."""
    z = (psi - psi.mean()) / psi.std()
    z.name = "z"
    return z


def classify(z: pd.Series, warn: float = 1.0, alert: float = 2.0) -> pd.Series:
    """0 = fair/normal, 1 = warning, 2 = regime shift detected."""
    out = pd.Series(0, index=z.index, dtype=int, name="signal")
    out[z.abs() >= warn] = 1
    out[z.abs() >= alert] = 2
    out[z.isna()] = -1  # insufficient baseline
    return out


def analyze(bars: pd.DataFrame, measure: str, window: int = 60, step: int = 1,
            bins: int = 12, baseline: int = 1950, min_periods: int = 390,
            mode: str = "rolling") -> pd.DataFrame:
    """End-to-end: bars -> income -> psi -> z -> signal.

    baseline=1950 rolling windows = ~5 trading days of 390 one-minute bars.
    mode='hourly' uses non-overlapping clock hours (baseline then counts hours,
    so pass e.g. baseline=35, min_periods=14 for ~5 days of 7 trading hours).
    """
    income = income_series(bars, measure)
    if mode == "rolling":
        psi = rolling_psi(income, window=window, step=step, bins=bins)
    elif mode == "hourly":
        psi = hourly_psi(income, bins=bins)
    else:
        raise ValueError("mode must be 'rolling' or 'hourly'")
    z = trailing_zscore(psi, baseline=baseline, min_periods=min_periods)
    sig = classify(z)
    return pd.DataFrame({"psi": psi, "z": z, "signal": sig})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_minute_bars(path: str, tz: str | None = None) -> pd.DataFrame:
    """Load 1-minute OHLCV bars from CSV with flexible column naming.

    Accepts either a single datetime column (datetime/timestamp/date_time/time)
    or separate date + time columns.  Column names are case-insensitive.
    Returns a DataFrame indexed by timestamp with columns open, high, low,
    close, volume, restricted to regular trading hours 09:30-16:00.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    dt_col = next((c for c in ("datetime", "timestamp", "date_time", "dt", "time")
                   if c in df.columns), None)
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif dt_col is not None:
        ts = pd.to_datetime(df[dt_col])
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")

    rename = {"vol": "volume", "adj_close": "close", "last": "close", "price": "close"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns
                            and v not in df.columns})
    missing = [c for c in ("close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns: {list(df.columns)}")

    df.index = pd.DatetimeIndex(ts)
    if tz is not None:
        df.index = df.index.tz_localize(tz)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.between_time("09:30", "16:00")
    return df
