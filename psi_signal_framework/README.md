# Psi-Signal Framework — Detecting Informed Trading / Dislocated Regimes in SPY 1-Minute Data

Implementation of the fairness-deviation ("psi") framework of
Venkatasubramanian (2010, 2015, 2017, 2019) and Kanbur & Venkatasubramanian
(2020), applied to intraday market data. The May 6, 2010 flash crash is the
target event.

## Idea

Statistical teleodynamics shows that the **maximum-entropy (fairest)
distribution of positive "incomes" under log-moment constraints is the
lognormal**. We treat each hour of trading as a small economy of 60 agents —
the 60 one-minute bars — where each minute's "income" is:

| measure | income of minute *i* | interpretation |
|---|---|---|
| `dollar_volume` | close_i × volume_i | equity-wise (price–volume) income |
| `price` | close_i | price-only |
| `abs_return` | \|log return_i\| | volatility shape (optional) |

For every window we fit the ideal lognormal from the window's **own**
log-moments (mu, sigma of ln x), and compute

```
psi = KL(empirical || ideal lognormal) >= 0
```

psi = 0 means the hour is at its fair maximum-entropy equilibrium; large psi
means the shape of the distribution is dislocated (bimodal prices during a
crash, a few minutes capturing an outsized share of dollar volume — the
signature of informed/forced flow). Because the ideal is refit per window,
psi is scale-free: standardizing incomes by the window mean leaves it
unchanged. The z-score of psi against a trailing baseline (default: 5
trading days) is the signal:

- |z| < 1  fair / normal regime
- 1 ≤ |z| < 2  warning
- |z| ≥ 2  regime shift / dislocation detected

Two window modes: **rolling** 60-minute windows stepping 1 minute (minute-level
detection latency), and **hourly** non-overlapping clock hours (the literal
"hour as a collection of 60 incomes" test).

## Files

- `psi_signal.py` — core library: psi score, income measures, rolling/hourly
  psi, z-scoring, flexible CSV loader.
- `run_psi_analysis.py` — CLI: point it at a 1-minute OHLCV CSV, get result
  CSVs + plots; automatically zooms on 2010-05-06 if present.
- `synthetic_flashcrash_test.py` — validation on synthetic data with an
  injected May-6-style crash (-9% in 13 min, 20–40× volume, partial rebound).
- `synthetic_flashcrash_result.png` — validation output.

## Validation result (synthetic crash)

- `dollar_volume` incomes: alert (|z| ≥ 2) at the **first minute** of the
  plunge, peak |z| ≈ 13.
- `price` incomes: alert 5 minutes in, peak |z| ≈ 8.
- Hourly mode: the crash hour scores z ≈ 3.1 vs its trailing baseline.
- False-alarm rate on calm days ≈ 3% of windows (a 2-sigma rule under a
  normal null allows 4.6%).

## Running on real data

```bash
pip install numpy pandas scipy matplotlib
python3 run_psi_analysis.py path/to/spy_1min.csv --baseline-days 5
```

### Expected CSV format

One row per minute bar, any of these header styles (case-insensitive):

```
datetime,open,high,low,close,volume
2010-05-06 09:30:00,113.61,113.65,113.51,113.55,1226300
```

or separate `date` and `time` columns. Only `close` and `volume` are strictly
required plus a timestamp. Timestamps should be US/Eastern; bars outside
09:30–16:00 are dropped automatically.

### Data needed for the flash-crash test

At least ~2 weeks before 2010-05-06 (to form the trailing baseline) through a
few days after — i.e. roughly **2010-04-15 to 2010-05-14**. More history
(e.g. all of 2010) gives a more stable baseline and a better false-alarm
estimate.

## Caveats

- With 60 observations per window, binned KL has a positive small-sample
  bias; the trailing z-score absorbs it (the baseline has the same bias).
- Minute dollar-volume is heavy-tailed even on calm days, so the *level* of
  psi is not meaningful by itself — only its deviation from baseline is.
- This is a distributional signal, not a directional one: it flags *that*
  the hour is unfair/dislocated, not which way price will go next.
