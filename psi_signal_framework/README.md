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
| `gross_return` | close_i / close_{i−1} | fair-game test: lognormal fit of gross returns = normality of minute returns |
| `dollar_volume` | vwap_i × volume_i | equity-wise (price–volume) income |
| `volume` | volume_i | share-volume income |
| `price` | close_i | price-only |
| `abs_return` | \|log return_i\| | volatility shape |

Returns are differenced **within each day** so the first bar never carries the
overnight gap (otherwise every opening window on a gap day is a false alarm).

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

## Results on real SPY data (2010, Kaggle `spy_1min_2008_2021_cleaned`)

Data: `data/spy_1min_2010.csv` (98,147 RTH bars, 252 days; feed timestamps
are auto-shifted +2h to US/Eastern by the loader). Rolling 60-minute
windows stepping 1 minute, 12 bins, 5-day trailing baseline.

**The May 6 flash crash is the #1 dislocation of 2010 under the
`gross_return` measure**: 8 of the year's 10 highest-|z| windows are crash
minutes, peak z = 10.7 at 14:47 ET, first alert 14:44 (the cleaned-data
price bottom), sustained |z| ≥ 2 for 27 consecutive windows. A warning
blip (z = 2.6) appears at ~12:35, two hours before the crash. The two
non-crash windows in the top 10 are genuine information events: 14:17 on
2010-08-10 (two minutes after the FOMC "QE-lite" statement) and 12:41 on
2010-05-28 (Fitch's midday downgrade of Spain).

Other measures on the real crash: `dollar_volume` peaks at z = 2.0 exactly
at 14:44; `price` never alerts (peak z = 1.1); the literal non-overlapping
clock-hour mode scores z = −0.5 for the 14:00–15:00 hour.

Two lessons the synthetic validation (peak z ≈ 13 on injected crash,
`synthetic_flashcrash_result.png`) did not show:

1. **psi is scale-invariant, and real panics are broad.** The crash raised
   volume in nearly every minute proportionally, so the hour's dollar-volume
   *shape* stayed close to lognormal — "fair" — while the return distribution
   became wildly non-normal. Shape-based fairness detection wants returns.
2. **Window alignment matters.** The clock hour 14:00–15:00 blends plunge
   and rebound into one symmetric fat population that looks almost fair;
   rolling windows that mix calm minutes with the first crash minutes are
   maximally bimodal and light up. Populations must roll, not snap to hours.

Known behavior: in very quiet periods many minute returns are exactly 0
(price pinned at a penny tick), producing a point mass in the distribution
and elevated psi at the *compressed* end — the "artificially suppressed
volatility" regime of the framework document, opposite in nature to a crash
but flagged by the same statistic.

## Full 2008–2021 run: psi + number of classes (mixture extension)

`detect_flashcrashes.py` runs `rolling_panel()` over the full dataset
(1,301,308 bars → 1,000,078 sliding 90-minute windows, step 1 minute, ~299
per day). For every window it reports, besides psi:

- **`n_classes`** — the BIC-optimal number of lognormal mixture components
  (1–3) fitted by vectorized EM to the window's **log-standardized** incomes
  ((ln x − μ)/σ, so the count is fully scale- and dispersion-free): the
  number of "classes of society" trading in that window.
- **`sigma`** — the window's log-income dispersion, used to split alert
  episodes into **dispersed** (crash/panic) vs **compressed** (pinned,
  artificially suppressed volatility) regimes.

Alert windows (z ≥ 3 on a 5-day trailing baseline) cluster into 1,337
intraday episodes: 204 dispersed, ~15/year compressed, rest mixed.

Key findings (gross-return incomes):

- **Class splitting is the crash signature**: 68% of alert windows have
  ≥ 2 classes vs 8.6% of calm windows; every top-30 dispersed episode has
  100% multi-class alert windows. Overall, 87.6% of all windows are
  single-class (fair), 11.9% two-class, 0.5% three-class.
- **The #1 episode of 14 years is the AP-Twitter-hack flash crash**
  (2013-04-23 13:10, z = 20.7) — on a mere 1% price dip. The 2010 flash
  crash (z = 13.6), Volmageddon (2018-02-05 15:12, z = 12.3), the failed
  TARP vote (2008-09-29), and COVID circuit-breaker #2 (2020-03-12) all
  make the dispersed list.
- **20 of the top 30 dispersed episodes peak in the 13:55–14:20 ET Fed
  slot** (24% of all dispersed episodes vs ~6% expected by chance) — FOMC
  statement reactions: QE1 expansion (2009-03-18), QE2 (2010-11-03),
  no-taper (2013-09-18), liftoff (2015-12-16), "autopilot" (2018-12-19)…
  The detector is effectively an information-event seismograph.
- **Known misses, both structural**: (1) *opening crashes* (2015-08-24
  ETF crash, 2014-10-15, COVID limit-down opens) — the first window of a
  day completes at 11:00, and by then the crash is blended with its own
  rebound (2015-08-24 is caught only late, z = 3.1 at 15:40); (2)
  *baseline saturation* — in sustained chaos (mid-March 2020, September
  2008) the trailing baseline itself explodes, so nothing is anomalous
  *relative to its own week*. The z-score detects transitions into
  dislocation, not steady-state chaos.
- Compressed episodes are dominated by half-day/holiday sessions
  (Christmas Eve 2020, post-Thanksgiving Fridays) and pre-FOMC pauses —
  the "suppressed volatility" end of the framework's table.

Outputs in `results_full/`: `episodes_gross_return.csv` (all 1,337
episodes), `panel_gross_return.csv.gz` (full per-window psi / n_classes /
sigma / z panel), `timeline_2008_2021.png`, `flashcrash_zooms.png`.

### Population counts

- Rolling mode: ~330 populations/day of 60 minute-incomes; 83,279 windows
  in 2010; ≈ 1.11 M across the full 2008–2021 file (3,347 trading days).
- Hourly mode: 7 populations/day, ~55.7 members each on average; 1,761 in
  2010; ≈ 23,400 across the full file.

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
