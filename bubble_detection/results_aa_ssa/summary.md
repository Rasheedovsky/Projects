# Bubble detection run summary

Data: `AA_h.csv` — 1749 bars, 2025-07-11 to 2026-07-10.
Label horizon: 21 bars. Walk-forward folds: 4 (purge = horizon).

## Full-sample GSADF test
GSADF = **2.187** vs Monte-Carlo critical values 90/95/99% = 2.275 / 2.535 / 3.175
→ explosiveness not detected at the 5% level over the sample
(sup-statistic cv simulated on the stationary-tail length — a lower bound for samples much longer
than the window cap, so treat borderline rejections cautiously; per-bar flags are unaffected).
Explosive bars (BSADF > cv95): 7.3% of the sample.

## PSY date-stamped episodes (min duration 7 bars)
| start | end | bars |
|---|---|---|
| 2025-12-22 14:30 | 2025-12-26 20:30 | 24 |
| 2026-01-02 18:30 | 2026-01-15 20:30 | 66 |
| 2026-06-24 13:30 | 2026-06-25 13:30 | 8 |
| 2026-07-01 13:30 | 2026-07-01 19:30 | 7 |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 579 | 150 | 101 | 0 | 0.00631 ± 0.01848 |
| 1 | 729 | 150 | 101 | 0 | 0.01699 ± 0.00932 |
| 2 | 879 | 150 | 101 | 0 | -0.00586 ± 0.01349 |
| 3 | 1029 | 150 | 101 | 18 | -0.04295 ± 0.01846 |

Pooled treated test bars: 18
- direction hit rate of sign(tau): **0.444**
- mean fwd 21-bar log return — sign(tau) strategy: **0.00892** (NW t = 0.76) vs always-long: -0.04010 (NW t = -1.52)
- corr(tau, realized fwd ret): -0.976

All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: 0.603 | corr(tau, fwd ret): 0.214 | CI excludes 0 on 60.2% of bars

## Event-driven strategy backtest (out-of-sample span, 2 bps/side, 1-bar delay)
span 2026-02-26 18:30:00+00:00 → 2026-07-01 14:30:00+00:00 (600 bars, ~1753 bars/yr) · exposure 1.8% · 3 trades · win rate 0.33

| series | total log ret | ann ret | ann vol | Sharpe | max DD (log) |
|---|---|---|---|---|---|
| event strategy | -0.0136 | -0.040 | 0.049 | -0.80 | -0.0346 |
| buy & hold | -0.2753 | -0.804 | 0.675 | -1.19 | -0.5847 |
| long whenever flagged | -0.0058 | -0.017 | 0.071 | -0.24 | -0.0545 |

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.495 (base rate up = 0.473), AUC = 0.434, n = 600

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| qlr_f | 0.1065 |
| chow_f | 0.0936 |
| cusum_sq | 0.0862 |
| vol_35 | 0.0730 |
| rtadf | 0.0643 |
| mom_7 | 0.0638 |
| bp_dmean | 0.0572 |
| rup_since | 0.0452 |
| rup_lvr | 0.0441 |
| qlr_loc | 0.0441 |

## Caveats
- Forward labels overlap (h = 21); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
