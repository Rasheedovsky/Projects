# Bubble detection run summary

Data: `SP500_real_monthly.csv` — 1833 bars, 1871-01-01 to 2023-09-01.
Label horizon: 12 bars. Walk-forward folds: 6 (purge = horizon).

## Full-sample GSADF test
GSADF = **2.961** vs Monte-Carlo critical values 90/95/99% = 2.715 / 3.010 / 3.660
→ explosiveness not detected at the 5% level over the sample.
Explosive bars (BSADF > cv95): 9.0% of the sample.

## PSY date-stamped episodes (min duration 8 bars)
| start | end | bars |
|---|---|---|
| 1928-07-01 00:00 | 1929-10-01 00:00 | 16 |
| 1954-09-01 00:00 | 1957-01-01 00:00 | 29 |
| 1995-07-01 00:00 | 1998-08-01 00:00 | 38 |
| 1998-10-01 00:00 | 2000-11-01 00:00 | 26 |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 588 | 150 | 38 | 30 | 0.11231 ± 0.06916 |
| 1 | 738 | 150 | 68 | 2 | 0.01448 ± 0.05544 |
| 2 | 888 | 150 | 70 | 7 | 0.07356 ± 0.07400 |
| 3 | 1038 | 150 | 77 | 46 | 0.13741 ± 0.05860 |
| 4 | 1188 | 150 | 112 | 34 | 0.10040 ± 0.06976 |
| 5 | 1338 | 136 | 157 | 1 | 0.06210 ± 0.04261 |

Pooled treated test bars: 120
- direction hit rate of sign(tau): **0.700**
- mean fwd 12-bar log return — sign(tau) strategy: **0.08374** (NW t = 2.19) vs always-long: 0.08366 (NW t = 2.15)
- corr(tau, realized fwd ret): 0.166

All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: 0.620 | corr(tau, fwd ret): 0.120 | CI excludes 0 on 49.0% of bars

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.546 (base rate up = 0.665), AUC = 0.471, n = 886

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| bsadf_gap | 0.0769 |
| chow_f | 0.0734 |
| qlr_loc | 0.0734 |
| cusum | 0.0723 |
| rtadf | 0.0717 |
| bsadf | 0.0620 |
| vol_35 | 0.0570 |
| mom_35 | 0.0554 |
| bp_dmean | 0.0514 |
| hmm_p_bear | 0.0507 |

## Caveats
- Forward labels overlap (h = 12); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
