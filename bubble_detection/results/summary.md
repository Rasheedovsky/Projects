# Bubble detection run summary

Data: `AA_h.csv` — 1749 hourly bars, 2025-07-11 to 2026-07-10.
Label horizon: 21 bars. Walk-forward folds: 5 (purge = horizon).

## Full-sample GSADF test
GSADF = **2.187** vs Monte-Carlo critical values 90/95/99% = 2.768 / 3.027 / 3.660
→ explosiveness not detected at the 5% level over the sample.
Explosive bars (BSADF > cv95): 6.9% of the sample.

## PSY date-stamped episodes (min duration 7 bars)
| start | end | bars |
|---|---|---|
| 2025-12-19 15:30 | 2025-12-23 20:30 | 20 |
| 2026-01-02 20:30 | 2026-01-08 16:30 | 25 |
| 2026-01-08 18:30 | 2026-01-15 20:30 | 38 |
| 2026-06-24 13:30 | 2026-06-25 13:30 | 8 |
| 2026-07-01 13:30 | 2026-07-01 19:30 | 7 |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 579 | 150 | 90 | 0 | 0.00212 ± 0.00712 |
| 1 | 729 | 150 | 91 | 0 | -0.06961 ± 0.02185 |
| 2 | 879 | 150 | 91 | 1 | 0.04525 ± 0.02981 |
| 3 | 1029 | 150 | 92 | 0 | 0.04774 ± 0.01314 |
| 4 | 1179 | 150 | 92 | 19 | 0.00008 ± 0.01168 |

Pooled treated test bars: 20
- direction hit rate of sign(tau): **0.650**
- mean fwd 21-bar log return — sign(tau) strategy: **0.03053** (NW t = 2.49) vs always-long: -0.03833 (NW t = -1.66)
- corr(tau, realized fwd ret): 0.721

All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: 0.523 | corr(tau, fwd ret): 0.271 | CI excludes 0 on 47.6% of bars

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.389 (base rate up = 0.479), AUC = 0.338, n = 750

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| cusum_sq | 0.1201 |
| qlr_loc | 0.0980 |
| vol_35 | 0.0845 |
| rtadf | 0.0671 |
| mom_7 | 0.0601 |
| qlr_f | 0.0563 |
| mom_35 | 0.0509 |
| chow_f | 0.0469 |
| rup_lvr | 0.0468 |
| rup_since | 0.0451 |

## Caveats
- Forward labels overlap (h = 21); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- One asset, one year of hourly data; upgrade path is a cross-sectional panel.
