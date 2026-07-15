# Bubble detection run summary

Data: `BMNR_h.csv` — 1740 bars, 2025-07-15 to 2026-07-15.
Label horizon: 21 bars. Walk-forward folds: 4 (purge = horizon).

## Full-sample GSADF test
GSADF = **2.861** vs Monte-Carlo critical values 90/95/99% = 2.275 / 2.535 / 3.175
→ explosiveness **detected** at the 5% level over the sample
(sup-statistic cv simulated on the stationary-tail length — a lower bound for samples much longer
than the window cap, so treat borderline rejections cautiously; per-bar flags are unaffected).
Explosive bars (BSADF > cv95): 4.5% of the sample.

## PSY date-stamped episodes (min duration 7 bars)
| start | end | bars |
|---|---|---|
| 2025-08-08 13:30 | 2025-08-12 17:30 | 19 |
| 2025-11-17 14:30 | 2025-11-17 20:30 | 7 |
| 2025-11-19 15:30 | 2025-11-21 20:30 | 20 |
| 2026-02-04 15:30 | 2026-02-06 14:30 | 14 |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 579 | 150 | 48 | 0 | 0.11307 ± 0.05093 |
| 1 | 729 | 150 | 48 | 0 | 0.16026 ± 0.09518 |
| 2 | 879 | 150 | 48 | 0 | 0.05686 ± 0.06053 |
| 3 | 1029 | 150 | 48 | 2 | 0.01846 ± 0.06424 |

Pooled treated test bars: 2
- too few treated test bars for strategy evaluation.

All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: 0.427 | corr(tau, fwd ret): 0.015 | CI excludes 0 on 47.0% of bars

## Event-driven strategy backtest (out-of-sample span, 20 bps/side, 1-bar delay)
span 2026-03-04 14:30:00+00:00 → 2026-07-07 17:30:00+00:00 (600 bars, ~1749 bars/yr) · exposure 0.0%
trades: 0 (0 long / 0 short) · win rate n/a · profit factor nan · avg win +nan / avg loss +nan (log) · avg duration nan bars

| series | total ret | total log ret | ann ret | ann vol | Sharpe | Sortino | Calmar | max DD (log) | bar hit rate |
|---|---|---|---|---|---|---|---|---|---|
| event strategy | +0.0% | +0.0000 | +0.000 | 0.000 | +nan | +nan | +nan | +0.0000 | nan |
| buy & hold | -21.6% | -0.2437 | -0.710 | 0.791 | -0.90 | -1.32 | -1.18 | -0.6026 | 0.475 |
| long whenever flagged | -1.4% | -0.0143 | -0.042 | 0.025 | -1.65 | -1.66 | -2.87 | -0.0146 | 0.500 |

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.553 (base rate up = 0.440), AUC = 0.537, n = 600

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| vol_35 | 0.1399 |
| qlr_loc | 0.0812 |
| ssa_noise_vol | 0.0807 |
| bp_since | 0.0787 |
| chow_f | 0.0710 |
| cusum_sq | 0.0679 |
| qlr_f | 0.0656 |
| mom_35 | 0.0632 |
| rtadf | 0.0609 |
| bp_dmean | 0.0597 |

## Caveats
- Forward labels overlap (h = 21); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
