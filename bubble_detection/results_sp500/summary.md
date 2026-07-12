# Bubble detection run summary

Data: `SP500_real_monthly.csv` — 1833 bars, 1871-01-01 to 2023-09-01.
Label horizon: 12 bars. Walk-forward folds: 6 (purge = horizon).

## Full-sample GSADF test
GSADF = **2.961** vs Monte-Carlo critical values 90/95/99% = 2.275 / 2.535 / 3.175
→ explosiveness **detected** at the 5% level over the sample
(sup-statistic cv simulated on the stationary-tail length — a lower bound for samples much longer
than the window cap, so treat borderline rejections cautiously; per-bar flags are unaffected).
Explosive bars (BSADF > cv95): 8.3% of the sample.

## PSY date-stamped episodes (min duration 8 bars)
| start | end | bars |
|---|---|---|
| 1928-07-01 00:00 | 1929-10-01 00:00 | 16 |
| 1954-10-01 00:00 | 1956-12-01 00:00 | 27 |
| 1995-07-01 00:00 | 1998-08-01 00:00 | 38 |
| 1998-10-01 00:00 | 2000-11-01 00:00 | 26 |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 588 | 150 | 32 | 27 | 0.08698 ± 0.07661 |
| 1 | 738 | 150 | 59 | 3 | -0.03087 ± 0.05006 |
| 2 | 888 | 150 | 62 | 7 | 0.03838 ± 0.06099 |
| 3 | 1038 | 150 | 69 | 45 | 0.11520 ± 0.05423 |
| 4 | 1188 | 150 | 103 | 31 | 0.09761 ± 0.06116 |
| 5 | 1338 | 136 | 145 | 0 | 0.04686 ± 0.03577 |

Pooled treated test bars: 113
- direction hit rate of sign(tau): **0.690**
- mean fwd 12-bar log return — sign(tau) strategy: **0.07449** (NW t = 1.79) vs always-long: 0.08685 (NW t = 2.23)
- corr(tau, realized fwd ret): 0.169

All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: 0.613 | corr(tau, fwd ret): 0.164 | CI excludes 0 on 45.4% of bars

## Event-driven strategy backtest (out-of-sample span, 2 bps/side, 1-bar delay)
span 1948-12-01 00:00:00+00:00 → 2022-09-01 00:00:00+00:00 (886 bars, ~12 bars/yr) · exposure 7.9% · 11 trades · win rate 0.64

| series | total log ret | ann ret | ann vol | Sharpe | max DD (log) |
|---|---|---|---|---|---|
| event strategy | +0.8272 | +0.011 | 0.040 | +0.28 | -0.1558 |
| buy & hold | +3.0220 | +0.041 | 0.123 | +0.33 | -0.9835 |
| long whenever flagged | +1.0958 | +0.015 | 0.050 | +0.30 | -0.2857 |

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.540 (base rate up = 0.665), AUC = 0.459, n = 886

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| qlr_loc | 0.0747 |
| cusum | 0.0746 |
| chow_f | 0.0733 |
| rtadf | 0.0706 |
| bsadf_gap | 0.0676 |
| bsadf | 0.0606 |
| vol_35 | 0.0589 |
| mom_35 | 0.0539 |
| bp_dmean | 0.0516 |
| cusum_sq | 0.0509 |

## Caveats
- Forward labels overlap (h = 12); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
