# Bubble detection run summary

Data: `BMNR_d.csv` — 281 bars, 2025-06-04 to 2026-07-14.
Label horizon: 5 bars. Walk-forward folds: 1 (purge = horizon).

## Full-sample GSADF test
GSADF = **3.024** vs Monte-Carlo critical values 90/95/99% = 2.865 / 3.155 / 4.269
→ explosiveness not detected at the 5% level over the sample
(sup-statistic cv simulated on the stationary-tail length — a lower bound for samples much longer
than the window cap, so treat borderline rejections cautiously; per-bar flags are unaffected).
Explosive bars (BSADF > cv95): 5.2% of the sample.

## PSY date-stamped episodes (min duration 6 bars)
| start | end | bars |
|---|---|---|
| (none) | | |

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
| 0 | 135 | 38 | 9 | 3 | skipped |

Pooled treated test bars: 0
- too few treated test bars for strategy evaluation.


## Event-driven strategy backtest (out-of-sample span, 20 bps/side, 1-bar delay)
skipped: not enough out-of-sample bars

## Benchmark RF direction classifier (all bars, all features)
accuracy = 0.447 (base rate up = 0.368), AUC = 0.515, n = 38

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
| cusum_sq | 0.1232 |
| chow_f | 0.1160 |
| vol_21 | 0.0840 |
| bp_since | 0.0749 |
| hmm_sig | 0.0713 |
| qlr_f | 0.0634 |
| rtadf | 0.0616 |
| mom_21 | 0.0563 |
| bp_dmean | 0.0546 |
| bsadf | 0.0490 |

## Caveats
- Forward labels overlap (h = 5); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
