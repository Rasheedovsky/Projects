# Leakage gates

| gate | result | evidence |
|---|---|---|
| (b) feature timestamps < 11:00 (<= 15:30 contract) | PASS | features bit-identical under corruption of 11:00-15:29 + 15:32-15:58; labels intact |
| (g) fitted statistics use train days only | PASS | bar threshold and all train-row features bit-identical when every non-train day is corrupted |
| (a) shuffled-label collapse (production scale) | PASS | acc=0.498 on 847 test days (95% CI half-width 0.034) |
| (c) injected-leak canary detects | PASS | acc=0.973 with label injected into a feature |
| (f) synthetic-signal positive control (soft) | WEAK | pipeline recovers acc=0.387 of a planted signal with Bayes acc≈0.611 (n=847); recovery above half-CI counts as pass |
| (d) train-on-future canary (purge+embargo verified) | PASS | clean CPCV passes incl. embargo window; corrupted split raises |
| (e) SSL corpus provenance | PASS | last corpus day 2010-12-31 < first supervised day 2011-01-01 |
