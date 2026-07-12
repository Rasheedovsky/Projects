# Leakage gates

| gate | result | evidence |
|---|---|---|
| (b) feature timestamps < 11:00 (<= 15:30 contract) | PASS | features bit-identical under corruption of 11:00-15:29 + 15:32-15:58; labels intact |
| (g) fitted statistics use train days only | PASS | bar threshold and all train-row features bit-identical when every non-train day is corrupted |
| (a) shuffled-label collapse (production scale) | PASS | acc=0.508 on 847 test days (95% CI half-width 0.034) |
| (c) injected-leak canary detects | PASS | acc=0.941 with label injected into a feature |
| (f) synthetic-signal positive control (soft) | PASS | planted signal Bayes acc≈0.611: in-era holdout recovery acc=0.531 (n=256); cross-era test recovery acc=0.372 (n=847) — pass judged on in-era capability; cross-era transfer is reported as a power diagnostic |
| (d) train-on-future canary (purge+embargo verified) | PASS | clean CPCV passes incl. embargo window; corrupted split raises |
| (e) SSL corpus provenance | PASS | last corpus day 2010-12-31 < first supervised day 2011-01-01 |
