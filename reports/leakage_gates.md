# Leakage gates

| gate | result | evidence |
|---|---|---|
| (b) feature timestamps < 11:00 (<= 15:30 contract) | PASS | features bit-identical under corruption of 11:00-15:29 + 15:32-15:58; labels intact |
| (a) shuffled-label collapse | PASS | acc=0.590 on shuffled labels (CI half-width 0.100) |
| (c) injected-leak canary detects | PASS | acc=1.000 with label injected into a feature |
| (d) train-on-future canary | PASS | clean CPCV passes; corrupted split raises |
| (e) SSL corpus provenance | PASS | last corpus day 2010-12-31 < first supervised day 2011-01-01 |
