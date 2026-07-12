# Phase 1 — falsifiability diagnostic

Supervised era: 2011-01-03 → 2021-05-06 (2579 days); K=16, L=32; calibration on first 60% (diagnostic only).

| scheme | d* clock | d* info | median bars/day | median morning bars | pad rate | floor rate | cap rate | median |corr| clock-info | frac>0.95 |
|---|---|---|---|---|---|---|---|---|---|
| dollar | 0.8 | 0.5 | 14 | 14 | 0.58 | 0.074 | 0.012 | 0.608 | 0.014 |
| dollar_imbalance | 0.8 | 0.5 | 15 | 15 | 0.54 | 0.299 | 0.101 | 0.578 | 0.015 |

**Primary bar scheme for the pilot: `dollar`** (lowest floor/cap/pad degeneracy, then lowest axis similarity).

**Falsifiability gate: PASS** — the info axis is distinguishable from the clock axis (median |corr| < 0.95); the dual-time hypothesis is testable on this data.
