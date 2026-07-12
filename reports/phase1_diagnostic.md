# Phase 1 — falsifiability diagnostic

Supervised era: 2011-01-03 → 2021-05-06 (2579 days); K=16, L=32; calibration on first 60% (diagnostic only).

| scheme | d* clock | d* info | median bars/day | median morning bars | pad rate | floor rate | cap rate | median |corr| clock-info | frac>0.95 |
|---|---|---|---|---|---|---|---|---|---|
| dollar | 0.8 | 0.8 | 67 | 20 | 0.21 | 0.198 | 0.000 | 0.426 | 0.000 |
| dollar_imbalance | 0.8 | 0.8 | 71 | 22 | 0.17 | 0.432 | 0.027 | 0.425 | 0.000 |

**Primary bar scheme for the pilot: `dollar`** (lowest floor/cap/pad degeneracy, then lowest axis similarity).

**Falsifiability gate: PASS** — the info axis is distinguishable from the clock axis (median |corr| < 0.95); the dual-time hypothesis is testable on this data.
