# Deep LPPLS (arXiv:2405.12803) vs Boulder-Investment-Technologies/lppls — tested on TASI

End-to-end implementation of the calibration methods in **"Deep LPPLS: Forecasting of
temporal critical points in natural, engineering and financial systems"**
(J. Nielsen, D. Sornette, M. Raissi, arXiv:2405.12803), compared against the reference
implementation in [Boulder-Investment-Technologies/lppls](https://github.com/Boulder-Investment-Technologies/lppls),
with both evaluated on the **Tadawul All Share Index (TASI)** 2021–22 bubble.

_Results, figures and the full comparison write-up are generated into `results/` by the
scripts below; the summary of findings is at the bottom of this file._

## Layout

```
deep_lppls/          paper implementation
  core.py            LPPLS function + analytic linear-parameter solve (Eq. 4–9)
  synthetic.py       synthetic LPPLS series w/ white & AR(1) noise (Table 1)
  lm.py              paper benchmark: multistart Levenberg–Marquardt (App. A.1)
  mlnn.py            Mono-LPPLS-NN — per-series PINN-style network (Sec. 2.1)
  plnn.py            Poly-LPPLS-NN — supervised network, input 252 (Sec. 2.2)
scripts/
  train_plnn.py            trains P-LNN-100K / -AR1 / -BOTH (100k series each)
  benchmark_synthetic.py   250-scenario error CDFs + timing (Fig. 3 / Table 2 analogue)
  compare_tasi.py          TASI bubble: fits + t_c PDFs + timing (Fig. 4 protocol)
  tasi_confidence_repo.py  Boulder package confidence indicator on TASI (bonus)
data/
  TASI_daily_2020_2024.csv daily TASI closes (validated against public records)
models/                    trained P-LNN weights
results/                   figures + CSV tables
```

## Reproduction

```bash
pip install jax optax numpy pandas scipy matplotlib
pip install -e <clone of Boulder-Investment-Technologies/lppls>
python scripts/train_plnn.py            # ~20 min on 4 CPU cores
python scripts/benchmark_synthetic.py   # ~15 min
python scripts/compare_tasi.py          # ~1 min
python scripts/tasi_confidence_repo.py  # ~5–20 min (bonus)
```

## Implementation decisions where the paper is silent

The paper specifies the architectures' depth, losses, optimizer, learning rates,
epochs, batch size and parameter bounds, but leaves several details open. These
were chosen as follows (all documented in module docstrings):

| Detail | Paper | This implementation |
|---|---|---|
| Hidden layer widths | not given | M-LNN: 128/128, P-LNN: 256×4 |
| M-LNN epochs | "specified number" | 1500, best state kept (per paper) |
| M-LNN penalty coefficient α | not given | 10 |
| Synthetic A, B, C, φ | not given | A=0, B=−1, \|C\|~U(0.05,0.3), φ~U(0,2π) (A,B scale out after min-max rescaling) |
| LM multistart budget | not given | 25 random inits — same protocol/bounds as the reference repo's `fit(max_searches=25)` |
| Framework | not given (GPU) | JAX (CPU), float32 |

## TASI test episode

Daily TASI closes 2020-01-01 → 2024-12-31, sourced from a public GitHub dataset
(investing.com export) and validated against public records: COVID trough
5,959.69 on 2020-03-16, 2021 close 11,281.71, bubble peak **13,820.35 on
2022-05-08**, followed by a drawdown into an October 2022 trough. Yahoo/stooq
were unreachable from this environment; see the report for details.

Protocol (paper Sec. 3.2): 30 calibration windows — every combination of window
length {150, 200, 252, 300, 350} trading days and end date t₂ set {5, 10, 15,
20, 25, 30} trading days before the realised peak; each window linearly
resampled to 252 observations, log-price min-max scaled; every method estimates
(t_c, m, ω) per window and the distribution of predicted t_c is compared with
the realised peak→trough interval.

<!-- RESULTS -->
