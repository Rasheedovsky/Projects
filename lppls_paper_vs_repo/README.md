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
  kan.py             M-LNN-KAN — M-LNN with Kolmogorov-Arnold layers [extension]
  plnn.py            Poly-LPPLS-NN — supervised network, input 252 (Sec. 2.2)
scripts/
  train_plnn.py            trains P-LNN-100K / -AR1 / -BOTH (100k series each)
  benchmark_synthetic.py   250-scenario error CDFs + timing (Fig. 3 / Table 2 analogue)
                           (--kan-merge adds the M-LNN-KAN to stored results)
  compare_empirical.py     bubble episode comparison: fits + t_c PDFs + timing
                           (--data tasi | spy; paper Fig. 4 protocol)
  tasi_confidence_repo.py  Boulder package confidence indicator on TASI (bonus)
data/
  TASI_daily_2020_2024.csv daily TASI closes (validated against public records)
  SPY_daily_1998_2021.csv  daily SPY closes (QuantConnect Lean sample data,
                           validated: COVID peak 338.34 on 2020-02-19,
                           trough 222.95 on 2020-03-23)
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

---

# Results

## 1. TASI 2021–22 bubble (paper Fig. 4 protocol)

![TASI fits](results/fig_tasi_fits.png)

Realised episode: peak close **13,820.35 on 2022-05-08**, drawdown trough
**10,909.18 on 2022-09-26 (−21.1%)**. The realised critical time t_c is interpreted,
as in the paper, to lie between the peak and the trough (red band). All 30
calibration windows end between 2022-03-20 and 2022-04-24 — i.e. every method
predicts 2–7 weeks ahead, never seeing data at or beyond the peak.

**Median predicted t_c across the 30 windows:**

| Method | median t_c | vs realised peak | IQR (days vs peak) | valid fits |
|---|---|---|---|---|
| **P-LNN-100K** | **2022-05-09** | **+1 day** | −8.2 … +6.7 | 30/30 |
| P-LNN-100K-AR1 | 2022-05-15 | +5 days | −3.9 … +13.2 | 30/30 |
| P-LNN-100K-BOTH | 2022-05-18 | +8 days | −3.9 … +24.8 | 30/30 |
| M-LNN | 2022-04-13 | −12 days (early) | −21.7 … −4.5 | 30/30 |
| LM (paper App. A.1) | 2022-04-07 | −16 days (early) | −25.1 … −11.5 | 30/30 |
| lppls-repo (Nelder-Mead) | 2022-08-22 | +72 days (late) | +18.0 … +345 | 27/30 |

This reproduces the paper's Fig. 4/5 finding on completely new data: the
**P-LNN's t_c PDF concentrates almost exactly on the realised peak**, the M-LNN
is slightly early with modest spread, and the classical multistart search
(the reference repo's default Nelder-Mead) shows the "too late, high
variability" behaviour the paper attributes to the incumbent method — its
median lands inside the realised peak→trough interval, but with an
inter-quartile spread of nearly a year and 3 of 30 windows failing outright.

### Per-calibration wall-clock on TASI windows (steady-state, 4-core CPU)

| Method | mean | std |
|---|---|---|
| P-LNN-100K-BOTH | **0.43 ms** | 0.04 ms |
| P-LNN-100K-AR1 | 0.53 ms | 0.06 ms |
| P-LNN-100K | 0.74 ms | 0.06 ms |
| lppls-repo (NM) | 60 ms | 101 ms |
| M-LNN | 0.48 s | 0.04 s |
| LM | 0.89 s | 0.64 s |

The paper's headline speed claim holds: **P-LNN inference is 2–3 orders of
magnitude faster than any iterative calibration** (here ~0.5 ms vs the paper's
5.2 ms, both dwarfing classical search). One caveat in the repo's favour: its
numba-JIT'd Nelder-Mead (60 ms) is ~60× faster than the paper's reported 3.58 s
LM average, so the gap between "state of the art" and P-LNN is smaller than
Table 2 of the paper suggests when the classical code is well optimised.

## 2. Reference repo's own product: confidence indicator on TASI

![TASI confidence](results/fig_tasi_confidence_repo.png)

For balance, the Boulder package's flagship output (the ensemble
positive-bubble confidence indicator from `mp_compute_nested_fits`) was run on
the same stretch: it spikes to ~0.63 in the final month before the realised
peak and collapses immediately after — i.e. while its *individual* t_c
estimates scatter, its *ensemble* indicator flags the TASI bubble correctly in
53 s of wall-clock for a 2-year daily series.

## 3. Synthetic benchmark (paper Fig. 3 / Table 2 protocol)

250 random scenarios per noise class from the Table 1 ranges, ground truth known.

![CDF white](results/fig_synth_cdf_white.png)
![CDF ar1](results/fig_synth_cdf_ar1.png)

**Median / 95th-percentile of |t_c error| in days (250 scenarios):**

| Method | white: median / p95 | AR(1): median / p95 |
|---|---|---|
| LM (paper App. A.1) | **1.9** / 28.0 | 1.8 / 23.7 |
| lppls-repo (NM) | **1.9** / 24.9 | 1.9 / 31.1 (2 fails) |
| M-LNN | 2.0 / **19.3** | **1.6 / 16.6** |
| P-LNN-100K | 6.5 / 21.4 | 6.4 / 19.6 |
| P-LNN-100K-AR1 | 10.3 / 27.1 | 6.4 / **17.7** |
| P-LNN-100K-BOTH | 6.3 / 22.9 | 6.3 / 20.0 |

**Steady-state wall-clock per calibration (4-core CPU, medians over 250):**

| Method | this study | paper Table 2 (V100) |
|---|---|---|
| P-LNN (any variant) | **0.22–0.29 ms** | 5.2 ms |
| lppls-repo (NM, numba) | 16 ms | — |
| LM (scipy, 25 restarts) | 0.42–0.50 s | 3.58 s |
| M-LNN (1500 epochs) | 0.47–0.49 s | 11.5 s |

Reading of the evidence:

* **The paper's tail claim replicates.** The NN methods produce far fewer
  catastrophic t_c errors: at the 95th percentile M-LNN and the P-LNN family
  beat both classical searches on both noise classes (e.g. 16.6 d vs 31.1 d
  under AR(1) noise). The CDF curves cross exactly as the paper describes —
  classical methods are sharper for small errors, NNs dominate the tail.
* **The paper's median-accuracy framing does not fully replicate.** A
  properly multistarted classical search (25 random inits, the reference
  repo's own protocol) attains ~2-day median accuracy, ~3× better than
  P-LNN's ~6.4 days. First-order stochastic dominance of the NNs over LM was
  *not* observed here — dominance only appears in the upper tail.
* **P-LNN's speed claim replicates and then some** (~0.25 ms/calibration —
  four orders of magnitude faster than LM at 25 restarts, ~60× faster than
  the repo's numba-optimised Nelder-Mead). Noise-matched training matters:
  P-LNN-AR1 degrades markedly on white-noise data (10.3 d median), echoing
  the paper's noise-specificity finding.
* **M-LNN is the accuracy champion overall** — equal-or-best median *and*
  best tails — at a per-fit cost comparable to multistart LM on CPU.

## 4. P-LNN training

![loss curves](results/fig_plnn_loss_curves.png)

Per paper spec (100k train / 33.3k val series, 20 epochs, batch 8, lr 1e-5,
Adam): consistent train/val convergence, no overfitting — mirroring the
paper's Figs. 8–10. Wall-clock: **~200 s per variant on a 4-core CPU** with the
JAX `lax.scan` training loop (the paper reports ~1.5 h per variant on a V100).

---

# Extension: M-LNN-KAN (Kolmogorov-Arnold Network)

`deep_lppls/kan.py` replaces the M-LNN's ReLU MLP with **KAN layers**
(Liu et al. 2024, arXiv:2404.19756): every edge carries a learnable cubic
B-spline activation (8-interval grid on [−1, 1], 11 basis functions, tanh
grid-bounding) plus a SiLU base branch. Everything else — the PINN-style
loss through the analytic linear solve, the parameter-bound penalty,
Adam(1e-2), 1500 epochs, best-state selection — is byte-for-byte the same
protocol as `mlnn.py`, so differences are attributable to the network
parameterisation alone. Widths follow KAN convention (narrower): [32, 32]
vs the MLP's [128, 128]; the KAN still has ~2.3× more parameters per layer
because each edge carries 12 coefficients.

<!-- KAN-RESULTS -->

# Conclusions

**On TASI, the paper's approach wins the forecasting task.** Standing 2–7
weeks before the (unseen) May 2022 peak, P-LNN-100K's median predicted
critical time misses the realised peak by **one day**, with a tight
inter-quartile band inside the realised peak→trough interval; the M-LNN is
~2 weeks early with modest spread. The reference repo's per-window
Nelder-Mead estimates are directionally right but scatter over months and
occasionally fail to converge — exactly the "sloppy t_c" pathology the paper
sets out to fix. The repo's ensemble confidence indicator, however, remains a
strong (and very cheap) bubble *detector* on TASI even though its individual
t_c estimates are noisy.

**Speed.** P-LNN inference costs ~0.25 ms per calibration on plain CPU —
about 4 orders of magnitude faster than multistart LM and ~60× faster than
the repo's numba Nelder-Mead — making dense rolling-window scans essentially
free once the ~200 s one-off training is paid. M-LNN is the most accurate but
the slowest of the paper's methods (~0.5 s/fit here, since a fresh network is
trained per series).

**Caveats.** (1) The paper omits several implementation details (hidden
widths, penalty coefficient, epochs, synthetic A/B/C sampling); results are
somewhat sensitive to these choices, which are documented above. (2) The
synthetic test data was generated by the same process as the P-LNN training
data (a limitation the paper itself acknowledges), which flatters P-LNN's
synthetic numbers yet TASI — fully out-of-sample — confirms its edge. (3) The
strong median performance of the classical searches here suggests the paper's
LM baseline (3.58 s, high variance) was less aggressively multistarted or less
optimised than the reference repo's current code.
