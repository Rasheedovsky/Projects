# Results — SPY dual-time 3D CNN (PILOT)

**Overall verdict: NULL (honest).** Registry: 17 trials (effective trials from return-correlation spectrum: 3.9); CSCV PBO: **0.64** (S=16; S=8 reference 0.63); headline cost 1.0 bp/side flat; measured era cost (half of 1-cent spread at 15:31 entry + 0.2 bp fees) averages 0.45 bp/side.

## Trial table (5 CPCV paths, net @1bp)

| trial | hit | AUC | mean path Sharpe ± sd | path Sharpes | DSR | Sortino | maxDD | exposure |
|---|---|---|---|---|---|---|---|---|
| vb_mae | 0.510 | 0.522 | -0.61 ± 0.14 | [-0.617, -0.423, -0.534, -0.837, -0.638] | 0.01 | -0.67 | 0.402 | 1.00 |
| vb_scratch | 0.500 | 0.520 | -0.37 ± 0.03 | [-0.337, -0.411, -0.337, -0.367, -0.397] | 0.02 | -0.48 | 0.379 | 1.00 |
| vb2d | 0.493 | 0.526 | -0.72 ± 0.06 | [-0.623, -0.721, -0.789, -0.672, -0.781] | 0.00 | -0.88 | 0.415 | 1.00 |
| vc | 0.505 | 0.515 | -0.56 ± 0.15 | [-0.552, -0.797, -0.464, -0.646, -0.355] | 0.00 | -0.81 | 0.423 | 1.00 |
| vc2d | 0.505 | 0.504 | -0.72 ± 0.13 | [-0.86, -0.894, -0.568, -0.627, -0.634] | 0.00 | -1.24 | 0.484 | 1.00 |
| va | 0.500 | 0.522 | -0.44 ± 0.06 | [-0.493, -0.529, -0.349, -0.411, -0.427] | 0.00 | -0.73 | 0.404 | 1.00 |
| clock_only | 0.494 | 0.525 | -0.57 ± 0.25 | [-0.546, -0.808, -0.836, -0.524, -0.149] | 0.01 | -0.62 | 0.393 | 1.00 |
| info_only | 0.498 | 0.518 | -0.44 ± 0.09 | [-0.535, -0.335, -0.414, -0.557, -0.365] | 0.02 | -0.52 | 0.373 | 1.00 |
| vb_shuffled | 0.513 | 0.522 | -0.82 ± 0.23 | [-0.778, -1.165, -0.991, -0.68, -0.506] | 0.00 | -1.06 | 0.446 | 1.00 |
| gao_ols | 0.503 | 0.522 | -0.41 ± 0.03 | [-0.438, -0.442, -0.366, -0.42, -0.388] | 0.01 | -0.68 | 0.302 | 1.00 |
| gao_sign | 0.505 | 0.504 | -0.65 ± 0.00 | [-0.654, -0.654, -0.654, -0.654, -0.654] | 0.00 | -0.89 | 0.388 | 1.00 |
| gao_vol | 0.501 | 0.514 | -0.42 ± 0.07 | [-0.448, -0.52, -0.43, -0.328, -0.377] | 0.01 | -0.59 | 0.279 | 1.00 |
| logistic | 0.512 | 0.520 | -1.05 ± 0.27 | [-1.412, -1.074, -0.801, -1.267, -0.711] | 0.00 | -1.51 | 0.638 | 1.00 |
| xgboost | 0.516 | 0.522 | -0.85 ± 0.04 | [-0.862, -0.827, -0.849, -0.79, -0.901] | 0.00 | -1.07 | 0.533 | 1.00 |
| minirocket | 0.516 | 0.509 | -0.81 ± 0.36 | [-1.095, -1.252, -0.944, -0.396, -0.379] | 0.00 | -1.09 | 0.512 | 1.00 |
| always_long | 0.512 | 0.500 | -1.25 ± 0.00 | [-1.249, -1.249, -1.249, -1.249, -1.249] | 0.00 | -1.61 | 0.652 | 1.00 |
| meta_vb_mae | 0.507 | 0.512 | -0.63 ± 0.31 | [-0.391, -0.724, -0.786, -1.087, -0.187] | 0.00 | -0.83 | 0.370 | 0.97 |

## Cost sensitivity (consensus Sharpe)

| trial | 0.5bp | 1.0bp | 2.0bp | 3.0bp | measured |
|---|---|---|---|---|---|
| vb_mae | 0.00 | -0.52 | -1.55 | -2.58 | 0.06 |
| vb_scratch | 0.16 | -0.35 | -1.39 | -2.42 | 0.22 |
| vb2d | -0.15 | -0.67 | -1.70 | -2.73 | -0.09 |
| vc | -0.14 | -0.66 | -1.69 | -2.73 | -0.09 |
| vc2d | -0.45 | -0.96 | -2.00 | -3.03 | -0.39 |
| gao_ols | 0.02 | -0.49 | -1.53 | -2.57 | 0.08 |
| xgboost | -0.33 | -0.85 | -1.89 | -2.93 | -0.28 |
| minirocket | -0.32 | -0.84 | -1.87 | -2.90 | -0.27 |
| always_long | -0.73 | -1.25 | -2.29 | -3.32 | -0.68 |

## Subperiod Sharpe (consensus @1bp)

| trial | 2011-2013 | 2014-2017 | 2018-2021 |
|---|---|---|---|
| vb_mae | -0.79 | -1.93 | 0.27 |
| gao_ols | 0.25 | -1.71 | -0.41 |
| xgboost | 0.10 | -1.70 | -1.12 |
| always_long | -1.19 | -2.32 | -0.93 |

## Model-comparison tests (primary = vb_mae, daily net returns)

- DM vs Gao OLS: t=-0.07, p=0.944
- DM vs XGBoost: t=0.82, p=0.410
- DM vs MiniRocket: t=0.80, p=0.426
- McNemar vs XGBoost: b01=628, p=0.468
- Hit rate on |move| > round-trip cost days (primary): 0.513

## Secondary hypotheses (direction + HAC-DM significance)

- S1 3D beats 2D control (vb_scratch > vb2d): **directional only, n.s.** (DM p=0.293)
- S2a dual beats clock-only: **directional only, n.s.** (DM p=0.512)
- S2b dual beats info-only: **directional only, n.s.** (DM p=0.812)
- S3a V-C beats identical-content 2D (vc > vc2d): **directional only, n.s.** (DM p=0.325)
- S3b V-C beats no-cross-plane (vc > va): **FAIL** (DM p=1.000)
- S4 primary beats Gao OLS: **FAIL** (DM p=1.000)
- S4b primary beats XGBoost: **directional only, n.s.** (DM p=0.410)
- S5 primary beats MiniRocket: **directional only, n.s.** (DM p=0.426)
- S6 SSL pretraining helps: **FAIL** (DM p=1.000)
- S7 depth shuffle hurts (3-seed matched): **directional only, n.s.** (DM p=0.104)

## Success criteria

- hit >= 52.5% with CI excluding 50%: **FAIL**
- net Sharpe > 0 on >= 4/5 paths @1bp: **FAIL**
- mean path Sharpe >= 0.8: **FAIL**
- DSR >= 0.95: **FAIL**
- PBO < 40%: **FAIL**
- design claim (S1, S2a, S2b, S3a, S3b all directional AND p<0.05): **FAIL**
- beats sequence baselines (S5, p<0.05): **FAIL**

**Verdict: NULL (honest)**

## Discussion — honest reading

**Every registered trial, including the Gao OLS replication and always-long, has negative net Sharpe at the headline cost.** In this regime the secondary-hypothesis 'PASS' marks (S1/S2/S3/S7) are orderings among losing strategies — relative rankings of noise — and must NOT be read as evidence for the dual-time design claim. The pilot's substantive conclusions are: (1) the last-half-hour direction is not predictable net of costs in 2011-2021 from morning information under this protocol — consistent with the documented post-2013 decay of intraday momentum (Gao OLS is net-positive only in the 2011-2013 subperiod, matching the decay literature); (2) SSL pretraining does not help (S6): with the v2 corpus — construction-matched to the supervised pipeline — the pretrained primary still trails the from-scratch variant (n.s.); the remaining explanation is era non-stationarity (a 2008-2010 crisis-era corpus vs 2011-2021 evaluation), which gate (f)'s cross-era inversion independently corroborates; (3) the falsifiability diagnostic and the leakage gates passed — look-ahead is excluded and a gross leak is provably detectable, so this is a *gated* null. Critically, the positive-control gate (f) bounds the study's POWER: from a planted series-borne signal with Bayes accuracy 0.611, the pipeline recovers only 0.531 on an in-era holdout, and the recovery INVERTS to 0.372 on the cross-era CPCV test block. Two consequences: (i) this pilot would likely miss a true signal of realistic (52-56%) size, so the null is honest but LOW-POWERED; (ii) whatever weak structure the encodings carry is strongly era-non-stationary under train-fit scaling — a plausible mechanism for the uniformly ~50% out-of-sample hit rates and a first-order target for the full protocol (per-era normalization, adaptation layers); (4) the registry PBO of 0.64 says any in-sample winner here would likely be backtest overfitting.

## Disclosures (execution history)

- **Mid-study rerun:** a first, complete evaluation run (including all 45 primary-trial fits) was executed before leakage gate (b) was first run; the gate then caught a feature-timestamp violation (bar threshold calibration/FFD d*/bar construction used full sessions, letting a train day's own afternoon into its features). Everything derived was wiped and rebuilt under the morning-only construction; the reported matrix is the second, clean run. Recorded in git history (commits around the gate fix).
- **MAE v2:** the SSL checkpoint was retrained after the fix under the corrected construction (per-window bars, era-morning threshold, per-offset FFD lanes); the v1 checkpoint predated the fix and was discarded as irreproducible.
- **Registry amendment v2 (post-referee, disclosed):** vc2d control added (the original S3 comparison mistakenly used V-B content); vb_shuffled promoted to 3 seeds; S3 re-specified as vc>vc2d AND vc>va. All amended trials enter the DSR count.
- **Known unfixed pilot limitations:** r_on carries ~-20 bp artifacts on ~44 quarterly ex-div days (dividend correction deferred; affects the Gao baselines' r_on regressor and one tabular feature); MOC exit is proxied by the 15:59 bar close (auction slippage not modeled); fitted statistics are train-fold-only but not past-only (standard CPCV practice; a walk-forward robustness pass is deferred to the full run); the embargo (5 days) is justified by the ~500-observation FFD weight window (~6 morning-days of memory) — one day short in the strictest reading, immaterial at ~430-day groups but fixed to 6 in the full protocol; CSCV S=16 as registered (S=8 shown for reference).

![equity](figures/equity_curves.png)

![sharpes](figures/sharpe_by_config.png)


## Pilot scope

Pilot compressions vs the full protocol (documented, not silent): CPCV N=6
(15 splits, 5 paths) instead of N=10; 3 seeds instead of 5; max 25 epochs;
GAF encoder only (DWT/MTF not run); FFD input only (vol-scaled ablation not
run); denoise none; R(2+1)D not run; DINOv2/I3D exploratory comparisons not
run; imbalance bars not run as primary (post-fix Phase-1 diagnostic: 29.9%
floor-degenerate vs 7.4% for dollar bars).

