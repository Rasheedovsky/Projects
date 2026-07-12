# PLAN.md — SPY Dual-Time 3D CNN (Publication-Grade)

**Target venue:** Journal of Financial Data Science
**Status:** PHASE-0 GATE — awaiting your approval before any code is written.
**Branch:** `claude/spy-dual-time-3d-cnn-6eq0ge`

The fixed centerpiece is unchanged and not up for debate: a three-dimensional tensor
encoding the same trading morning in **clock time** and **information time**, convolved
jointly by a 3D CNN. Everything below serves that construction. This document covers
(1) what I found in your data, (2) open questions you must answer, (3) the phase plan
with gates and tests, (4) design disagreements with reasons, (5) compute budget for
this environment, and (6) the literature/novelty scan for the Cross-Time Gramian field.

---

## 1. Data reality check (what I actually found)

I located your data in Google Drive (no `./data/raw/` exists in the repo yet):

| File | Size | Coverage | Schema |
|---|---|---|---|
| `spy_1min_2008_2021_cleaned.csv` | 134 MB | 2008-01-22 → 2021 | `date,open,high,low,close,volume` (assumed same as 5-min) |
| `spy_5min_2008_2021_cleaned.csv` | 15 MB | 2008-01-22 → 2021 | `date,open,high,low,close,volume` |

Facts that change the plan as written:

1. **Coverage is 2008–2021, not 2008–present.** Usable labeled days ≈ **3,400**
   (~3,500 sessions minus ~125 half days), not ~4,300. All power calculations and the
   regime split ("2016–present" → **2016–2021**) are adjusted accordingly. Gao et al.'s
   sample ends ~2013, so 2014–2021 still provides ~8 years of post-sample evidence.
2. **Timestamps are not ET.** Sessions run 07:30→~14:10 in file time (e.g. 2008-01-22).
   That is consistent with US Mountain time (ET−2h year-round) or a vendor fixed offset.
   The 14:05/14:10 bars look like post-close prints (16:05/16:10 ET). Phase 0 will
   verify the offset against known event minutes (2010-05-06 flash crash ~14:45 ET)
   and normalize everything to tz-aware ET. **Hard rule:** the session must reconstruct
   to exactly 09:30–16:00 ET with half days detected by early close at 13:00 ET.
3. **No auxiliary 1-min tickers exist in your Drive.** (A `QQQ.csv` exists but is daily;
   the Netflix intraday file is one day of 15-min bars.) The prompt's SSL default —
   "pretrain on non-SPY tickers only" — is **impossible with the data on hand**.
   See Open Question 2.
4. **The Drive files are private.** Direct download from this environment fails
   (redirects to Google sign-in), and the Drive MCP connector cannot stream a 134 MB
   file. See Open Question 1.
5. **Adjustment status unknown.** "cleaned" suggests processing, but I cannot yet tell
   whether prices are split/dividend adjusted. This matters for `r_on` (overnight
   return) and the Gao OLS replication: SPY goes ex-dividend ~quarterly and an
   unadjusted overnight return has ~30 bp artificial drops. Intraday features are
   unaffected either way. Phase 0 audit will scan for ex-div signatures; if unadjusted,
   `r_on` gets a dividend correction from SPY's distribution history (fetched or
   provided). SPY has never split in this window, so split risk is nil.

Repo state: the repository currently contains only standalone showcase notebooks at the
root. Phase 0 will move them to `notebooks/legacy/` (reversible; nothing is deleted).

## 2. Open questions — please answer these with your approval

1. **Data access (blocking).** Pick one:
   a. *(Recommended, ~30 seconds)* In Drive, set both CSVs to "Anyone with the link
      can view". I then pull them with `gdown`/`curl` into `data/raw/`, record SHA-256,
      and you can revoke sharing afterward.
   b. Provide a Kaggle API key (`KAGGLE_USERNAME`/`KAGGLE_KEY` as env vars in this
      environment's settings) and name the dataset you sourced these files from.
   c. Commit the CSVs to the repo yourself (134 MB requires Git LFS — not recommended).
2. **SSL corpus (design decision).** With no auxiliary tickers on hand, choose:
   a. *(Recommended)* I fetch free 1-min history for liquid ETFs (QQQ, IWM, DIA, XLF,
      XLE, …) from a public source (candidates verified in §7; Kaggle needs your API
      key, GitHub mirrors need none). Pretraining then uses **non-SPY tickers only**,
      exactly as the prompt intended — cleanest leakage story, biggest corpus.
   b. SPY-only SSL with a **temporal firewall**: pretrain exclusively on 2008–2012
      windows; run all supervised CPCV evaluation on 2013–2021 (~2,250 labels). Clean,
      but sacrifices ~1/3 of labels and the corpus shrinks to ~60k volumes.
      (Note: SPY-only SSL on *all* years violates the prompt's own provenance rule —
      under CPCV every group is a test group in some split. Not an option.)
3. **Compute (affects scope).** This remote environment has **no GPU** (4 CPU cores,
   15 GB RAM). The tiered protocol in §6 makes the project feasible on CPU (~1–2 weeks
   wall-clock across phases, mostly unattended), but a GPU (free Colab/Kaggle T4, or
   your own) collapses Phase 3+7 from ~a week to ~a day. Options:
   a. Run everything here on CPU with the tiered protocol (reduced corpus stride,
      finalists-only full CPCV).
   b. I build everything here with tests on CPU; you run the two heavy scripts
      (`pretrain.py`, `run_matrix.py`) on a GPU box and drop the checkpoints back.
   c. You attach a GPU-enabled environment for Phases 3 and 7.
4. **Registry:** MLflow requires a running server; default is a **CSV/JSON registry**
   (append-only, keyed by config-hash + seed, committed to the repo). Object if you
   specifically want MLflow.
5. **Skills:** the superpowers/quant/probabl skills from your setup block are not
   installed in this remote environment (that setup ran on your machine). I substitute:
   TDD discipline enforced by the phase gates below, the built-in `code-review` +
   `security-review` skills, and a written adversarial referee pass at Phase 8
   (same function as backtest-review/strategy-critique). I can additionally clone the
   two skill repos into the container if you want their checklists applied verbatim.

**Default if you say "proceed" without specifics:** 1a + 2a + 3a + CSV registry.

## 3. Phase plan

Phases match your prompt; deviations are flagged **[Δ]** and argued in §4.
Every phase ends with its tests green before the next begins; gated phases stop for you.

### Phase 0 — Scaffold & audit *(gate: STOP for your review)*
- Scaffold `src/`, `configs/`, `tests/`, `notebooks/`, `reports/`, `data/{raw,interim,tensors,pretrain}`;
  pinned `requirements.txt` (pandas, numpy, scipy, statsmodels, scikit-learn, pyts,
  PyWavelets, torch, torchvision, timm, sktime, xgboost, matplotlib, pyyaml); move
  legacy notebooks to `notebooks/legacy/`.
- Ingest data (per Open Q1), record SHA-256, write immutable copy to `data/raw/`.
- `reports/data_audit.md`: timezone/DST resolution (verified against event minutes),
  missing-minute census (report per-year), half-day detection (drop list), adjustment
  check (ex-div scan vs SPY distribution dates), outlier scan (price spikes > 20σ,
  zero/negative volume), event-day sanity panels (2010-05-06, 2015-08-24, 2018-02-05,
  2020-03-16), usable N per ticker, label-balance by year.
- **[Δ]** Additional drop rule: days missing >10% of minutes in 09:30–11:00 or any
  minute in 15:25–16:00 (the decision/execution leg must be intact).
- Tests: session reconstruction (390 RTH minutes), TZ round-trip, half-day detector
  against known NYSE early closes, audit functions on synthetic corrupted fixtures.

### Phase 1 — Two time axes
- Sessionize; `r_on = ln(open_t / adj_close_{t-1})` (dividend-corrected if needed).
- Clock axis: 1-min log returns 09:30–11:00 (90 pts) → PAA to L ∈ {32, 45}.
- Information axis per AFML ch. 2.3 behind one `BarScheme` interface:
  tick-rule signing on 1-min closes; **dollar bars** (simple threshold); **dollar
  imbalance bars** (EWMA-updated E₀[T] and expected signed flow, warm-start on first
  train month, 3×E₀[T] cap **plus a floor of 3 underlying minutes per bar [Δ]**);
  volume-bar and tick-bar variants.
- Calibration on train folds so median day yields K bars in 09:30–11:00;
  **[Δ] K ∈ {16, 32}** (not {32, 64} — see §4.2), take first K bars, pad-flag days
  with fewer (report frequency and clock-span distribution). Clock span of the K bars
  kept as tabular feature.
- FFD (AFML ch. 5) per axis: **[Δ] applied to the continuous intraday log-price stream
  per axis, then window-sliced** (not per-90-obs-window — see §4.5); d* = smallest
  d ∈ [0,1] step 0.05 passing ADF 95%, train folds only, per axis. Ablation: FFD vs
  vol-scaled returns.
- Optional denoise per axis (config): SSA (pyts, top-r) or 1-D conv DAE, train-fold fit.
- Vol scaling by train-fit EWMA σ; realized 90-min vol as tabular feature.
- Tests: bar-boundary correctness on hand-built fixtures, imbalance-threshold update
  math, EWMA warm-up isolation from test data, K-bar extraction determinism, FFD
  weights vs closed form, tick-rule zero-change carry-forward, PAA exactness.

### Phase 2 — Dual-time 3D tensor *(gate: STOP — I show you example volumes from all three constructions)*
- Per-axis encoders (both axes length-normalized to L): GAF = GASF+GADF (per-window
  min-max to [−1,1]; amplitude lives in tabular features), optional MTF; DWT db4 L4–5
  coefficient image (alt: CWT morlet), train-fold standardized.
- **V-A** direct dual-time stack: L×L×2 volume, depth = time-representation axis,
  channels carry encodings; (k,k,2) kernels convolve both geometries in one op.
- **V-B (primary)** deep dual-time volume: 5 overlapping sub-windows per axis
  (clock 0–30/15–45/30–60/45–75/60–90 min; info the analogous bar-index spans),
  each L′×L′, depth ordering [clock₁..₅, info₁..₅] → L′×L′×10. Also the R(2+1)D
  factorization of the same volume.
- **V-C (novelty candidate)** Cross-Time Gramian field: φᵢ = arccos(clock̃ᵢ),
  ψⱼ = arccos(iñfoⱼ), **C_ij = cos(φᵢ + ψⱼ)** — rows are clock time, columns are
  information time. Volume = [GASF_clock, C_cross, GASF_info], depth traverses
  clock → joint → info. Novelty status: see §7.
- Mandatory ablation control: identical content as multi-channel 2D (C×H×W) → 2D CNN.
- **[Δ]** Single-axis controls (clock-only and info-only volumes) are built here too —
  success criterion 5 needs them and they share the caching machinery.
- Tensor cache keyed by config hash; **[Δ]** stored uint8-quantized (GAF is bounded
  [−1,1]) to respect this container's ~30 GB free disk; slice grids →
  `reports/figures/`.
- Tests: GASF/GADF closed-form on tiny fixtures, C_cross symmetry/range/agreement with
  GASF when both axes are identical, depth-ordering invariants, encode-decode
  determinism, cache-hash stability.

### Phase 3 — SSL pretraining
- Corpus: rolling 90-min windows, 5-min stride (CPU fallback: 15-min stride), built
  into the same dual-time volumes; per-ticker bar thresholds calibrated on train data.
  Provenance manifest (ticker, day, window start) logged per corpus build.
- Leakage rules enforced in code: corpus filtered against every CPCV test period;
  default corpus = non-SPY tickers only (per Open Q2 decision).
- Primary: **3D MAE**, 75% tube masking, on the 3D CNN backbone below (conv-MAE style:
  dense encoder, masked reconstruction loss). Alternative: SimCLR-3D (crops/jitter in
  time domain before encoding — never image flips). Checkpoints + reconstruction
  figures saved.
- Cheap mandatory comparisons: frozen **DINOv2 ViT-S/14** per-depth-slice embeddings,
  concatenated → logistic/ridge probe; **ImageNet ResNet-18 inflated to 3D**
  (I3D-style), frozen + linear head, and fine-tune-last-block variant.
- Pooled supervised option (config): train on all tickers' labeled volumes, evaluate
  held-out on SPY (Jiang–Kelly–Xiu-style cross-sectional pooling).
- No generative VLMs as predictors; one line in related work.
- Tests: corpus/test-period intersection is provably empty, mask-ratio correctness,
  tube-mask geometry, checkpoint reload equivalence.

### Phase 4 — Models & regularization
- **Primary:** SSL-pretrained 3D CNN on V-B + tiny head, fine-tuned. 3–4 conv blocks,
  (3,3,3) kernels, stride-2 depth pooling only after the full clock↔info depth extent
  has been traversed twice, 8→16→32→64 filters, 3D GAP → logit. Budget ≤ 300k params
  (current sketch ≈ 60k — headroom documented). R(2+1)D same budget. From-scratch = the
  no-pretraining ablation.
- V-A and V-C through the same trunk (first-layer depth adjusted); all three reported.
- Tabular fusion: [r_on, realized_vol_90m, info_bar_clockspan, pad_flag, day_of_week]
  concatenated to pooled features before the logit.
- Regularization (default-on): AdamW wd 1e-2 (log-tuned), dropout 0.3–0.5, label
  smoothing 0.1, early stopping on purged val split, mixup α=0.2 on volumes
  **[Δ] config-off-able — see §4.7**, time-domain augmentation before encoding only
  (jitter, magnitude warp, window warp; never image flips/rotations), cosine LR, grad
  clipping, 5-seed averaging + CPCV-fold ensembling.
- Sequence baselines: MiniRocket on [clock; info] fracdiff sequences, 1-D CNN, small
  LSTM. Tabular baselines: Gao OLS replication (r_last30 ~ r_first30 + r_on),
  logistic + XGBoost on handcrafted features, always-long, coin flip.
- 2D multi-channel control from Phase 2.
- Tests: parameter-budget assertion, augmentation-invariant (augment→encode ≠
  encode→augment guard), deterministic seeding, tabular-fusion shape, head-only vs
  full fine-tune flags.

### Phase 5 — Validation (CPCV)
- Purged K-Fold + CPCV per AFML ch. 7 & 12: N=10 groups, k=2 → 45 splits → 9 paths;
  purge 1 day, embargo 3 days. **[Δ]** Screening tier uses a cheaper protocol (§6);
  full CPCV runs for finalists and all baselines.
- Everything fits inside training folds: d* per axis, bar thresholds/EWMA spans,
  scalers, SSA/DAE, encoder stats, SSL corpus selection, nested hyperparameters.
- Leakage tests gating ALL reporting:
  (a) shuffled-label collapse to chance; (b) max feature timestamp < 15:30 ET per
  sample; (c) train-on-future canary the purge code must catch; (d) SSL corpus
  provenance vs test periods; (e) bar-threshold calibration uses train days only.
- Tests: purge/embargo boundary proofs on synthetic calendars, path reconstruction
  (every group appears in exactly 9 test paths), split determinism.

### Phase 6 — Meta-labeling
- Primary probability → meta-model (logistic/XGBoost) on [primary_prob, r_on,
  realized_vol, info_bar_clockspan, day_of_week] for trade/no-trade + de Prado
  probability bet sizing. Report primary-only vs meta-labeled.
- Tests: meta-features computed strictly from information available at 15:30;
  bet-size monotonicity in probability.

### Phase 7 — Experiment matrix
- YAML grid pruned to ~40 runs × 5 seeds: construction {V-A, V-B, V-C} × encoder
  {gaf, dwt, both} × input {ffd, vol-scaled} × denoise {none, ssa, dae} × arch
  {3d, (2+1)d, 2d-control} × pretraining {none, mae3d, dino-probe, inflated-imagenet}
  × bars {dollar, dollar-imbalance}. **Every trial logged for DSR correction**
  (screening trials included); registry keyed by config hash + seed.
- **[Δ]** Two-tier execution per §6 (screen → finalists), forced by CPU budget.

### Phase 8 — Reporting *(gate: adversarial referee pass; every issue answered in `reports/critique_response.md`)*
- `reports/results.md` + LaTeX: baseline table; the central V-A/V-B/V-C vs 2D-control
  table; single-axis (clock-only/info-only) comparison; ablations; CPCV path-Sharpe
  distributions; PBO plot; cost sensitivity (0.5/1/2 bp **+ 3 bp stress [Δ]**); regime
  split 2008–2015 vs 2016–2021; SSL-benefit learning curves; example volumes;
  architecture diagram; limitations.
- Success metrics and criteria exactly as specified in your prompt (hit rate ≥ 52.5%
  with CI excluding 50%; net Sharpe > 0 at 1 bp on ≥ 7/9 paths, mean ≥ 0.8; DSR ≥ 0.95,
  PBO < 40%; DM vs Gao OLS p < 0.05 and beats XGBoost; dual-time 3D beats 2D control
  AND both single-axis volumes; volumetric beats MiniRocket/1-D CNN; all five leakage
  tests pass) — else the honest null/methodology paper.

## 4. Design disagreements & risk register (reasons attached)

The centerpiece stays. These are amendments where the spec as written would hurt the
paper or cannot be executed on the actual data. Each needs your sign-off or veto.

1. **Dollar-imbalance bars as *primary* info clock is risky on 1-min inputs.**
   With ~90 signed observations per morning, tick-rule signs are near-balanced, so the
   expected-imbalance term E₀[T]·|2v⁺−E[p·v]| hovers near zero → degenerate one-tick
   bars or (with the cap) mostly cap-triggered bars, i.e. the "information clock"
   becomes an artifact of the cap. **Proposal:** implement both exactly as specced,
   but decide primary-vs-secondary at the Phase 1 gate from measured degeneracy stats
   (share of cap-triggered bars, bar-count dispersion). Plain dollar bars are the
   defensible default primary; imbalance bars stay as the config variant. I also add a
   floor of 3 underlying minutes per bar to both schemes.
2. **K = 64 information bars by 11:00 is infeasible.** The window contains only 90
   1-min observations; K=64 forces median bar ≈ 1.4 observations — the info axis
   degenerates into a noisy copy of the clock axis, which would *manufacture* the
   dual-time redundancy the 2D control is supposed to test. **Proposal:** K ∈ {16, 32}
   with a hard bound K ≤ 45 (half the observation count).
3. **Full CPCV × full matrix is not computable here** (9,000 fits ≈ 2,000+ CPU-hours
   on this box). **Proposal (§6):** tier 1 screens all ~40 configs × 3 seeds on a
   5-split purged CV; tier 2 runs the full 45-split CPCV × 5 seeds for ~8 finalists
   plus *all* baselines and controls. Screening trials still enter the DSR trial count.
   This changes execution, not the reported statistics.
4. **Label count and sample split.** ~3,400 usable labels (not ~4,300); regime split
   2008–2015 vs 2016–2021. Power at hit-rate 52.5%: with N≈680 per path union and
   9 paths this is detectable, but the 95%-CI-excludes-50% criterion will require the
   effect to be real, which is the point.
5. **FFD per 90-obs window is numerically unsound.** With |w|<1e-4 truncation, the
   effective weight window at d*≈0.3–0.5 spans hundreds of observations — longer than
   the window itself; per-window FFD would be all warm-up artifact. **Proposal:** FFD
   runs on each axis's *continuous* intraday stream (train-fold-estimated d*), then
   windows are sliced from the differentiated stream. Same math, correct warm-up.
6. **SSL non-SPY default requires acquiring data** (Open Q2). If you choose SPY-only,
   the temporal-firewall variant replaces the prompt's provenance rule, and the paper
   says so explicitly.
7. **Mixup on Gramian volumes is semantically odd** (a convex mix of GAFs is not the
   GAF of any series). Acceptable as pure regularization, but stacked with label
   smoothing + dropout on a ≤300k-param model it risks underfitting. Keep default-on
   as specced, but expose `mixup: off` in the config and include it in the ablation
   appendix rather than silently trusting it.
8. **Costs.** MOC exit has no spread but NYSE MOC fees; 15:30 entry pays ~half-spread
   (SPY: 1 cent on ~$300–450 ≈ 0.15–0.3 bp) + fees. Your 0.5/1/2 bp grid brackets this
   well; I add a 3 bp stress column so a referee cannot claim cost optimism.
9. **Adjustment risk** (§1.5): if the series is dividend-adjusted with *backward*
   adjustment, early-year prices are distorted for dollar-bar thresholds (p·v uses
   adjusted p × raw v). Audit decides; if adjusted, dollar bars use adjustment-ratio
   de-scaled prices. Flagged now so it isn't a surprise at Phase 1.

## 5. Success criteria

Adopted verbatim from your prompt (§3 Phase 8 lists them) — no changes. The only
addition: single-axis controls are explicitly built in Phase 2 so criterion 5 is
testable, and the DSR trial registry provably includes screening-tier trials.

## 6. Compute budget (this environment: 4 CPU cores, 15 GB RAM, no GPU)

Measured/estimated with the ~3,400-day corpus:

| Item | Size | CPU (this box) | T4 GPU |
|---|---|---|---|
| Tensor build, all 3 constructions (labeled) | ~3,400 × ≤10×32×32×2 | ~1–2 h | — |
| SSL corpus build (SPY-only, 5-min stride) | ~205k volumes ≈ 4 GB uint8 | ~4–8 h | — |
| 3D MAE pretraining, 150 epochs @ ~250 MFLOPs/step | ~7.5 PFLOPs | ~2–4 days (15-min stride + 100 ep → **8–16 h**) | ~1–3 h |
| DINOv2 ViT-S/14 slice embeddings (one-time, cached) | ~34k images | ~2–3 h | ~10 min |
| One supervised fit (≤300k params, ~2,700 train days) | ~30 TFLOPs | ~10–15 min | <1 min |
| Tier-1 screen: 40 configs × 3 seeds × 5 splits | 600 fits | ~3–5 days unattended | ~4 h |
| Tier-2 CPCV: ~8 finalists + baselines × 5 seeds × 45 splits | ~2,000 fits (cheap baselines ≈ free) | ~1 week unattended | ~1 day |
| MiniRocket / XGB / OLS baselines | full CPCV | minutes–hours | — |

CPU-only is feasible with the tiered protocol and reduced corpus stride, ~2 weeks
wall-clock mostly unattended; a GPU collapses the two heavy rows to ~a day (Open Q3).
Disk: tensor caches stored uint8 with config-hash keys; at most two encoder configs
cached simultaneously (~30 GB free here); GAF encoding is cheap enough to re-encode
on the fly in the dataloader when the cache is evicted.

## 7. Novelty & literature scan (Cross-Time Gramian field)

*Being verified right now by a parallel literature-search workflow; this section will
be finalized from its output before you approve.*

Preliminary position from my own knowledge: GAF/MTF imaging of single series
(Wang & Oates 2015) and multi-channel 2D GAF stacks for finance are well-trodden;
information-driven bars (López de Prado 2018) are standard; but a *Gramian cross-field
between two time-warped representations of the same instrument* (C_ij = cos(φᵢ+ψⱼ)
with rows in clock time, columns in information time) does not match anything I know.
Closest families to check and cite: cross-recurrence plots, joint recurrence
quantification, multivariate GAF fusion, and video-style 3D CNNs on stacked financial
images. Final verdict + closest-work table lands here.

## 8. References (core)

- Gao, Han, Li & Zhou (2018), "Market intraday momentum", JFE.
- Lou, Polk & Skouras (2019), "A tug of war: Overnight versus intraday expected returns", JFE.
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 2, 5, 7, 12.
- Wang & Oates (2015), "Imaging time-series to improve classification and imputation", IJCAI.
- Jiang, Kelly & Xiu (2023), "(Re-)Imag(in)ing price trends", JF.
- Tran et al. (2018), "A closer look at spatiotemporal convolutions" (R(2+1)D), CVPR.
- He et al. (2022), "Masked autoencoders are scalable vision learners", CVPR;
  Tong et al. (2022), "VideoMAE", NeurIPS.
- Bailey & López de Prado (2014), "The deflated Sharpe ratio"; Bailey et al. (2017),
  "The probability of backtest overfitting".
- Dempster, Schmidt & Webb (2021), "MiniRocket", KDD.

## 9. Approval checklist (what I need from you)

- [ ] Answer Open Questions 1–3 (data access, SSL corpus, compute); 4–5 optional.
- [ ] Approve or veto disagreements §4.1–4.9 (silence = approve as proposed).
- [ ] Approve Phase 0 start.

Nothing beyond this document has been built. On approval I begin Phase 0
(scaffold + audit) and stop at its gate with `reports/data_audit.md`.
