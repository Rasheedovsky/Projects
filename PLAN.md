# PLAN.md — SPY Dual-Time 3D CNN (Publication-Grade)

**Target venue:** Journal of Financial Data Science
**Status:** PHASE-0 GATE — awaiting your approval before any code is written.
**Branch:** `claude/spy-dual-time-3d-cnn-6eq0ge`

The fixed centerpiece is unchanged: a three-dimensional tensor encoding the same
trading morning in **clock time** and **information time**, convolved jointly by a
3D CNN. Everything below serves that construction.

This plan was stress-tested before writing: a parallel research pass ran
(1) a literature novelty scan for the Cross-Time Gramian field (~20 query families
across arXiv/SSRN/Scholar surfaces), (2) live verification of data sources reachable
from this environment, and (3) an adversarial referee critique of the full design
(6 blockers, 11 majors, 4 minors — all resolved or surfaced below). Sections:
§1 data findings, §2 open questions, §3 phase plan, §4 design amendments and
disagreements, §5 success criteria, §6 compute budget, §7 novelty scan, §8 references.

---

## 1. Data reality check (what I actually found)

**Your Drive** (located; no `./data/raw/` exists in the repo yet):

| File | Size | Coverage | Schema |
|---|---|---|---|
| `spy_1min_2008_2021_cleaned.csv` | 134 MB | 2008-01-22 → 2021 | `date,open,high,low,close,volume` |
| `spy_5min_2008_2021_cleaned.csv` | 15 MB | 2008-01-22 → 2021 | same |

Both files are **private** (direct download from this box redirects to Google
sign-in, and the Drive connector cannot stream 134 MB). However —

**Verified alternative (live-tested from this box, no credentials):** the Kaggle
dataset `gratefuldata/intraday-stock-data-1-min-sp-500-200821` (SPY-only despite the
name; 2.07M one-minute bars, 2008–2021, CC0 license, Interactive-Brokers-style schema
with `barCount`/`average` columns, ~612 bars/day ⇒ includes extended hours) downloads
anonymously via `curl -L https://www.kaggle.com/api/v1/datasets/download/...`.
Your Drive file is almost certainly a cleaned derivative of this dataset (same window,
same base schema, RTH-trimmed) — matching your "use github to access the data in
kaggle" instruction. A near-duplicate (`rockinbrock/spy-1-minute-data`) exists as
cross-check. **Data access is therefore not a blocker.**

Facts that change the plan as written:

1. **Coverage is 2008-01-22 → 2021-05-06, not 2008–present.** Measured in the
   Phase-0 audit: **3,316 usable labeled days** (3,347 sessions − 29 half days −
   2 gap days), not ~4,300. Gao et al.'s sample ends 2013, so 2014–2021 remains
   genuine post-sample evidence.
2. **Timestamps are not ET.** Drive sessions run 07:30→~14:10 file time — consistent
   with US Mountain time (ET−2h) or a vendor offset; the Kaggle original includes
   extended hours. Phase 0 verifies the offset against known event minutes
   (2010-05-06 flash crash ~14:32–14:45 ET) and normalizes to tz-aware ET; sessions
   must reconstruct to exactly 09:30–16:00 ET with half days closing 13:00 ET.
3. **No auxiliary 1-min tickers in your Drive** (the QQQ.csv there is daily; the
   Netflix intraday file is one day). Aux-ticker options verified in §1.1.
4. **Adjustment status unknown.** IB-provenance data is typically split-adjusted but
   NOT dividend-adjusted. Policy (referee-mandated, adopted): **unadjusted prices
   everywhere** that nominal quantities matter — dollar bars (p·v), execution prices,
   spreads; dividend-corrected series only where total-return matters — `r_on`
   (SPY goes ex-div quarterly; ~30 bp artificial overnight drops otherwise) and the
   Gao OLS replication. Phase-0 audit scans for ex-div signatures to classify the file;
   SPY dividend history fetched to build the correction. SPY never split in-window.

**§1.1 Auxiliary tickers for SSL (verified sources):**
- **HF Data Library** (hfdatalibrary.com; CC BY 4.0): 1,391 US stocks/ETFs, 1-min,
  Dec 2002–present, split/div-adjusted, clean-RTH and raw versions, curl-able REST API
  with bulk bundles (e.g. `quintile-5` = 278 most-liquid names). Caveats: one-time
  registration has a Cloudflare Turnstile challenge (you'd do the signup in a browser
  once, then everything is headless); data switches to IEX-only feed at 2022-03
  (irrelevant to us — our window ends 2021).
- **HuggingFace `mito0o852/OHLCV-1m`**: thousands of tickers 1992–2026, anonymous
  download, but **no license** and unverified quality — not fit for a paper's data
  section. Not recommended.
- FirstRate free samples (2 weeks/ticker): schema-validation/QC only.

**Repo state:** only standalone showcase notebooks at root. Phase 0 moves them to
`notebooks/legacy/` (reversible; nothing deleted).

## 2. Open questions — answer these with your approval

1. **Data source.** Default: Kaggle `gratefuldata` original (CC0, anonymous download,
   raw = auditable from source). Optionally *also* flip link-sharing on your two Drive
   CSVs so I can cross-check your cleaning against my Phase-0 audit (30-second action,
   revocable after download). Any objection to the default?
2. **SSL corpus scale** (the temporal firewall in §4-B2 applies regardless):
   a. *(Recommended)* You register once at hfdatalibrary.com (browser needed for the
      Turnstile challenge), set `HFDL_API_KEY` in this environment → SSL corpus =
      278 liquid tickers × SSL-era days ≈ millions of windows, CC BY 4.0, and the
      "pretrain on non-SPY tickers only" default survives.
   b. SPY-only SSL-era corpus (~45k volumes) — no action needed, weaker pretraining.
3. **Supervised evaluation window.** The SSL leakage fix (§4-B2) requires pretraining
   data to predate every CPCV test date. Proposal: **SSL era = 2008–2010 (737 days),
   supervised CPCV = 2011–2021 (2,579 labels; both measured in the audit)**. Bonus: evaluating 2011–2021 directly tests
   whether nonlinear signal exists *after* the published linear anomaly began decaying
   — a stronger claim than re-finding Gao's effect in 2008–09 volatility. The Gao OLS
   replication itself still runs on all years for the classic-result table.
   Alternative (needs GPU): per-path SSL purging, 9 pretraining runs, keeps 2008–2021
   supervised. Which one?
4. **Compute.** This environment: 4 CPU cores, 15 GB RAM, **no GPU**, ~30 GB free disk.
   Feasible with the tiered protocol in §6 (~2 weeks wall-clock, mostly unattended).
   A GPU (free Colab/Kaggle T4 works) collapses Phase 3+7 to ~1 day. Options:
   (a) all-CPU here; (b) I build and test everything here, you run `pretrain.py` +
   `run_matrix.py` on a GPU and drop checkpoints back; (c) attach a GPU environment.
5. **Registry:** MLflow needs a running server; default = append-only CSV/JSON registry
   keyed by config-hash + seed, committed to the repo. Object if you want MLflow.
6. **Skills note:** the superpowers/quant/probabl skills from your setup block aren't
   installed in this remote environment (that setup ran locally). Substitutes: TDD
   enforced by phase gates, built-in `code-review`/`security-review`, and the §4
   adversarial referee pass (already run once pre-code; re-run at Phase 8). I can clone
   the two skill repos into the container if you want their checklists verbatim.

**Default if you reply "proceed" without specifics:** 1-default + 2a-if-key-else-2b +
3-proposal + 4a + CSV registry.

## 3. Phase plan

Phases match your prompt; amendments are flagged **[Δ]** and argued in §4.
Green tests gate every phase; **STOP** gates halt for your review.
Gates: Phase 0 (audit), **Phase 1 (falsifiability diagnostic — added)**, Phase 2
(example volumes), **Phase 7 (registry freeze — added)**, Phase 8 (referee pass).

### Phase 0 — Scaffold & audit *(STOP gate)*
- Scaffold `src/`, `configs/`, `tests/`, `notebooks/`, `reports/`, `data/{raw,interim,tensors,pretrain}`;
  pinned `requirements.txt` (pandas, numpy, scipy, statsmodels, scikit-learn, pyts,
  PyWavelets, torch, torchvision, timm, sktime, xgboost, matplotlib, pyyaml);
  legacy notebooks → `notebooks/legacy/`.
- Ingest per Open Q1; SHA-256 recorded; immutable copy in `data/raw/`.
- `reports/data_audit.md`: timezone/DST resolution verified against event minutes;
  missing-minute census per year; half-day detection vs the NYSE early-close calendar
  (drop list); adjustment classification (ex-div scan vs SPY distribution dates);
  outlier scan (>20σ prints, zero/negative volume); event-day sanity panels
  (2010-05-06, 2015-08-24, 2018-02-05, 2020-03-09/12/16 circuit-breaker days);
  usable N per ticker; label balance by year.
- **[Δ]** Exclusion rules fixed ex ante: half days; days missing >10% of minutes in
  09:30–11:00; days with any missing minute in 15:25–16:00; circuit-breaker-halt
  mornings handled by a named rule (pad-flag, not silent drop) — all logged.
- **[Δ]** Execution defined precisely now: features hard-capped at timestamps
  ≤ 15:30:00 ET; **entry = 15:31 bar open** (eliminates the 15:30-close leak the
  referee flagged); exit = official closing print via MOC. Cost model: per-era
  half-spread measured from the data + MOC fee; sensitivity at 0.5/1/2/3 bp per side.
- Tests: session reconstruction (390 RTH minutes), TZ round-trip, half-day detector vs
  known early closes, audit functions on synthetic corrupted fixtures.

### Phase 1 — Two time axes *(STOP gate — falsifiability diagnostic [Δ])*
- Sessionize; `r_on = ln(open_t / div_adjusted_close_{t-1})`.
- Clock axis: 1-min log returns 09:30–11:00 (90 pts) → PAA to L ∈ {32, 45}
  (**[Δ]** fractional PAA for the non-integer 90/32 case, exactness-tested).
- Info axis per AFML ch. 2.3 behind one `BarScheme` interface: tick-rule signing on
  1-min closes (zero-change carry-forward); **dollar bars**; **dollar imbalance bars**
  (EWMA-updated E₀[T] and expected signed flow, warm-start on first train month,
  3×E₀[T] cap **[Δ] plus a 3-minute floor per bar**); volume/tick-bar variants.
- **[Δ]** K ∈ {16, 24} (not {32, 64}): the window has only 90 one-minute observations;
  K=64 forces ~1.4-obs bars and K=32 ~2.8-obs bars — information time would collapse
  onto clock time, manufacturing the redundancy the 2D control is supposed to test.
  Hard bound K ≤ 45. First K bars = the information clock; pad-flag days with fewer
  (**[Δ]** padding uses a mask channel, not zeros — arccos(0) is a real angle);
  clock-span of the K bars kept as tabular feature.
- **[Δ] Falsifiability diagnostic (this phase's gate):** per fold, report bars/day
  distribution, fraction of cap-triggered bars, pad-flag frequency, and clock-vs-info
  axis similarity (correlation + DTW distance). **If the info axis is statistically
  indistinguishable from the clock axis, the dual-time hypothesis is untestable on
  this data and we stop and redesign rather than proceed to a foregone null.** The
  same stats decide dollar vs dollar-imbalance as primary bar scheme (§4-B1).
- FFD (AFML ch. 5): **[Δ]** applied to each axis's **continuous intraday log-price
  stream** (not per-90-obs window — the |w|<1e-4 weight window spans hundreds of
  observations, longer than the window itself), then windows sliced from the
  differentiated stream; burn-in handled by backward extension with documented
  overnight-gap treatment. d* = smallest d ∈ [0,1] step 0.05 passing ADF 95%, train
  folds only, per axis. Ablation: FFD vs vol-scaled returns.
- Optional denoise per axis (config): SSA (pyts, top-r) or 1-D conv DAE, train-fold fit.
- Vol scaling by train-fit EWMA σ; realized 90-min vol as tabular feature.
  **[Δ]** Embargo (Phase 5) is tied to the longest EWMA half-life used here; EWMA
  state warm-starts from train-side data only at every fold boundary.
- Tests: bar-boundary correctness on hand-built fixtures, imbalance-threshold update
  math, EWMA warm-up isolation, K-bar determinism, FFD weights vs closed form,
  tick-rule carry-forward, fractional PAA exactness.

### Phase 2 — Dual-time 3D tensor *(STOP gate: example volumes, all three constructions)*
- Per-axis encoders (both axes length-normalized to L): GAF = GASF+GADF, optional MTF;
  DWT db4 L4–5 coefficient image (alt: CWT morlet).
- **[Δ] Scaling specification (was the #1 leakage trap):** rescaling to [−1,1] uses
  **train-fold-fit robust statistics, frozen, applied with clipping to [−1+ε, 1−ε]**
  before arccos; per-day min-max exists only as a named ablation (it destroys cross-day
  amplitude, which otherwise lives in the tabular vol features — trade-off reported).
  Same rule for MTF quantile edges and DWT coefficient standardization. Leakage gate:
  recompute encodings with test data ablated, assert bit-identical train tensors.
- **V-A** direct dual-time stack: L×L×2 volume, depth = time-representation axis,
  channels carry encodings; (k,k,2) kernels convolve both geometries in one op.
- **V-B (registered primary)** deep dual-time volume: 5 overlapping sub-windows per
  axis (clock 0–30/15–45/30–60/45–75/60–90 min; info the analogous bar-index spans),
  each L′×L′, depth ordering [clock₁..₅, info₁..₅] → L′×L′×10; plus the R(2+1)D
  factorization. **[Δ]** Mandatory depth-structure ablations (referee): shuffled depth
  order; two-stream variant with no cross-boundary kernels; depth-slice mutual
  correlation reported (overlapping sub-windows make adjacent slices near-duplicated —
  if 3D beats 2D we must show it isn't a parameter artifact, hence the 2D control is
  parameter- AND budget-matched, not just architecture-matched).
- **V-C (registered secondary hypothesis)** Cross-Time Gramian field:
  φᵢ = arccos(clock̃ᵢ), ψⱼ = arccos(iñfoⱼ), **C_ij = cos(φᵢ + ψⱼ)** — rows clock time,
  columns information time. **[Δ] L = K enforced** (default L′ = K = 32 grid... see
  §4-B6: C is L×K, so the stacked volume [GASF_clock, C_cross, GASF_info] requires
  equal axis lengths; the config grid pins the V-C cells to L = K). Depth traverses
  clock → joint → info. Novelty positioning per §7 (narrow claim, DS-GAF and CRP cited
  and differentiated; the angle-sum identity means C adds *inductive bias*, not new
  information — the registered V-C claim is therefore: 3D conv over the depth-3 volume
  beats a 2D CNN given the identical 3-channel stack AND beats the two within-axis
  GASFs without the cross plane).
- Mandatory controls built here: 2D multi-channel (C×H×W) versions of everything;
  **[Δ]** clock-only and info-only single-axis volumes (success criterion needs them).
- Tensor cache keyed by config hash, stored uint8 (GAF is bounded); slice grids →
  `reports/figures/`.
- Tests: GASF/GADF closed form on fixtures; C_cross range/symmetry properties and
  agreement with GASF when both axes are identical; depth-ordering invariants;
  mask-channel propagation; cache-hash stability; the scaling leakage gate above.

### Phase 3 — SSL pretraining
- **[Δ] Temporal firewall (blocker fix):** contemporaneous non-SPY tickers leak
  test-era regime structure — different ticker ≠ different information. The corpus is
  therefore restricted to the **SSL era (2008–2010), strictly before every CPCV test
  date**, for ALL tickers; plus ticker exclusion of SPY by default. Provenance
  manifest (ticker, day, window-start) logged; corpus ∩ test-period = ∅ is an
  executable test, not a policy statement.
- Corpus: rolling 90-min windows, 5-min stride (CPU fallback 15-min), built into the
  same dual-time volumes; per-ticker bar thresholds calibrated inside the SSL era.
- Primary: **3D MAE**, 75% tube masking, conv-MAE style (dense encoder, masked
  reconstruction loss — tube masking as specced is ViT-native; the conv adaptation is
  noted in the paper). Alternative: SimCLR-3D with time-domain augmentations only.
  Checkpoints + reconstruction figures saved.
- **[Δ]** DINOv2 frozen-slice probe and ImageNet-inflated I3D ResNet-18: kept, but as
  clearly-labeled **exploratory comparisons outside the primary registry** (32×32 GAF
  slices upsampled to 224 are far outside DINOv2's training distribution; I3D
  inflation assumes depth smoothness the clock→info boundary violates — referee is
  right that these are weak as anything more; upsampling method specified: bicubic).
- Pooled supervised option (config): all tickers' labeled volumes, evaluated held-out
  on SPY (Jiang–Kelly–Xiu-style pooling), same temporal firewall.
- No generative VLMs as predictors; one line in related work.
- Tests: corpus/test-period empty intersection (proof, not assertion), mask-ratio and
  tube-geometry correctness, checkpoint reload equivalence.

### Phase 4 — Models & regularization
- **Primary:** SSL-pretrained 3D CNN on V-B + tiny head, fine-tuned. 3–4 conv blocks,
  (3,3,3) kernels, stride-2 depth pooling only after the full clock↔info depth extent
  is traversed twice, 8→16→32→64 filters, 3D GAP → logit. Budget ≤ 300k params
  (sketch ≈ 60k; headroom documented). R(2+1)D same budget. From-scratch = the
  no-pretraining ablation. V-A and V-C through the same trunk (first-layer depth
  adjusted).
- Tabular fusion: [r_on, realized_vol_90m, info_bar_clockspan, pad_flag, day_of_week]
  → concatenated to pooled features before the logit.
- **[Δ] Label economics (referee):** sign label kept for Gao comparability, but
  training uses |return|-magnitude sample weights; dead-zone three-class variant as
  ablation; hit rate additionally reported conditional on |return| > round-trip cost.
  Economic criteria are primary (§5); raw hit rate is secondary.
- Regularization (default-on): AdamW wd 1e-2 (log-tuned), dropout 0.3–0.5, label
  smoothing 0.1, early stopping on purged inner-val split, cosine LR, grad clipping,
  5-seed averaging + CPCV-fold ensembling. **[Δ] Mixup moves to the time domain**
  (mix raw series pairs before encoding — a convex mix of GAF volumes is not the GAF
  of any series and contradicts the prompt's own "time-domain augmentation only"
  rule; volume-mixup demoted to a named ablation). Augmentations: jitter, magnitude
  warp, window warp; never image flips/rotations.
- **[Δ]** Probability calibration (Platt/isotonic, fit on inner-train validation)
  before any probability is consumed by meta-labeling or bet sizing.
- Sequence baselines: MiniRocket on [clock; info] fracdiff sequences, 1-D CNN, small
  LSTM. Tabular baselines: Gao OLS replication (r_last30 ~ r_first30 + r_on),
  **[Δ] Gao sign rule and vol-scaled Gao** (referee: the cheap conditional baselines
  reviewers ask for), logistic + XGBoost on handcrafted features, always-long
  (reported as the close-drift benchmark it is), coin flip (de-emphasized).
- **[Δ] One trading-rule mapping for every model** (sign of forecast, unit notional,
  identical costs/calendar) so the DM test compares signal quality, not
  implementation choices.
- Tests: parameter-budget assertion, augment-before-encode invariant, deterministic
  seeding, tabular-fusion shapes, calibration monotonicity, head-only vs full
  fine-tune flags.

### Phase 5 — Validation (CPCV)
- Purged K-Fold + CPCV per AFML ch. 7 & 12: N=10 contiguous groups, k=2 → 45 splits →
  9 paths; purge 1 day; **[Δ]** embargo = max(3 days, longest EWMA half-life used in
  preprocessing) — the number is derived, not asserted; EWMA state never crosses fold
  boundaries.
- Everything fits inside training folds: d* per axis, bar thresholds/EWMA spans,
  scalers, SSA/DAE, encoder stats, corpus selection, nested hyperparameters.
  Per-split preprocessing refit pipeline (thresholds → bars → FFD d* → scaling) is
  budgeted in §6, cached per (config, split).
- **[Δ] Leakage gates as executable assertions with expected outcomes** (referee: a
  canary must be a deliberately injected leak, else it tests nothing):
  (a) shuffled labels → accuracy CI covers 50%;
  (b) max feature timestamp < 15:30:00 ET per sample → hard fail otherwise;
  (c) injected leak canary (label appended to a feature) → accuracy > 90%, proving
      the harness *can* detect leaks; train-on-future canary → purge code must throw;
  (d) SSL corpus provenance ∩ test periods = ∅;
  (e) bar-threshold calibration provably train-days-only (hash of days used).
- Tests: purge/embargo boundary proofs on synthetic calendars, path reconstruction
  (every group in exactly 9 test paths), split determinism.

### Phase 6 — Meta-labeling
- Calibrated primary probability → meta-model (logistic/XGBoost) on [primary_prob,
  r_on, realized_vol, info_bar_clockspan, day_of_week] for trade/no-trade + de Prado
  probability bet sizing. Report primary-only vs meta-labeled. **[Δ]** Sized and
  unsized strategies are two distinct registry trials (both registered ex ante).
- Tests: meta-features strictly ≤ 15:30 ET; bet-size monotonicity in probability.

### Phase 7 — Experiment matrix *(STOP gate: registry freeze [Δ])*
- **[Δ] Registry freeze before any test-fold evaluation** (blocker fix — forking-path
  control): a written, committed registry defining (i) THE primary configuration
  (V-B + best-on-inner-val settings — chosen on train/inner-val data only);
  (ii) the enumerated secondary hypotheses (V-C claim, 3D-vs-2D, dual-vs-single-axis,
  vs-MiniRocket), each with its own registered test; (iii) the fixed ablation list;
  (iv) the unit of a trial = **the 5-seed fold-ensembled strategy** (per-seed
  dispersion reported as robustness, not as extra trials); (v) everything else marked
  exploratory. You sign off on the frozen registry at this gate.
- Grid pruned to ~40 registered runs: construction {V-A, V-B, V-C} × encoder
  {gaf, dwt, both} × input {ffd, vol-scaled} × denoise {none, ssa, dae} × arch
  {3d, (2+1)d, 2d-control} × pretraining {none, mae3d} × bars {dollar,
  dollar-imbalance} — pruned, not the full cross product; exploratory items
  (dino-probe, inflated-imagenet, volume-mixup, per-day scaling) run outside the
  registry and are labeled as such in the paper.
- **[Δ] Two-tier execution (compute; referee-endorsed):** tier 1 screens on
  **inner-train validation only — screening never touches test folds**, so it does
  not contaminate the trial count; tier 2 = full 45-split CPCV × 5 seeds for the
  frozen registry (primary + secondaries + all baselines and controls).
- Every registry trial logged for DSR; **[Δ]** DSR uses López de Prado's
  effective-trials estimate (clustered trial correlations — registry trials are
  highly correlated and treating them as independent is wrong in the strict
  direction, but we compute it, not assume it) and the skew/kurtosis-expanded PSR.

### Phase 8 — Reporting *(STOP gate: adversarial referee pass; every issue answered in `reports/critique_response.md`)*
- `reports/results.md` + LaTeX: baseline table; the central V-A/V-B/V-C vs 2D-control
  vs single-axis table; depth-structure ablations; CPCV path-Sharpe distributions
  (annualization convention stated: daily net strategy returns × √252, flat outside
  15:31–16:00); **[Δ]** PBO via CSCV specified concretely — S=16 partitions on the
  matrix of daily net returns of all registry trials (persisted per-trial artifacts
  are a pipeline requirement from day one), reported alongside, not conflated with,
  CPCV path stats; cost sensitivity 0.5/1/2/**3** bp with era-based cost as the
  headline; **[Δ]** pre-specified subperiods 2011–2013 / 2014–2017 / 2018–2021 with
  the explicit "signal after linear decay" test (does the CNN find signal where OLS
  no longer does — the interesting claim, tested directly); SSL-benefit learning
  curves; example volumes; architecture diagram; limitations.
- DM (HAC-adjusted, on daily net strategy returns under the single trading-rule
  mapping) vs Gao OLS; McNemar vs XGBoost.

## 4. Design amendments & disagreements

### 4-B. Referee blockers — resolutions adopted into §3

| # | Blocker | Resolution (where) |
|---|---|---|
| B1 | Dollar-imbalance bars near-degenerate on 1-min inputs; K∈{32,64} arithmetic collapse | K → {16,24}, 3-min bar floor, degeneracy stats + **falsifiability gate** at Phase 1 STOP; primary bar scheme decided there by measured stats, not by fiat (Phase 1) |
| B2 | SSL corpus leaks test-era structure through correlated non-SPY tickers | **Temporal firewall**: SSL era 2008–2010 strictly precedes all test dates, any ticker; supervised CPCV 2011–2021 (Open Q3) (Phase 3) |
| B3 | "V-B primary AND V-C headline" is a forking path; trial count undefined | Registry freeze gate; V-B = sole registered primary, V-C = registered secondary with its own test; trial = seed-ensemble; effective-trials DSR (Phase 7) |
| B4 | Entry mechanics inconsistent; flat 1bp optimistic for 2008–11; half-days | Entry = 15:31 bar open, features ≤ 15:30:00; exit MOC; era-based cost + 3bp stress; half-day/halt rules ex ante (Phase 0) |
| B5 | GAF/MTF rescaling fit unspecified — the classic leakage bug | Train-fold robust scaling, frozen, clipped before arccos; per-day scaling = ablation; bit-identical ablation gate (Phase 2) |
| B6 | V-C dimensionally inconsistent (L×K vs L×L stack); angle-sum identity ⇒ no new information | L=K enforced; claim reframed as **inductive bias** with the two registered ablations that make it survive the objection (Phase 2, §7) |

### 4-M. Referee majors — all adopted
V-B depth-redundancy ablations (shuffled depth, two-stream, param-matched 2D control);
FFD on continuous log-price streams with burn-in handling; |return|-weighted training
+ cost-conditional hit rate; embargo derived from EWMA half-lives; pre-specified
subperiods + post-decay test; single trading-rule mapping for DM; tiered success
criteria (§5); time-domain mixup + calibration before sizing; concrete CSCV spec;
staged compute protocol with screening isolated from test folds; unadjusted-price
policy for dollar bars/execution; fractional PAA + mask-channel padding; sign-rule and
vol-scaled Gao baselines; DINOv2/I3D demoted to exploratory; leakage gates as
executable assertions with expected outcomes.

### 4-D. Remaining judgment calls (your sign-off; silence = adopt as proposed)

1. **Dollar-imbalance bars.** Referee says drop them outright without tick data; your
   prompt makes them primary. **Proposal: implement both exactly as specced; the
   Phase-1 falsifiability diagnostic decides primary-vs-variant on measured
   degeneracy** (share of cap/floor-triggered bars, bar-count dispersion, axis
   similarity). If imbalance bars are stable, they stay primary as you intended; the
   paper's one-line note covers the 1-min tick-rule construction either way.
2. **Supervised window 2011–2021** (Open Q3). Sacrifices 2008–2010 labels (~650) to
   the SSL firewall but converts the paper's evaluation into a cleaner post-decay
   test. Alternative preserves all labels at the cost of 9 GPU pretraining runs.
3. **Success-criteria structure** (§5): your prompt's all-seven-conjunction stays as
   the definition of a "full positive result", but the referee is right that its
   joint power at plausible effect sizes is low; §5 adds a registered primary claim
   so the paper has a defensible headline even when the conjunction fails partially.
4. **K = {16, 24} and L′ = K = 32 for V-C cells** replacing K ∈ {32, 64} — forced by
   the 90-observation arithmetic; veto only if you want K=32 kept as an upper cell.
5. **2016+ regime split → subperiods 2011–13 / 2014–17 / 2018–21** (follows from #2).

## 5. Success criteria (tiered [Δ])

**Registered primary claim (determines "positive result" for the paper's headline):**
the registered primary model's net-of-era-cost strategy achieves **DSR ≥ 0.95**
(effective-trials-corrected, skew/kurt-adjusted PSR vs 0) across CPCV paths.

**Registered secondary hypotheses (each with its own test):**
S1 dual-time 3D (V-B) beats the parameter-matched 2D control; S2 V-B beats clock-only
and info-only single-axis volumes; S3 V-C beats 2D-on-identical-3-channels and beats
within-axis-GASFs-without-cross-plane; S4 beats Gao OLS net (HAC-DM p < 0.05) and
XGBoost (McNemar); S5 beats MiniRocket/1-D CNN on the same inputs.

**Full positive result (the paper's strongest form) = your original conjunction:**
hit rate ≥ 52.5% (95% CI excluding 50%) — also reported conditional on
|return| > cost; net Sharpe > 0 at 1 bp on ≥ 7/9 paths with mean ≥ 0.8 (annualization
convention per Phase 8); DSR ≥ 0.95; PBO < 40%; S1–S5 all pass; all leakage gates
pass. Failure of any tier is reported exactly as such — the honest-null/methodology
paper (dual-time construction + SSL + CPCV framework) is the committed fallback.
Never torture the data; never report a metric absent from the registry.

## 6. Compute budget (4 CPU cores, 15 GB RAM, no GPU, ~30 GB free disk)

| Item | Size | CPU (this box) | T4 GPU |
|---|---|---|---|
| Tensor build, 3 constructions + controls (labeled) | ~2,750 days | ~1–2 h | — |
| SSL corpus build (era 2008–2010) | SPY-only ~45k vols / +aux ≈ 10M (subsample to ~500k) | 4–12 h | — |
| 3D MAE, 100–150 epochs | ~1–8 PFLOPs (corpus-dependent) | 8 h – 3 days | 1–3 h |
| DINOv2 slice embeddings (exploratory, cached once) | ~28k images | ~2 h | 10 min |
| One supervised fit (≤300k params, ~2,200 train days) | ~25 TFLOPs | ~10 min | <1 min |
| Tier-1 screen (inner-val only): ~40 × 3 seeds × 5 inner splits | 600 fits | 3–5 days unattended | ~4 h |
| Tier-2 CPCV: registry (~10 configs) × 5 seeds × 45 splits | ~2,250 NN fits + cheap baselines | ~1 week unattended | ~1 day |
| MiniRocket / XGB / OLS, full CPCV | — | minutes–hours | — |

Disk: uint8 tensor caches, ≤2 encoder configs cached simultaneously, GAF re-encoded
on the fly on cache eviction. Per-split preprocessing refits are cached per
(config, split) — bar thresholds and d* are cheap to refit; only NN training
dominates. CPU-only total ≈ 2 weeks wall-clock, mostly unattended (Open Q4).

## 7. Novelty scan — Cross-Time Gramian Angular Field (CT-GAF)

**Verdict from the literature workflow (~20 query families, arXiv/SSRN/Scholar
surfaces, EN+ZH): NOVEL, claim it narrowly.** No published work builds a Gramian-type
cross field between two time-warped representations of the same instrument; none
applies GAF to information-driven bars at all; none convolves in 3D across two time
representations. But three families foreclose the broad claim:

| Closest work | What it is | Differentiation |
|---|---|---|
| Li et al. 2024, "Dual-source GAF", ESWA 237:121521 | Mixes arccos-angles of TWO sensors (vibration/strain) in Gramian matrices via spherical coordinates, RGB→CNN | Two *different physical sensors on one shared clock*; not two time indexings of one series; no rectangular clock×info field; no 3D conv. **Must cite** — forecloses "first to mix two signals' angles in a Gramian field" |
| Shabani et al. 2023, NCAA (arXiv:2210.14605) | Cross recurrence plots of stock *pairs* → deep nets, in finance | The established rectangular two-series cross field; thresholded phase-space distance, two assets, 2D. Position CT-GAF as the angular/Gramian analogue of a CRP applied to *a series and its time-deformed self* |
| Wang & Oates 2015, IJCAI | GASF/GADF/MTF origin; multichannel stacking | The within-axis building block; C_ij = cos(φᵢ+ψⱼ) = xᵢzⱼ − √(1−xᵢ²)√(1−zⱼ²) is the standard GASF identity on a cross pair — a referee **will** call the formula trivial |
| Barra et al. 2020, IEEE/CAA JAS | GAF+CNN on S&P futures, multi-resolution ensemble | Multiple clock-time *aggregations* of one series, each its own square GAF into an ensemble — no cross field, no event time |
| GAF-FusionNet 2025 (ICONIP'24) | Multi-view GAF fusion via split attention | Confirms multi-view GAF fusion is standard — always parallel square images on one time axis |
| Zeng et al. 2021, ICAIF (arXiv:2102.12061) | Financial series → video → spatiotemporal conv | Closest 3D/volumetric use; third dim is rolling clock frames, not a second time *representation* |

**Consequences adopted into the plan:** (i) the contribution is the clock-time ×
information-time pairing with its microstructure motivation (volume clock /
subordination: Easley–López de Prado–O'Hara 2012; Clark 1973; Ané & Geman 2000) plus
the 3D convolution across representations — not the cosine-sum formula; (ii) by the
angle-sum identity C adds inductive bias, not information, so V-C's registered claim
is exactly the two ablations in Phase 2 (beat 2D-on-identical-channels; beat
within-axis-GASFs-without-cross-plane); (iii) DS-GAF and the CRP literature are cited
and differentiated explicitly. "Cross-Time Gramian Angular Field" is featherweight as
a *named* contribution under these conditions — verdict supports featuring it.

**Phase-3 evidence notes (SSL):** the workflow's SSL-literature agent failed on an
API-side flag and its ground is covered from general knowledge, flagged for
verification in the Phase-3 literature pass: VideoMAE-style tube masking is
ViT-native — the conv adaptation (dense encoder, masked loss; SparK/ConvNeXt-V2 show
masked pretraining works for conv nets) is what Phase 3 implements; DINOv2 frozen
features degrade far off natural images (GAF slices qualify) — consistent with its
demotion to exploratory; I3D inflation is standard (Carreira & Zisserman 2017) but
assumes depth smoothness our clock→info boundary violates — also exploratory;
MiniRocket remains the small-N time-series-classification baseline to beat honestly.

## 8. References (core)

- Gao, Han, Li & Zhou (2018), "Market intraday momentum", JFE 129(2), 394–414. Sample: SPY 1993–2013.
- Lou, Polk & Skouras (2019), "A tug of war: Overnight versus intraday expected returns", JFE 134(1).
- López de Prado (2018), *Advances in Financial Machine Learning*, Wiley — ch. 2, 5, 7, 12.
- Easley, López de Prado & O'Hara (2012), "The volume clock", JPM 39(1); Clark (1973),
  Econometrica 41(1); Ané & Geman (2000), JF 55(5) — subordination/event-time theory.
- Wang & Oates (2015), "Imaging time-series...", IJCAI, 3939–3945.
- Jiang, Kelly & Xiu (2023), "(Re-)Imag(in)ing price trends", **JF** 78(6), 3193–3249.
- Barra et al. (2020), IEEE/CAA JAS 7(3); Chen & Tsai (2020), Financial Innovation 6:26 — GAF+CNN finance.
- Li et al. (2024), "Dual-source GAF...", Expert Systems with Applications 237:121521.
- Shabani et al. (2023), "...Cross Recurrence Plots", Neural Comput. & Applic. (arXiv:2210.14605).
- Tran et al. (2018), "A closer look at spatiotemporal convolutions" (R(2+1)D), CVPR.
- He et al. (2022), "Masked autoencoders...", CVPR; Tong et al. (2022), "VideoMAE", NeurIPS.
- Bailey & López de Prado (2014), "The deflated Sharpe ratio", JPM; Bailey et al.
  (2017), "The probability of backtest overfitting", J. Computational Finance.
- Dempster, Schmidt & Webb (2021), "MiniRocket", KDD.
- Zeng et al. (2021), "Deep video prediction for time series forecasting", ICAIF.

## 9. Approval checklist

- [ ] Open Questions 1–4 (§2): data source / SSL corpus scale / supervised window /
      compute. (5–6 optional; defaults stated.)
- [ ] Judgment calls §4-D 1–5 (silence = adopt as proposed).
- [ ] Approve Phase 0 start.

Nothing beyond this document has been built. On approval I begin Phase 0
(scaffold + audit) and stop at its gate with `reports/data_audit.md`.
