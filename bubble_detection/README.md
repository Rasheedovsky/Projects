# Bubble Explosiveness Detection — statistical tests + Causal Random Forest

A research system that turns eight econometric tests for explosiveness /
structural change into **causal (walk-forward safe) features**, then uses a
**Causal Random Forest** (econml `CausalForestDML`) as the meta-model to
predict the *direction* of a detected bubble episode.

```
data (hourly OHLCV)
  └─ statistical-test features, all trailing-window / filtered  ── docs/TESTS.md
        GSADF/BSADF · right-tailed ADF · CUSUM · Chow · QLR ·
        Bai-Perron · ruptures(PELT) · HMM(filtered)
  └─ labels: forward h-bar return, PSY episode termination
  └─ purged walk-forward folds (expanding train, purge = horizon)
        ├─ Causal Forest:  Y = fwd return, T = 1{BSADF > cv95}, X = context
        │    τ̂(x) > 0 → bubble growth leg (long) · τ̂(x) < 0 → collapse (short)
        └─ benchmark RF direction classifier (all features)
  └─ results/: metrics.json · summary.md · plots · features_predictions.csv
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py                      # uses data/AA_h.csv
python run_pipeline.py --csv path/to.csv --horizon 21 --mc-sims 200
```

First run simulates BSADF Monte-Carlo critical values (~2 min) and caches
them in `results/cache_bsadf_cv.json`; subsequent runs take ~3 min total.

## Layout

| path | contents |
|---|---|
| `src/bubbles/prefix_ols.py` | O(1)-per-window OLS engine (prefix-sum sufficient statistics) behind all tests |
| `src/bubbles/psy.py` | right-tailed ADF, BSADF/GSADF, MC critical values, episode date-stamping |
| `src/bubbles/stability.py` | CUSUM & CUSUM-of-squares, Chow, QLR (sup-F) |
| `src/bubbles/bai_perron.py` | multiple-break DP with BIC selection |
| `src/bubbles/ruptures_feats.py` | PELT change-point features |
| `src/bubbles/hmm_feats.py` | walk-forward *filtered* Gaussian HMM |
| `src/bubbles/features.py` / `labels.py` | feature/label assembly |
| `src/bubbles/walkforward.py` | purged expanding walk-forward folds |
| `src/bubbles/causal_model.py` | CausalForestDML meta-model + RF benchmark |
| `run_pipeline.py` | end-to-end orchestration, metrics, plots |
| `docs/TESTS.md` | what each test is, its literature, and how it becomes a feature |
| `results/` | output of the committed reference run on AA hourly data |

## Reference run (AA, hourly, 2025-07 → 2026-07)

- GSADF = 2.19 < cv95 = 3.03 → no *full-sample* explosiveness, but the
  real-time BSADF date-stamps **5 episodes** (the Dec-25/Jan-26 run-up from
  ~$45 to $66, and the June-26 breakdown), ~7% of bars flagged.
- Causal forest on treated test bars (n=20 — small!): sign(τ̂) direction hit
  rate 0.65; sign(τ̂) strategy mean fwd return +0.031 (NW t = 2.5) vs
  always-long −0.038. Across all test bars: corr(τ̂, realized fwd ret) = 0.27.
- Benchmark RF direction classifier: AUC 0.34 on 750 test bars — *below*
  chance, the expected overfit signature for a plain predictive model on a
  single asset-year. The causal framing is doing the useful work here.

Full numbers: `results/summary.md`, `results/metrics.json`.

## Verification

The prefix-sum engine (ADF t-stats, AR SSRs), CUSUM, and Chow statistics are
tested to 1e-8 agreement against explicit statsmodels regressions, and the
BSADF correctly explodes (>13) on synthetic explosive-root series.

## Known limitations / upgrade path

1. **One asset-year** — treated bars are scarce (~95 per fold). The natural
   upgrade is a cross-sectional panel (many tickers), which multiplies
   treated observations and lets the forest learn cross-asset heterogeneity.
2. Overlapping labels (h=21) — Newey-West t-stats partially correct;
   non-overlapping evaluation or triple-barrier labels are the next step.
3. Treatment is binary BSADF exceedance; dose-response (continuous
   `bsadf_gap` treatment) is supported by econml and worth trying.
4. Nuisance CV inside DML is blocked-but-not-purged; a purged splitter
   would tighten it further.
