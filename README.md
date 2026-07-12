# SPY Dual-Time 3D CNN

Publication-grade research codebase (target: Journal of Financial Data Science).
Central object: a 3D tensor encoding the same trading morning in **clock time**
and **information time** (López de Prado information-driven bars), convolved
jointly by a 3D CNN. See `PLAN.md` for the full phase plan, design amendments,
and the pre-registered evaluation protocol.

## Layout

```
src/spydt/       library code (data, bars, encodings, models, cv)
configs/         YAML configs — every parameter lives here, no magic numbers
tests/           pytest suite; green tests gate every phase
scripts/         phase runners (run_audit.py, ...)
reports/         data_audit.md, figures, results
data/            raw/interim/tensors/pretrain — never committed (.gitignore)
notebooks/legacy pre-project showcase notebooks, untouched
```

## Reproduction

```bash
pip install -r requirements.txt   # torch CPU wheel: --extra-index-url https://download.pytorch.org/whl/cpu

# Phase 0 — data ingest + audit
curl -sL "https://www.kaggle.com/api/v1/datasets/download/gratefuldata/intraday-stock-data-1-min-sp-500-200821" \
  -o data/raw/spy_1min_gratefuldata.zip          # CC0, anonymous
unzip -o data/raw/spy_1min_gratefuldata.zip -d data/raw/
sha256sum -c <(echo "8bd52867b248359db437f2fd809ac1faafc3f69449679aae9d499423487ccd5e  data/raw/spy_1min_gratefuldata.zip")

python -m pytest                   # 28 tests
python scripts/run_audit.py        # -> reports/data_audit.md + event figures
```

## Data contract (enforced in code)

- RTH 09:30–16:00 ET, tz verified against independent evidence (audit hard-fails
  on contradiction). Decision 15:30:00 ET; every feature timestamp ≤ 15:30:00;
  entry at the 15:31 bar open; MOC exit.
- Unadjusted prices for dollar bars and execution; dividend-corrected overnight
  returns only. Volume is IB lots (×100 shares).
- SSL pretraining era 2008–2010 (temporal firewall); supervised CPCV 2011–2021.
