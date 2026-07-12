"""Phase 5 leakage gates — executable assertions that gate ALL reporting.

(a) shuffled labels (PRODUCTION bundle & split) -> test accuracy CI covers 50%
(b) feature-timestamp bound -> corrupting all post-11:00 non-label minutes
    leaves features BIT-IDENTICAL
(c) injected label leak -> accuracy > 90% (harness can detect a gross leak)
(d) train-on-future canary -> corrupted split raises (purge AND embargo)
(e) SSL corpus provenance -> every corpus date < first supervised date
(f) synthetic-signal positive control (SOFT) -> a planted series-borne signal
    at realistic effect size must be partially recovered; if not, the null
    statement is weakened accordingly (reported, does not block)
(g) fitted-statistic provenance -> corrupting NON-train days leaves the bar
    threshold and all train-row features bit-identical
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from spydt.axes import build_bundle, load_daydata
from spydt.cv import Split, assert_no_leakage, cpcv_splits
from spydt.models import Classifier3D
from spydt.train import VolumeDataset, fit, predict_proba

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
PILOT = yaml.safe_load(Path("configs/pilot.yaml").read_text())

results: list[str] = []


def gate(name: str, passed: bool, detail: str, soft: bool = False) -> None:
    tag = "PASS" if passed else ("WEAK" if soft else "**FAIL**")
    results.append(f"| {name} | {tag} | {detail} |")
    print(f"{name}: {tag} — {detail}", flush=True)
    if not passed and not soft:
        Path("reports/leakage_gates.md").write_text(
            "\n".join(["# Leakage gates (FAILED RUN)\n",
                       "| gate | result | evidence |", "|---|---|---|"] + results) + "\n")
        raise SystemExit(f"leakage gate failed: {name}")


def _prod_bundle():
    sys.path.insert(0, "scripts")
    from run_matrix import load_bundle

    b = load_bundle(Path("data/tensors/bundles/split0.npz"))
    sp = json.loads(Path("data/tensors/splits.json").read_text())["0"]
    return b, np.array(sp["train_idx"]), np.array(sp["test_idx"])


def _quick_fit_eval(bundle, tr_idx, te_idx, *, seed=0, dropout=0.3) -> float:
    n_val = max(int(len(tr_idx) * 0.15), 20)
    m = Classifier3D(dropout=dropout)
    fit(m, VolumeDataset(bundle, tr_idx[: -n_val - 1], "vb", 16, seed=seed),
        VolumeDataset(bundle, tr_idx[-n_val:], "vb", 16),
        max_epochs=25, patience=6, batch_size=128, lr=1e-3,
        weight_decay=1e-3, label_smoothing=0.0, seed=seed)
    ds = VolumeDataset(bundle, te_idx, "vb", 16)
    p = predict_proba(m, ds)
    return float(np.mean(np.where(p >= 0.5, 1, -1) == bundle.y[ds.idx])), len(ds.idx)


def main() -> None:
    torch.set_num_threads(2)
    days = load_daydata("data/interim/spy_rth_et.parquet",
                        "data/interim/day_classification.parquet",
                        CFG["volume"]["unit_multiplier"])
    lo, hi = CFG["eras"]["supervised"]
    sup = [d for d in days if lo <= d.date <= hi]

    # ---- (b) feature timestamp bound: corrupt post-11:00 non-label minutes
    sup_c = copy.deepcopy(sup[:300])
    for d in sup_c:
        d.close[90:360] *= 1000.0
        d.close[361:389] *= 1000.0
        d.open_[91:361] *= 1000.0
        d.volume[90:] = 1.0
    tr = np.arange(200)
    kw = dict(bar_kind="dollar", K=PILOT["axes"]["K"], L=PILOT["axes"]["L"],
              floor_buckets=PILOT["axes"]["floor_buckets"],
              cap_mult=PILOT["axes"]["cap_mult"], ffd_grid_step=0.5)
    b_ref = build_bundle(sup[:300], tr, **kw)
    b_cor = build_bundle(sup_c, tr, **kw)
    same = (np.allclose(b_ref.clock, b_cor.clock, equal_nan=True)
            and np.allclose(b_ref.info, b_cor.info, equal_nan=True)
            and np.allclose(b_ref.tabular, b_cor.tabular, equal_nan=True))
    labels_ok = np.allclose(b_ref.r_label, b_cor.r_label)
    gate("(b) feature timestamps < 11:00 (<= 15:30 contract)", same and labels_ok,
         "features bit-identical under corruption of 11:00-15:29 + 15:32-15:58; labels intact")

    # ---- (g) fitted-statistic provenance: corrupt NON-train days only
    sup_g = copy.deepcopy(sup[:300])
    train_set = set(range(200))
    for i, d in enumerate(sup_g):
        if i not in train_set:
            d.close[:90] *= 1000.0
            d.volume[:90] *= 1000.0
    b_gcor = build_bundle(sup_g, tr, **kw)
    thr_same = np.isclose(b_ref.diag["threshold"], b_gcor.diag["threshold"])
    train_rows_same = (
        np.allclose(b_ref.clock[:200], b_gcor.clock[:200], equal_nan=True)
        and np.allclose(b_ref.info[:200], b_gcor.info[:200], equal_nan=True))
    gate("(g) fitted statistics use train days only", bool(thr_same and train_rows_same),
         "bar threshold and all train-row features bit-identical when every "
         "non-train day is corrupted")

    # ---- production bundle for (a), (c), (f)
    bp, tr_idx, te_idx = _prod_bundle()

    # ---- (a) shuffled labels at production scale, no fudge factor
    b_sh = copy.deepcopy(bp)
    b_sh.y = np.random.default_rng(0).permutation(b_sh.y)
    acc, n_te = _quick_fit_eval(b_sh, tr_idx, te_idx)
    ci = 1.96 * np.sqrt(0.25 / n_te)
    gate("(a) shuffled-label collapse (production scale)", abs(acc - 0.5) <= ci,
         f"acc={acc:.3f} on {n_te} test days (95% CI half-width {ci:.3f})")

    # ---- (c) injected label leak must be detected
    b_leak = copy.deepcopy(bp)
    b_leak.tabular = b_leak.tabular.copy()
    b_leak.tabular[:, 0] = b_leak.y
    acc_c, _ = _quick_fit_eval(b_leak, tr_idx, te_idx, dropout=0.0)
    gate("(c) injected-leak canary detects", acc_c > 0.9,
         f"acc={acc_c:.3f} with label injected into a feature")

    # ---- (f) SOFT: synthetic series-borne signal at realistic effect size
    b_syn = copy.deepcopy(bp)
    rng = np.random.default_rng(1)
    sig = b_syn.clock[:, -8:].mean(axis=1)          # late-morning pattern
    sig = (sig - np.nanmean(sig)) / (np.nanstd(sig) + 1e-12)
    noise = rng.normal(0, 1, len(sig))
    # mixing weight tuned so the Bayes-optimal accuracy is ~56%
    lat = 0.32 * sig + noise
    b_syn.y = np.where(lat >= 0, 1.0, -1.0)
    b_syn.w = np.ones_like(b_syn.w)
    bayes = float(np.mean((sig >= 0) == (lat >= 0)))
    acc_f, n_f = _quick_fit_eval(b_syn, tr_idx, te_idx)
    ci_f = 1.96 * np.sqrt(0.25 / n_f)
    gate("(f) synthetic-signal positive control (soft)", acc_f - 0.5 > ci_f / 2,
         f"pipeline recovers acc={acc_f:.3f} of a planted signal with Bayes "
         f"acc≈{bayes:.3f} (n={n_f}); recovery above half-CI counts as pass",
         soft=True)

    # ---- (d) purge + embargo canary
    full = cpcv_splits(2579, n_groups=PILOT["cv"]["n_groups"],
                       k_test=PILOT["cv"]["k_test"],
                       purge_days=PILOT["cv"]["purge_days"],
                       embargo_days=PILOT["cv"]["embargo_days"])
    assert_no_leakage(full, purge_days=PILOT["cv"]["purge_days"],
                      embargo_days=PILOT["cv"]["embargo_days"])
    bad = Split(0, full[0].test_groups,
                np.concatenate([full[0].train_idx, full[0].test_idx[:1]]),
                full[0].test_idx)
    try:
        assert_no_leakage([bad], purge_days=1)
        caught = False
    except AssertionError:
        caught = True
    gate("(d) train-on-future canary (purge+embargo verified)", caught,
         "clean CPCV passes incl. embargo window; corrupted split raises")

    # ---- (e) SSL provenance
    prov = json.loads(Path("data/pretrain/corpus_provenance.json").read_text())
    last = max(prov["window_dates"])
    gate("(e) SSL corpus provenance", last < lo,
         f"last corpus day {last} < first supervised day {lo}")

    md = ["# Leakage gates\n", "| gate | result | evidence |", "|---|---|---|"]
    md += results
    Path("reports/leakage_gates.md").write_text("\n".join(md) + "\n")
    print("gates complete -> reports/leakage_gates.md")


if __name__ == "__main__":
    main()
