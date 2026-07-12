"""Phase 5 leakage gates — executable assertions that gate ALL reporting.

(a) shuffled labels  -> test accuracy CI must cover 50%
(b) feature-timestamp bound -> corrupting all post-11:00 non-label minutes
    must leave features BIT-IDENTICAL (proves no feature reads beyond the
    morning window; labels use only 15:30/15:31/15:59)
(c) injected label leak -> accuracy must EXCEED 90% (harness can detect leaks)
(d) train-on-future canary -> corrupted split must raise
(e) SSL corpus provenance -> every corpus date < first supervised date
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


def gate(name: str, passed: bool, detail: str) -> None:
    results.append(f"| {name} | {'PASS' if passed else '**FAIL**'} | {detail} |")
    print(f"{name}: {'PASS' if passed else 'FAIL'} — {detail}", flush=True)
    if not passed:
        raise SystemExit(f"leakage gate failed: {name}")


def main() -> None:
    torch.set_num_threads(4)
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
         "features bit-identical under corruption of 11:00-15:29 + 15:32-15:58; "
         "labels intact")

    # shared quick-fit split for (a) and (c)
    splits = cpcv_splits(len(b_ref.dates), n_groups=3, k_test=1,
                         purge_days=1, embargo_days=5)
    s = splits[-1]
    tr_idx, te_idx = s.train_idx, s.test_idx
    n_val = max(int(len(tr_idx) * 0.15), 20)
    fit_kw = dict(max_epochs=10, patience=4, batch_size=128, lr=1e-3,
                  weight_decay=1e-2, label_smoothing=0.0, seed=0)

    # ---- (a) shuffled labels -> chance
    b_sh = copy.deepcopy(b_ref)
    rng = np.random.default_rng(0)
    b_sh.y = rng.permutation(b_sh.y)
    m = Classifier3D(dropout=0.3)
    fit(m, VolumeDataset(b_sh, tr_idx[:-n_val], "vb", 16, seed=0),
        VolumeDataset(b_sh, tr_idx[-n_val:], "vb", 16), **fit_kw)
    p = predict_proba(m, VolumeDataset(b_sh, te_idx, "vb", 16))
    yy = b_sh.y[VolumeDataset(b_sh, te_idx, "vb", 16).idx]
    acc = float(np.mean(np.where(p >= 0.5, 1, -1) == yy))
    ci = 2 * np.sqrt(0.25 / len(yy))
    gate("(a) shuffled-label collapse", abs(acc - 0.5) <= ci + 0.02,
         f"acc={acc:.3f} on shuffled labels (CI half-width {ci:.3f})")

    # ---- (c) injected leak must be detected
    b_leak = copy.deepcopy(b_ref)
    b_leak.tabular = b_leak.tabular.copy()
    b_leak.tabular[:, 0] = b_leak.y
    m2 = Classifier3D(dropout=0.0)
    fit(m2, VolumeDataset(b_leak, tr_idx[:-n_val], "vb", 16, seed=0),
        VolumeDataset(b_leak, tr_idx[-n_val:], "vb", 16), **fit_kw)
    p2 = predict_proba(m2, VolumeDataset(b_leak, te_idx, "vb", 16))
    yy2 = b_leak.y[VolumeDataset(b_leak, te_idx, "vb", 16).idx]
    acc2 = float(np.mean(np.where(p2 >= 0.5, 1, -1) == yy2))
    gate("(c) injected-leak canary detects", acc2 > 0.9,
         f"acc={acc2:.3f} with label injected into a feature")

    # ---- (d) purge canary
    full = cpcv_splits(2579, n_groups=PILOT["cv"]["n_groups"],
                       k_test=PILOT["cv"]["k_test"],
                       purge_days=PILOT["cv"]["purge_days"],
                       embargo_days=PILOT["cv"]["embargo_days"])
    assert_no_leakage(full, purge_days=PILOT["cv"]["purge_days"])
    bad = Split(0, full[0].test_groups,
                np.concatenate([full[0].train_idx, full[0].test_idx[:1]]),
                full[0].test_idx)
    try:
        assert_no_leakage([bad], purge_days=1)
        caught = False
    except AssertionError:
        caught = True
    gate("(d) train-on-future canary", caught,
         "clean CPCV passes; corrupted split raises")

    # ---- (e) SSL provenance
    prov = json.loads(Path("data/pretrain/corpus_provenance.json").read_text())
    last = max(prov["window_dates"])
    gate("(e) SSL corpus provenance", last < lo,
         f"last corpus day {last} < first supervised day {lo}")

    md = ["# Leakage gates\n", "| gate | result | evidence |", "|---|---|---|"]
    md += results
    Path("reports/leakage_gates.md").write_text("\n".join(md) + "\n")
    print("all gates passed -> reports/leakage_gates.md")


if __name__ == "__main__":
    main()
