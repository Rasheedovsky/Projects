"""Phases 5-7: CPCV pilot matrix over the frozen registry.

Stages (resumable; completed artifacts are skipped):
  --stage bundles    build + cache the per-split axis bundles
  --stage baselines  deterministic baselines on every split
  --stage nn         neural trials (multiprocessing over (config, seed, split))
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spydt.axes import AxisBundle, build_bundle, load_daydata
from spydt.cv import cpcv_splits

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
PILOT = yaml.safe_load(Path("configs/pilot.yaml").read_text())
REG = yaml.safe_load(Path("configs/registry.yaml").read_text())

BUNDLE_DIR = Path("data/tensors/bundles")
PRED_DIR = Path("data/registry/preds")
SPLIT_FILE = Path("data/tensors/splits.json")


# ---------------------------------------------------------------------- utils
def get_supervised_days():
    days = load_daydata("data/interim/spy_rth_et.parquet",
                        "data/interim/day_classification.parquet",
                        CFG["volume"]["unit_multiplier"])
    lo, hi = CFG["eras"]["supervised"]
    return [d for d in days if lo <= d.date <= hi]


def get_splits(n_days: int):
    return cpcv_splits(n_days, n_groups=PILOT["cv"]["n_groups"],
                       k_test=PILOT["cv"]["k_test"],
                       purge_days=PILOT["cv"]["purge_days"],
                       embargo_days=PILOT["cv"]["embargo_days"])


def save_bundle(b: AxisBundle, path: Path) -> None:
    np.savez_compressed(
        path, clock=b.clock, info=b.info, info_mask=b.info_mask,
        tabular=b.tabular, y=b.y, w=b.w, r_label=b.r_label, r_exec=b.r_exec,
        dates=np.array(b.dates), diag=json.dumps(b.diag, default=str),
    )


def load_bundle(path: Path) -> AxisBundle:
    z = np.load(path, allow_pickle=False)
    return AxisBundle(
        dates=[str(d) for d in z["dates"]], clock=z["clock"], info=z["info"],
        info_mask=z["info_mask"], tabular=z["tabular"], y=z["y"], w=z["w"],
        r_label=z["r_label"], r_exec=z["r_exec"],
        diag=json.loads(str(z["diag"])),
    )


# -------------------------------------------------------------------- bundles
def stage_bundles() -> None:
    days = get_supervised_days()
    splits = get_splits(len(days))
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(
        {str(s.split_id): {"test_groups": list(s.test_groups),
                           "train_idx": s.train_idx.tolist(),
                           "test_idx": s.test_idx.tolist()} for s in splits}
    ))
    for s in splits:
        out = BUNDLE_DIR / f"split{s.split_id}.npz"
        if out.exists():
            continue
        t0 = time.time()
        b = build_bundle(days, s.train_idx, bar_kind="dollar",
                         K=PILOT["axes"]["K"], L=PILOT["axes"]["L"],
                         floor_buckets=PILOT["axes"]["floor_buckets"],
                         cap_mult=PILOT["axes"]["cap_mult"],
                         ffd_grid_step=PILOT["axes"]["ffd_grid_step"])
        save_bundle(b, out)
        print(f"split {s.split_id}: {time.time() - t0:.0f}s "
              f"d*=({b.diag['d_clock']},{b.diag['d_info']}) "
              f"pad={b.diag['pad_rate']:.2f}", flush=True)


# ------------------------------------------------------------------ baselines
def _baseline_features(days):
    r1, r_on, r_lm, rvol, rng_, dow = [], [], [], [], [], []
    import pandas as pd

    for d in days:
        pc = d.prev_close if d.prev_close else d.open_[0]
        r1.append(np.log(d.close[29] / pc))
        r_on.append(np.log(d.open_[0] / pc))
        r_lm.append(np.log(d.close[89] / d.close[59]))
        rets = np.diff(np.log(d.close[:90]))
        rvol.append(np.std(rets) * np.sqrt(390))
        rng_.append((d.close[:90].max() - d.close[:90].min()) / d.open_[0])
        dow.append(pd.Timestamp(d.date).dayofweek / 4.0)
    return np.column_stack([r1, r_on, r_lm, rvol, rng_, dow])


def stage_baselines() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    days = get_supervised_days()
    splits = get_splits(len(days))
    X = _baseline_features(days)
    r_label = np.array([np.log(d.close[389] / d.close[360]) for d in days])
    y = np.where(r_label >= 0, 1.0, -1.0)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for s in splits:
        tr, te = s.train_idx, s.test_idx
        outs: dict[str, np.ndarray] = {}

        # gao_ols: r_last30 ~ r1 + r_on (train fit)
        A = np.column_stack([np.ones(len(tr)), X[tr, 0], X[tr, 1]])
        beta, *_ = np.linalg.lstsq(A, r_label[tr], rcond=None)
        pred = beta[0] + X[te, 0] * beta[1] + X[te, 1] * beta[2]
        outs["gao_ols"] = 1 / (1 + np.exp(-pred / (r_label[tr].std() + 1e-12)))

        # gao_vol: vol-scaled predictors
        Av = np.column_stack([np.ones(len(tr)), X[tr, 0] / (X[tr, 3] + 1e-9),
                              X[tr, 1] / (X[tr, 3] + 1e-9)])
        bv, *_ = np.linalg.lstsq(Av, r_label[tr], rcond=None)
        predv = bv[0] + X[te, 0] / (X[te, 3] + 1e-9) * bv[1] \
            + X[te, 1] / (X[te, 3] + 1e-9) * bv[2]
        outs["gao_vol"] = 1 / (1 + np.exp(-predv / (r_label[tr].std() + 1e-12)))

        outs["gao_sign"] = (X[te, 0] > 0).astype(float)
        outs["always_long"] = np.ones(len(te))

        sc = StandardScaler().fit(X[tr])
        lo = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), y[tr] > 0)
        outs["logistic"] = lo.predict_proba(sc.transform(X[te]))[:, 1]

        xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=0)
        xgb.fit(X[tr], (y[tr] > 0).astype(int))
        outs["xgboost"] = xgb.predict_proba(X[te])[:, 1]

        outs["minirocket"] = _minirocket_probs(s)

        for name, prob in outs.items():
            np.savez(PRED_DIR / f"{name}_s0_split{s.split_id}.npz",
                     idx=te, prob=prob.astype(np.float32))
        print(f"baselines split {s.split_id} done", flush=True)


def _minirocket_probs(s) -> np.ndarray:
    from sklearn.linear_model import RidgeClassifierCV
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    b = load_bundle(BUNDLE_DIR / f"split{s.split_id}.npz")
    ok = ~np.isnan(b.clock).any(axis=1) & ~np.isnan(b.info).any(axis=1)
    X3 = np.stack([b.clock, b.info], axis=1)  # (n, 2, L)
    tr = np.array([i for i in s.train_idx if ok[i]])
    te = s.test_idx
    rocket = MiniRocketMultivariate(random_state=0)
    rocket.fit(X3[tr])
    ftr = rocket.transform(X3[tr])
    fte = rocket.transform(np.nan_to_num(X3[te]))
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(ftr, b.y[tr] > 0)
    margin = clf.decision_function(fte)
    prob = 1 / (1 + np.exp(-margin / (np.abs(margin).mean() + 1e-9)))
    prob[~ok[te]] = 0.5
    return prob


# ------------------------------------------------------------------------- nn
def _nn_task(task) -> str:
    config, seed, split_id = task
    import torch

    torch.set_num_threads(1)
    from spydt.models import Classifier2D, Classifier3D, transfer_trunk
    from spydt.train import VolumeDataset, fit, predict_proba

    out = PRED_DIR / f"{config}_s{seed}_split{split_id}.npz"
    if out.exists():
        return f"skip {out.name}"
    b = load_bundle(BUNDLE_DIR / f"split{split_id}.npz")
    with open(SPLIT_FILE) as f:
        sp = json.load(f)[str(split_id)]
    train_idx = np.array(sp["train_idx"])
    test_idx = np.array(sp["test_idx"])

    n_val = int(len(train_idx) * PILOT["train"]["val_frac"])
    fit_idx, val_idx = train_idx[: -n_val - 1], train_idx[-n_val:]  # 1-day purge

    construction = {"vb_mae": "vb", "vb_scratch": "vb"}.get(config, config)
    l_sub = PILOT["axes"]["l_sub"]
    torch.manual_seed(seed)
    if config == "vb2d":
        model = Classifier2D(in_channels=30, dropout=PILOT["train"]["dropout"])
    elif config == "vc2d":
        model = Classifier2D(in_channels=9, dropout=PILOT["train"]["dropout"])
    else:
        model = Classifier3D(dropout=PILOT["train"]["dropout"])
    if config == "vb_mae":
        state = torch.load("data/pretrain/mae3d.pt", weights_only=True)
        transfer_trunk(model, state)

    aug = PILOT["train"]["augment"]
    t0 = time.time()
    res = fit(
        model,
        VolumeDataset(b, fit_idx, construction, l_sub, augment=aug, seed=seed),
        VolumeDataset(b, val_idx, construction, l_sub),
        max_epochs=PILOT["train"]["max_epochs"], patience=PILOT["train"]["patience"],
        batch_size=PILOT["train"]["batch_size"], lr=PILOT["train"]["lr"],
        weight_decay=PILOT["train"]["weight_decay"],
        label_smoothing=PILOT["train"]["label_smoothing"], seed=seed,
    )
    ds_te = VolumeDataset(b, test_idx, construction, l_sub)
    prob_te = predict_proba(model, ds_te)
    ds_val = VolumeDataset(b, val_idx, construction, l_sub)
    prob_val = predict_proba(model, ds_val)
    np.savez(out, idx=ds_te.idx, prob=prob_te.astype(np.float32),
             val_idx=ds_val.idx, val_prob=prob_val.astype(np.float32))
    return (f"{config} s{seed} split{split_id}: val_loss={res.best_val_loss:.4f} "
            f"epochs={res.epochs_ran} {time.time() - t0:.0f}s")


def stage_nn(workers: int, only: list[str] | None = None) -> None:
    from multiprocessing import get_context

    tasks = []
    configs = [c for c in REG["nn_trials"] if not only or c in only]
    for config in configs:
        seeds = PILOT["train"]["seeds"]
        for seed in seeds:
            for split_id in range(len(json.load(open(SPLIT_FILE)))):
                out = PRED_DIR / f"{config}_s{seed}_split{split_id}.npz"
                if not out.exists():
                    tasks.append((config, seed, split_id))
    print(f"{len(tasks)} nn tasks, {workers} workers", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        for msg in pool.imap_unordered(_nn_task, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["bundles", "baselines", "nn"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()
    if args.stage == "bundles":
        stage_bundles()
    elif args.stage == "baselines":
        stage_baselines()
    else:
        stage_nn(args.workers, args.configs)
