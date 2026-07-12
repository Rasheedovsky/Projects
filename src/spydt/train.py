"""Training loop: on-the-fly encoding, time-domain augmentation, early stop.

Augmentation happens on the SERIES before encoding (never image-space ops),
matching the Gramian-semantics constraint. Determinism: one seed controls
torch, numpy and the augmentation stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spydt.axes import AxisBundle
from spydt.encode import EPS, build_va, build_vb, build_vc, build_single_axis, vb_as_2d

CONSTRUCTIONS = ("vb", "vb2d", "va", "vc", "vc2d", "clock_only", "info_only",
                 "vb_shuffled")

_VB_SHUFFLE = np.array([7, 2, 9, 0, 5, 3, 8, 1, 6, 4])  # fixed depth permutation


def encode_one(name: str, clock: np.ndarray, info: np.ndarray, mask: np.ndarray,
               l_sub: int) -> np.ndarray:
    if name == "vb":
        return build_vb(clock, info, mask, l_sub)
    if name == "vb2d":
        return vb_as_2d(build_vb(clock, info, mask, l_sub))
    if name == "vb_shuffled":
        return build_vb(clock, info, mask, l_sub)[:, _VB_SHUFFLE]
    if name == "va":
        return build_va(clock, info, mask)
    if name == "vc":
        return build_vc(clock, info, mask)
    if name == "vc2d":  # identical V-C content, 2D layout (S3 control)
        return vb_as_2d(build_vc(clock, info, mask))
    if name == "clock_only":
        return build_single_axis(clock, l_sub)
    if name == "info_only":
        return build_single_axis(info, l_sub, valid=mask)
    raise ValueError(name)


class VolumeDataset(Dataset):
    def __init__(self, bundle: AxisBundle, idx: np.ndarray, construction: str,
                 l_sub: int, augment: dict | None = None, seed: int = 0) -> None:
        ok = ~np.isnan(bundle.clock).any(axis=1) & ~np.isnan(bundle.info).any(axis=1)
        self.idx = np.array([i for i in idx if ok[i]])
        self.b = bundle
        self.construction = construction
        self.l_sub = l_sub
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.idx)

    def _aug(self, x: np.ndarray) -> np.ndarray:
        a = self.augment
        x = x + self.rng.normal(0, a["jitter_sigma"], len(x))
        x = x * (1.0 + self.rng.normal(0, a["scale_sigma"]))
        return np.clip(x, -1 + EPS, 1 - EPS)

    def __getitem__(self, j: int):
        i = int(self.idx[j])
        clock, info = self.b.clock[i], self.b.info[i]
        if self.augment is not None:
            clock, info = self._aug(clock), self._aug(info)
        vol = encode_one(self.construction, clock, info, self.b.info_mask[i], self.l_sub)
        y01 = 0.5 * (self.b.y[i] + 1.0)
        return (
            torch.from_numpy(vol),
            torch.from_numpy(self.b.tabular[i].astype(np.float32)),
            torch.tensor(y01, dtype=torch.float32),
            torch.tensor(self.b.w[i], dtype=torch.float32),
        )


@dataclass
class FitResult:
    best_val_loss: float
    epochs_ran: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit(model, ds_train: Dataset, ds_val: Dataset, *, max_epochs: int, patience: int,
        batch_size: int, lr: float, weight_decay: float, label_smoothing: float,
        seed: int = 0) -> FitResult:
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    dv = DataLoader(ds_val, batch_size=512, shuffle=False)
    best, best_state, bad, ran = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        ran = epoch + 1
        model.train()
        for vol, tab, y, w in dl:
            opt.zero_grad()
            logit = model(vol, tab)
            ys = y * (1 - label_smoothing) + 0.5 * label_smoothing
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, ys, weight=w
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        vl, wsum = 0.0, 0.0
        with torch.no_grad():
            for vol, tab, y, w in dv:
                logit = model(vol, tab)
                l = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, y, weight=w, reduction="sum"
                )
                vl += float(l)
                wsum += float(w.sum())
        vl /= max(wsum, 1.0)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(best, ran)


@torch.no_grad()
def predict_proba(model, ds: Dataset) -> np.ndarray:
    model.eval()
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    out = []
    for vol, tab, _, _ in dl:
        out.append(torch.sigmoid(model(vol, tab)).numpy())
    return np.concatenate(out) if out else np.empty(0)
