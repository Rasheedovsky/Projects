"""Phase 3: build SSL corpus (era-firewalled) and train the 3D MAE."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spydt.axes import load_daydata
from spydt.encode import build_vb
from spydt.models import MAE3D, mae_loss, tube_mask
from spydt.ssl import build_ssl_windows

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
PILOT = yaml.safe_load(Path("configs/pilot.yaml").read_text())


def main() -> None:
    torch.set_num_threads(4)
    days = load_daydata("data/interim/spy_rth_et.parquet",
                        "data/interim/day_classification.parquet",
                        CFG["volume"]["unit_multiplier"])
    lo, hi = PILOT["ssl"]["era"]
    era = [d for d in days if lo <= d.date <= hi]
    sup_start = CFG["eras"]["supervised"][0]
    assert max(d.date for d in era) < sup_start, "SSL era violates temporal firewall"

    K, L, l_sub = PILOT["axes"]["K"], PILOT["axes"]["L"], PILOT["axes"]["l_sub"]
    print(f"building corpus from {len(era)} era days ...")
    corpus = build_ssl_windows(era, K=K, L=L,
                               stride_min=PILOT["ssl"]["window_stride_min"],
                               floor_buckets=PILOT["axes"]["floor_buckets"],
                               cap_mult=PILOT["axes"]["cap_mult"],
                               ffd_grid_step=PILOT["axes"]["ffd_grid_step"])
    meta = corpus["meta"]
    print("corpus:", json.dumps(meta, default=str))
    Path("data/pretrain").mkdir(exist_ok=True, parents=True)
    with open("data/pretrain/corpus_provenance.json", "w") as f:
        json.dump({"meta": meta,
                   "window_dates": sorted({p[0] for p in corpus["provenance"]})},
                  f, default=str)

    # pre-encode volumes once (corpus is small enough): (n, 3, 10, 16, 16)
    n = len(corpus["clock"])
    vols = np.empty((n, 3, 10, l_sub, l_sub), dtype=np.float32)
    for i in range(n):
        vols[i] = build_vb(corpus["clock"][i], corpus["info"][i],
                           corpus["mask"][i], l_sub)
    vols_t = torch.from_numpy(vols)

    model = MAE3D()
    opt = torch.optim.AdamW(model.parameters(), lr=PILOT["ssl"]["lr"], weight_decay=1e-4)
    epochs = PILOT["ssl"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = PILOT["ssl"]["batch_size"]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    hist = []
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        model.train()
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            batch = vols_t[idx]
            m = np.stack([tube_mask(batch.shape[1:], PILOT["ssl"]["mask_ratio"],
                                    PILOT["ssl"]["patch"], rng)
                          for _ in range(len(idx))])
            loss = mae_loss(model, batch, torch.from_numpy(m))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        hist.append(tot / nb)
        print(f"epoch {ep + 1}/{epochs} loss {hist[-1]:.5f} ({time.time() - t0:.0f}s)",
              flush=True)

    torch.save(model.state_dict(), "data/pretrain/mae3d.pt")

    # reconstruction figure
    model.eval()
    with torch.no_grad():
        sample = vols_t[:4]
        m = np.stack([tube_mask(sample.shape[1:], PILOT["ssl"]["mask_ratio"],
                                PILOT["ssl"]["patch"], np.random.default_rng(1))
                      for _ in range(4)])
        masked = sample * (~torch.from_numpy(m)[:, None, None, :, :])
        rec = model(masked)
    fig, ax = plt.subplots(4, 3, figsize=(9, 10))
    for i in range(4):
        ax[i, 0].imshow(sample[i, 0, 2], vmin=-1, vmax=1)
        ax[i, 1].imshow(masked[i, 0, 2], vmin=-1, vmax=1)
        ax[i, 2].imshow(rec[i, 0, 2], vmin=-1, vmax=1)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    ax[0, 0].set_title("original"), ax[0, 1].set_title("masked (75%)")
    ax[0, 2].set_title("reconstruction")
    fig.tight_layout()
    fig.savefig("reports/figures/mae_reconstruction.png", dpi=110)
    with open("data/pretrain/mae_history.json", "w") as f:
        json.dump(hist, f)
    print("saved data/pretrain/mae3d.pt")


if __name__ == "__main__":
    main()
