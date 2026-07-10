"""Train the three P-LNN variants per the paper (Sec. 2.2, Table 1):
100,000 training / 33,333 validation synthetic series each, 20 epochs,
batch 8, lr 1e-5. Saves weights to models/ and loss curves to results/.

Usage: python scripts/train_plnn.py [--quick]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import plnn, synthetic

VARIANTS = [
    ("P-LNN-100K", "white"),
    ("P-LNN-100K-AR1", "ar1"),
    ("P-LNN-100K-BOTH", "both"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=33_333)
    args = ap.parse_args()
    n_train, n_val, epochs = (2_000, 500, 2) if args.quick else (args.n_train, args.n_val, plnn.EPOCHS)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, noise) in zip(axes, VARIANTS):
        print(f"=== {name} (noise={noise}) ===", flush=True)
        t0 = time.time()
        X, y = synthetic.make_dataset(n_train, noise=noise, seed=1)
        Xv, yv = synthetic.make_dataset(n_val, noise=noise, seed=2)
        print(f"  data generated in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        params, hist = plnn.train(X, y, Xv, yv, epochs=epochs, log=lambda s: print(s, flush=True))
        train_time = time.time() - t0
        print(f"  trained in {train_time:.1f}s", flush=True)
        plnn.save_params(params, ROOT / "models" / f"{name}.npz")
        ax.plot(range(1, len(hist["train"]) + 1), hist["train"], label="train")
        ax.plot(range(1, len(hist["val"]) + 1), hist["val"], label="validation")
        ax.set_title(f"{name} (train {train_time:.0f}s)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (tc, m, w)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("P-LNN training curves (paper Figs. 8-10 analogue)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_plnn_loss_curves.png", dpi=140)
    print("saved models + loss curves")


if __name__ == "__main__":
    main()
