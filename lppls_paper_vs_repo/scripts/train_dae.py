"""Train the denoising autoencoder on synthetic (noisy -> clean) LPPLS pairs
(white + AR(1) mixture, Table 1 amplitudes), then benchmark its noise
reduction and endpoint fidelity against SSA on held-out data.

Usage: python scripts/train_dae.py [--n-train 100000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import dae, ssa, synthetic


def make_pairs(n, seed):
    """(noisy, clean) pairs, both min-max scaled; 50/50 white / AR(1)."""
    rng = np.random.default_rng(seed)
    tc, m, w, c1, c2 = synthetic.sample_params(rng, n)
    clean = np.empty((n, synthetic.N_POINTS), dtype=np.float32)
    for i in range(n):
        clean[i] = synthetic.clean_series(tc[i], m[i], w[i], c1[i], c2[i])
    noisy = clean.copy()
    half = n // 2
    noisy[:half] += synthetic.white_noise(rng, noisy[:half].shape).astype(np.float32)
    noisy[half:] += synthetic.ar1_noise(rng, noisy[half:].shape).astype(np.float32)
    return noisy, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    args = ap.parse_args()

    t0 = time.time()
    Xn, Xc = make_pairs(args.n_train, seed=11)
    Vn, Vc = make_pairs(args.n_val, seed=12)
    print(f"pairs generated in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    params, hist = dae.train(Xn, Xc, Vn, Vc, log=lambda s: print(s, flush=True))
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    dae.save_params(params, ROOT / "models" / "DAE-cleaner.npz")

    # held-out benchmark vs SSA: overall and endpoint fidelity
    Tn, Tc = make_pairs(2000, seed=13)
    rec = np.stack([dae.clean(params, x) for x in Tn])
    ssa_rec = np.stack([ssa.ssa_clean(x) for x in Tn])
    def mse(a, b, sl=slice(None)):
        return float(np.mean((a[:, sl] - b[:, sl]) ** 2))
    print(f"noisy    : MSE {mse(Tn, Tc):.6f} | tail-10 {mse(Tn, Tc, slice(-10, None)):.6f}")
    print(f"DAE      : MSE {mse(rec, Tc):.6f} | tail-10 {mse(rec, Tc, slice(-10, None)):.6f}")
    print(f"SSA      : MSE {mse(ssa_rec, Tc):.6f} | tail-10 {mse(ssa_rec, Tc, slice(-10, None)):.6f}")


if __name__ == "__main__":
    main()
