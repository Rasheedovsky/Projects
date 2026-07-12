"""Training: determinism, augmentation order, dataset encoding, overfit sanity."""
import numpy as np
import torch

from spydt.axes import AxisBundle
from spydt.models import Classifier3D
from spydt.train import VolumeDataset, encode_one, fit, predict_proba


def _bundle(n=80, L=32, seed=0):
    rng = np.random.default_rng(seed)
    clock = rng.uniform(-0.9, 0.9, (n, L))
    info = rng.uniform(-0.9, 0.9, (n, L))
    y = np.where(rng.normal(0, 1, n) > 0, 1.0, -1.0)
    # plant a learnable signal in a tabular feature (GASF is sign-flip
    # invariant, so series-mean signals are a poor fixture for a quick test)
    tabular = rng.normal(0, 1, (n, 5))
    tabular[:, 0] = y * 0.8 + rng.normal(0, 0.2, n)
    return AxisBundle(
        dates=[f"d{i}" for i in range(n)],
        clock=clock, info=info, info_mask=np.ones((n, L), dtype=bool),
        tabular=tabular,
        y=y, w=np.ones(n), r_label=y * 1e-3, r_exec=y * 1e-3,
    )


def test_dataset_encodes_on_the_fly():
    b = _bundle()
    ds = VolumeDataset(b, np.arange(10), "vb", l_sub=16)
    vol, tab, y, w = ds[0]
    assert vol.shape == (3, 10, 16, 16)
    assert tab.shape == (5,)


def test_augmentation_is_series_level_and_seeded():
    b = _bundle()
    aug = {"jitter_sigma": 0.05, "scale_sigma": 0.05}
    d1 = VolumeDataset(b, np.arange(10), "vb", 16, augment=aug, seed=7)
    d2 = VolumeDataset(b, np.arange(10), "vb", 16, augment=aug, seed=7)
    v1, *_ = d1[0]
    v2, *_ = d2[0]
    torch.testing.assert_close(v1, v2)  # same seed -> same augmentation
    d3 = VolumeDataset(b, np.arange(10), "vb", 16, augment=aug, seed=8)
    v3, *_ = d3[0]
    assert not torch.allclose(v1, v3)   # different seed -> different draw
    # mask channel is never augmented
    torch.testing.assert_close(v1[2], v3[2])


def test_fit_is_deterministic_and_learns():
    b = _bundle(n=160)
    tr, va = np.arange(120), np.arange(120, 160)

    def run(seed):
        torch.manual_seed(seed)
        m = Classifier3D(dropout=0.1)
        fit(m, VolumeDataset(b, tr, "vb", 16, seed=seed),
            VolumeDataset(b, va, "vb", 16),
            max_epochs=25, patience=25, batch_size=16, lr=3e-3,
            weight_decay=1e-3, label_smoothing=0.0, seed=seed)
        return predict_proba(m, VolumeDataset(b, va, "vb", 16))

    p1, p2 = run(0), run(0)
    np.testing.assert_allclose(p1, p2, atol=1e-6)  # deterministic
    acc = np.mean((p1 > 0.5) == (b.y[va] > 0))     # planted signal learnable
    assert acc > 0.7


def test_encode_one_depth_shuffle_differs():
    b = _bundle()
    v = encode_one("vb", b.clock[0], b.info[0], b.info_mask[0], 16)
    s = encode_one("vb_shuffled", b.clock[0], b.info[0], b.info_mask[0], 16)
    assert v.shape == s.shape
    assert not np.allclose(v, s)
    np.testing.assert_allclose(np.sort(v.ravel()), np.sort(s.ravel()))
