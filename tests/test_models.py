"""Models: budgets, shapes across constructions, MAE mask/loss, transfer."""
import numpy as np
import pytest
import torch

from spydt.models import (
    MAE3D,
    Classifier2D,
    Classifier3D,
    mae_loss,
    n_params,
    transfer_trunk,
    tube_mask,
    PARAM_BUDGET,
)


def test_param_budgets_and_match():
    m3 = Classifier3D()
    m2 = Classifier2D(in_channels=30)
    assert n_params(m3) <= PARAM_BUDGET and n_params(m2) <= PARAM_BUDGET
    # param-matched control: within 25%
    assert abs(n_params(m3) - n_params(m2)) / n_params(m3) < 0.25


@pytest.mark.parametrize("shape", [(3, 10, 16, 16), (3, 2, 32, 32), (3, 3, 32, 32),
                                   (3, 5, 16, 16)])
def test_classifier3d_all_construction_shapes(shape):
    m = Classifier3D()
    x = torch.randn(4, *shape)
    tab = torch.randn(4, 5)
    out = m(x, tab)
    assert out.shape == (4,)


def test_classifier2d_shape():
    m = Classifier2D(in_channels=30)
    out = m(torch.randn(4, 30, 16, 16), torch.randn(4, 5))
    assert out.shape == (4,)


def test_tube_mask_ratio_and_geometry():
    rng = np.random.default_rng(0)
    m = tube_mask((3, 10, 16, 16), 0.75, 4, rng)
    assert m.shape == (16, 16)
    assert abs(m.mean() - 0.75) < 0.01
    # patch structure: every 4x4 patch is uniform
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            assert m[i : i + 4, j : j + 4].std() == 0


def test_mae_reconstruction_and_loss():
    model = MAE3D()
    vol = torch.randn(2, 3, 10, 16, 16)
    rec = model(vol)
    assert rec.shape == (2, 2, 10, 16, 16)
    mask = torch.from_numpy(tube_mask((3, 10, 16, 16), 0.75, 4,
                                      np.random.default_rng(1)))
    loss = mae_loss(model, vol, mask.unsqueeze(0).expand(2, -1, -1))
    assert loss.item() > 0


def test_trunk_transfer():
    mae = MAE3D()
    clf = Classifier3D()
    before = clf.trunk.b1[0].weight.detach().clone()
    n = transfer_trunk(clf, mae.state_dict())
    assert n > 10
    assert not torch.allclose(before, clf.trunk.b1[0].weight)
    torch.testing.assert_close(clf.trunk.b1[0].weight, mae.encoder.b1[0].weight)
