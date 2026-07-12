"""Models: parameter-disciplined 3D CNN, param-matched 2D control, MAE-3D.

Budget: every classifier <= 300k parameters (asserted); pilot trunks ~60k.
Input volumes are (C=3, D, H, W): channels [GASF, GADF, validity-mask].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PARAM_BUDGET = 300_000
N_TABULAR = 5


def _block3d(cin: int, cout: int, stride) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class Trunk3D(nn.Module):
    """8->16->32->48 conv3d trunk; depth stride only in block 3 so kernels
    traverse the full clock<->info depth extent twice before pooling depth."""

    def __init__(self) -> None:
        super().__init__()
        self.b1 = _block3d(3, 8, 1)
        self.b2 = _block3d(8, 16, (1, 2, 2))
        self.b3 = _block3d(16, 32, (2, 2, 2))
        self.b4 = _block3d(32, 48, (1, 2, 2))
        self.out_ch = 48

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b4(self.b3(self.b2(self.b1(x))))


class Classifier3D(nn.Module):
    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()
        self.trunk = Trunk3D()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.trunk.out_ch + N_TABULAR, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        assert n_params(self) <= PARAM_BUDGET, f"over budget: {n_params(self)}"

    def forward(self, vol: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        f = self.trunk(vol).mean(dim=(2, 3, 4))  # 3D GAP
        return self.head(torch.cat([f, tab], dim=1)).squeeze(-1)


def _block2d(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Classifier2D(nn.Module):
    """Param-matched 2D control: identical content as (C*D)-channel image."""

    def __init__(self, in_channels: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _block2d(in_channels, 20, 1),
            _block2d(20, 40, 2),
            _block2d(40, 56, 2),
            _block2d(56, 64, 2),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 + N_TABULAR, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        assert n_params(self) <= PARAM_BUDGET, f"over budget: {n_params(self)}"

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        f = self.trunk(img).mean(dim=(2, 3))
        return self.head(torch.cat([f, tab], dim=1)).squeeze(-1)


class MAE3D(nn.Module):
    """Conv-MAE: dense encoder over tube-masked input, light conv decoder,
    MSE on masked voxels of the GASF/GADF channels."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Trunk3D()
        self.decoder = nn.Sequential(
            nn.Conv3d(48, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 4, 4), mode="trilinear", align_corners=False),
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(16, 2, 3, padding=1),
        )

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        z = self.encoder(vol)
        rec = self.decoder(z)
        # crop/pad depth to input depth (upsample uses fixed factors)
        d_in = vol.shape[2]
        rec = rec[:, :, :d_in]
        return rec


def tube_mask(shape, mask_ratio: float, patch: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask (H, W): True = masked. Tubes span all depth slices."""
    _, _, h, w = shape
    ph, pw = h // patch, w // patch
    n = ph * pw
    k = int(round(mask_ratio * n))
    flat = np.zeros(n, dtype=bool)
    flat[rng.choice(n, size=k, replace=False)] = True
    grid = flat.reshape(ph, pw)
    return np.kron(grid, np.ones((patch, patch), dtype=bool))


def mae_loss(model: MAE3D, vol: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    """MSE on masked voxels only, channels 0-1 (GASF/GADF)."""
    masked = vol.clone()
    m = mask_hw[:, None, None, :, :]  # (B,1,1,H,W) broadcast over C,D
    masked = masked * (~m)
    rec = model(masked)
    target = vol[:, :2]
    diff = (rec - target) ** 2
    m2 = m.expand_as(target).float()
    return (diff * m2).sum() / m2.sum().clamp(min=1.0)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def transfer_trunk(classifier: Classifier3D, mae_state: dict) -> int:
    """Copy MAE encoder weights into the classifier trunk; returns #tensors."""
    enc = {k.removeprefix("encoder."): v for k, v in mae_state.items()
           if k.startswith("encoder.")}
    missing = classifier.trunk.load_state_dict(enc, strict=False)
    return len(enc) - len(missing.missing_keys)
