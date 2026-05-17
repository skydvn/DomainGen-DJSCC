"""Common utilities: metrics, seeding, simple logger."""
from __future__ import annotations

import math
import random
import os
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB. Inputs are (B, C, H, W) in [0, max_val]."""
    mse = (x_hat.clamp(0, max_val) - x).pow(2).mean(dim=[1, 2, 3])
    return 10.0 * torch.log10(max_val ** 2 / (mse + 1e-12))


try:
    from pytorch_msssim import ms_ssim as _ms_ssim

    @torch.no_grad()
    def ms_ssim(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return _ms_ssim(x_hat.clamp(0, 1), x, data_range=1.0, size_average=False)

except ImportError:  # pragma: no cover
    @torch.no_grad()
    def ms_ssim(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # pytorch-msssim isn't installed; return NaNs so training still runs.
        return torch.full((x.shape[0],), float("nan"), device=x.device)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


def save_ckpt(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
              extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_ckpt(path: str, model: torch.nn.Module,
              optimizer: Optional[torch.optim.Optimizer] = None,
              map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("extra", {})
