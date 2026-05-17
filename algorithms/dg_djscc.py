"""
Basic Domain-Generalized DJSCC.

Implements the algorithm:

    z = PowerNorm(E_phi(x))
    for each channel domain k:
        z_tilde_k = h_k * z + n_k       (h_k, n_k sampled fresh per step)
        x_hat_k   = D_theta(z_tilde_k)
    L = 1/(BK) * sum_{i,k} d(x_hat_{i,k}, x_i)
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from .base import BaseAlgorithm
from utils import psnr


class DGDJSCC(BaseAlgorithm):
    """Multi-source training over all configured channels."""

    name = "dg_djscc"

    def __init__(self, model, channels, optimizer, cfg):
        super().__init__(model, channels, optimizer, cfg)
        # `train_channels` may select a subset; default = all channels.
        sel = cfg.get("algorithm", {}).get("train_channels", "all")
        if sel == "all" or sel is None or (isinstance(sel, list) and not sel):
            self._train_names = list(channels.keys())
        elif isinstance(sel, str):
            self._train_names = [sel]
        else:
            self._train_names = list(sel)
        missing = [n for n in self._train_names if n not in channels]
        if missing:
            raise ValueError(
                f"train_channels lists {missing} not present in configured "
                f"channels {list(channels)}"
            )

    def in_domain_channels(self) -> Iterable[str]:
        return list(self._train_names)

    def train_step(self, x: torch.Tensor, snr_db: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.model.train()
        z = self.model.encode(x)

        K = len(self._train_names)
        loss = 0.0
        per_ch_psnr: Dict[str, float] = {}
        for name in self._train_names:
            ch = self.channels[name]
            x_hat = self.model.decode(ch(z, snr_db))
            loss = loss + F.mse_loss(x_hat, x)
            with torch.no_grad():
                per_ch_psnr[name] = psnr(x_hat, x).mean().item()
        loss = loss / K

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        logs = {
            "train/loss": loss.item(),
            "train/psnr_avg": sum(per_ch_psnr.values()) / K,
            **{f"train/psnr_{n}": v for n, v in per_ch_psnr.items()},
        }
        return loss.detach(), logs
