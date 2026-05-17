"""
Single-source DJSCC baseline.

Trains on exactly one channel (e.g., AWGN) and is later evaluated on every
channel. The gap between in-domain and OOD test PSNR is the generalization
gap that DG-DJSCC aims to close.

In practice this is just DG-DJSCC with K=1, but kept as its own class so:
  - config files clearly read as "this is a baseline";
  - registry has separate names for ablations/results tables;
  - future single-source-specific tricks (e.g., domain randomization on SNR
    only) have a natural home.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from .base import BaseAlgorithm
from utils import psnr


class SingleSource(BaseAlgorithm):
    name = "single_source"

    def __init__(self, model, channels, optimizer, cfg):
        super().__init__(model, channels, optimizer, cfg)
        sel = cfg.get("algorithm", {}).get("train_channel")
        if sel is None:
            raise ValueError(
                "SingleSource requires `algorithm.train_channel` (a single "
                "channel name)."
            )
        if isinstance(sel, list):
            if len(sel) != 1:
                raise ValueError(
                    f"SingleSource needs exactly one channel; got {sel}. "
                    "Use the dg_djscc algorithm with `train_channels` for "
                    "multi-source training."
                )
            sel = sel[0]
        if sel not in channels:
            raise ValueError(
                f"train_channel '{sel}' is not in configured channels "
                f"{list(channels)}."
            )
        self._train_name = sel

    def in_domain_channels(self) -> Iterable[str]:
        return [self._train_name]

    def train_step(self, x: torch.Tensor, snr_db: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.model.train()
        z = self.model.encode(x)
        ch = self.channels[self._train_name]
        x_hat = self.model.decode(ch(z, snr_db))
        loss = F.mse_loss(x_hat, x)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            p = psnr(x_hat, x).mean().item()
        return loss.detach(), {
            "train/loss": loss.item(),
            "train/psnr_avg": p,
            f"train/psnr_{self._train_name}": p,
        }
