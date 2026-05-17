"""
Deep JSCC for Wireless Image Transmission (Bourtsoulatze et al., 2019).

The base algorithm for image transmission over a noisy channel:
encoder -> channel -> decoder, trained end-to-end with mean-squared error.

This class only defines the *per-channel forward + MSE loss*. The training
scope is decided by :class:`BaseAlgorithm` via ``algorithm.mode``:

  - ``single_source``: run on one channel per step. With a fixed
    ``algorithm.train_snr_db``, this reproduces the canonical Bourtsoulatze /
    chunbaobao protocol.
  - ``multi_source`` : run on all selected channels per step and average the
    MSE. With every channel selected, this is **Basic Domain-Generalized
    DJSCC**.

So "DJSCC-WIT" and "DG-DJSCC" are the same algorithm, differing only by
mode. The registry exposes both names (``djscc_wit`` and ``dg_djscc``) for
convenience, with ``dg_djscc`` defaulting ``mode=multi_source``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgorithm, MULTI


class DJSCC_WIT(BaseAlgorithm):
    name = "djscc_wit"
    # Default to single_source unless the config overrides.
    default_mode = None  # falls back to BaseAlgorithm's default (single_source)

    def compute_per_channel_loss(self, x: torch.Tensor, channel_name: str,
                                 channel: nn.Module, snr_db: float
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.model.encode(x)
        z_tilde = channel(z, snr_db)
        x_hat = self.model.decode(z_tilde)
        loss = F.mse_loss(x_hat, x)
        return loss, x_hat


class DGDJSCC(DJSCC_WIT):
    """Alias of DJSCC-WIT with multi-source mode as the default.

    Configuring ``algorithm.name: dg_djscc`` is equivalent to
    ``algorithm.name: djscc_wit`` with ``algorithm.mode: multi_source``.
    """

    name = "dg_djscc"
    default_mode = MULTI
