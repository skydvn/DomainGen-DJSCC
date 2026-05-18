"""
DeepJSCC-L++ (aprilbian/deepjscc-lplusplus, Globecom 2023).

The only thing this algorithm changes vs. plain DJSCC-WIT is that **the SNR
is passed into the encoder and decoder as side information**. The actual
side-info conditioning lives in the model backbone (typically
``SwinJSCC``); this algorithm just plumbs ``snr_db`` through. Use the
``swin_jscc`` model to get the real L++ behaviour.

What's intentionally not implemented (v1)
-----------------------------------------
- **DWA** (Dynamic Weight Assignment): scales per-(SNR, bw) losses on the
  fly. Belongs here as a loss-weighting refinement; deferred until the
  base loop is settled.
- **Variable bandwidth per step**: the paper varies the number of latent
  symbols on the fly. Our latent shape is fixed by ``model.cr``, so a
  fixed bw per run is the natural slot. To use multi-bw training you'd
  pass a sampled ``bw`` to ``encode/decode``; the SwinJSCC backbone
  supports this via the ``bw`` kwarg.

Composition with modes
----------------------
- ``mode: single_source`` — train on one channel with SNR conditioning
  (matches the paper's vanilla setup).
- ``mode: multi_source``  — DG-DJSCC-L++: SNR-conditioned model trained
  across multiple channel domains. Strictly stronger than either piece
  alone when channel and SNR both vary.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgorithm, MULTI


def _model_accepts_side_info(model) -> bool:
    """Check whether ``model.encode`` takes the snr_db kwarg.

    Lets us reuse this algorithm with backbones that don't condition (they
    just ignore the side info and behave like djscc_wit).
    """
    import inspect
    try:
        sig = inspect.signature(model.encode)
        return "snr_db" in sig.parameters
    except (TypeError, ValueError):
        return False


class DJSCC_LPP(BaseAlgorithm):
    """SNR-conditioned DJSCC (DeepJSCC-L++)."""
    name = "djscc_lpp"
    default_mode = None  # caller picks single_source vs multi_source

    def __init__(self, model, channels, optimizer, cfg):
        super().__init__(model, channels, optimizer, cfg)
        self._cond = _model_accepts_side_info(model)
        if not self._cond:
            # Don't fail — but warn that the user picked an algorithm whose
            # whole point is side-info conditioning, paired with a backbone
            # that ignores it.
            print(
                f"[djscc_lpp] warning: model {model.__class__.__name__} does "
                f"not accept `snr_db` in encode/decode; falling back to "
                f"plain DJSCC-WIT behaviour. Pair with `model.name: swin_jscc` "
                f"for the real L++ setup."
            )

    def compute_per_channel_loss(self, x: torch.Tensor, channel_name: str,
                                 channel: nn.Module, snr_db: float
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cond:
            z = self.model.encode(x, snr_db=snr_db)
            z_tilde = channel(z, snr_db)
            x_hat = self.model.decode(z_tilde, snr_db=snr_db)
        else:
            z = self.model.encode(x)
            z_tilde = channel(z, snr_db)
            x_hat = self.model.decode(z_tilde)
        loss = F.mse_loss(x_hat, x)
        return loss, x_hat


class DG_DJSCC_LPP(DJSCC_LPP):
    """Multi-source DJSCC-L++ — SNR conditioning + cross-channel training."""
    name = "dg_djscc_lpp"
    default_mode = MULTI