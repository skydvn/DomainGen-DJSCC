"""
Dynamic JSCC algorithm (Yang & Kim, ICASSP 2022).

Adds a channel-usage regularizer to plain DJSCC-WIT:

    L = E[ ||x - x_hat||^2 ] + lambda_reward * E[ active_groups ]

where ``active_groups`` is sampled per image via a policy network and the
Gumbel-Softmax trick. As the regularizer pulls the policy toward smaller
active counts, the encoder/decoder learn to communicate well at lower
bandwidth, while the policy learns to spend bandwidth where it matters
(low SNR or complex images).

Pair with ``model.name: dynamic_jscc``. The model exposes
``encode_with_policy`` so this algorithm can read the active-group count
without re-running the encoder.

Mode compatibility
------------------
- ``single_source`` reproduces the paper protocol (AWGN, fixed-channel).
- ``multi_source`` runs the algorithm across multiple channel domains.
  This is a research extension, not a paper claim — the paper considers
  AWGN only. Use it deliberately, not by accident.

Loss schedule
-------------
The Gumbel-Softmax temperature ``tau`` is annealed linearly from
``tau_init`` to ``tau_final`` over ``anneal_epochs``. ``lambda_reward``
is constant (the paper uses 2e-3 as a default).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlgorithm, MULTI, SINGLE
from utils import psnr as _psnr_fn


def _psnr_scalar(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    return float(_psnr_fn(x_hat, x).mean().item())


class DynamicJSCC(BaseAlgorithm):
    name = "dynamic_jscc"
    # The paper runs single-channel. Default to that, but allow override.
    default_mode = SINGLE

    def __init__(self, model, channels, optimizer, cfg):
        super().__init__(model, channels, optimizer, cfg)
        algo_cfg = cfg.get("algorithm", {}) or {}

        # Sanity check: the model must support `encode_with_policy`.
        if not hasattr(model, "encode_with_policy"):
            raise ValueError(
                f"dynamic_jscc algorithm requires a model with "
                f"`encode_with_policy`; got {model.__class__.__name__}. "
                "Pair with `model.name: dynamic_jscc`."
            )

        self.lambda_reward = float(algo_cfg.get("lambda_reward", 2e-3))
        self.tau_init = float(algo_cfg.get("tau_init", 5.0))
        self.tau_final = float(algo_cfg.get("tau_final", 0.5))
        self.anneal_epochs = int(algo_cfg.get("anneal_epochs", 100))
        self.warmup_epochs = int(algo_cfg.get("warmup_epochs", 5))
        # Hard sampling (straight-through) by default — gives binary masks.
        self.hard = bool(algo_cfg.get("hard", True))

        self._epoch = 0  # set by on_epoch_end + on_train_start

        # Quiet warning when used with multi-source.
        if self.mode == MULTI:
            print("[dynamic_jscc] note: running in multi_source mode. "
                  "The original paper claims numbers for AWGN only; "
                  "Rayleigh/Rician results here are a research extension.")

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _tau(self) -> float:
        if self.anneal_epochs <= 0:
            return self.tau_final
        t = min(self._epoch / max(self.anneal_epochs, 1), 1.0)
        return self.tau_init + t * (self.tau_final - self.tau_init)

    def _in_warmup(self) -> bool:
        return self._epoch < self.warmup_epochs

    def on_train_start(self) -> None:
        self._epoch = 0

    def on_epoch_end(self, epoch: int) -> None:
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Per-channel forward + extra regularizer
    # ------------------------------------------------------------------

    def compute_per_channel_loss(self, x: torch.Tensor, channel_name: str,
                                 channel: nn.Module, snr_db: float
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self._tau()
        force_all = self._in_warmup()
        z, n_active_real, logits, _one_hot = self.model.encode_with_policy(
            x, snr_db, tau=tau, hard=self.hard, force_all_active=force_all,
        )
        x_hat = self.model.decode(channel(z, snr_db), snr_db=snr_db)

        rec = F.mse_loss(x_hat, x)

        # Channel-usage regularizer: mean number of *active groups* across the
        # batch. n_active_real = active_groups * L, so we divide by L to get
        # the active group count. Detach `force_all` case to avoid no-op grad.
        n_active_groups = n_active_real / float(self.model.L)
        usage = n_active_groups.mean()
        if force_all:
            # Don't push the policy yet — warmup.
            reg = usage.detach() * 0.0
        else:
            reg = self.lambda_reward * usage

        loss = rec + reg

        # Stash diagnostics for train_step to log.
        self._last_aux = {
            "rec_loss": float(rec.detach()),
            "reg_loss": float((reg if not force_all else torch.zeros(())).detach()),
            "active_groups_mean": float(usage.detach()),
            "tau": tau,
            "warmup": bool(force_all),
        }
        return loss, x_hat

    # ------------------------------------------------------------------
    # Override train_step purely to surface the aux logs
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, snr_db: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, logs = super().train_step(x, snr_db)
        aux = getattr(self, "_last_aux", {})
        # Prefix with train/ to keep W&B groups consistent.
        logs.update({
            f"train/{k}": v
            for k, v in aux.items()
            if isinstance(v, (int, float))
        })
        return loss, logs