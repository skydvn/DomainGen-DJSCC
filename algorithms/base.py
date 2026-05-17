"""
Algorithm interface.

Algorithm vs. Mode
------------------
- **Algorithm**: how the per-channel loss is computed, what regularizers it
  adds, whether it consumes side info (e.g., channel-aware methods). Concrete
  algorithms subclass :class:`BaseAlgorithm` and implement
  :meth:`compute_per_channel_loss`.
- **Mode**: training-time scope over channels.
    - ``single_source``: each step trains on exactly one channel.
    - ``multi_source`` (a.k.a. DG): each step trains on every selected source
      channel; per-channel losses are averaged. With every channel selected
      and a plain MSE algorithm, this is "Basic Domain-Generalized DJSCC".

This factorization means a new algorithm only needs to write one forward
pass; the harness reuses it for both modes for free. To produce "DG-MyMethod"
from "MyMethod", just set ``algorithm.mode: multi_source`` in the YAML.

Config schema (read by BaseAlgorithm)
-------------------------------------
algorithm:
  name: <registered_name>
  mode: single_source | multi_source       # default: single_source
  # When mode = single_source:
  train_channel: awgn                      # required (exactly one)
  train_snr_db: 10.0                       # optional; pin training SNR
  # When mode = multi_source:
  train_channels: all                      # 'all' | [list of names]
  # Algorithm-specific knobs go under the same `algorithm:` block.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


SINGLE = "single_source"
MULTI = "multi_source"


class BaseAlgorithm(ABC):
    name: str = "base"
    # Subclasses can pin a default mode (e.g., dg_djscc -> 'multi_source').
    # If set, an explicit `algorithm.mode` in the config still wins.
    default_mode: Optional[str] = None

    def __init__(self, model: nn.Module, channels: Dict[str, nn.Module],
                 optimizer: torch.optim.Optimizer, cfg: dict):
        self.model = model
        self.channels = channels
        self.optimizer = optimizer
        self.cfg = cfg

        algo_cfg = cfg.get("algorithm", {}) or {}
        self.mode: str = algo_cfg.get("mode") or self.default_mode or SINGLE
        if self.mode not in (SINGLE, MULTI):
            raise ValueError(
                f"algorithm.mode must be '{SINGLE}' or '{MULTI}'; "
                f"got {self.mode!r}"
            )

        # Resolve which channels to TRAIN on. Eval always uses all channels.
        if self.mode == SINGLE:
            self._train_names = self._resolve_single(algo_cfg, channels)
        else:
            self._train_names = self._resolve_multi(algo_cfg, channels)

        # Optional fixed training SNR (e.g., paper DJSCC-WIT protocol).
        snr = algo_cfg.get("train_snr_db", None)
        self._fixed_snr_db: Optional[float] = None if snr is None else float(snr)

    # ------------------------------------------------------------------
    # Channel-set resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_single(algo_cfg, channels) -> List[str]:
        sel = algo_cfg.get("train_channel")
        if sel is None:
            raise ValueError(
                "single_source mode requires `algorithm.train_channel` "
                "(a single channel name)."
            )
        if isinstance(sel, list):
            if len(sel) != 1:
                raise ValueError(
                    f"single_source mode needs exactly one channel; got "
                    f"{sel}. Use mode='multi_source' for >1 channel."
                )
            sel = sel[0]
        if sel not in channels:
            raise ValueError(
                f"train_channel '{sel}' not in configured channels "
                f"{list(channels)}."
            )
        return [sel]

    @staticmethod
    def _resolve_multi(algo_cfg, channels) -> List[str]:
        sel = algo_cfg.get("train_channels", "all")
        if sel == "all" or sel is None or (isinstance(sel, list) and not sel):
            return list(channels.keys())
        if isinstance(sel, str):
            sel = [sel]
        missing = [n for n in sel if n not in channels]
        if missing:
            raise ValueError(
                f"train_channels lists {missing} not in configured channels "
                f"{list(channels)}."
            )
        return list(sel)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_per_channel_loss(self, x: torch.Tensor, channel_name: str,
                                 channel: nn.Module, snr_db: float
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One forward pass through one channel.

        Returns (loss, x_hat). The loss is a scalar tensor (mean over the
        batch). x_hat is returned so the base class can compute PSNR for
        logging; algorithms that don't produce a reconstruction can return
        any 4D tensor (it will only be used in a torch.no_grad() block).
        """
        raise NotImplementedError

    def in_domain_channels(self) -> Iterable[str]:
        """Names of channels the model was trained on (drives ID/OOD tags)."""
        return list(self._train_names)

    # ------------------------------------------------------------------
    # Generic train step — shared by all algorithms
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, snr_db: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One optimizer step over the configured mode."""
        self.model.train()

        if self.mode == SINGLE:
            name = self._train_names[0]
            loss, x_hat = self.compute_per_channel_loss(
                x, name, self.channels[name], snr_db
            )
            per_ch_psnr = {name: _psnr(x_hat, x)}
        else:
            losses = []
            per_ch_psnr: Dict[str, float] = {}
            for name in self._train_names:
                loss_k, x_hat_k = self.compute_per_channel_loss(
                    x, name, self.channels[name], snr_db
                )
                losses.append(loss_k)
                per_ch_psnr[name] = _psnr(x_hat_k, x)
            loss = torch.stack(losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        logs = {
            "train/loss": float(loss.detach()),
            "train/psnr_avg": sum(per_ch_psnr.values()) / len(per_ch_psnr),
            **{f"train/psnr_{n}": v for n, v in per_ch_psnr.items()},
            "train/mode": self.mode,
        }
        if self._fixed_snr_db is not None:
            logs["train/snr_db_fixed"] = self._fixed_snr_db
        return loss.detach(), logs

    # ------------------------------------------------------------------
    # SNR override (paper protocol = fixed SNR per run)
    # ------------------------------------------------------------------

    def override_snr_db(self, default_snr_db: float) -> float:
        if self._fixed_snr_db is not None:
            return self._fixed_snr_db
        return default_snr_db

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional)
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass


@torch.no_grad()
def _psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """Local PSNR helper to avoid a circular import with utils."""
    mse = (x_hat.clamp(0, 1) - x).pow(2).mean(dim=[1, 2, 3])
    psnr_db = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return float(psnr_db.mean().item())
