"""
Algorithm interface.

An *algorithm* owns the training-step semantics. Everything else (data, eval,
ckpt, logging) lives in engine.py. To add a new method, subclass BaseAlgorithm,
implement `train_step`, and register it in algorithms/__init__.py.

Contract
--------
- `train_step(batch, snr_db) -> (loss, log_dict)`:
    runs one optimization step on a mini-batch and returns the scalar loss
    (already detached for logging) plus a dict of scalars to log.
- `in_domain_channels()`:
    names of channels treated as in-domain for this run (drives ID/OOD tags
    in eval). For baselines this is the single trained-on channel; for DG it
    is all channels.

The algorithm holds references to `model`, `optimizer`, and `channels`, so the
engine doesn't need to know what subset is used or how the loss is composed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


class BaseAlgorithm(ABC):
    name: str = "base"

    def __init__(self, model: nn.Module, channels: Dict[str, nn.Module],
                 optimizer: torch.optim.Optimizer, cfg: dict):
        self.model = model
        self.channels = channels
        self.optimizer = optimizer
        self.cfg = cfg

    @abstractmethod
    def train_step(self, x: torch.Tensor, snr_db: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One gradient step. Returns (loss, logs)."""
        raise NotImplementedError

    @abstractmethod
    def in_domain_channels(self) -> Iterable[str]:
        """Channel names the model was trained on (for ID/OOD tagging)."""
        raise NotImplementedError

    # Optional hook: called once at the start of training. Algorithms that need
    # warmups, schedule init, etc. can override.
    def on_train_start(self) -> None:
        pass

    # Optional hook: called at the end of each epoch. Useful for things like
    # EMA updates, schedule stepping, or domain-curriculum changes.
    def on_epoch_end(self, epoch: int) -> None:
        pass
