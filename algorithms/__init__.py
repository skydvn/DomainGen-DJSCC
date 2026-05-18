"""
Algorithm registry.

Each algorithm is a *method*: how the per-channel loss is computed.
**Training scope** (one channel vs. many) is independent and chosen via
``algorithm.mode`` in the config.

Algorithms shipped
------------------
- ``djscc_wit`` — Deep JSCC for Wireless Image Transmission
  (Bourtsoulatze et al., 2019). MSE forward; mode-agnostic.
- ``dg_djscc``  — alias of ``djscc_wit`` with ``mode='multi_source'`` as the
  default. Reads as the Basic Domain-Generalized DJSCC algorithm directly.

Adding a new algorithm
----------------------
1. Create algorithms/my_method.py with a subclass of :class:`BaseAlgorithm`.
   Implement only ``compute_per_channel_loss(x, name, channel, snr_db)``.
2. Import and register it here.
3. In a config: ``algorithm.name: my_method``. Optionally set
   ``algorithm.mode`` and any method-specific knobs under ``algorithm.*``.

The same algorithm automatically works in both single-source and multi-source
modes — the base class handles the loop and loss aggregation.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseAlgorithm
from .djscc_wit import DGDJSCC, DJSCC_WIT
from .djscc_lpp import DJSCC_LPP, DG_DJSCC_LPP
from .dynamic_jscc import DynamicJSCC


ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "djscc_wit": DJSCC_WIT,
    "dg_djscc":  DGDJSCC,
    "djscc_lpp": DJSCC_LPP,
    "dynamic_jscc": DynamicJSCC,
}


def build_algorithm(cfg, model, channels, optimizer) -> BaseAlgorithm:
    name = cfg.get("algorithm", {}).get("name")
    if not name:
        raise ValueError(
            "Config must specify `algorithm.name`. Available: "
            f"{list(ALGORITHMS)}"
        )
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {list(ALGORITHMS)}"
        )
    return ALGORITHMS[name](model, channels, optimizer, cfg)


__all__ = ["BaseAlgorithm", "ALGORITHMS", "build_algorithm"]
