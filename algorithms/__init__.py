"""
Algorithm registry.

Adding a new algorithm
----------------------
1. Create algorithms/my_method.py with a BaseAlgorithm subclass.
2. Import it here and add to ALGORITHMS under the desired config name.
3. Set `algorithm.name: my_method` in a YAML config; pass any algorithm-
   specific knobs under `algorithm.*`.

Engine and main do not need to change.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseAlgorithm
from .dg_djscc import DGDJSCC
from .single_source import SingleSource


ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "dg_djscc": DGDJSCC,
    "single_source": SingleSource,
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
