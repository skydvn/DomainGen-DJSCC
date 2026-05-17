"""
Model registry.

Map ``model.name`` in a YAML config to a backbone class. Each backbone must
expose ``.encode(x) -> z`` and ``.decode(z_tilde) -> x_hat`` so the engine can
inject any channel between them, and ``.c_out`` for logging.

Adding a new backbone
---------------------
1. Create models/my_backbone.py with a class exposing encode/decode/c_out.
2. Import and register here under a name.
3. In a YAML config, set ``model.name: my_backbone`` and any constructor args
   under ``model.*``.

Backbones shipped
-----------------
- ``baseline``    : our wider GroupNorm+PReLU CNN (was the default). Good when
                    you want a stronger backbone for DG-DJSCC.
- ``chunbaobao``  : exact architecture from chunbaobao/Deep-JSCC-PyTorch
                    (paper-faithful Bourtsoulatze et al. 2019).
"""
from __future__ import annotations

from typing import Type

import torch.nn as nn

from .baseline import DJSCC as BaselineDJSCC
from .chunbaobao import ChunbaobaoDJSCC


MODELS: dict = {
    "baseline":  BaselineDJSCC,
    "chunbaobao": ChunbaobaoDJSCC,
}


def build_model(cfg) -> nn.Module:
    """Build the model backbone from a config dict.

    Reads ``cfg.model.name`` (defaults to 'baseline' for backwards
    compatibility) and forwards every other key under ``cfg.model.*`` to the
    constructor.
    """
    model_cfg = dict(cfg.get("model", {}))
    name = model_cfg.pop("name", "baseline")
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODELS)}"
        )
    return MODELS[name](**model_cfg)


__all__ = ["MODELS", "build_model", "BaselineDJSCC", "ChunbaobaoDJSCC"]
