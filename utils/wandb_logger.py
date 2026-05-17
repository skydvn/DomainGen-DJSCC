"""Thin Weights & Biases wrapper.

Use this so the rest of the code doesn't have to care whether W&B is enabled,
installed, or offline. All methods are no-ops when ``cfg.wandb.enabled`` is
false or the ``wandb`` package is missing.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


class WandbLogger:
    def __init__(self, cfg: Dict[str, Any], job_type: str = "train"):
        wb_cfg = cfg.get("wandb", {}) or {}
        self.enabled = bool(wb_cfg.get("enabled", False)) and _WANDB_AVAILABLE
        self.run = None
        self.cfg = cfg
        self.wb_cfg = wb_cfg

        if not self.enabled:
            if wb_cfg.get("enabled") and not _WANDB_AVAILABLE:
                print("[wandb] enabled in config but the package isn't installed -> skipping.")
            return

        self.run = wandb.init(
            project=wb_cfg.get("project", "dg-djscc"),
            entity=wb_cfg.get("entity"),
            name=cfg.get("experiment", {}).get("name"),
            mode=wb_cfg.get("mode", "online"),
            tags=wb_cfg.get("tags", []),
            job_type=job_type,
            config=cfg,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_images(self, key: str, images, captions=None, step: Optional[int] = None) -> None:
        """Log a list/tensor of images to W&B. ``images`` is a (N, C, H, W) tensor in [0,1]."""
        if not self.enabled:
            return
        wb_images = []
        for i in range(images.shape[0]):
            cap = captions[i] if captions is not None else None
            wb_images.append(wandb.Image(images[i].clamp(0, 1).cpu(), caption=cap))
        wandb.log({key: wb_images}, step=step)

    def log_table(self, key: str, columns, rows, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({key: table}, step=step)

    def watch(self, model, log: str = "gradients", log_freq: int = 200) -> None:
        if not self.enabled:
            return
        wandb.watch(model, log=log, log_freq=log_freq)

    def save_artifact(self, path: str, name: str, kind: str = "model") -> None:
        if not self.enabled:
            return
        artifact = wandb.Artifact(name=name, type=kind)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()
