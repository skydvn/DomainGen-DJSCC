"""
Unified entry point for DG-DJSCC.

Usage
-----
# Train + auto-evaluate the best checkpoint, log everything to W&B
python main.py --config configs/dg_djscc_cifar10.yaml --mode train_eval

# Train only
python main.py --config configs/dg_djscc_cifar10.yaml --mode train

# Evaluate an existing checkpoint
python main.py --config configs/dg_djscc_cifar10.yaml --mode eval \
               --ckpt runs/dg_djscc_cifar10/best.pt

# Override W&B settings from the CLI without editing YAML
python main.py --config configs/dg_djscc_cifar10.yaml --mode train_eval \
               --wandb-project my-project --wandb-mode offline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Project root on sys.path so 'channels', 'models', etc. resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import run_eval, run_train  # noqa: E402
from utils.wandb_logger import WandbLogger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="DG-DJSCC: train and/or evaluate.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--mode", choices=["train", "eval", "train_eval"],
                   default="train_eval",
                   help="What to run. Default trains then evaluates best.pt.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint for eval mode. Defaults to "
                        "<out_dir>/best.pt for train_eval.")

    # W&B overrides
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default=None,
                   choices=["online", "offline", "disabled"])
    p.add_argument("--no-wandb", action="store_true",
                   help="Force-disable W&B regardless of config.")
    p.add_argument("--debug-cuda", action="store_true",
                   help="Sync CUDA launches so errors point at the real op. "
                        "Slower; use only when diagnosing crashes.")
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    cfg.setdefault("wandb", {})
    if args.wandb_project is not None:
        cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        cfg["wandb"]["entity"] = args.wandb_entity
    if args.wandb_mode is not None:
        cfg["wandb"]["mode"] = args.wandb_mode
        if args.wandb_mode == "disabled":
            cfg["wandb"]["enabled"] = False
    if args.no_wandb:
        cfg["wandb"]["enabled"] = False
    return cfg


def main():
    args = parse_args()

    if args.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        try:
            import torch
            torch.cuda.set_sync_debug_mode("error")
        except Exception:  # pragma: no cover
            pass

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cli_overrides(cfg, args)

    out_dir = cfg["experiment"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "train":
        logger = WandbLogger(cfg, job_type="train")
        try:
            run_train(cfg, logger)
        finally:
            logger.finish()

    elif args.mode == "eval":
        ckpt = args.ckpt or os.path.join(out_dir, "best.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        logger = WandbLogger(cfg, job_type="eval")
        try:
            run_eval(cfg, ckpt, logger)
        finally:
            logger.finish()

    elif args.mode == "train_eval":
        # One W&B run that covers both phases. Eval metrics show up alongside
        # training curves in the same run, which is what you usually want.
        logger = WandbLogger(cfg, job_type="train_eval")
        try:
            best_ckpt, _ = run_train(cfg, logger)
            ckpt = args.ckpt or best_ckpt
            print(f"\nEvaluating {ckpt} on SNR grid...")
            run_eval(cfg, ckpt, logger)
        finally:
            logger.finish()


if __name__ == "__main__":
    main()
