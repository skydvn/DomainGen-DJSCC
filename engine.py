"""
Training/evaluation harness.

Engine knows about: data, model, channels, optimizer, eval loop, checkpointing,
W&B. It does NOT know how a training step is computed — that lives in the
algorithm (see algorithms/). To add a new method, add a file in algorithms/
and register it; engine stays untouched.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from algorithms import BaseAlgorithm, build_algorithm
from channels import build_channels
from data import get_loaders
from models import DJSCC
from utils import AverageMeter, load_ckpt, ms_ssim, psnr, save_ckpt, set_seed
from utils.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_runtime(cfg):
    """Build model + channels + loaders + device from a config dict."""
    set_seed(cfg["seed"])
    device = torch.device(
        cfg["device"] if (torch.cuda.is_available() or cfg["device"] == "cpu") else "cpu"
    )
    train_loader, test_loader = get_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
    )
    model = DJSCC(cr=cfg["model"]["cr"]).to(device)
    channels = {name: ch.to(device)
                for name, ch in build_channels(cfg["channels"]).items()}
    return model, channels, train_loader, test_loader, device


def _build_optimizer(model, cfg):
    name = cfg["train"].get("optimizer", "adam").lower()
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _sample_snr(cfg) -> float:
    strategy = cfg["train"]["snr_strategy"]
    if strategy == "fixed":
        return float(cfg["train"]["snr_db_fixed"])
    if strategy == "uniform":
        lo, hi = cfg["train"]["snr_db_range"]
        return random.uniform(float(lo), float(hi))
    raise ValueError(f"Unknown snr_strategy: {strategy}")


# ---------------------------------------------------------------------------
# Evaluation (algorithm-agnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_at_snr(model, channels, loader, device, snr_db: float):
    model.eval()
    pm = {n: AverageMeter() for n in channels}
    sm = {n: AverageMeter() for n in channels}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model.encode(x)
        for name, ch in channels.items():
            x_hat = model.decode(ch(z, snr_db)).clamp(0, 1)
            pm[name].update(psnr(x_hat, x).mean().item(), n=x.shape[0])
            sm[name].update(ms_ssim(x_hat, x).mean().item(), n=x.shape[0])
    return {n: (pm[n].avg, sm[n].avg) for n in channels}


@torch.no_grad()
def sweep_snr(model, channels, loader, device, snr_grid: List[float]):
    return {snr: evaluate_at_snr(model, channels, loader, device, snr)
            for snr in snr_grid}


@torch.no_grad()
def log_qualitative(model, channels, loader, device, logger: WandbLogger,
                    n_samples: int, snr_db: float, step: int):
    if not logger.enabled:
        return
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_samples].to(device)
    z = model.encode(x)
    logger.log_images("samples/original", x,
                      captions=[f"orig_{i}" for i in range(n_samples)], step=step)
    for name, ch in channels.items():
        x_hat = model.decode(ch(z, snr_db)).clamp(0, 1)
        logger.log_images(f"samples/{name}_recon@{snr_db}dB", x_hat,
                          captions=[f"{name}_{i}" for i in range(n_samples)],
                          step=step)


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------

def _run_epoch(algo: BaseAlgorithm, loader, cfg, device, epoch,
               logger: WandbLogger, global_step: int):
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for it, (x, _) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        snr_db = _sample_snr(cfg)

        loss, logs = algo.train_step(x, snr_db)

        loss_meter.update(float(loss), n=x.shape[0])
        psnr_meter.update(logs.get("train/psnr_avg", 0.0), n=x.shape[0])
        global_step += 1

        if it % cfg["train"]["log_every"] == 0:
            logger.log({**logs, "train/snr_db": snr_db, "epoch": epoch},
                       step=global_step)
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                psnr=f"{psnr_meter.avg:.2f}dB",
                snr=f"{snr_db:.1f}dB",
            )

    return loss_meter.avg, psnr_meter.avg, global_step


def run_train(cfg, logger: WandbLogger):
    model, channels, train_loader, test_loader, device = build_runtime(cfg)
    optimizer = _build_optimizer(model, cfg)
    algo: BaseAlgorithm = build_algorithm(cfg, model, channels, optimizer)

    out_dir = cfg["experiment"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    in_domain = set(algo.in_domain_channels())
    ood = [n for n in channels if n not in in_domain]

    print(f"Algorithm: {algo.name}")
    print(f"Model: DJSCC, c_out={model.c_out}, CR={model.cr:.4f}")
    print(f"Train channels: {', '.join(in_domain)}")
    print(f"Eval channels : {', '.join(channels.keys())}")
    if ood:
        print(f"  -> OOD test channels: {', '.join(ood)}")

    logger.watch(model)
    logger.log({
        "setup/algorithm": algo.name,
        "setup/n_train_channels": len(in_domain),
        "setup/n_eval_channels": len(channels),
    })

    best_psnr = -float("inf")
    best_ckpt = os.path.join(out_dir, "best.pt")
    global_step = 0

    log_img_every = cfg.get("wandb", {}).get("log_images_every_epochs", 10)
    n_samples = cfg.get("wandb", {}).get("num_image_samples", 8)

    algo.on_train_start()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss, train_psnr, global_step = _run_epoch(
            algo, train_loader, cfg, device, epoch, logger, global_step
        )
        algo.on_epoch_end(epoch)

        val = evaluate_at_snr(model, channels, test_loader, device, snr_db=10.0)
        val_psnr = sum(p for p, _ in val.values()) / len(val)
        val_ssim = sum(s for _, s in val.values()) / len(val)
        in_p = [p for n, (p, _) in val.items() if n in in_domain]
        ood_p = [p for n, (p, _) in val.items() if n not in in_domain]

        payload = {
            "val/psnr_avg@10dB": val_psnr,
            "val/msssim_avg@10dB": val_ssim,
            "val/in_domain_psnr@10dB": sum(in_p) / max(len(in_p), 1),
            "epoch": epoch,
        }
        if ood_p:
            payload["val/ood_psnr@10dB"] = sum(ood_p) / len(ood_p)
            payload["val/generalization_gap@10dB"] = (
                payload["val/in_domain_psnr@10dB"] - payload["val/ood_psnr@10dB"]
            )
        for n, (p, s) in val.items():
            tag = "id" if n in in_domain else "ood"
            payload[f"val/psnr_{n}@10dB"] = p
            payload[f"val/msssim_{n}@10dB"] = s
            payload[f"val/psnr_{n}_{tag}@10dB"] = p
        logger.log(payload, step=global_step)

        per_ch = " | ".join(
            f"{k}{'*' if k in in_domain else ''}: {p:.2f}dB"
            for k, (p, _) in val.items()
        )
        print(f"[ep {epoch:03d}] loss {loss:.4f}  train_psnr {train_psnr:.2f}dB  "
              f"val@10dB {val_psnr:.2f}dB  ({per_ch})  [* = in-domain]")

        if epoch % log_img_every == 0:
            log_qualitative(model, channels, test_loader, device, logger,
                            n_samples=n_samples, snr_db=10.0, step=global_step)

        # Model selection by in-domain PSNR (mean over all for DG).
        sel_psnr = (sum(in_p) / len(in_p)) if len(in_domain) < len(channels) else val_psnr
        if sel_psnr > best_psnr:
            best_psnr = sel_psnr
            save_ckpt(best_ckpt, model, optimizer,
                      extra={"epoch": epoch, "val_psnr": val_psnr,
                             "in_domain": sorted(in_domain),
                             "selection_psnr": sel_psnr,
                             "algorithm": algo.name})

        if epoch % cfg["train"]["ckpt_every_epochs"] == 0:
            save_ckpt(os.path.join(out_dir, f"ep{epoch:03d}.pt"),
                      model, optimizer,
                      extra={"epoch": epoch, "val_psnr": val_psnr})

    logger.save_artifact(best_ckpt, name=f"{cfg['experiment']['name']}-best")
    print(f"Done. Best selection PSNR = {best_psnr:.2f} dB")
    return best_ckpt, best_psnr


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------

def run_eval(cfg, ckpt_path: str, logger: WandbLogger):
    model, channels, _, test_loader, device = build_runtime(cfg)
    extra = load_ckpt(ckpt_path, model, map_location=device)

    in_domain = set(extra.get("in_domain") or [])
    if not in_domain:
        # Fall back: rebuild algorithm from config to recover in-domain split.
        try:
            algo = build_algorithm(cfg, model, channels,
                                   _build_optimizer(model, cfg))
            in_domain = set(algo.in_domain_channels())
        except Exception:
            in_domain = set(channels.keys())

    snr_grid = cfg["eval"]["snr_db_grid"]
    results = sweep_snr(model, channels, test_loader, device, snr_grid)
    channel_names = list(channels.keys())

    _print_table(results, channel_names, in_domain=in_domain)

    rows_psnr, rows_ssim = [], []
    for snr_db in sorted(results):
        psnr_vals = [results[snr_db][n][0] for n in channel_names]
        ssim_vals = [results[snr_db][n][1] for n in channel_names]
        mean_p = sum(psnr_vals) / len(psnr_vals)
        mean_s = sum(ssim_vals) / len(ssim_vals)
        rows_psnr.append([snr_db, *psnr_vals, mean_p])
        rows_ssim.append([snr_db, *ssim_vals, mean_s])

        scalar = {}
        in_v, ood_v = [], []
        for n, v in zip(channel_names, psnr_vals):
            tag = "id" if n in in_domain else "ood"
            scalar[f"eval/psnr_{n}"] = v
            scalar[f"eval/psnr_{n}_{tag}"] = v
            (in_v if n in in_domain else ood_v).append(v)
        for n, v in zip(channel_names, ssim_vals):
            scalar[f"eval/msssim_{n}"] = v
        scalar["eval/psnr_mean"] = mean_p
        scalar["eval/msssim_mean"] = mean_s
        if in_v:
            scalar["eval/psnr_in_domain"] = sum(in_v) / len(in_v)
        if ood_v:
            scalar["eval/psnr_ood"] = sum(ood_v) / len(ood_v)
            scalar["eval/generalization_gap"] = (
                scalar["eval/psnr_in_domain"] - scalar["eval/psnr_ood"]
            )
        scalar["eval/snr_db"] = snr_db
        logger.log(scalar)

    tagged_cols = [f"{n} ({'ID' if n in in_domain else 'OOD'})"
                   for n in channel_names]
    logger.log_table("eval/psnr_table",
                     columns=["snr_db", *tagged_cols, "mean"], rows=rows_psnr)
    logger.log_table("eval/msssim_table",
                     columns=["snr_db", *tagged_cols, "mean"], rows=rows_ssim)

    mid_snr = snr_grid[len(snr_grid) // 2]
    n_samples = cfg.get("wandb", {}).get("num_image_samples", 8)
    log_qualitative(model, channels, test_loader, device, logger,
                    n_samples=n_samples, snr_db=mid_snr, step=0)

    return results


def _print_table(results, channel_names, in_domain=None):
    in_domain = in_domain or set()
    tagged = [f"{n} (ID)" if n in in_domain else f"{n} (OOD)" for n in channel_names]
    header = ["SNR(dB)"] + tagged + ["Mean"]
    col_w = max(12, max(len(h) for h in header) + 2)

    def fmt(row):
        return "".join(str(c).ljust(col_w) for c in row)

    if in_domain and len(in_domain) < len(channel_names):
        print("\n[legend] ID = trained on; OOD = held-out generalization test")

    print("\nPSNR (dB)")
    print(fmt(header))
    print("-" * (col_w * len(header)))
    for snr_db in sorted(results):
        vals = [results[snr_db][n][0] for n in channel_names]
        mean = sum(vals) / len(vals)
        print(fmt([snr_db, *[f"{v:.2f}" for v in vals], f"{mean:.2f}"]))

    print("\nMS-SSIM")
    print(fmt(header))
    print("-" * (col_w * len(header)))
    for snr_db in sorted(results):
        vals = [results[snr_db][n][1] for n in channel_names]
        mean = sum(vals) / len(vals)
        print(fmt([snr_db, *[f"{v:.4f}" for v in vals], f"{mean:.4f}"]))

    if in_domain and 0 < len(in_domain) < len(channel_names):
        print("\nIn-Domain vs OOD (PSNR, dB)")
        sub = ["SNR(dB)", "in-domain", "OOD", "gap"]
        print(fmt(sub))
        print("-" * (col_w * len(sub)))
        for snr_db in sorted(results):
            iv = [results[snr_db][n][0] for n in channel_names if n in in_domain]
            ov = [results[snr_db][n][0] for n in channel_names if n not in in_domain]
            i_m = sum(iv) / len(iv)
            o_m = sum(ov) / len(ov)
            print(fmt([snr_db, f"{i_m:.2f}", f"{o_m:.2f}", f"{i_m - o_m:.2f}"]))
