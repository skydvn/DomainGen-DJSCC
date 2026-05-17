"""
Train Basic Domain-Generalized DJSCC.

Faithful implementation of the algorithm:

    for each iter:
        sample mini-batch B of size B
        z = PowerNorm(E_phi(x))                       # shared per sample
        for each channel k in {AWGN, Rayleigh, Rician}:
            sample h_k ~ p_k(h), n_{i,k} ~ p_k(n)     # E[|h_k|^2] = 1
            z_tilde_{i,k} = h_k * z_i + n_{i,k}
            x_hat_{i,k}   = D_theta(z_tilde_{i,k})
        L_rec = 1/(BK) * sum_{i,k} d(x_hat_{i,k}, x_i)
        update phi, theta to minimize L_rec
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channels import build_channels                          # noqa: E402
from data import get_loaders                                 # noqa: E402
from models import DJSCC                                     # noqa: E402
from utils import AverageMeter, psnr, save_ckpt, set_seed   # noqa: E402


def sample_snr(cfg) -> float:
    strategy = cfg["train"]["snr_strategy"]
    if strategy == "fixed":
        return float(cfg["train"]["snr_db_fixed"])
    if strategy == "uniform":
        lo, hi = cfg["train"]["snr_db_range"]
        return random.uniform(float(lo), float(hi))
    raise ValueError(f"Unknown snr_strategy: {strategy}")


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(x_hat, x) in the algorithm. MSE on RGB pixels in [0, 1]."""
    return F.mse_loss(x_hat, x, reduction="mean")


def train_one_epoch(model, channels, loader, optimizer, cfg, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for it, (x, _) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        B = x.shape[0]

        # Line 3: encode once and normalize (shared across channels)
        z = model.encode(x)

        # Lines 4-8: forward through each channel domain
        loss = 0.0
        per_channel_psnr = {}
        K = len(channels)
        snr_db = sample_snr(cfg)

        for name, ch in channels.items():
            z_tilde = ch(z, snr_db)              # h_k * z + n_{i,k}
            x_hat = model.decode(z_tilde)
            loss_k = reconstruction_loss(x_hat, x)
            loss = loss + loss_k
            with torch.no_grad():
                per_channel_psnr[name] = psnr(x_hat, x).mean().item()

        # Line 9: 1/(BK) sum_{i,k} d(.) — F.mse_loss already averages over i.
        loss = loss / K

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=B)
        psnr_meter.update(
            sum(per_channel_psnr.values()) / K, n=B,
        )

        if it % cfg["train"]["log_every"] == 0:
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                psnr=f"{psnr_meter.avg:.2f}dB",
                snr=f"{snr_db:.1f}dB",
                **{f"psnr_{k}": f"{v:.2f}" for k, v in per_channel_psnr.items()},
            )

    return loss_meter.avg, psnr_meter.avg


@torch.no_grad()
def evaluate(model, channels, loader, cfg, device, snr_db: float):
    """Evaluate PSNR per channel domain at a given SNR."""
    model.eval()
    meters = {name: AverageMeter() for name in channels}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model.encode(x)
        for name, ch in channels.items():
            x_hat = model.decode(ch(z, snr_db))
            meters[name].update(psnr(x_hat, x).mean().item(), n=x.shape[0])
    return {name: m.avg for name, m in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available()
                          or cfg["device"] == "cpu" else "cpu")

    out_dir = cfg["experiment"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_loader, test_loader = get_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
    )

    model = DJSCC(cr=cfg["model"]["cr"]).to(device)
    channels = {name: ch.to(device)
                for name, ch in build_channels(cfg["channels"]).items()}

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])

    print(f"Model: DJSCC, c_out = {model.c_out}, CR = {model.cr:.4f}")
    print(f"Channel domains (K = {len(channels)}): "
          + ", ".join(channels.keys()))

    best_psnr = -float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss, train_psnr = train_one_epoch(
            model, channels, train_loader, optimizer, cfg, device, epoch
        )

        # Quick validation at SNR = 10 dB
        val_per_ch = evaluate(model, channels, test_loader, cfg, device, snr_db=10.0)
        val_psnr = sum(val_per_ch.values()) / len(val_per_ch)

        per_ch_str = " | ".join(f"{k}: {v:.2f}dB" for k, v in val_per_ch.items())
        print(f"[ep {epoch:03d}]  loss {loss:.4f}  "
              f"train_psnr {train_psnr:.2f}dB  "
              f"val_psnr@10dB {val_psnr:.2f}dB  ({per_ch_str})")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_ckpt(os.path.join(out_dir, "best.pt"),
                      model, optimizer,
                      extra={"epoch": epoch, "val_psnr": val_psnr})

        if epoch % cfg["train"]["ckpt_every_epochs"] == 0:
            save_ckpt(os.path.join(out_dir, f"ep{epoch:03d}.pt"),
                      model, optimizer,
                      extra={"epoch": epoch, "val_psnr": val_psnr})

    print(f"Done. Best val PSNR (avg over channels @10dB) = {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()