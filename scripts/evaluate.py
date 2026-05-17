"""
Evaluate a trained DG-DJSCC model.

Produces a table:
    SNR\\Channel | AWGN | Rayleigh | Rician | Mean
    0 dB         | ...  | ...      | ...    | ...
    ...

Metrics: PSNR (dB) and MS-SSIM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channels import build_channels                       # noqa: E402
from data import get_loaders                              # noqa: E402
from models import DJSCC                                  # noqa: E402
from utils import AverageMeter, load_ckpt, ms_ssim, psnr  # noqa: E402


@torch.no_grad()
def sweep(model, channels, loader, snr_grid, device):
    """Returns dict[snr_db] = dict[channel_name] = (psnr_avg, msssim_avg)."""
    results = {}
    model.eval()
    for snr_db in snr_grid:
        meters_psnr = {name: AverageMeter() for name in channels}
        meters_ssim = {name: AverageMeter() for name in channels}
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            z = model.encode(x)
            for name, ch in channels.items():
                x_hat = model.decode(ch(z, snr_db)).clamp(0, 1)
                meters_psnr[name].update(psnr(x_hat, x).mean().item(), n=x.shape[0])
                meters_ssim[name].update(ms_ssim(x_hat, x).mean().item(), n=x.shape[0])
        results[snr_db] = {
            name: (meters_psnr[name].avg, meters_ssim[name].avg)
            for name in channels
        }
    return results


def print_table(results, channel_names):
    header = ["SNR(dB)"] + list(channel_names) + ["Mean"]
    col_w = max(10, max(len(h) for h in header) + 2)

    def fmt_row(row):
        return "".join(str(c).ljust(col_w) for c in row)

    print("\nPSNR (dB)")
    print(fmt_row(header))
    print("-" * (col_w * len(header)))
    for snr_db in sorted(results):
        psnr_vals = [results[snr_db][n][0] for n in channel_names]
        mean = sum(psnr_vals) / len(psnr_vals)
        print(fmt_row([f"{snr_db}"] + [f"{v:.2f}" for v in psnr_vals]
                      + [f"{mean:.2f}"]))

    print("\nMS-SSIM")
    print(fmt_row(header))
    print("-" * (col_w * len(header)))
    for snr_db in sorted(results):
        ssim_vals = [results[snr_db][n][1] for n in channel_names]
        mean = sum(ssim_vals) / len(ssim_vals)
        print(fmt_row([f"{snr_db}"] + [f"{v:.4f}" for v in ssim_vals]
                      + [f"{mean:.4f}"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available()
                          or cfg["device"] == "cpu" else "cpu")

    _, test_loader = get_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=False,
    )

    model = DJSCC(cr=cfg["model"]["cr"]).to(device)
    load_ckpt(args.ckpt, model, map_location=device)

    channels = {name: ch.to(device)
                for name, ch in build_channels(cfg["channels"]).items()}

    snr_grid = cfg["eval"]["snr_db_grid"]
    results = sweep(model, channels, test_loader, snr_grid, device)
    print_table(results, list(channels.keys()))


if __name__ == "__main__":
    main()