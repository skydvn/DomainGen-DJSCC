"""
DJSCC encoder/decoder for CIFAR-10 (32x32 RGB).

A standard CNN backbone in the spirit of Bourtsoulatze et al. (2019):
- Encoder: 5 conv blocks, downsample 32 -> 8 spatially, output channels = c_out
  controlled by the compression ratio.
- Decoder: mirror of the encoder with transposed convs.

Compression ratio (CR) is defined as
    CR = n / m,
where m = H * W * 3 is the source dimension (real numbers per image),
and n is the number of *real* channel uses per image.
For a feature map of shape (c_out, 8, 8), n = c_out * 64.
For CIFAR-10, m = 32 * 32 * 3 = 3072, so c_out = round(CR * m / 64).

Note on "channel uses": the channel module packs pairs of real values into
complex symbols, so the number of complex channel uses is n / 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from channels import PowerNorm


def cout_from_cr(cr: float, source_dim: int = 32 * 32 * 3, spatial: int = 8 * 8) -> int:
    """Resolve encoder output channels from a target compression ratio.

    Forced to be even so the channel layer can pack into complex symbols.
    """
    c_out = round(cr * source_dim / spatial)
    if c_out % 2 == 1:
        c_out += 1
    return max(c_out, 2)


def _gdn_substitute(num_features: int) -> nn.Module:
    """Use GroupNorm as a light-weight stand-in for GDN.

    GDN is the original DJSCC nonlinearity but introduces an extra dep.
    GroupNorm + PReLU gives a good practical proxy.
    """
    return nn.Sequential(
        nn.GroupNorm(num_groups=min(8, num_features), num_channels=num_features),
        nn.PReLU(num_features),
    )


class Encoder(nn.Module):
    def __init__(self, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # 32 -> 16
            _gdn_substitute(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 16 -> 8
            _gdn_substitute(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            _gdn_substitute(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            _gdn_substitute(128),
            nn.Conv2d(128, c_out, kernel_size=5, stride=1, padding=2),  # (c_out, 8, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, kernel_size=5, stride=1, padding=2),
            _gdn_substitute(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            _gdn_substitute(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            _gdn_substitute(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1),  # 8 -> 16
            _gdn_substitute(64),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1),   # 16 -> 32
            nn.Sigmoid(),
        )

    def forward(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return self.net(z_tilde)


class DJSCC(nn.Module):
    """Full DJSCC: encoder -> PowerNorm -> [channel] -> decoder.

    The channel is *not* part of this module: training/eval code injects the
    received signal (because we have multiple channel domains per iteration).
    Call .encode() and .decode() separately.
    """

    def __init__(self, cr: float = 1 / 6):
        super().__init__()
        self.c_out = cout_from_cr(cr)
        self.cr = cr
        self.encoder = Encoder(self.c_out)
        self.decoder = Decoder(self.c_out)
        self.power_norm = PowerNorm()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.power_norm(self.encoder(x))

    def decode(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_tilde)

    def forward(self, x: torch.Tensor, channel, snr_db: float) -> torch.Tensor:
        z = self.encode(x)
        z_tilde = channel(z, snr_db)
        return self.decode(z_tilde)
