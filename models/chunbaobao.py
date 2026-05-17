"""
DJSCC backbone from chunbaobao/Deep-JSCC-PyTorch, adapted to this benchmark.

Reference
---------
- Repo:  https://github.com/chunbaobao/Deep-JSCC-PyTorch
- Paper: Bourtsoulatze, Burth Kurka, Gunduz, "Deep Joint Source-Channel Coding
         for Wireless Image Transmission", IEEE Trans. Cogn. Commun. Netw.,
         2019.

Differences from our `BaselineDJSCC`
------------------------------------
- Narrower encoder: 3 -> 16 -> 32 -> 32 -> 32 -> 2c (vs. 64/128/128/128 here).
  This matches the original paper's "Architecture A" widths.
- PReLU activations with Kaiming init; final decoder layer uses Sigmoid with
  Xavier init.
- Normalization: per-sample total-energy normalization to ``P * k`` where
  ``k = c * H_out * W_out``. Equivalent to our PowerNorm (E[|z|^2] = P) but
  computed directly per the repo's formula so reproduced numbers match.

Compression
-----------
The repo parameterizes capacity by ``c`` (encoder produces ``2c`` planes).
We accept either:
  - ``c_inner: int`` directly  (matches the repo's CLI), or
  - ``cr: float``  resolved via :func:`cr_to_cinner` (round to even count of
    real values per sample so the channel layer can complex-pack).

For CIFAR-10 (32x32) the encoder downsamples 32 -> 8, so the latent has
shape (2c, 8, 8) i.e. ``n_real = 2c * 64`` real values per image.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="leaky_relu")

    def forward(self, x):
        return self.prelu(self.conv(x))


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activate=None, padding=0, output_padding=0):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding,
        )
        if activate is None:
            activate = nn.PReLU()
        self.activate = activate
        if isinstance(activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode="fan_out",
                                    nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        return self.activate(self.transconv(x))


# ---------------------------------------------------------------------------
# Normalization (per-sample total energy, repo convention)
# ---------------------------------------------------------------------------

class _TotalEnergyNorm(nn.Module):
    """Normalize each sample so ||z||_2^2 = P * k where k = numel-per-sample.

    Equivalent to E[|z|^2] = P. The repo uses P = 1.
    """

    def __init__(self, P: float = 1.0):
        super().__init__()
        self.P = P

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        b = z.shape[0]
        k = z[0].numel()
        flat = z.reshape(b, -1)
        norm = flat.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        scale = math.sqrt(self.P * k)
        out = (flat * scale / norm).reshape_as(z).contiguous()
        return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Encoder / Decoder (chunbaobao widths)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, c: int, P: float = 1.0):
        super().__init__()
        self.conv1 = _ConvWithPReLU(3,   16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(16,  32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(32,  32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(32,  32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(32, 2*c, kernel_size=5, padding=2)
        self.norm = _TotalEnergyNorm(P=P)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.norm(x)


class _Decoder(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.tconv1 = _TransConvWithPReLU(2*c, 32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(32, 32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(32, 32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(32, 16, kernel_size=5, stride=2,
                                          padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(16,  3, kernel_size=5, stride=2,
                                          padding=2, output_padding=1,
                                          activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def cr_to_cinner(cr: float, image_hw=(32, 32), n_channels: int = 3,
                 latent_hw=(8, 8),
                 cr_convention: str = "complex") -> int:
    """Resolve ``c`` (inner channels) from a target compression ratio.

    Two conventions are commonly used:

    - ``cr_convention='complex'`` (repo / paper convention):
        cr = (number of complex channel uses) / (number of real source pixels)
        For CIFAR-10 (32x32 RGB, latent 8x8):
            n_complex = c * 8 * 8 = 64 c,   m_real = 3072
            cr = 64 c / 3072 = c / 48
          => c = round(cr * 48). e.g. cr=1/6 -> c=8.

    - ``cr_convention='real'`` (our default convention used by
      :class:`BaselineDJSCC`):
        cr = (real values in latent) / (real values in source)
        For CIFAR-10: n_real = 2c * 8 * 8 = 128 c
          => c = round(cr * 24). e.g. cr=1/3 -> c=8.

    Both yield the same model when consistently interpreted, but a "ratio of
    1/6" means different ``c`` values under the two conventions. Default is
    'complex' here because chunbaobao's repo and the Bourtsoulatze paper use
    that convention.

    Returns at least c=1.
    """
    H, W = image_hw
    h_lat, w_lat = latent_hw
    source_real = n_channels * H * W
    if cr_convention == "complex":
        complex_per_c = h_lat * w_lat
        c = round(cr * source_real / complex_per_c)
    elif cr_convention == "real":
        real_per_c = 2 * h_lat * w_lat
        c = round(cr * source_real / real_per_c)
    else:
        raise ValueError(
            f"cr_convention must be 'complex' or 'real'; got {cr_convention!r}"
        )
    return max(c, 1)


# ---------------------------------------------------------------------------
# Composed model
# ---------------------------------------------------------------------------

class ChunbaobaoDJSCC(nn.Module):
    """Encoder + total-energy norm + (channel injected by engine) + decoder.

    The channel is NOT part of this module: training/eval code injects the
    received signal so multi-channel benchmarks work. Use ``.encode()`` and
    ``.decode()`` separately, just like our other backbones.

    Args
    ----
    cr : float, optional
        Target compression ratio; resolves ``c`` via :func:`cr_to_cinner`.
    c_inner : int, optional
        Direct override of ``c`` (the encoder produces ``2c`` planes). One of
        ``cr`` or ``c_inner`` must be provided.
    P : float
        Per-symbol average signal power constraint. Defaults to 1.0.
    """

    def __init__(self, cr: float = None, c_inner: int = None, P: float = 1.0,
                 cr_convention: str = "complex"):
        super().__init__()
        if c_inner is None:
            if cr is None:
                raise ValueError("Either `cr` or `c_inner` must be set.")
            c_inner = cr_to_cinner(cr, cr_convention=cr_convention)
        self.c_inner = int(c_inner)
        self.c_out = 2 * self.c_inner       # encoder produces this many planes
        self.cr = cr
        self.cr_convention = cr_convention
        self.encoder = _Encoder(c=self.c_inner, P=P)
        self.decoder = _Decoder(c=self.c_inner)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_tilde)

    def forward(self, x: torch.Tensor, channel, snr_db: float) -> torch.Tensor:
        z = self.encode(x)
        z_tilde = channel(z, snr_db)
        return self.decode(z_tilde)
