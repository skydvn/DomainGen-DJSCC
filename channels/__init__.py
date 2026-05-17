"""
Channel layers for DJSCC.

Convention
----------
- Encoder output z is a real tensor of shape (B, C, ...) with an EVEN number of
  real values per sample. We pack consecutive (real, imag) pairs into complex
  symbols. Number of complex channel uses per sample: n = (numel/B) / 2.
- After PowerNorm, average per *complex symbol* power is 1: E[|z|^2] = 1.
- Each channel returns a real tensor with the same shape as its input.
- All fading gains satisfy E[|h|^2] = 1.
- sigma^2 = 10^{-SNR_dB/10}; complex circularly symmetric noise
  (real/imag ~ N(0, sigma^2 / 2)).

CUDA safety
-----------
We use ``torch.view_as_complex`` / ``torch.view_as_real`` instead of
``torch.complex`` + ``.real`` / ``.imag`` views. The view-as variants require a
contiguous trailing dim of size 2 and produce CUDA-safe layouts. An earlier
version used ``flat.chunk(2, dim=1)`` which returned strided views and could
trigger ``CUDA error: misaligned address`` on some driver/torch combinations.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Real <-> Complex packing (CUDA-safe via view_as_complex / view_as_real)
# ---------------------------------------------------------------------------

def _to_complex(z: torch.Tensor) -> torch.Tensor:
    """Pack a real tensor (B, ...) with even total per sample into complex (B, n)."""
    b = z.shape[0]
    n_real = z.numel() // b
    assert n_real % 2 == 0, (
        f"Encoder output must have an even number of real values per sample; "
        f"got {n_real}."
    )
    n = n_real // 2
    # contiguous + reshape so the last dim of size 2 has stride 1
    real_pairs = z.contiguous().reshape(b, n, 2)
    return torch.view_as_complex(real_pairs)  # zero-copy, shape (B, n)


def _from_complex(zc: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of _to_complex; reshape to the original real shape."""
    flat = torch.view_as_real(zc.contiguous())   # (B, n, 2) real, contiguous
    return flat.reshape_as(like)


def snr_db_to_sigma2(snr_db: float) -> float:
    """sigma^2 with E[|h z|^2] = 1  =>  SNR_linear = 1 / sigma^2."""
    return 10.0 ** (-snr_db / 10.0)


def _complex_randn(shape, device, dtype) -> torch.Tensor:
    """Allocate a contiguous complex tensor with real,imag ~ N(0, 1)."""
    real_pairs = torch.randn(*shape, 2, device=device, dtype=dtype)
    return torch.view_as_complex(real_pairs)


# ---------------------------------------------------------------------------
# Power normalization
# ---------------------------------------------------------------------------

class PowerNorm(nn.Module):
    """Per-sample normalization to E[|z|^2] = 1 over complex symbols."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        flat = z.reshape(b, -1)
        n_real = flat.shape[1]
        power = flat.pow(2).sum(dim=1, keepdim=True) / (n_real / 2)
        scale = torch.rsqrt(power + 1e-12)
        out = (flat * scale).reshape_as(z)
        return out.contiguous()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

class BaseChannel(nn.Module):
    name: str = "base"

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        raise NotImplementedError


def _add_awgn(zc: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add complex circularly symmetric noise of variance sigma^2 to zc."""
    sigma2 = snr_db_to_sigma2(snr_db)
    std = math.sqrt(sigma2 / 2.0)
    real_dtype = torch.float32 if zc.dtype == torch.complex64 else torch.float64
    noise = _complex_randn(zc.shape, device=zc.device, dtype=real_dtype) * std
    return zc + noise


class AWGNChannel(BaseChannel):
    """y = z + n,  n ~ CN(0, sigma^2)."""

    name = "awgn"

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        zc = _to_complex(z)
        return _from_complex(_add_awgn(zc, snr_db), z)


class RayleighChannel(BaseChannel):
    """y = h * z + n,  h ~ CN(0, 1). Block fading by default."""

    name = "rayleigh"

    def __init__(self, per_symbol: bool = False):
        super().__init__()
        self.per_symbol = per_symbol

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        zc = _to_complex(z)                                      # (B, n) complex
        b, n = zc.shape
        real_dtype = torch.float32 if zc.dtype == torch.complex64 else torch.float64

        h_shape = (b, n) if self.per_symbol else (b, 1)
        h = _complex_randn(h_shape, device=z.device, dtype=real_dtype) / math.sqrt(2.0)

        y = h * zc
        return _from_complex(_add_awgn(y, snr_db), z)


class RicianChannel(BaseChannel):
    """Rician fading; K-factor in dB; E[|h|^2] = 1.

    h = mu + sigma_h * scatter,
        |mu|^2 = K/(K+1),  2 sigma_h^2 = 1/(K+1).
    Random LOS phase per sample.
    """

    name = "rician"

    def __init__(self, k_factor_db: float = 4.0, per_symbol: bool = False):
        super().__init__()
        self.k_lin = 10.0 ** (k_factor_db / 10.0)
        self.per_symbol = per_symbol

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        zc = _to_complex(z)
        b, n = zc.shape
        real_dtype = torch.float32 if zc.dtype == torch.complex64 else torch.float64
        K = self.k_lin
        mu_amp = math.sqrt(K / (K + 1.0))
        sigma_h = math.sqrt(1.0 / (2.0 * (K + 1.0)))

        # LOS with random per-sample phase, packed as a (B, 1) complex tensor.
        phi = torch.rand(b, 1, device=z.device, dtype=real_dtype) * (2 * math.pi)
        mu_pairs = torch.stack(
            [mu_amp * torch.cos(phi), mu_amp * torch.sin(phi)],
            dim=-1,
        ).contiguous()
        mu = torch.view_as_complex(mu_pairs)                     # (B, 1)

        h_shape = (b, n) if self.per_symbol else (b, 1)
        scatter = _complex_randn(h_shape, device=z.device, dtype=real_dtype) * sigma_h
        h = mu + scatter

        y = h * zc
        return _from_complex(_add_awgn(y, snr_db), z)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_channels(specs) -> Dict[str, BaseChannel]:
    channels: Dict[str, BaseChannel] = {}
    for spec in specs:
        name = spec["name"].lower()
        if name == "awgn":
            ch = AWGNChannel()
        elif name == "rayleigh":
            ch = RayleighChannel(per_symbol=spec.get("per_symbol", False))
        elif name == "rician":
            ch = RicianChannel(
                k_factor_db=spec.get("k_factor_db", 4.0),
                per_symbol=spec.get("per_symbol", False),
            )
        else:
            raise ValueError(f"Unknown channel: {name}")
        channels[name] = ch
    return channels
