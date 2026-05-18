"""
Channel layers for DJSCC — real-arithmetic implementation.

Convention
----------
- Encoder output ``z`` is a real tensor with an EVEN number of real values per
  sample. The first half holds the real parts of the complex symbols, the
  second half holds the imaginary parts. (Equivalently: we view the latent as
  ``(B, 2, n_complex)`` for the math but keep the on-disk layout as-is.)
- After ``PowerNorm``, the average per **complex** symbol power is 1:
  E[|z|^2] = 1, i.e. mean(z_re^2 + z_im^2) = 1.
- Each channel returns a real tensor with the same shape as its input.
- All fading gains satisfy E[|h|^2] = 1.
- sigma^2 = 10^{-SNR_dB/10}; complex circularly symmetric noise
  (real/imag ~ N(0, sigma^2 / 2)).

Why no ``view_as_complex``?
---------------------------
Earlier versions used ``torch.view_as_complex`` on freshly-reshaped tensors.
Even when the input was made contiguous, the **alignment** the CUDA complex
kernels require can drift after many iterations as the caching allocator
fragments — producing ``misaligned address`` or ``illegal instruction``
crashes hundreds of steps in. Doing complex arithmetic by hand on real-valued
tensors sidesteps the entire issue and costs ~0 in our regime (small
latents). This is also what the original DJSCC implementations
(chunbaobao, Bourtsoulatze) do.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Real <-> "complex pair" layout (no torch.complex, no view_as_complex)
# ---------------------------------------------------------------------------

def _split_re_im(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Reshape a real tensor (B, ...) with even per-sample count into the
    pair (z_re, z_im), each (B, n) where n = numel/B/2.

    Layout convention: the first half of the per-sample values are real,
    the second half imag. This matches how PowerNorm averages.

    Returns ``(z_re, z_im, original_shape)`` so the caller can reassemble.
    """
    b = z.shape[0]
    orig_shape = z.shape
    n_real = z.numel() // b
    if n_real % 2 != 0:
        raise ValueError(
            f"Channel input must have an even per-sample count; got {n_real}."
        )
    n = n_real // 2
    # contiguous(): make sure the slice halves are clean.
    flat = z.contiguous().view(b, n_real)
    z_re = flat[:, :n].contiguous()
    z_im = flat[:, n:].contiguous()
    return z_re, z_im, orig_shape


def _merge_re_im(z_re: torch.Tensor, z_im: torch.Tensor,
                 orig_shape: torch.Size) -> torch.Tensor:
    """Inverse of _split_re_im. Always returns a contiguous tensor."""
    b = z_re.shape[0]
    flat = torch.cat([z_re, z_im], dim=1).contiguous()
    return flat.view(orig_shape)


def snr_db_to_sigma2(snr_db: float) -> float:
    """sigma^2 with E[|h z|^2] = 1  =>  SNR_linear = 1 / sigma^2."""
    return 10.0 ** (-snr_db / 10.0)


# ---------------------------------------------------------------------------
# PowerNorm: E[|z|^2] = 1 per complex symbol (i.e. mean(re^2 + im^2) = 1)
# ---------------------------------------------------------------------------

class PowerNorm(nn.Module):
    """Per-sample normalization to E[|z|^2] = 1 over complex symbols.

    With the (real, imag) split convention, ``|z_i|^2 = z_re_i^2 + z_im_i^2``,
    averaged over the n_complex = n_real/2 symbols per sample.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        flat = z.reshape(b, -1)
        n_real = flat.shape[1]
        # mean per complex symbol power = sum(z^2) / n_complex = 2*sum(z^2)/n_real
        power = flat.pow(2).sum(dim=1, keepdim=True) * (2.0 / n_real)
        scale = torch.rsqrt(power + 1e-12)
        out = (flat * scale).view_as(z)
        return out.contiguous()


# ---------------------------------------------------------------------------
# Complex-arithmetic helpers (real-valued)
# ---------------------------------------------------------------------------

def _add_awgn_re_im(z_re: torch.Tensor, z_im: torch.Tensor,
                    snr_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add CN(0, sigma^2) noise to a (z_re, z_im) pair."""
    sigma2 = snr_db_to_sigma2(snr_db)
    std = math.sqrt(sigma2 / 2.0)
    n_re = torch.randn_like(z_re) * std
    n_im = torch.randn_like(z_im) * std
    return z_re + n_re, z_im + n_im


def _complex_mul(a_re, a_im, b_re, b_im):
    """(a_re + j a_im) * (b_re + j b_im)."""
    return (a_re * b_re - a_im * b_im,
            a_re * b_im + a_im * b_re)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

class BaseChannel(nn.Module):
    name: str = "base"

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        raise NotImplementedError


class AWGNChannel(BaseChannel):
    """y = z + n,  n ~ CN(0, sigma^2)."""
    name = "awgn"

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        z_re, z_im, shape = _split_re_im(z)
        y_re, y_im = _add_awgn_re_im(z_re, z_im, snr_db)
        return _merge_re_im(y_re, y_im, shape)


class RayleighChannel(BaseChannel):
    """y = h * z + n,  h ~ CN(0, 1). Block fading by default."""
    name = "rayleigh"

    def __init__(self, per_symbol: bool = False):
        super().__init__()
        self.per_symbol = per_symbol

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        z_re, z_im, shape = _split_re_im(z)
        B, n = z_re.shape
        n_h = n if self.per_symbol else 1

        # CN(0,1):  h = (a + j b) / sqrt(2),  a,b ~ N(0,1)  =>  E[|h|^2] = 1
        h_re = torch.randn(B, n_h, device=z.device, dtype=z.dtype) / math.sqrt(2.0)
        h_im = torch.randn(B, n_h, device=z.device, dtype=z.dtype) / math.sqrt(2.0)
        y_re, y_im = _complex_mul(h_re, h_im, z_re, z_im)
        y_re, y_im = _add_awgn_re_im(y_re, y_im, snr_db)
        return _merge_re_im(y_re, y_im, shape)


class RicianChannel(BaseChannel):
    """Rician fading with E[|h|^2] = 1; K-factor in dB.

    h = mu + sigma_h * scatter,
        |mu|^2 = K/(K+1),  2*sigma_h^2 = 1/(K+1).
    LOS phase randomized per sample.
    """
    name = "rician"

    def __init__(self, k_factor_db: float = 4.0, per_symbol: bool = False):
        super().__init__()
        self.k_lin = 10.0 ** (k_factor_db / 10.0)
        self.per_symbol = per_symbol

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        z_re, z_im, shape = _split_re_im(z)
        B, n = z_re.shape
        K = self.k_lin
        mu_amp = math.sqrt(K / (K + 1.0))
        sigma_h = math.sqrt(1.0 / (2.0 * (K + 1.0)))

        # LOS with random per-sample phase.
        phi = torch.rand(B, 1, device=z.device, dtype=z.dtype) * (2 * math.pi)
        mu_re = mu_amp * torch.cos(phi)
        mu_im = mu_amp * torch.sin(phi)

        n_h = n if self.per_symbol else 1
        scatter_re = torch.randn(B, n_h, device=z.device, dtype=z.dtype) * sigma_h
        scatter_im = torch.randn(B, n_h, device=z.device, dtype=z.dtype) * sigma_h

        h_re = mu_re + scatter_re
        h_im = mu_im + scatter_im

        y_re, y_im = _complex_mul(h_re, h_im, z_re, z_im)
        y_re, y_im = _add_awgn_re_im(y_re, y_im, snr_db)
        return _merge_re_im(y_re, y_im, shape)


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