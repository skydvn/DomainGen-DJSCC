"""
Dynamic JSCC backbone — adaptive-rate DJSCC with a policy network.

Reference
---------
- Paper: Yang, Kim. "Deep Joint Source-Channel Coding for Wireless Image
         Transmission with Adaptive Rate Control", ICASSP 2022.
         https://arxiv.org/abs/2110.04456
- Repo:  https://github.com/mingyuyng/Dynamic_JSCC

Architecture
------------
- Source-encoder ``E_s``: CNN that downsamples the image to a feature map.
- Channel-encoder ``E_c``: 1×1 conv producing ``(G_s + G_n) * L`` planes,
  arranged as ``G = G_s + G_n`` "groups" of ``L`` features each. The first
  ``G_s`` groups are *selective* (can be zeroed by the policy mask); the
  last ``G_n`` groups are *non-selective* (always active).
- Policy network ``P``: takes ``X_s`` (the source-encoder output) and the
  SNR scalar, returns logits over ``G_s + 1`` rate levels (number of active
  selective groups, from 0 to G_s inclusive). At training time we sample via
  Gumbel-Softmax (straight-through), at test time we take argmax.
- SNR-adaptive modules ("AF modules"): squeeze + concat(snr) + MLP gate,
  applied between conv blocks of E_c and D_c. Same idea as ADJSCC.
- Power normalization: per-sample, only over the *active* groups.

This module exposes the framework's standard interface
(``encode(x, snr_db) -> z``, ``decode(z, snr_db) -> x_hat``) plus an extra
``encode_with_policy(x, snr_db)`` that also returns policy diagnostics that
the algorithm needs for its regularizer.

What's intentionally simplified (v1)
------------------------------------
- **Mask-in-place**: we keep the encoded latent shape constant and zero out
  the inactive features rather than physically transmitting a variable
  number of complex symbols. For AWGN this is numerically identical to
  the variable-shape implementation (zero * noise contributes noise that
  the decoder learns to ignore). The "true bandwidth saving" is captured
  by the regularizer, which measures ``mean(active_groups)`` — not by the
  size of the tensor passed to the channel.
- **AWGN only**: the paper claims numbers only for AWGN. The base
  framework will let you run this with Rayleigh/Rician (and the model
  will not crash), but those numbers are research speculation, not a
  paper reproduction.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SNR-adaptive module (AF-module style)
# ---------------------------------------------------------------------------

class _AFModule(nn.Module):
    """Per-channel gating conditioned on SNR (Eqn. similar to ADJSCC).

    Squeeze (global pool) -> concat(SNR) -> 2-layer MLP -> sigmoid gate.
    Output = input * gate.
    """

    def __init__(self, num_features: int, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(num_features + 1, hidden)
        self.fc2 = nn.Linear(hidden, num_features)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c = x.shape[0], x.shape[1]
        # Global average pool over spatial dims, plus broadcast SNR per sample.
        s = x.mean(dim=[2, 3])                                       # (B, C)
        snr = torch.full((b, 1), snr_db, device=x.device, dtype=x.dtype)
        h = F.relu(self.fc1(torch.cat([s, snr], dim=1)))
        gate = torch.sigmoid(self.fc2(h)).view(b, c, 1, 1)
        return x * gate


# ---------------------------------------------------------------------------
# Source encoder / decoder (CNN, chunbaobao-style narrow)
# ---------------------------------------------------------------------------

class _SourceEncoder(nn.Module):
    """3x32x32 -> ngf*8 x 8 x 8 (down by 4)."""

    def __init__(self, in_chans: int = 3, ngf: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, ngf, 5, stride=2, padding=2),  # 32 -> 16
            nn.PReLU(),
            nn.Conv2d(ngf, ngf * 2, 5, stride=2, padding=2),   # 16 -> 8
            nn.PReLU(),
            nn.Conv2d(ngf * 2, ngf * 4, 5, stride=1, padding=2),
            nn.PReLU(),
        )
        self.out_channels = ngf * 4

    def forward(self, x):
        return self.net(x)


class _SourceDecoder(nn.Module):
    """ngf*4 x 8 x 8 -> 3 x 32 x 32."""

    def __init__(self, out_chans: int = 3, ngf: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, 5, stride=1, padding=2),
            nn.PReLU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(ngf, out_chans, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Channel encoder / decoder with AF modules + group structure
# ---------------------------------------------------------------------------

class _ChannelEncoder(nn.Module):
    """Outputs G * L planes, with one AF module conditioned on SNR."""

    def __init__(self, in_channels: int, g_total: int, L: int = 4,
                 hidden: int = 64):
        super().__init__()
        self.g_total = g_total
        self.L = L
        out = g_total * L
        self.conv1 = nn.Conv2d(in_channels, hidden, 1)
        self.af1 = _AFModule(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 1)
        self.af2 = _AFModule(hidden)
        self.conv3 = nn.Conv2d(hidden, out, 1)

    def forward(self, x_s: torch.Tensor, snr_db: float) -> torch.Tensor:
        h = F.relu(self.conv1(x_s))
        h = self.af1(h, snr_db)
        h = F.relu(self.conv2(h))
        h = self.af2(h, snr_db)
        return self.conv3(h)                                # (B, G*L, H, W)


class _ChannelDecoder(nn.Module):
    """Inverse of _ChannelEncoder."""

    def __init__(self, out_channels: int, g_total: int, L: int = 4,
                 hidden: int = 64):
        super().__init__()
        self.g_total = g_total
        self.L = L
        in_ = g_total * L
        self.conv1 = nn.Conv2d(in_, hidden, 1)
        self.af1 = _AFModule(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 1)
        self.af2 = _AFModule(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, y: torch.Tensor, snr_db: float) -> torch.Tensor:
        h = F.relu(self.conv1(y))
        h = self.af1(h, snr_db)
        h = F.relu(self.conv2(h))
        h = self.af2(h, snr_db)
        return self.conv3(h)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class _PolicyNetwork(nn.Module):
    """Takes (X_s, SNR), outputs logits over (G_s + 1) rate levels.

    The output level ``k`` means "activate the first k selective groups"
    (thermometer encoding). Levels: 0 ... G_s inclusive.
    """

    def __init__(self, in_channels: int, g_s: int, hidden: int = 64):
        super().__init__()
        self.g_s = g_s
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, g_s + 1)

    def forward(self, x_s: torch.Tensor, snr_db: float) -> torch.Tensor:
        b = x_s.shape[0]
        s = self.pool(x_s).view(b, -1)
        snr = torch.full((b, 1), snr_db, device=x_s.device, dtype=x_s.dtype)
        h = F.relu(self.fc1(torch.cat([s, snr], dim=1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)                                  # (B, G_s + 1)


# ---------------------------------------------------------------------------
# Composed Dynamic JSCC model
# ---------------------------------------------------------------------------

class DynamicJSCC(nn.Module):
    """Encoder + policy + decoder. Channel is injected by the algorithm.

    Args
    ----
    g_s : int
        Number of *selective* feature groups (rate is variable over these).
    g_n : int
        Number of *non-selective* groups (always active).
    L : int
        Length of each feature group along the channel dim of the latent.
        Total transmitted real planes per spatial position = (G_s + G_n) * L.
    ngf : int
        Width multiplier for the source encoder/decoder.

    Latent / CR
    -----------
    For CIFAR-10 with source-encoder downsample 4×, the latent spatial
    resolution is 8×8. Max real values transmitted per image:
        n_real_max = (G_s + G_n) * L * 8 * 8
    => max CR = n_real_max / (2 * 3072)  (real-value convention).
    Minimum CR is given by ``G_n / (G_s + G_n)`` of this.
    """

    def __init__(self,
                 in_chans: int = 3,
                 ngf: int = 32,
                 g_s: int = 8,
                 g_n: int = 2,
                 L: int = 4,
                 P: float = 1.0):
        super().__init__()
        if g_s < 1 or g_n < 0:
            raise ValueError("Need g_s >= 1 and g_n >= 0.")
        if L < 2 or L % 2 != 0:
            raise ValueError("L must be even (we pack pairs into complex).")
        self.g_s = g_s
        self.g_n = g_n
        self.g_total = g_s + g_n
        self.L = L
        self.P = P

        self.source_encoder = _SourceEncoder(in_chans, ngf=ngf)
        self.channel_encoder = _ChannelEncoder(
            self.source_encoder.out_channels, self.g_total, L=L
        )
        self.policy = _PolicyNetwork(self.source_encoder.out_channels, g_s)
        self.channel_decoder = _ChannelDecoder(
            self.source_encoder.out_channels, self.g_total, L=L
        )
        self.source_decoder = _SourceDecoder(in_chans, ngf=ngf)

        # For framework logging (max latent real-channel count).
        self.c_out = self.g_total * L
        self.cr = None  # variable rate; report max via .max_cr() if asked.

    # ------------------------------------------------------------------
    # Mask sampling (Gumbel-Softmax straight-through)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_thermometer_mask(logits: torch.Tensor, tau: float,
                                hard: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a thermometer mask of length G_s from logits over G_s+1 levels.

        Returns (mask, levels_one_hot) where:
          mask:           (B, G_s)  in [0,1], differentiable via Gumbel-Softmax.
          levels_one_hot: (B, G_s+1) the sampled level one-hot.

        Thermometer code: if level k is selected, mask[:, :k] = 1, rest 0.
        """
        # one_hot is differentiable (straight-through if hard=True).
        one_hot = F.gumbel_softmax(logits, tau=tau, hard=hard)        # (B, G_s+1)
        # Cumulative sum from the *right* turns a one-hot at index k into
        # a prefix-1 vector of length G_s with the first k entries = 1.
        # Specifically: mask[i, j] = sum_{k > j} one_hot[i, k]
        # => mask[:, j] = 1 iff sampled level > j.
        g_s = logits.shape[-1] - 1
        # Build a (G_s+1, G_s) accumulator matrix once.
        levels = torch.arange(g_s + 1, device=logits.device).unsqueeze(1)  # (G_s+1, 1)
        positions = torch.arange(g_s, device=logits.device).unsqueeze(0)   # (1, G_s)
        acc = (levels > positions).to(one_hot.dtype)                  # (G_s+1, G_s)
        mask = one_hot @ acc                                          # (B, G_s)
        return mask, one_hot

    # ------------------------------------------------------------------
    # Power normalization over the active features
    # ------------------------------------------------------------------

    def _power_normalize(self, z: torch.Tensor, n_active_real: torch.Tensor) -> torch.Tensor:
        """Per-sample normalization to mean(|z_active|^2) = P over the active
        complex symbols.

        z: (B, G_total * L, H, W) — inactive features are already zeroed by the
           caller.
        n_active_real: (B,) — number of *real* values currently active per sample.
        """
        b = z.shape[0]
        flat = z.reshape(b, -1)
        energy = flat.pow(2).sum(dim=1, keepdim=True)                # (B, 1)
        n_complex = (n_active_real / 2).clamp_min(1.0).view(b, 1)
        # mean complex power = energy / n_complex
        scale = torch.rsqrt((energy / n_complex).clamp_min(1e-12)) * (self.P ** 0.5)
        return (flat * scale).view_as(z).contiguous()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_with_policy(self, x: torch.Tensor, snr_db: float,
                           tau: float = 1.0, hard: bool = True,
                           force_all_active: bool = False
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode pass returning everything the algorithm needs.

        Returns
        -------
        z : (B, G_total * L, H, W)
            Transmitted symbols (with inactive features zeroed and power
            normalized over the active ones).
        n_active : (B,)
            Number of active groups per sample (G_n <= n_active <= G_total).
        logits : (B, G_s + 1)
            Policy network logits, for any auxiliary regularization.
        levels_one_hot : (B, G_s + 1)
            Sampled level one-hot (straight-through differentiable).
        """
        x_s = self.source_encoder(x)
        feats = self.channel_encoder(x_s, snr_db)                    # (B, G_total*L, H, W)

        b = feats.shape[0]
        device = feats.device
        dtype = feats.dtype

        if force_all_active:
            # Bypass the policy (used for warmup / pre-training).
            mask_s = torch.ones(b, self.g_s, device=device, dtype=dtype)
            one_hot = F.one_hot(torch.full((b,), self.g_s, device=device,
                                           dtype=torch.long),
                                num_classes=self.g_s + 1).to(dtype)
            logits = torch.zeros(b, self.g_s + 1, device=device, dtype=dtype)
        else:
            logits = self.policy(x_s, snr_db)
            mask_s, one_hot = self.sample_thermometer_mask(logits, tau=tau, hard=hard)

        # Always-on mask for the G_n non-selective groups.
        mask_n = torch.ones(b, self.g_n, device=device, dtype=dtype)
        # Concat to per-group mask of length G_total. (B, G_total)
        mask = torch.cat([mask_s, mask_n], dim=1)
        # Broadcast to (B, G_total*L, H, W).
        H, W = feats.shape[2], feats.shape[3]
        mask_full = mask.repeat_interleave(self.L, dim=1).view(b, -1, 1, 1)
        z = feats * mask_full

        # Active counts (in real values, not complex).
        n_active = mask.sum(dim=1) * self.L                          # (B,) reals
        z = self._power_normalize(z, n_active)

        return z, n_active, logits, one_hot

    def encode(self, x: torch.Tensor, snr_db: float = 10.0,
               tau: float = 1.0, hard: bool = True,
               force_all_active: bool = False) -> torch.Tensor:
        """Framework-standard encode. Discards policy outputs."""
        z, _, _, _ = self.encode_with_policy(
            x, snr_db, tau=tau, hard=hard, force_all_active=force_all_active
        )
        return z

    def decode(self, z_tilde: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        feats = self.channel_decoder(z_tilde, snr_db)
        return self.source_decoder(feats)