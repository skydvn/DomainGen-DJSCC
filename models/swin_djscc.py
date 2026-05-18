"""
SwinJSCC backbone — Swin Transformer encoder/decoder with side-info
conditioning, ported from aprilbian/deepjscc-lplusplus (Globecom 2023).

What's faithful to the paper
----------------------------
- Swin Transformer encoder + decoder with patch merging / reverse merging.
- Side-info (SNR, bandwidth) embedded via a 2-layer MLP and **concatenated**
  with patch features at the encoder input and decoder input.
- Power normalization to unit per-symbol power on the transmitted side.
- 4 stages, default ``embed_dims=[96,192,384,768], depths=[2,2,6,2]``.
- Same patch_size=2 → 16× downsampling.

What's intentionally NOT here (v1)
----------------------------------
- **Dynamic Weight Assignment (DWA)** — a loss-weighting trick across
  (SNR, bw) operating points. Belongs in the algorithm, not the model.
- **Variable bandwidth per step (the masking trick)** — the repo keeps only
  ``bw * unit_trans_feat`` features per latent token. In this framework the
  latent shape is fixed by ``model.cr``, so a fixed ``bw`` per run is the
  natural fit. Variable-bw is a larger refactor of the channel/data path.
- **Channel logic** — the repo bundles AWGN into the model. We keep it in
  ``channels/`` so any algorithm can swap channels.

CIFAR-10 caveats
----------------
The Swin defaults target 256×256. For CIFAR-10 (32×32), use the
``cifar_small`` preset, which shrinks depths/embed_dims and uses ``patch_size=2``
so the 4-stage Swin downsamples 32 → 2 (a 2×2 latent). This is intentionally
tight; for paper-grade numbers train on a larger dataset.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal local copies of timm helpers (so we don't add a dep)
# ---------------------------------------------------------------------------

def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def _trunc_normal_(t: torch.Tensor, std=0.02):
    nn.init.trunc_normal_(t, std=std)


class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


# ---------------------------------------------------------------------------
# Swin building blocks (close port of swin_module_bw.py, trimmed)
# ---------------------------------------------------------------------------

class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index",
                             relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=4,
                 shift_size=0, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        if min(input_resolution) <= window_size:
            shift_size = 0
            window_size = min(input_resolution)
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = _WindowAttention(
            dim, _to_2tuple(window_size), num_heads,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self._build_attn_mask()

    def _build_attn_mask(self):
        H, W = self.input_resolution
        if self.shift_size == 0:
            self.register_buffer("attn_mask", torch.zeros(0))
            self._has_mask = False
            return
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        for h in (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None)):
            for w in (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = _window_partition(img_mask, self.window_size)
        mw = mw.view(-1, self.window_size * self.window_size)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        am = am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)
        self.register_buffer("attn_mask", am)
        self._has_mask = True

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))
        windows = _window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        mask = self.attn_mask if self._has_mask else None
        windows = self.attn(windows, mask=mask)
        x = _window_reverse(
            windows.view(-1, self.window_size, self.window_size, C),
            self.window_size, H, W,
        )
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        x = x.view(B, L, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        out_dim = out_dim or dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]
        # cat produces contiguous output; .reshape (not .view) is safe even if
        # an intermediate path produces strided memory.
        x = torch.cat([x0, x1, x2, x3], dim=-1).reshape(B, H * W // 4, 4 * C)
        return self.reduction(self.norm(x))


class _PatchReverseMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        out_dim = out_dim or dim
        self.norm = norm_layer(dim // 4)
        self.increment = nn.Linear(dim // 4, out_dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # PixelShuffle expects (B, C, H, W); view() requires contiguous input.
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = nn.PixelShuffle(2)(x).contiguous()
        # After PixelShuffle: (B, C//4, 2H, 2W). Flatten the spatial dims and
        # bring channels last for LayerNorm. Force contiguity at every step.
        x = x.reshape(B, C // 4, -1).permute(0, 2, 1).contiguous()
        return self.increment(self.norm(x))


class _PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class _EncoderStage(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4.0, drop_path=0.0,
                 downsample: Optional[type] = None):
        super().__init__()
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim)
            block_dim = out_dim
            block_res = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.downsample = None
            block_dim = out_dim
            block_res = input_resolution
        self.blocks = nn.ModuleList([
            _SwinBlock(block_dim, block_res, num_heads, window_size,
                       shift_size=0 if (i % 2 == 0) else window_size // 2,
                       mlp_ratio=mlp_ratio, drop_path=drop_path)
            for i in range(depth)
        ])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _DecoderStage(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4.0, drop_path=0.0,
                 upsample: Optional[type] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(dim, input_resolution, num_heads, window_size,
                       shift_size=0 if (i % 2 == 0) else window_size // 2,
                       mlp_ratio=mlp_ratio, drop_path=drop_path)
            for i in range(depth)
        ])
        self.upsample = (upsample(input_resolution, dim=dim, out_dim=out_dim)
                         if upsample is not None else None)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Side-info embedding
# ---------------------------------------------------------------------------

class _AdaptEmbedding(nn.Module):
    """Encodes a (snr_db, bw) pair into a vector to be broadcast over tokens.

    Following the repo, a single ``nn.Linear(2, n_adapt_embed)`` is used. We
    keep ``bw`` as a normalized scalar (bw / max_bw) so it stays O(1) in
    magnitude — the repo passes raw int bw, which works only because the
    range is small.
    """

    def __init__(self, n_adapt_embed: int = 8):
        super().__init__()
        self.linear = nn.Linear(2, n_adapt_embed)
        self.n_adapt_embed = n_adapt_embed

    def forward(self, snr_db: float, bw_norm: float, device, dtype):
        v = torch.tensor([snr_db, bw_norm], dtype=dtype, device=device)
        return self.linear(v).unsqueeze(0)        # (1, n_adapt_embed)


# ---------------------------------------------------------------------------
# Composed Swin-JSCC model
# ---------------------------------------------------------------------------

# Presets: (embed_dims, depths, num_heads, window_size)
_PRESETS = {
    # Repo default (256x256 input, 16× downsample, ~16M params)
    "paper": ([96, 192, 384, 768], [2, 2, 6, 2], [3, 6, 12, 24], 4),
    # CIFAR-friendly: shallower + narrower; same 16× downsample so 32 -> 2.
    # Latent is 2×2 (tiny); use this only for benchmarking, not paper numbers.
    "cifar_small": ([48, 96, 192, 384], [2, 2, 2, 2], [3, 6, 6, 12], 2),
}


class SwinJSCC(nn.Module):
    """Swin Transformer encoder + decoder with (SNR, bw) side-info embedding.

    Inputs to ``encode``/``decode`` accept the optional ``snr_db`` and ``bw``
    kwargs (broadcast over tokens via ``_AdaptEmbedding``). When omitted,
    snr_db defaults to 10 dB and bw_norm defaults to 1.0 — usable but the
    side-info channel is degenerate, so set them in the algorithm.

    Compression-ratio convention follows the chunbaobao "complex" convention:
    ``cr = n_complex / m_real_source``. For CIFAR-10 (3072 real pixels) and a
    2×2 latent (4 tokens), cr=1/6 ⇒ n_complex = 512 ⇒ 128 complex per token
    ⇒ ``n_trans_feat = 2 * 128 = 256`` real values per token.
    """

    def __init__(self,
                 img_size: int = 32,
                 in_chans: int = 3,
                 patch_size: int = 2,
                 preset: str = "cifar_small",
                 cr: Optional[float] = None,
                 n_trans_feat: Optional[int] = None,
                 n_adapt_embed: int = 8,
                 max_bw: int = 1,
                 P: float = 1.0):
        super().__init__()
        if preset not in _PRESETS:
            raise ValueError(f"Unknown SwinJSCC preset {preset!r}; "
                             f"available: {list(_PRESETS)}")
        embed_dims, depths, num_heads, window_size = _PRESETS[preset]
        self.preset = preset
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_layers = len(embed_dims)
        self.embed_dims = embed_dims
        self.max_bw = max_bw
        self.P = P
        self.n_adapt_embed = n_adapt_embed

        # Latent spatial resolution after the 4 stages.
        # patch_embed downsamples by patch_size, then 3 patch-merge stages
        # downsample by 2 each => total = patch_size * 8.
        self.latent_hw = img_size // (patch_size * 2 ** (self.num_layers - 1))
        n_tokens = self.latent_hw ** 2

        # Resolve n_trans_feat (per-token real-valued latent dim).
        if n_trans_feat is None:
            if cr is None:
                raise ValueError("Pass either `cr` or `n_trans_feat`.")
            source_real = in_chans * img_size * img_size
            n_complex_total = int(round(cr * source_real))
            n_real_total = 2 * n_complex_total
            n_trans_feat = max(2, int(round(n_real_total / n_tokens)))
            if n_trans_feat % 2 == 1:
                n_trans_feat += 1  # even so it packs to complex
        self.n_trans_feat = n_trans_feat
        self.cr = cr
        self.c_out = n_trans_feat                # for logging consistency

        # Side-info embedding
        self.adapt = _AdaptEmbedding(n_adapt_embed)

        # ---- Encoder ----
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans,
                                       embed_dims[0])
        # Adapt projection: cat(patch_feat, embed) -> embed_dims[0]
        self.enc_adapt_proj = nn.Linear(embed_dims[0] + n_adapt_embed,
                                        embed_dims[0])
        self.enc_layers = nn.ModuleList()
        res = (img_size // patch_size, img_size // patch_size)
        for i in range(self.num_layers):
            dim_in = embed_dims[i - 1] if i > 0 else embed_dims[0]
            dim_out = embed_dims[i]
            self.enc_layers.append(_EncoderStage(
                dim=dim_in, out_dim=dim_out, input_resolution=res,
                depth=depths[i], num_heads=num_heads[i],
                window_size=window_size,
                downsample=_PatchMerging if i > 0 else None,
            ))
            if i > 0:
                res = (res[0] // 2, res[1] // 2)
        self.enc_norm = nn.LayerNorm(embed_dims[-1])
        self.enc_proj = nn.Linear(embed_dims[-1], n_trans_feat)

        # ---- Decoder ----
        # Decoder upsamples 3× and ends at (img_size//patch_size, ...)
        self.dec_proj = nn.Linear(n_trans_feat + n_adapt_embed, embed_dims[-1])
        self.dec_layers = nn.ModuleList()
        res = (self.latent_hw, self.latent_hw)
        for i in range(self.num_layers):
            stage_dim = embed_dims[self.num_layers - 1 - i]
            next_dim = (embed_dims[self.num_layers - 2 - i]
                        if i < self.num_layers - 1 else embed_dims[0])
            self.dec_layers.append(_DecoderStage(
                dim=stage_dim, out_dim=next_dim, input_resolution=res,
                depth=depths[self.num_layers - 1 - i],
                num_heads=num_heads[self.num_layers - 1 - i],
                window_size=window_size,
                upsample=_PatchReverseMerging if i < self.num_layers - 1 else None,
            ))
            if i < self.num_layers - 1:
                res = (res[0] * 2, res[1] * 2)
        # After the 3 upsamples the resolution is (img_size//patch_size).
        # Reshape to (B, C, H, W) and use a transposed conv with stride =
        # patch_size to recover the original image size.
        self.out_proj = nn.ConvTranspose2d(embed_dims[0], in_chans,
                                           kernel_size=patch_size,
                                           stride=patch_size)

    # ------------------------------------------------------------------
    # Power norm (per-sample, average complex-symbol power = P)
    # ------------------------------------------------------------------

    def _power_normalize(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure a contiguous buffer up-front: reshape(-1) on a non-contiguous
        # tensor can silently allocate a copy with weird strides on CUDA,
        # which has caused "illegal instruction" / "misaligned address" later
        # in the channel layer.
        z = z.contiguous()
        b = z.shape[0]
        flat = z.view(b, -1)
        n_real = flat.shape[1]
        power = flat.pow(2).sum(dim=1, keepdim=True) / (n_real / 2)
        scale = torch.rsqrt(power + 1e-12) * (self.P ** 0.5)
        return (flat * scale).view_as(z).contiguous()

    # ------------------------------------------------------------------
    # Public API matching the framework's encode/decode contract
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, snr_db: float = 10.0,
               bw: Optional[int] = None) -> torch.Tensor:
        if bw is None:
            bw = self.max_bw
        bw_norm = bw / max(self.max_bw, 1)

        B = x.shape[0]
        embed = self.adapt(snr_db, bw_norm, x.device, x.dtype)    # (1, A)
        z = self.patch_embed(x)                                   # (B, N, E)
        N = z.shape[1]
        embed_b = embed.expand(B, N, -1)
        z = self.enc_adapt_proj(torch.cat([z, embed_b], dim=-1))
        for layer in self.enc_layers:
            z = layer(z)
        z = self.enc_norm(z)
        z = self.enc_proj(z)                                      # (B, n_tokens, n_trans_feat)
        # Reshape to (B, n_trans_feat, latent_hw, latent_hw) so it fits
        # the channel module's (B, C, H, W) expectation.
        # NOTE: transpose() produces a non-contiguous view; reshape() on it
        # may or may not copy. Force a contiguous buffer here so downstream
        # ops (PowerNorm + view_as_complex in the channel) see clean strides
        # — non-contiguous complex packing has caused CUDA crashes on some
        # driver / torch combos.
        z = z.transpose(1, 2).contiguous().view(
            B, self.n_trans_feat, self.latent_hw, self.latent_hw
        )
        return self._power_normalize(z)

    def decode(self, z_tilde: torch.Tensor, snr_db: float = 10.0,
               bw: Optional[int] = None) -> torch.Tensor:
        if bw is None:
            bw = self.max_bw
        bw_norm = bw / max(self.max_bw, 1)

        B = z_tilde.shape[0]
        # (B, C, H, W) -> (B, N, C). Same contiguous-after-transpose pattern.
        n_tokens = self.latent_hw ** 2
        z = z_tilde.contiguous().view(B, self.n_trans_feat, n_tokens)
        z = z.transpose(1, 2).contiguous()                          # (B, N, C)

        embed = self.adapt(snr_db, bw_norm, z.device, z.dtype)
        embed_b = embed.expand(B, n_tokens, -1)
        z = self.dec_proj(torch.cat([z, embed_b], dim=-1))
        for layer in self.dec_layers:
            z = layer(z)
        # (B, N_full, embed_dims[0]) -> (B, C, H, W) at (img_size//patch_size)
        Hpe = self.img_size // self.patch_size
        z = z.transpose(1, 2).contiguous().view(
            B, self.embed_dims[0], Hpe, Hpe
        )
        x = self.out_proj(z)
        return torch.sigmoid(x)

    def forward(self, x, channel, snr_db, bw=None):
        z = self.encode(x, snr_db=snr_db, bw=bw)
        z_tilde = channel(z, snr_db)
        return self.decode(z_tilde, snr_db=snr_db, bw=bw)