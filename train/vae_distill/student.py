"""
train/vae_distill/student.py — Lightweight proxy VAE encoder.

Architecture (Option B / Option D hybrid from plans/precomp2-proxy-vae-design.md):
  Mirrors the Flux2Encoder block structure (ResBlocks + GroupNorm + stride-2
  Downsample) but at quarter-scale channel widths.  This preserves the
  teacher's inductive bias (same spatial layout, same normalisation, same
  residual pattern) without the full parameter cost.

Default config (channels=[64,128,256,256], layers_per_block=1, groups=16):
  ~6.5M parameters, expected inference ~20ms/image at batch=16 on M1 Max.

Spatial schedule for 512px input (mirrors teacher):
  After block 0: [B,  64, 256, 256]
  After block 1: [B, 128, 128, 128]
  After block 2: [B, 256,  64,  64]
  After block 3: [B, 256,  64,  64]  (no downsample — matches teacher stride-8)
  Final latent:  [B,  32,  64,  64]
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks  (same conventions as mflux: tensors are [B,C,H,W];
# transposed to [B,H,W,C] only for MLX conv/groupnorm ops)
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 16, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array) -> mx.array:
        residual = mx.transpose(x, (0, 2, 3, 1))
        h = mx.transpose(x, (0, 2, 3, 1))
        h = self.norm1(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            residual = self.skip(residual)
        return mx.transpose(h + residual, (0, 3, 1, 2))


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        return mx.transpose(x, (0, 3, 1, 2))


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_layers: int,
        groups: int,
        downsample: bool,
    ):
        super().__init__()
        self.resnets = [
            _ResBlock(in_ch if i == 0 else out_ch, out_ch, groups=groups)
            for i in range(num_layers)
        ]
        self.downsampler = _Downsample(out_ch) if downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x


class _SelfAttention(nn.Module):
    """Single-head self-attention at final resolution for spatial coherence."""

    def __init__(self, ch: int, groups: int = 16, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch, eps=eps, pytorch_compatible=True)
        self.to_q  = nn.Linear(ch, ch, bias=False)
        self.to_k  = nn.Linear(ch, ch, bias=False)
        self.to_v  = nn.Linear(ch, ch, bias=False)
        self.out   = nn.Linear(ch, ch, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.transpose(x, (0, 2, 3, 1))
        B, H, W, C = h.shape
        n = self.norm(h.astype(mx.float32)).astype(x.dtype)
        q = self.to_q(n).reshape(B, H * W, 1, C)
        k = self.to_k(n).reshape(B, H * W, 1, C)
        v = self.to_v(n).reshape(B, H * W, 1, C)
        q, k, v = (mx.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))
        scale = float(C) ** -0.5
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn = mx.transpose(attn, (0, 2, 1, 3)).reshape(B, H, W, C)
        return x + mx.transpose(self.out(attn), (0, 3, 1, 2))


class _MidBlock(nn.Module):
    def __init__(self, ch: int, groups: int, use_attention: bool):
        super().__init__()
        self.res0 = _ResBlock(ch, ch, groups=groups)
        self.attn = _SelfAttention(ch, groups=groups) if use_attention else None
        self.res1 = _ResBlock(ch, ch, groups=groups)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res0(x)
        if self.attn is not None:
            x = self.attn(x)
        return self.res1(x)


# ---------------------------------------------------------------------------
# Student encoder
# ---------------------------------------------------------------------------

class StudentEncoder(nn.Module):
    """
    Lightweight proxy for the Flux2Encoder.

    Parameters
    ----------
    channels : list of 4 ints — output channels per down-block
    layers_per_block : int — ResBlocks per down-block (teacher uses 2)
    norm_groups : int — GroupNorm groups (teacher uses 32)
    latent_channels : int — output channels (must be 32 to match teacher)
    use_mid_attention : bool — include self-attention in mid block
    """

    def __init__(
        self,
        channels: list[int] = (64, 128, 256, 256),
        layers_per_block: int = 1,
        norm_groups: int = 16,
        latent_channels: int = 32,
        use_mid_attention: bool = True,
    ):
        super().__init__()
        assert len(channels) == 4, "channels must have exactly 4 entries"

        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        self._in_norm = nn.GroupNorm(norm_groups, channels[0], pytorch_compatible=True)

        self.down_blocks = [
            _DownBlock(
                in_ch=channels[i - 1] if i > 0 else channels[0],
                out_ch=channels[i],
                num_layers=layers_per_block,
                groups=norm_groups,
                downsample=(i < 3),  # blocks 0-2 downsample; block 3 does not
            )
            for i in range(4)
        ]

        self.mid = _MidBlock(channels[3], norm_groups, use_mid_attention)

        self.norm_out = nn.GroupNorm(norm_groups, channels[3],
                                     pytorch_compatible=True)
        self.conv_out = nn.Conv2d(channels[3], latent_channels * 2, 3, padding=1)
        self.quant    = nn.Conv2d(latent_channels * 2, latent_channels * 2,
                                   1, padding=0)

    def _conv_in_forward(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv_in(x)
        return mx.transpose(x, (0, 3, 1, 2))

    def _conv_out_forward(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv_out(x)
        return mx.transpose(x, (0, 3, 1, 2))

    def _quant_forward(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.quant(x)
        return mx.transpose(x, (0, 3, 1, 2))

    def __call__(
        self,
        x: mx.array,
        return_features: bool = False,
        qat_bits: Optional[int] = None,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        """
        Forward pass.

        Parameters
        ----------
        x : [B, 3, H, W] float32 in [-1, 1]
        return_features : if True, also return list of 4 intermediate feature maps
        qat_bits : if set, simulate uniform quantisation noise on outputs
                   to mimic int<qat_bits> precision during training

        Returns
        -------
        latent : [B, 32, H/8, W/8] — the proxy latent (mean only, unscaled)
        features (optional) : 4 × [B, Ci, Hi, Wi] intermediate feature maps
        """
        h = self._conv_in_forward(x)
        feats: list[mx.array] = []
        for db in self.down_blocks:
            h = db(h)
            if return_features:
                feats.append(h)

        h = self.mid(h)
        if return_features:
            feats[3] = h  # replace last entry with post-mid features

        h = mx.transpose(h, (0, 2, 3, 1))
        h = self.norm_out(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = mx.transpose(h, (0, 3, 1, 2))

        h = self._conv_out_forward(h)
        h = self._quant_forward(h)

        mean, _ = mx.split(h, 2, axis=1)
        latent = mean  # caller applies teacher scaling/shifting as needed

        if qat_bits is not None:
            latent = _fake_quantize(latent, bits=qat_bits)

        return (latent, feats) if return_features else latent

    def param_count(self) -> int:
        import mlx.utils as mx_utils
        return sum(v.size for _, v in mx_utils.tree_flatten(self.parameters()))


# ---------------------------------------------------------------------------
# Fake quantisation for QAT
# ---------------------------------------------------------------------------

def _fake_quantize(x: mx.array, bits: int = 8) -> mx.array:
    """
    Simulate uniform int<bits> quantisation on x.

    Clips to [-1, 1] (standard latent range), quantises to 2^bits levels,
    adds straight-through gradient via (x + noise - stop_gradient(noise)).
    Does NOT actually change the stored dtype.
    """
    n_levels = 2 ** bits - 1
    x_clamp  = mx.clip(x, -1.0, 1.0)
    step     = 2.0 / n_levels
    q        = mx.round(x_clamp / step) * step
    # Straight-through estimator: gradient passes through unmodified
    noise    = mx.stop_gradient(q - x_clamp)
    return x_clamp + noise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_student(cfg: dict) -> StudentEncoder:
    """
    Construct a StudentEncoder from a config dict.

    Expected keys (all optional, defaults shown):
        student.channels         = [64, 128, 256, 256]
        student.layers_per_block = 1
        student.norm_groups      = 16
        student.latent_channels  = 32
        student.use_mid_attention = true
    """
    sc = cfg.get("student", {})
    return StudentEncoder(
        channels         = sc.get("channels",          [64, 128, 256, 256]),
        layers_per_block = sc.get("layers_per_block",  1),
        norm_groups      = sc.get("norm_groups",       16),
        latent_channels  = sc.get("latent_channels",   32),
        use_mid_attention = sc.get("use_mid_attention", True),
    )
