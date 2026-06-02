"""
train/vae_distill/student.py — Lightweight proxy VAE encoder.

Architecture (Option B / Option D hybrid from plans/precomp2-proxy-vae-design.md):
  Mirrors the Flux2Encoder block structure (ResBlocks + GroupNorm + stride-2
  Downsample) at reduced channel widths.  This preserves the teacher's inductive
  bias (same spatial layout, same normalisation, same residual pattern).

Model variants (choose via build_student_small / build_student_medium):

  small   [48, 96, 192, 192]  1 layer/block  ~3.3M params   maximum speed
  default [64,128, 256, 256]  1 layer/block  ~6.0M params   speed/quality balance
  medium  [80,160, 320, 320]  1 layer/block  ~9.1M params   better fidelity

Performance on M1 Max (512px, batch=8):
  After mx.compile warm-up and BF16 weight conversion, expect ~10-20ms/image.
  Call student.to_bfloat16() and student.make_compiled() before inference.

Spatial schedule for 512px input (all variants, mirrors teacher):
  After block 0 (stride-2): [B, C0, 256, 256]
  After block 1 (stride-4): [B, C1, 128, 128]
  After block 2 (stride-8): [B, C2,  64,  64]
  After block 3 (stride-8): [B, C3,  64,  64]  no downsample
  Final latent:              [B, 32,  64,  64]
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Model variant presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "small": {
        "channels":          [48, 96, 192, 192],
        "layers_per_block":  1,
        "norm_groups":       12,
        "latent_channels":   32,
        "use_mid_attention": True,
    },
    "default": {
        "channels":          [64, 128, 256, 256],
        "layers_per_block":  1,
        "norm_groups":       16,
        "latent_channels":   32,
        "use_mid_attention": True,
    },
    "medium": {
        "channels":          [80, 160, 320, 320],
        "layers_per_block":  1,
        "norm_groups":       16,
        "latent_channels":   32,
        "use_mid_attention": True,
    },
}


# ---------------------------------------------------------------------------
# Building blocks  (mflux conventions: external [B,C,H,W], internal [B,H,W,C])
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
    def __init__(self, in_ch: int, out_ch: int, num_layers: int,
                 groups: int, downsample: bool):
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
    """Single-head self-attention at the bottleneck for spatial coherence."""

    def __init__(self, ch: int, groups: int = 16, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(ch, ch, bias=False)
        self.to_k = nn.Linear(ch, ch, bias=False)
        self.to_v = nn.Linear(ch, ch, bias=False)
        self.out  = nn.Linear(ch, ch, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.transpose(x, (0, 2, 3, 1))
        B, H, W, C = h.shape
        n = self.norm(h.astype(mx.float32)).astype(x.dtype)
        q = self.to_q(n).reshape(B, H * W, 1, C)
        k = self.to_k(n).reshape(B, H * W, 1, C)
        v = self.to_v(n).reshape(B, H * W, 1, C)
        q, k, v = (mx.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=float(C) ** -0.5)
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
    channels         : 4-element list — output channels per down-block
    layers_per_block : ResBlocks per down-block (teacher uses 2)
    norm_groups      : GroupNorm groups (teacher uses 32)
    latent_channels  : output channels (must match teacher: 32)
    use_mid_attention: include self-attention in mid block
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

        self.down_blocks = [
            _DownBlock(
                in_ch=channels[i - 1] if i > 0 else channels[0],
                out_ch=channels[i],
                num_layers=layers_per_block,
                groups=norm_groups,
                downsample=(i < 3),
            )
            for i in range(4)
        ]

        self.mid      = _MidBlock(channels[3], norm_groups, use_mid_attention)
        self.norm_out = nn.GroupNorm(norm_groups, channels[3], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(channels[3], latent_channels * 2, 3, padding=1)
        self.quant    = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1, padding=0)

    def _c(self, conv: nn.Conv2d, x: mx.array) -> mx.array:
        """Run a Conv2d on a [B,C,H,W] tensor, return [B,C,H,W]."""
        x = mx.transpose(x, (0, 2, 3, 1))
        x = conv(x)
        return mx.transpose(x, (0, 3, 1, 2))

    def __call__(
        self,
        x: mx.array,
        return_features: bool = False,
        qat_bits: Optional[int] = None,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        """
        Forward pass.

        x              : [B, 3, H, W] float32 in [-1, 1]
        return_features: also return list of 4 intermediate [B,Ci,Hi,Wi] maps
        qat_bits       : simulate int<qat_bits> quantisation noise (QAT only)

        Returns latent [B, 32, H/8, W/8] or (latent, features) if return_features.
        """
        h = self._c(self.conv_in, x)
        feats: list[mx.array] = []
        for db in self.down_blocks:
            h = db(h)
            if return_features:
                feats.append(h)

        h = self.mid(h)
        if return_features:
            feats[3] = h

        h = mx.transpose(h, (0, 2, 3, 1))
        h = self.norm_out(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = mx.transpose(h, (0, 3, 1, 2))
        h = self._c(self.conv_out, h)
        h = self._c(self.quant, h)

        mean, _ = mx.split(h, 2, axis=1)
        if qat_bits is not None:
            mean = _fake_quantize(mean, bits=qat_bits)

        return (mean, feats) if return_features else mean

    # ── M1 Max optimisation helpers ─────────────────────────────────────────

    def to_bfloat16(self) -> "StudentEncoder":
        """
        Cast all Conv2d and Linear weights to bfloat16.

        GroupNorm weights stay float32 (they are cast internally in _ResBlock
        before normalisation for numerical stability).  BF16 convolutions run
        ~2x faster on M1 Max and halve activation memory.

        Returns self for chaining.
        """
        import mlx.utils as mu
        def _cast(params):
            return mu.tree_map(
                lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) else v,
                params,
            )
        self.update(_cast(self.parameters()))
        mx.eval(self.parameters())
        return self

    def make_compiled(self) -> Callable:
        """
        Return a compiled version of the forward pass via mx.compile.

        mx.compile traces the graph on first call and caches a Metal kernel
        per input shape.  Subsequent calls with the same shape skip Python
        tracing overhead (~10-30% throughput improvement for repeated encodes).

        Usage:
            _encode = student.make_compiled()
            latent  = _encode(image_bchw)
        """
        mx.eval(self.parameters())   # ensure weights are materialized first
        return mx.compile(lambda x: self(x))

    def quantize_attention(self, group_size: int = 64, bits: int = 8) -> "StudentEncoder":
        """
        INT-quantize the attention Linear layers (to_q, to_k, to_v, out).

        Only the _SelfAttention module contains nn.Linear layers; all Conv2d
        layers are unaffected.  Reduces attention memory and may improve speed
        on quantization-capable hardware.  Typical memory saving: <5% of total.

        Returns self for chaining.
        """
        if self.mid.attn is not None:
            nn.quantize(self.mid.attn, group_size=group_size, bits=bits)
            mx.eval(self.mid.attn.parameters())
        return self

    def param_count(self) -> int:
        import mlx.utils as mu
        return sum(v.size for _, v in mu.tree_flatten(self.parameters()))


# ---------------------------------------------------------------------------
# Fake quantisation for QAT
# ---------------------------------------------------------------------------

def _fake_quantize(x: mx.array, bits: int = 8) -> mx.array:
    """
    Simulate uniform int<bits> quantisation via straight-through estimator.

    Uses per-batch min/max as the quantisation range (adapts to the actual
    latent distribution, which is roughly [-13, +10] for Flux VAE latents).
    """
    n_levels = 2 ** bits - 1
    x_min  = mx.stop_gradient(mx.min(x))
    x_max  = mx.stop_gradient(mx.max(x))
    x_rng  = mx.maximum(x_max - x_min, mx.array(1e-6))
    step   = x_rng / n_levels
    q      = mx.round((x - x_min) / step) * step + x_min
    noise  = mx.stop_gradient(q - x)
    return x + noise


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_student(cfg: dict) -> "StudentEncoder":
    """Construct StudentEncoder from a config dict (reads cfg["student"])."""
    sc = cfg.get("student", {})
    variant = sc.get("variant")
    if variant and variant in PRESETS:
        base = dict(PRESETS[variant])
        base.update({k: v for k, v in sc.items() if k != "variant"})
        sc = base
    return StudentEncoder(
        channels          = sc.get("channels",          [64, 128, 256, 256]),
        layers_per_block  = sc.get("layers_per_block",  1),
        norm_groups       = sc.get("norm_groups",       16),
        latent_channels   = sc.get("latent_channels",   32),
        use_mid_attention = sc.get("use_mid_attention", True),
    )


def build_student_small() -> "StudentEncoder":
    """~3.3M param variant — maximum inference speed."""
    return StudentEncoder(**PRESETS["small"])


def build_student_medium() -> "StudentEncoder":
    """~9.1M param variant — better fidelity than default."""
    return StudentEncoder(**PRESETS["medium"])
