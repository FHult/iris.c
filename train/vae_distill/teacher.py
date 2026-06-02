"""
train/vae_distill/teacher.py — Flux VAE encoder wrapper for distillation.

Loads only the VAE weights (same as precompute_all._load_flux_vae_only) and
wraps the encoder to expose per-block intermediate features for feature-level
distillation loss.  All feature tensors are in [B, C, H, W] format.

Teacher spatial schedule for 512px input:
  f0: [B, 128, 256, 256]  — after down_blocks[0] (stride-2 output)
  f1: [B, 256, 128, 128]  — after down_blocks[1] (stride-4 output)
  f2: [B, 512,  64,  64]  — after down_blocks[2] (stride-8 output)
  f3: [B, 512,  64,  64]  — after down_blocks[3] + mid_block (stride-8, refined)
  latent: [B,  32,  64,  64]  — final encoder output (mean only, scaled)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------

class TeacherEncoder(nn.Module):
    """
    Wraps the mflux Flux2VAE encoder to expose intermediate block features.

    After load(), call encode_with_features(image) to get both the final
    latent and the list of intermediate feature maps.

    Usage:
        teacher = TeacherEncoder.load(flux_model_path)
        latent, feats = teacher.encode_with_features(image_bchw)
        # feats: list of 4 mx.arrays, [f0..f3] in [B,C,H,W]
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self._vae = vae

    @staticmethod
    def load(flux_model_path: str) -> "TeacherEncoder":
        """Load Flux VAE weights from flux_model_path (the Klein 4B model dir)."""
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mflux.models.flux2.weights.flux2_weight_definition import (
            Flux2KleinWeightDefinition,
        )
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        vae_comp = next(
            c for c in Flux2KleinWeightDefinition.get_components()
            if c.name == "vae"
        )

        class _VaeOnly:
            @staticmethod
            def get_components():
                return [vae_comp]
            @staticmethod
            def get_download_patterns():
                return ["vae/*.safetensors", "vae/*.json"]

        weights = WeightLoader.load(weight_definition=_VaeOnly,
                                    model_path=flux_model_path)
        vae = Flux2VAE()
        WeightApplier.apply_and_quantize_single(
            weights=weights, model=vae, component=vae_comp, quantize_arg=None,
        )
        vae.eval()
        mx.eval(vae.parameters())
        print(f"  [teacher] Flux2VAE loaded from {flux_model_path}")
        return TeacherEncoder(vae)

    def encode(self, image: mx.array) -> mx.array:
        """Standard encode — returns [B, 32, H/8, W/8] latent."""
        return self._vae.encode(image)

    def encode_with_features(
        self, image: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Encode image and return (latent, intermediate_features).

        intermediate_features: 4 tensors in [B, C, H, W]:
          [0]: after down_blocks[0] — stride-2
          [1]: after down_blocks[1] — stride-4
          [2]: after down_blocks[2] — stride-8
          [3]: after down_blocks[3] + mid_block — stride-8 refined
        """
        enc = self._vae.encoder

        h = enc.conv_in(image)
        feats: list[mx.array] = []
        for down_block in enc.down_blocks:
            h = down_block(h)
            feats.append(h)

        h = enc.mid_block(h)
        feats[3] = h  # replace last entry with post-mid-block features

        h = enc.conv_norm_out(h)
        h = nn.silu(h)
        h = enc.conv_out(h)

        # Apply quant_conv (expects [B, H, W, C] layout)
        h_t = mx.transpose(h, (0, 2, 3, 1))
        h_t = self._vae.quant_conv(h_t)
        h = mx.transpose(h_t, (0, 3, 1, 2))

        mean, _ = mx.split(h, 2, axis=1)
        latent = (mean - self._vae.shift_factor) * self._vae.scaling_factor

        return latent, feats

    def decode(self, latent: mx.array) -> mx.array:
        """Decode latent back to image space for perceptual loss."""
        return self._vae.decode(latent)

    @property
    def scaling_factor(self) -> float:
        return self._vae.scaling_factor

    @property
    def shift_factor(self) -> float:
        return self._vae.shift_factor
