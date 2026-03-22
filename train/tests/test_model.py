"""
train/tests/test_model.py — Shape and correctness tests for IP-Adapter model.

Tests:
  - PerceiverResampler: forward pass shape [B, 729, 1152] → [B, 128, 3072]
  - IPAdapterKlein: construction, parameter counts
  - get_kv_all: einsum output shapes [B, num_blocks, 128, 3072]
  - inject: SDPA output shape [B, img_seq, 3072]
"""

import os
import sys

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.model import PerceiverResampler, IPAdapterKlein


# ---------------------------------------------------------------------------
# PerceiverResampler
# ---------------------------------------------------------------------------

class TestPerceiverResampler:
    def _make_model(self):
        return PerceiverResampler(
            hidden_dim=3072,
            num_heads=24,
            num_queries=128,
            siglip_dim=1152,
        )

    def test_output_shape(self):
        model = self._make_model()
        B = 2
        siglip = mx.random.normal((B, 729, 1152))
        out = model(siglip)
        mx.eval(out)
        assert out.shape == (B, 128, 3072)

    def test_batch_size_1(self):
        model = self._make_model()
        siglip = mx.random.normal((1, 729, 1152))
        out = model(siglip)
        mx.eval(out)
        assert out.shape == (1, 128, 3072)

    def test_output_dtype(self):
        model = self._make_model()
        siglip = mx.random.normal((1, 729, 1152))
        out = model(siglip)
        mx.eval(out)
        assert out.dtype in (mx.float32, mx.bfloat16, mx.float16)

    def test_output_finite(self):
        """No NaN or Inf in initial output (random init)."""
        model = self._make_model()
        siglip = mx.random.normal((2, 729, 1152)) * 0.1
        out = model(siglip)
        mx.eval(out)
        arr = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(arr)), "PerceiverResampler output contains NaN/Inf"

    def test_different_batches_differ(self):
        """Model is not returning a constant — output varies with input."""
        model = self._make_model()
        rng = np.random.default_rng(99)
        a = mx.array(rng.standard_normal((1, 729, 1152)).astype(np.float32))
        b = mx.array(rng.standard_normal((1, 729, 1152)).astype(np.float32))
        out_a = model(a)
        out_b = model(b)
        mx.eval(out_a, out_b)
        assert not np.allclose(np.array(out_a.astype(mx.float32)),
                               np.array(out_b.astype(mx.float32)))


# ---------------------------------------------------------------------------
# IPAdapterKlein
# ---------------------------------------------------------------------------

class TestIPAdapterKlein:
    def _make_model(self, num_blocks=25, hidden_dim=3072):
        return IPAdapterKlein(
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            num_image_tokens=128,
            siglip_dim=1152,
            perceiver_heads=24,
        )

    def test_construction(self):
        model = self._make_model()
        assert model.num_blocks == 25
        assert model.hidden_dim == 3072
        assert model.num_image_tokens == 128

    def test_stacked_weight_shapes(self):
        model = self._make_model()
        mx.eval(model.to_k_ip_stacked, model.to_v_ip_stacked)
        assert model.to_k_ip_stacked.shape == (25, 3072, 3072)
        assert model.to_v_ip_stacked.shape == (25, 3072, 3072)

    def test_scale_initial_value(self):
        model = self._make_model()
        mx.eval(model.scale)
        s = np.array(model.scale)
        assert s.shape == (25,)
        assert np.allclose(s, 1.0, atol=1e-5)

    def test_get_image_embeds_shape(self):
        model = self._make_model()
        B = 2
        siglip = mx.random.normal((B, 729, 1152))
        ip_embeds = model.get_image_embeds(siglip)
        mx.eval(ip_embeds)
        assert ip_embeds.shape == (B, 128, 3072)

    def test_get_kv_all_shapes(self):
        model = self._make_model()
        B = 2
        ip_embeds = mx.random.normal((B, 128, 3072)) * 0.01
        k, v = model.get_kv_all(ip_embeds)
        mx.eval(k, v)
        # [B, num_blocks, T, D]
        assert k.shape == (B, 25, 128, 3072)
        assert v.shape == (B, 25, 128, 3072)

    def test_get_kv_all_k_v_differ(self):
        """K and V use separate weight matrices — outputs must differ."""
        model = self._make_model()
        ip_embeds = mx.random.normal((1, 128, 3072)) * 0.01
        k, v = model.get_kv_all(ip_embeds)
        mx.eval(k, v)
        k_arr = np.array(k.astype(mx.float32))
        v_arr = np.array(v.astype(mx.float32))
        assert not np.allclose(k_arr, v_arr), "K and V should differ (different weight matrices)"

    def test_inject_output_shape(self):
        model = self._make_model()
        B = 2
        num_heads = 24
        head_dim = 128  # 3072 / 24
        img_seq = 256   # e.g. 512×512 patchified image tokens

        # img_q: [B, num_heads, img_seq, head_dim]
        img_q = mx.random.normal((B, num_heads, img_seq, head_dim)) * 0.01
        ip_embeds = mx.random.normal((B, 128, 3072)) * 0.01
        k, v = model.get_kv_all(ip_embeds)

        # Slice one block's K/V: [B, 128, 3072]
        k_block = k[:, 0, :, :]
        v_block = v[:, 0, :, :]

        out = model.inject(img_q, k_block, v_block, block_idx=0)
        mx.eval(out)
        assert out.shape == (B, img_seq, 3072)

    def test_inject_scaled_by_block_scale(self):
        """inject output scales proportionally to self.scale[block_idx]."""
        model = self._make_model()
        B = 1
        num_heads, img_seq, head_dim = 24, 16, 128

        img_q = mx.random.normal((B, num_heads, img_seq, head_dim)) * 0.01
        ip_embeds = mx.random.normal((B, 128, 3072)) * 0.01
        k, v = model.get_kv_all(ip_embeds)
        k_block = k[:, 0, :, :]
        v_block = v[:, 0, :, :]

        # With scale=1
        out1 = model.inject(img_q, k_block, v_block, block_idx=0)
        mx.eval(out1)

        # Halve the scale
        model.scale = mx.concatenate([
            mx.array([0.5]),
            model.scale[1:],
        ])
        out2 = model.inject(img_q, k_block, v_block, block_idx=0)
        mx.eval(out1, out2)

        arr1 = np.array(out1.astype(mx.float32))
        arr2 = np.array(out2.astype(mx.float32))
        assert np.allclose(arr2, arr1 * 0.5, atol=1e-4)

    def test_smaller_model_for_fast_ci(self):
        """Smoke test with smaller dims (runs faster in CI)."""
        # 4-head, 256-hidden mini model
        model = IPAdapterKlein(
            num_blocks=3,
            hidden_dim=256,
            num_image_tokens=16,
            siglip_dim=64,
            perceiver_heads=4,
        )
        B = 1
        siglip = mx.random.normal((B, 10, 64))
        ip_embeds = model.get_image_embeds(siglip)
        k, v = model.get_kv_all(ip_embeds)
        mx.eval(ip_embeds, k, v)

        assert ip_embeds.shape == (B, 16, 256)
        assert k.shape == (B, 3, 16, 256)
        assert v.shape == (B, 3, 16, 256)
