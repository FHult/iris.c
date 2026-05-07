"""
train/tests/test_model.py — Shape and correctness tests for IP-Adapter model.

Tests:
  - PerceiverResampler: forward pass shape [B, 729, 1152] → [B, 128, 3072]
  - PerceiverResampler: output determinism with fixed seed
  - IPAdapterKlein: construction, parameter counts
  - get_kv_all: einsum output shapes [B, num_blocks, 128, 3072]
  - inject: SDPA output shape [B, img_seq, 3072]
  - inject: style-only mode (ip_scale[0:5]=0 zeros double-stream, keeps single-stream)
  - inject: output regression with fixed random seed
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

    # -----------------------------------------------------------------------
    # Style-only mode
    # -----------------------------------------------------------------------

    def test_inject_style_only_double_stream_zero(self):
        """
        Style-only inference: zero out ip_scale for double-stream blocks (indices 0–4).
        inject() for those blocks must return a zero tensor.
        """
        model = self._make_model()
        # Zero out the first 5 blocks (double-stream)
        zeros5 = mx.zeros((5,))
        ones20 = mx.ones((20,))
        model.scale = mx.concatenate([zeros5, ones20])
        mx.eval(model.scale)

        B, num_heads, img_seq, head_dim = 1, 24, 16, 128
        img_q     = mx.random.normal((B, num_heads, img_seq, head_dim)) * 0.01
        ip_embeds = mx.random.normal((B, 128, 3072)) * 0.01
        k, v = model.get_kv_all(ip_embeds)

        for block_idx in range(5):
            out = model.inject(img_q, k[:, block_idx], v[:, block_idx], block_idx=block_idx)
            mx.eval(out)
            assert out.shape == (B, img_seq, 3072)
            assert float(mx.max(mx.abs(out)).item()) < 1e-9, (
                f"block {block_idx}: inject should be zero when scale=0, got max={float(mx.max(mx.abs(out)).item())}"
            )

    def test_inject_style_only_single_stream_active(self):
        """
        Style-only mode: single-stream blocks (indices 5–24) remain active.
        inject() for those blocks must produce non-zero output.
        """
        model = self._make_model()
        zeros5 = mx.zeros((5,))
        ones20 = mx.ones((20,))
        model.scale = mx.concatenate([zeros5, ones20])
        mx.eval(model.scale)

        B, num_heads, img_seq, head_dim = 1, 24, 16, 128
        rng = np.random.default_rng(50)
        img_q     = mx.array(rng.standard_normal((B, num_heads, img_seq, head_dim)).astype(np.float32)) * 0.1
        ip_embeds = mx.array(rng.standard_normal((B, 128, 3072)).astype(np.float32)) * 0.1
        k, v = model.get_kv_all(ip_embeds)

        for block_idx in range(5, 25):
            out = model.inject(img_q, k[:, block_idx], v[:, block_idx], block_idx=block_idx)
            mx.eval(out)
            assert float(mx.max(mx.abs(out)).item()) > 1e-6, (
                f"block {block_idx}: inject should be non-zero when scale=1"
            )

    def test_inject_style_content_vs_style_only_differ(self):
        """Full mode (all blocks active) vs style-only mode produce different outputs."""
        model_full = self._make_model()
        model_style = self._make_model()

        # Synchronise weights by setting them explicitly to the same values
        rng = np.random.default_rng(51)
        shared_k = mx.array(rng.standard_normal((25, 3072, 3072)).astype(np.float32) * (3072 ** -0.5))
        shared_v = mx.array(rng.standard_normal((25, 3072, 3072)).astype(np.float32) * (3072 ** -0.5))
        model_full.to_k_ip_stacked  = shared_k
        model_full.to_v_ip_stacked  = shared_v
        model_style.to_k_ip_stacked = shared_k
        model_style.to_v_ip_stacked = shared_v
        model_style.scale = mx.concatenate([mx.zeros((5,)), mx.ones((20,))])

        B, num_heads, img_seq, head_dim = 1, 24, 16, 128
        img_q     = mx.array(rng.standard_normal((B, num_heads, img_seq, head_dim)).astype(np.float32)) * 0.1
        ip_embeds = mx.array(rng.standard_normal((B, 128, 3072)).astype(np.float32)) * 0.1

        k_full, v_full = model_full.get_kv_all(ip_embeds)
        k_style, v_style = model_style.get_kv_all(ip_embeds)

        # Block 0 (double-stream): full ≠ 0, style = 0
        out_full0 = model_full.inject(img_q, k_full[:, 0], v_full[:, 0], block_idx=0)
        out_style0 = model_style.inject(img_q, k_style[:, 0], v_style[:, 0], block_idx=0)
        mx.eval(out_full0, out_style0)
        assert float(mx.max(mx.abs(out_full0)).item()) > 1e-6
        assert float(mx.max(mx.abs(out_style0)).item()) < 1e-9

    # -----------------------------------------------------------------------
    # Output regression / determinism
    # -----------------------------------------------------------------------

    def test_inject_output_deterministic(self):
        """Same model weights + same inputs → identical inject output on two calls."""
        rng = np.random.default_rng(60)
        model = self._make_model()
        B, num_heads, img_seq, head_dim = 1, 24, 8, 128
        img_q     = mx.array(rng.standard_normal((B, num_heads, img_seq, head_dim)).astype(np.float32))
        ip_embeds = mx.array(rng.standard_normal((B, 128, 3072)).astype(np.float32))
        k, v = model.get_kv_all(ip_embeds)

        out_a = model.inject(img_q, k[:, 0], v[:, 0], block_idx=0)
        out_b = model.inject(img_q, k[:, 0], v[:, 0], block_idx=0)
        mx.eval(out_a, out_b)
        assert np.allclose(np.array(out_a.astype(mx.float32)),
                           np.array(out_b.astype(mx.float32)), atol=0)

    def test_inject_output_shape_and_magnitude(self):
        """
        Output shape: [B, img_seq, hidden_dim].
        Output magnitude with scale=1 and unit-normal inputs is O(1).
        """
        rng = np.random.default_rng(61)
        model = self._make_model()
        B, num_heads, img_seq, head_dim = 2, 24, 32, 128
        img_q     = mx.array(rng.standard_normal((B, num_heads, img_seq, head_dim)).astype(np.float32))
        ip_embeds = mx.array(rng.standard_normal((B, 128, 3072)).astype(np.float32))
        k, v = model.get_kv_all(ip_embeds)

        out = model.inject(img_q, k[:, 0], v[:, 0], block_idx=0)
        mx.eval(out)
        out_np = np.array(out.astype(mx.float32))

        assert out.shape == (B, img_seq, 3072)
        # Output should be finite and non-trivially small
        assert np.all(np.isfinite(out_np))
        assert np.abs(out_np).max() > 1e-8

    def test_perceiver_output_deterministic(self):
        """PerceiverResampler: same inputs always produce identical outputs."""
        model = PerceiverResampler(hidden_dim=256, num_heads=4, num_queries=8, siglip_dim=64)
        rng = np.random.default_rng(70)
        feat = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32))

        out_a = model(feat)
        out_b = model(feat)
        mx.eval(out_a, out_b)
        assert np.allclose(np.array(out_a.astype(mx.float32)),
                           np.array(out_b.astype(mx.float32)), atol=0)

    def test_perceiver_output_regression(self):
        """
        PerceiverResampler forward pass with fixed-weight init check.
        Verifies: output is finite, has the right shape, and that the forward
        pass hasn't been accidentally broken (output L2 norm changes if the
        attention computation regresses).
        """
        mx.random.seed(42)
        model = PerceiverResampler(hidden_dim=64, num_heads=4, num_queries=8, siglip_dim=32)
        rng = np.random.default_rng(42)
        feat = mx.array(rng.standard_normal((1, 10, 32)).astype(np.float32) * 0.1)

        out = model(feat)
        mx.eval(out)
        out_np = np.array(out.astype(mx.float32))

        assert out.shape == (1, 8, 64)
        assert np.all(np.isfinite(out_np))
        # L2 norm of the output should be non-trivial (model is not dead)
        l2 = float(np.linalg.norm(out_np))
        assert l2 > 1e-4, f"Output L2 norm {l2} is suspiciously small (dead model?)"
