"""
train/tests/test_export.py — Smoke tests for export_adapter.py.

Creates a synthetic checkpoint (matching the expected training key layout),
runs the full export pipeline (bfloat16 and int8 modes), and validates the
output bundle. No real checkpoint or model weights required.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from export.export_adapter import (
    _KEY_MAP,
    _QUANTISED_TENSORS,
    _infer_dims,
    _quantise_int8_tensor,
    _to_bfloat16_numpy,
    apply_quant,
    apply_style_only,
    load_checkpoint,
    validate_bundle,
    export,
)


# ---------------------------------------------------------------------------
# Synthetic checkpoint helpers
# ---------------------------------------------------------------------------

# Tiny dimensions matching Flux Klein 4B structure but much smaller
_D  = 64     # hidden_dim  (real: 3072)
_S  = 32     # siglip_dim  (real: 1152)
_Q  = 4      # num_image_tokens (real: 16)
_N  = 25     # num_blocks  (real: 25)


def _make_synthetic_weights() -> dict[str, np.ndarray]:
    """Return a float32 weight dict matching training checkpoint key layout."""
    rng = np.random.default_rng(0)
    return {
        "image_proj.query_tokens":                  rng.standard_normal((_Q, _D)).astype(np.float32),
        "image_proj.cross_attn.query_proj.weight":  rng.standard_normal((_D, _D)).astype(np.float32),
        "image_proj.cross_attn.key_proj.weight":    rng.standard_normal((_D, _S)).astype(np.float32),
        "image_proj.cross_attn.value_proj.weight":  rng.standard_normal((_D, _S)).astype(np.float32),
        "image_proj.cross_attn.out_proj.weight":    rng.standard_normal((_D, _D)).astype(np.float32),
        "image_proj.norm.weight":                   rng.standard_normal((_D,)).astype(np.float32),
        "image_proj.norm.bias":                     np.zeros(_D, dtype=np.float32),
        "to_k_ip_stacked":                          rng.standard_normal((_N, _D, _D)).astype(np.float32),
        "to_v_ip_stacked":                          rng.standard_normal((_N, _D, _D)).astype(np.float32),
        "scale":                                    np.ones(_N, dtype=np.float32),
    }


def _write_synthetic_checkpoint(path: str, weights: dict[str, np.ndarray]) -> None:
    """Save synthetic weights as a safetensors file using MLX."""
    import mlx.core as mx
    mlx_w = {k: mx.array(v) for k, v in weights.items()}
    mx.eval(*mlx_w.values())
    mx.save_safetensors(path, mlx_w)


# ---------------------------------------------------------------------------
# Tests: low-level utilities
# ---------------------------------------------------------------------------

class TestQuantUtils:

    def test_int8_quant_roundtrip_2d(self):
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((8, 16)).astype(np.float32)
        q, scale = _quantise_int8_tensor(arr)
        assert q.shape == (8, 16)
        assert q.dtype == np.int8
        assert scale.shape == (8,)
        assert np.all(scale > 0)
        # Dequant error should be < 1 LSB
        dq = q.astype(np.float32) * scale[:, None]
        assert np.allclose(arr, dq, atol=scale.max())

    def test_int8_quant_roundtrip_3d(self):
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((4, 8, 8)).astype(np.float32)
        q, scale = _quantise_int8_tensor(arr)
        assert q.shape == (4, 8, 8)
        assert scale.shape == (4, 8)

    def test_int8_zero_row_no_div_error(self):
        arr = np.zeros((4, 8), dtype=np.float32)
        q, scale = _quantise_int8_tensor(arr)
        assert np.all(q == 0)
        assert np.all(np.isfinite(scale))

    def test_bf16_roundtrip_finite(self):
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((16,)).astype(np.float32)
        bf16 = _to_bfloat16_numpy(arr)
        assert bf16.dtype == np.uint16
        assert bf16.shape == arr.shape
        # Reconstruct float32 from uint16 (BF16 = upper 16 bits of float32)
        f32 = np.zeros_like(arr)
        f32.view(np.uint32)[:] = bf16.astype(np.uint32) << 16
        assert np.allclose(arr, f32, atol=1e-2)

    def test_apply_quant_bfloat16_small_tensors_stay_f32(self):
        weights = {k: np.ones(1, dtype=np.float32) for k in
                   ("perceiver.norm_weight", "perceiver.norm_bias", "ip_scale")}
        out = apply_quant(weights, "bfloat16")
        for k in ("perceiver.norm_weight", "perceiver.norm_bias", "ip_scale"):
            assert out[k].dtype == np.float32, f"{k} should be F32"

    def test_apply_quant_int8_scale_shape(self):
        rng = np.random.default_rng(4)
        weights = {
            "ip_k_stacked": rng.standard_normal((3, 8, 8)).astype(np.float32),
            "ip_v_stacked": rng.standard_normal((3, 8, 8)).astype(np.float32),
            "ip_scale":     np.ones(3, dtype=np.float32),
        }
        out = apply_quant(weights, "int8")
        assert out["ip_k_stacked"].dtype == np.int8
        assert "ip_k_stacked.scale" in out
        assert out["ip_k_stacked.scale"].shape == (3, 8)

    def test_infer_dims(self):
        weights = _make_synthetic_weights()
        # Map to export keys
        export_w = {v: weights[k] for k, v in _KEY_MAP.items()}
        dims = _infer_dims(export_w)
        assert dims["num_blocks"]       == _N
        assert dims["hidden_dim"]       == _D
        assert dims["num_image_tokens"] == _Q
        assert dims["siglip_dim"]       == _S

    def test_apply_style_only_zeros_double_blocks(self):
        weights = {"ip_scale": np.ones(_N, dtype=np.float32)}
        apply_style_only(weights, num_double_blocks=5)
        assert np.all(weights["ip_scale"][:5] == 0.0)
        assert np.all(weights["ip_scale"][5:]  == 1.0)


# ---------------------------------------------------------------------------
# Tests: full export pipeline
# ---------------------------------------------------------------------------

class TestExportPipeline:

    @pytest.fixture(scope="class")
    def checkpoint(self, tmp_path_factory):
        """Write a synthetic checkpoint to a temp file, return the path."""
        ckpt_dir  = tmp_path_factory.mktemp("ckpt")
        ckpt_path = str(ckpt_dir / "step_001000.safetensors")
        _write_synthetic_checkpoint(ckpt_path, _make_synthetic_weights())
        return ckpt_path

    def test_load_checkpoint_live_keys(self, checkpoint):
        w = load_checkpoint(checkpoint, use_ema=False)
        assert set(w.keys()) == set(_KEY_MAP.values())
        assert all(v.dtype == np.float32 for v in w.values())

    def test_load_checkpoint_ema_missing_raises(self, checkpoint):
        with pytest.raises(ValueError, match="ema"):
            load_checkpoint(checkpoint, use_ema=True)

    def test_export_bfloat16_creates_bundle(self, checkpoint, tmp_path_factory):
        out_dir = str(tmp_path_factory.mktemp("bundle_bf16"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="bfloat16",
               use_ema=False, do_validate=False)
        assert os.path.exists(os.path.join(out_dir, "adapter_weights.safetensors"))
        assert os.path.exists(os.path.join(out_dir, "adapter_meta.json"))

    def test_export_bfloat16_meta_fields(self, checkpoint, tmp_path_factory):
        out_dir = str(tmp_path_factory.mktemp("bundle_bf16_meta"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="bfloat16")
        with open(os.path.join(out_dir, "adapter_meta.json")) as f:
            meta = json.load(f)
        required = {"version", "adapter_type", "num_blocks", "hidden_dim",
                    "num_image_tokens", "siglip_dim", "quant", "tensors"}
        assert required <= set(meta.keys())
        assert meta["num_blocks"] == _N
        assert meta["hidden_dim"] == _D
        assert meta["quant"] == "bfloat16"
        assert meta["training_step"] == 1000  # auto-detected from filename

    def test_export_bfloat16_validates(self, checkpoint, tmp_path_factory):
        out_dir = str(tmp_path_factory.mktemp("bundle_bf16_val"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="bfloat16",
               do_validate=True)
        ok = validate_bundle(out_dir, quant="bfloat16")
        assert ok

    def test_export_float16_validates(self, checkpoint, tmp_path_factory):
        out_dir = str(tmp_path_factory.mktemp("bundle_f16"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="float16",
               do_validate=True)
        ok = validate_bundle(out_dir, quant="float16")
        assert ok

    def test_export_int8_creates_scale_tensors(self, checkpoint, tmp_path_factory):
        import mlx.core as mx
        out_dir = str(tmp_path_factory.mktemp("bundle_int8"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="int8")
        loaded = mx.load(os.path.join(out_dir, "adapter_weights.safetensors"))
        for name in _QUANTISED_TENSORS:
            assert name in loaded,            f"missing {name}"
            assert f"{name}.scale" in loaded, f"missing {name}.scale"
        # Main weight should be INT8
        assert loaded["ip_k_stacked"].dtype == mx.int8

    def test_export_int8_validates(self, checkpoint, tmp_path_factory):
        out_dir = str(tmp_path_factory.mktemp("bundle_int8_val"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="int8",
               do_validate=True)
        ok = validate_bundle(out_dir, quant="int8")
        assert ok

    def test_export_style_only_zeros_scale(self, checkpoint, tmp_path_factory):
        import mlx.core as mx
        out_dir = str(tmp_path_factory.mktemp("bundle_styleonly"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="bfloat16",
               style_only=True)
        loaded = mx.load(os.path.join(out_dir, "adapter_weights.safetensors"))
        sc = np.array(loaded["ip_scale"].astype(mx.float32))
        # Double-stream blocks (0–4) should be zero
        assert np.all(sc[:5] == 0.0), f"expected zeros, got {sc[:5]}"

    def test_export_int8_dequant_finite(self, checkpoint, tmp_path_factory):
        """Dequantised values from INT8 export must be finite."""
        import mlx.core as mx
        out_dir = str(tmp_path_factory.mktemp("bundle_int8_dq"))
        export(checkpoint=checkpoint, out_dir=out_dir, quant="int8")
        loaded = mx.load(os.path.join(out_dir, "adapter_weights.safetensors"))
        q8  = np.array(loaded["perceiver.query_proj"])
        sc8 = np.array(loaded["perceiver.query_proj.scale"].astype(mx.float32))
        dq  = q8.astype(np.float32) * sc8[:, None]
        assert np.all(np.isfinite(dq))

    def test_validate_bundle_bad_dir_fails(self, tmp_path_factory):
        empty = str(tmp_path_factory.mktemp("empty"))
        ok = validate_bundle(empty, quant="bfloat16")
        assert not ok
