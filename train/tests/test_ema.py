"""
train/tests/test_ema.py — Unit tests for EMA weight averaging.

Tests:
  - update_ema: decay formula, interpolation between endpoints
  - save_ema / load_ema: round-trip through safetensors
  - _flatten: nested dict flattening
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.ema import update_ema, save_ema, load_ema, _flatten


# ---------------------------------------------------------------------------
# Minimal model for testing
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def __call__(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_flat_dict(self):
        params = {"a": mx.array([1.0]), "b": mx.array([2.0])}
        flat = _flatten(params)
        keys = [k for k, _ in flat]
        assert "a" in keys and "b" in keys
        assert len(flat) == 2

    def test_nested_dict(self):
        params = {"layer": {"weight": mx.array([1.0, 2.0])}}
        flat = _flatten(params)
        assert flat[0][0] == "layer.weight"

    def test_nested_list(self):
        params = {"blocks": [mx.array([1.0]), mx.array([2.0])]}
        flat = _flatten(params)
        keys = [k for k, _ in flat]
        assert "blocks.0" in keys and "blocks.1" in keys

    def test_returns_mx_arrays(self):
        params = {"w": mx.array([3.0, 4.0])}
        flat = _flatten(params)
        for _, v in flat:
            assert isinstance(v, mx.array)


# ---------------------------------------------------------------------------
# update_ema
# ---------------------------------------------------------------------------

class TestUpdateEma:
    def test_decay_formula(self):
        """ema_new = decay * ema_old + (1 - decay) * model."""
        model = _TinyModel()
        # Force model weights to all-ones
        w = mx.ones_like(model.linear.weight)
        model.linear.weight = w

        # EMA starts at all-zeros
        ema_params = {"linear": {"weight": mx.zeros_like(model.linear.weight)}}
        decay = 0.9

        ema_params = update_ema(ema_params, model, decay=decay)
        mx.eval(ema_params)

        expected = decay * 0.0 + (1 - decay) * 1.0  # = 0.1
        result = ema_params["linear"]["weight"]
        mx.eval(result)
        assert np.allclose(np.array(result), expected, atol=1e-5)

    def test_ema_converges_to_model_at_decay_zero(self):
        """decay=0 → EMA equals model weights instantly."""
        model = _TinyModel()
        model.linear.weight = mx.full(model.linear.weight.shape, 5.0)

        ema_params = {"linear": {"weight": mx.zeros_like(model.linear.weight)}}
        ema_params = update_ema(ema_params, model, decay=0.0)
        mx.eval(ema_params)

        result = np.array(ema_params["linear"]["weight"])
        assert np.allclose(result, 5.0, atol=1e-5)

    def test_ema_unchanged_at_decay_one(self):
        """decay=1 → EMA does not change."""
        model = _TinyModel()
        model.linear.weight = mx.full(model.linear.weight.shape, 99.0)

        init_val = mx.full(model.linear.weight.shape, 42.0)
        ema_params = {"linear": {"weight": init_val}}
        ema_params = update_ema(ema_params, model, decay=1.0)
        mx.eval(ema_params)

        result = np.array(ema_params["linear"]["weight"])
        assert np.allclose(result, 42.0, atol=1e-5)

    def test_repeated_updates_track_model(self):
        """After many updates at high decay, EMA should be close to model."""
        model = _TinyModel()
        target_val = 3.0
        model.linear.weight = mx.full(model.linear.weight.shape, target_val)

        ema_params = {"linear": {"weight": mx.zeros_like(model.linear.weight)}}
        decay = 0.9
        for _ in range(100):
            ema_params = update_ema(ema_params, model, decay=decay)
        mx.eval(ema_params)

        result = np.array(ema_params["linear"]["weight"])
        # After 100 steps at decay=0.9: should be very close to 3.0
        # 3.0 * (1 - 0.9^100) ≈ 3.0
        assert np.allclose(result, target_val, atol=0.01)

    def test_does_not_mutate_original_ema(self):
        """update_ema returns a new dict, doesn't modify in place."""
        model = _TinyModel()
        model.linear.weight = mx.ones_like(model.linear.weight)

        ema_w = mx.zeros_like(model.linear.weight)
        ema_params = {"linear": {"weight": ema_w}}
        new_ema = update_ema(ema_params, model, decay=0.9)
        mx.eval(new_ema)

        # Original ema_params reference should be unmodified by Python semantics
        # (mx.tree_map returns new arrays)
        orig_val = np.array(ema_w)
        assert np.allclose(orig_val, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# save_ema / load_ema round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadEma:
    def _make_ema(self):
        rng = np.random.default_rng(42)
        return {
            "layer0": {"weight": mx.array(rng.standard_normal((4, 4)).astype(np.float32))},
            "layer1": {"weight": mx.array(rng.standard_normal((8, 4)).astype(np.float32))},
        }

    def test_round_trip(self, tmp_path):
        ema = self._make_ema()
        path = str(tmp_path / "ema.safetensors")
        save_ema(ema, path)
        loaded = load_ema(path)
        assert os.path.exists(path)

        # Keys should be in flat dot-notation
        assert "layer0.weight" in loaded
        assert "layer1.weight" in loaded

    def test_values_preserved(self, tmp_path):
        ema = self._make_ema()
        mx.eval(ema)
        orig_w0 = np.array(ema["layer0"]["weight"])

        path = str(tmp_path / "ema.safetensors")
        save_ema(ema, path)
        loaded = load_ema(path)

        loaded_w0 = np.array(loaded["layer0.weight"])
        assert np.allclose(orig_w0, loaded_w0, atol=1e-6)

    def test_file_created(self, tmp_path):
        ema = self._make_ema()
        path = str(tmp_path / "ema_check.safetensors")
        assert not os.path.exists(path)
        save_ema(ema, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
