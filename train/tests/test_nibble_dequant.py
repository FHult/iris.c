"""
train/tests/test_nibble_dequant.py — 4-bit nibble pack/unpack sign round-trip.

Guards the dequantisation sign bug that was the true root cause of the
patch-periodic grid (BUGS.md IP-ADAPTER-INFER-1). `precompute_all._quantize_4bit`
stores TWO'S-COMPLEMENT signed nibbles (values in [-8, 7]); the cache loaders in
ip_adapter.dataset MUST sign-extend nibbles >= 8 back to negative. The original
loaders masked with `& 0x0F` and read every negative quantised value as a large
positive — producing SigLIP/Qwen3 features ANTI-correlated with the truth, which
the IP-Adapter then trained on.

These tests are hermetic: pure numpy, no weights, no data volumes (flywheel-safe).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ip_adapter.dataset import (  # noqa: E402
    _unpack_signed_nibbles,
    _load_siglip_embed,
    _load_qwen3_embed,
)
from precompute_all import _quantize_4bit  # noqa: E402


def test_sign_extend_known_nibbles():
    # one byte per (low, high) nibble pair; verify two's-complement decode
    # low nibble of 0xF8 = 8 -> -8 ; high nibble = 0xF -> -1
    q = np.array([[0xF8]], dtype=np.uint8)
    full = _unpack_signed_nibbles(q)
    assert full[0, 0] == -8        # low nibble 8 sign-extends to -8
    assert full[0, 1] == -1        # high nibble 15 sign-extends to -1
    # positive nibbles pass through unchanged
    q2 = np.array([[0x71]], dtype=np.uint8)   # low=1, high=7
    full2 = _unpack_signed_nibbles(q2)
    assert full2[0, 0] == 1 and full2[0, 1] == 7


def test_quantize_roundtrip_is_positively_correlated():
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 5.0, size=(64, 256)).astype(np.float32)
    arr[:, 7] = 80.0  # massive-activation outlier dim (SigLIP-like)
    q, scale = _quantize_4bit(arr)
    full = _unpack_signed_nibbles(q)
    deq = full.astype(np.float32) * scale.astype(np.float32)
    corr = np.corrcoef(arr.ravel(), deq.ravel())[0, 1]
    # the buggy `& 0x0F` decode gives corr ~ -0.5; the correct decode must be high
    assert corr > 0.8, f"dequant corr {corr:.3f} too low — sign extension regressed"
    # signed data must round-trip to signed data, not all-positive
    assert (deq < 0).mean() > 0.05, "dequant produced no negatives — sign bug back"


def test_loaders_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.normal(0.0, 3.0, size=(32, 128)).astype(np.float32)
    q, scale = _quantize_4bit(arr)
    np.savez(tmp_path / "rec.npz", q=q, scale=scale)
    for loader in (_load_siglip_embed, _load_qwen3_embed):
        out = np.array(loader("rec", str(tmp_path))).astype(np.float32)
        corr = np.corrcoef(arr.ravel(), out.ravel())[0, 1]
        assert corr > 0.8, f"{loader.__name__} corr {corr:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
