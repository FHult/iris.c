"""
train/tests/test_bn_pack.py — VAE-Q1 packed-space parity (hermetic, no weights).

The IP-adapter trains on raw VAE-latents (mflux encode(), no BatchNorm) but C
inference operates the transformer in the BN'd packed space. `_bn_pack_latents`
applies a per-128-feature BatchNorm in [32,Lh,Lw] space (feature = c*4+(h%2)*2+(w%2))
so the trainer's existing patchify pack yields exactly C `iris_vae_encode`'s packed
output (see BUGS.md VAE-Q1, debug/vae_parity.c).

This pins that convention without needing the 168 MB VAE weights: it proves the
algebraic identity the fix relies on —

    patchify( bn_pack(raw) )  ==  BN_128( patchify(raw) )

i.e. checkerboard-BN-then-patchify equals patchify-then-per-128-channel-BN, which
is what the C path (`iris_patchify` then `iris_batch_norm` on 128 features) does.
A regression in either the feature mapping or the patchify order breaks it.

Flywheel-safe: pure numpy + mlx, no GPU, no weights, no data volumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
mx = pytest.importorskip("mlx.core")
from train_ip_adapter import _bn_pack_latents


def _patchify_np(x: np.ndarray) -> np.ndarray:
    """Replicate iris_patchify / the trainer pack: [C,H,W] -> [C*4,H/2,W/2] with
    out_c = c*4 + pi*2 + pj, out[ph,pw] = in[ph*2+pi, pw*2+pj]."""
    C, H, W = x.shape
    pH, pW = H // 2, W // 2
    out = np.empty((C * 4, pH, pW), dtype=x.dtype)
    for c in range(C):
        for pi in range(2):
            for pj in range(2):
                out[c * 4 + pi * 2 + pj] = x[c, pi::2, pj::2]
    return out


def _bn128(packed: np.ndarray, bn_mean: np.ndarray, bn_var: np.ndarray,
           eps: float) -> np.ndarray:
    """Per-128-feature BatchNorm, as C iris_batch_norm applies to the patchified form."""
    m = bn_mean.reshape(-1, 1, 1)
    s = np.sqrt(bn_var.reshape(-1, 1, 1) + eps)
    return (packed - m) / s


def test_bn_pack_matches_patchify_then_bn():
    rng = np.random.default_rng(0)
    C, H, W, eps = 32, 16, 16, 1e-4
    raw = rng.standard_normal((C, H, W)).astype(np.float32) * 1.7
    bn_mean = rng.standard_normal(128).astype(np.float32) * 0.1
    bn_var = (rng.uniform(2.5, 3.5, size=128)).astype(np.float32)

    # training-side path: checkerboard BN in [32,H,W] space, then patchify
    bn_mean_r = mx.array(bn_mean.reshape(32, 2, 2))
    bn_std_r = mx.sqrt(mx.array(bn_var.reshape(32, 2, 2)) + eps)
    packed_fix = _patchify_np(
        np.array(_bn_pack_latents(mx.array(raw[None]), bn_mean_r, bn_std_r)[0])
    )

    # C-side reference: patchify, then per-128-feature BN
    packed_ref = _bn128(_patchify_np(raw), bn_mean, bn_var, eps)

    assert packed_fix.shape == (128, 8, 8) == packed_ref.shape
    assert np.abs(packed_fix - packed_ref).max() < 1e-4


def test_bn_pack_noop_without_stats():
    # No stats -> identity (defensive path when bn stats are unavailable).
    raw = np.ones((1, 32, 8, 8), dtype=np.float32)
    out = np.array(_bn_pack_latents(mx.array(raw), None, None))
    assert np.array_equal(out, raw)


def test_bn_pack_normalizes_scale():
    # A raw latent at std~1.7 should come out near unit std (the whole point: match
    # C's packed convention so training and inference share a scale).
    rng = np.random.default_rng(1)
    raw = (rng.standard_normal((1, 32, 16, 16)).astype(np.float32) * 1.7)
    bn_mean_r = mx.zeros((32, 2, 2))
    bn_std_r = mx.ones((32, 2, 2)) * 1.7
    out = np.array(_bn_pack_latents(mx.array(raw), bn_mean_r, bn_std_r))
    assert 0.8 < out.std() < 1.25
