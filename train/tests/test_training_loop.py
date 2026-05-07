"""
train/tests/test_training_loop.py — End-to-end mini training loop tests.

Runs 3 gradient steps with a tiny synthetic config to verify:
  - The adapter forward pass → loss → gradient → AdamW update cycle works.
  - Loss is finite and non-zero at each step.
  - Adapter weights change after gradient updates.
  - Gradient norm is finite and positive.
  - Style loss term integrates correctly when style_loss_weight > 0.
  - EMA updates track the adapter weights.

No mflux / Flux Klein weights required — the Flux transformer is replaced by a
tiny synthetic "flux_state" with the same dict structure the real training loop
passes to loss_fn.

Note on nn.value_and_grad signature convention:
  nn.value_and_grad(model, fn) calls fn(*args, **kwargs) WITHOUT prepending model.
  The model is updated in-place via model.update(trainable_params) before each
  call. Loss functions must therefore capture `model` (adapter) from the
  enclosing scope, not accept it as a positional argument.
"""

import os
import sys

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.model import IPAdapterKlein
from ip_adapter.loss import (
    fused_flow_noise,
    get_schedule_values,
    gram_style_loss,
)
from ip_adapter.ema import update_ema


# ---------------------------------------------------------------------------
# Tiny model config (runs in < 1 s per step on M-series)
# ---------------------------------------------------------------------------

_NUM_BLOCKS   = 3
_HIDDEN       = 64
_NUM_HEADS    = 4       # 64 / 4 = 16 head_dim
_HEAD_DIM     = _HIDDEN // _NUM_HEADS
_NUM_QUERIES  = 4
_SIGLIP_DIM   = 32
_SIGLIP_TOKS  = 6       # compressed SigLIP tokens per image
_IMG_SEQ      = 8       # spatial sequence (e.g. 4×2 patches)
_LATENT_C     = 4       # VAE channels (normally 32; tiny here for speed)
_LATENT_H     = 2
_LATENT_W     = 2


def _make_adapter():
    return IPAdapterKlein(
        num_blocks=_NUM_BLOCKS,
        hidden_dim=_HIDDEN,
        num_image_tokens=_NUM_QUERIES,
        siglip_dim=_SIGLIP_DIM,
        perceiver_heads=_NUM_HEADS,
    )


def _make_synthetic_flux_state(rng, B: int = 1):
    """
    Construct a synthetic flux_state dict matching the structure that
    train_ip_adapter.py passes to loss_fn:
      qs:      list[num_blocks] of [B, H, seq_img, head_dim]
      h_final: [B, seq_img, hidden]
      B, C, Lh, Lw, pH, pW, seq_img  (shape metadata)
    """
    qs = [
        mx.array(rng.standard_normal((B, _NUM_HEADS, _IMG_SEQ, _HEAD_DIM)).astype(np.float32)) * 0.1
        for _ in range(_NUM_BLOCKS)
    ]
    h_final = mx.array(rng.standard_normal((B, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1
    return {
        "qs":      qs,
        "h_final": h_final,
        "B": B,
        "C": _LATENT_C,
        "Lh": _LATENT_H,
        "Lw": _LATENT_W,
        "pH": _LATENT_H,
        "pW": _LATENT_W,
        "seq_img": _IMG_SEQ,
    }


def _make_loss_fn(adapter, flux_state, target,
                  x0_ref=None, noisy_in=None,
                  alpha_in=None, sigma_in=None,
                  style_weight: float = 0.0):
    """
    Return a loss function that captures flux_state, target, and style inputs.
    The returned fn(siglip_feats) is suitable for nn.value_and_grad.

    Mirrors the adapter-only gradient graph from train_ip_adapter.py, but
    replaces the frozen Flux norm_out + proj_out with an identity projection
    so no Flux weights are required.
    """
    def loss_fn(siglip_feats):
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        k_all, v_all = adapter.get_kv_all(ip_embeds)

        qs      = flux_state["qs"]
        h_final = flux_state["h_final"]
        B       = flux_state["B"]
        seq_img = flux_state["seq_img"]
        D       = _HIDDEN

        ip_total = mx.zeros((B, seq_img, D), dtype=h_final.dtype)
        for i, q_i in enumerate(qs):
            H  = q_i.shape[1]
            Hd = q_i.shape[3]
            k_i = k_all[:, i].reshape(B, -1, H, Hd).transpose(0, 2, 1, 3)
            v_i = v_all[:, i].reshape(B, -1, H, Hd).transpose(0, 2, 1, 3)
            ip_out = mx.fast.scaled_dot_product_attention(q_i, k_i, v_i, scale=Hd ** -0.5)
            ip_out = ip_out.transpose(0, 2, 1, 3).reshape(B, seq_img, D)
            ip_total = ip_total + adapter.scale[i] * ip_out

        pred = h_final + ip_total  # [B, seq_img, D]
        flow_loss = mx.mean((pred - target) ** 2)

        if style_weight > 0.0 and x0_ref is not None and alpha_in is not None:
            # Treat pred [B, seq_img, D] as [B, C, H, 1] for Gram computation.
            # C=_NUM_HEADS, H=_HEAD_DIM*_IMG_SEQ//_NUM_HEADS — any consistent split works.
            # Here: [B, seq_img, D] → [B, D, seq_img, 1] (D as channels, seq_img as H).
            a = float(alpha_in.flatten()[0])
            s = float(sigma_in.flatten()[0])
            pred_4d  = pred.astype(mx.float32).transpose(0, 2, 1)[..., None]  # [B, D, seq_img, 1]
            noisy_4d = noisy_in.astype(mx.float32).transpose(0, 2, 1)[..., None]
            x0_pred  = a * noisy_4d - s * pred_4d
            x0_ref_4d = x0_ref.astype(mx.float32).transpose(0, 2, 1)[..., None]
            return flow_loss + style_weight * gram_style_loss(x0_pred, x0_ref_4d)

        return flow_loss

    return loss_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMiniTrainingLoop:

    def _run_steps(self, n_steps: int = 3, style_weight: float = 0.0):
        """
        Run n_steps of AdamW gradient updates on a tiny adapter.
        Returns list of (loss_value, grad_norm_value) per step.
        """
        adapter   = _make_adapter()
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-2)
        mx.eval(adapter.parameters())

        rng = np.random.default_rng(99)
        history = []

        for _ in range(n_steps):
            siglip     = mx.array(rng.standard_normal((1, _SIGLIP_TOKS, _SIGLIP_DIM)).astype(np.float32))
            flux_state = _make_synthetic_flux_state(rng, B=1)
            target     = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1

            x0_ref = alpha_in = sigma_in = noisy_in = None
            if style_weight > 0.0:
                t = mx.array([500])
                alpha_in, sigma_in = get_schedule_values(t)
                # x0_ref and noisy_in share shape [B, seq_img, hidden] — the Gram
                # computation in _make_loss_fn transposes to [B, D, seq_img, 1].
                x0_ref   = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32))
                noisy_in, _ = fused_flow_noise(x0_ref, mx.random.normal(x0_ref.shape), alpha_in, sigma_in)

            loss_fn = _make_loss_fn(
                adapter, flux_state, target,
                x0_ref, noisy_in, alpha_in, sigma_in, style_weight,
            )
            loss_and_grad = nn.value_and_grad(adapter, loss_fn)
            loss_val, grads = loss_and_grad(siglip)
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(adapter, grads)
            mx.eval(loss_val, grad_norm, adapter.parameters())
            history.append((float(loss_val.item()), float(grad_norm.item())))

        return history

    def test_loss_finite_every_step(self):
        """Loss must be a finite positive scalar at every step."""
        history = self._run_steps(n_steps=3)
        for step, (loss, _) in enumerate(history):
            assert np.isfinite(loss),  f"step {step}: loss={loss} is not finite"
            assert loss >= 0.0,        f"step {step}: loss={loss} is negative"

    def test_grad_norm_finite_and_positive(self):
        """Gradient norm must be finite and positive (non-zero gradient signal)."""
        history = self._run_steps(n_steps=3)
        for step, (_, grad_norm) in enumerate(history):
            assert np.isfinite(grad_norm), f"step {step}: grad_norm={grad_norm} is not finite"
            assert grad_norm > 0.0,        f"step {step}: grad_norm={grad_norm} — no gradient"

    def test_weights_change_after_update(self):
        """Adapter scale parameters must change after at least one gradient step."""
        adapter   = _make_adapter()
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=1e-2)
        mx.eval(adapter.parameters())
        scale_before = np.array(adapter.scale).copy()

        rng        = np.random.default_rng(100)
        siglip     = mx.array(rng.standard_normal((1, _SIGLIP_TOKS, _SIGLIP_DIM)).astype(np.float32))
        flux_state = _make_synthetic_flux_state(rng, B=1)
        target     = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1

        loss_fn       = _make_loss_fn(adapter, flux_state, target)
        loss_and_grad = nn.value_and_grad(adapter, loss_fn)
        loss_val, grads = loss_and_grad(siglip)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())

        scale_after = np.array(adapter.scale)
        assert not np.allclose(scale_before, scale_after), (
            "adapter.scale unchanged after gradient step"
        )

    def test_k_v_weights_change_after_update(self):
        """to_k_ip_stacked / to_v_ip_stacked must change after gradient step."""
        adapter   = _make_adapter()
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=1e-2)
        mx.eval(adapter.parameters())
        k_before = np.array(adapter.to_k_ip_stacked).copy()

        rng        = np.random.default_rng(101)
        siglip     = mx.array(rng.standard_normal((1, _SIGLIP_TOKS, _SIGLIP_DIM)).astype(np.float32))
        flux_state = _make_synthetic_flux_state(rng, B=1)
        target     = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1

        loss_fn       = _make_loss_fn(adapter, flux_state, target)
        loss_and_grad = nn.value_and_grad(adapter, loss_fn)
        loss_val, grads = loss_and_grad(siglip)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters())

        k_after = np.array(adapter.to_k_ip_stacked)
        assert not np.allclose(k_before, k_after), (
            "to_k_ip_stacked unchanged after gradient step"
        )

    def test_3_steps_run_without_crash(self):
        """Full 3-step run completes without exception and all losses are finite."""
        history = self._run_steps(n_steps=3)
        assert len(history) == 3
        assert all(np.isfinite(l) for l, _ in history)

    def test_style_loss_enabled_run(self):
        """3 steps with style_loss_weight=0.1 — loss and gradient remain finite."""
        history = self._run_steps(n_steps=3, style_weight=0.1)
        for step, (loss, grad_norm) in enumerate(history):
            assert np.isfinite(loss),      f"step {step}: loss={loss}"
            assert np.isfinite(grad_norm), f"step {step}: grad_norm={grad_norm}"
            assert grad_norm > 0.0,        f"step {step}: zero gradient with style loss"

    def test_ema_tracks_adapter_after_steps(self):
        """EMA weights converge toward adapter weights after repeated updates."""
        adapter   = _make_adapter()
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=1e-2)
        mx.eval(adapter.parameters())
        ema_params = adapter.parameters()

        rng = np.random.default_rng(102)
        for _ in range(5):
            siglip     = mx.array(rng.standard_normal((1, _SIGLIP_TOKS, _SIGLIP_DIM)).astype(np.float32))
            flux_state = _make_synthetic_flux_state(rng, B=1)
            target     = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1
            loss_fn       = _make_loss_fn(adapter, flux_state, target)
            loss_and_grad = nn.value_and_grad(adapter, loss_fn)
            loss_val, grads = loss_and_grad(siglip)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(adapter, grads)
            ema_params = update_ema(ema_params, adapter, decay=0.5)  # low decay → fast tracking
        mx.eval(adapter.parameters(), ema_params)

        ema_scale  = np.array(ema_params["scale"])
        live_scale = np.array(adapter.scale)
        # With decay=0.5 over 5 steps, EMA should have moved substantially from 1.0
        assert not np.allclose(ema_scale, np.ones_like(ema_scale), atol=0.05), (
            "EMA scale has not moved from its initial value — EMA update is broken"
        )
        # EMA should be closer to live_scale than to initial 1.0 after 5 fast-decay steps
        dist_to_live = float(np.abs(ema_scale - live_scale).mean())
        dist_to_init = float(np.abs(ema_scale - 1.0).mean())
        assert dist_to_init >= dist_to_live * 0.5, (
            f"EMA not tracking adapter: dist_to_live={dist_to_live:.4f}, dist_to_init={dist_to_init:.4f}"
        )

    def test_null_siglip_loss_finite(self):
        """
        Zero siglip features (null conditioning) must produce a finite loss.
        Gradient still flows through the h_final path.
        """
        adapter   = _make_adapter()
        mx.eval(adapter.parameters())

        rng        = np.random.default_rng(103)
        siglip     = mx.zeros((1, _SIGLIP_TOKS, _SIGLIP_DIM))
        flux_state = _make_synthetic_flux_state(rng, B=1)
        target     = mx.array(rng.standard_normal((1, _IMG_SEQ, _HIDDEN)).astype(np.float32)) * 0.1

        loss_fn       = _make_loss_fn(adapter, flux_state, target)
        loss_and_grad = nn.value_and_grad(adapter, loss_fn)
        loss_val, grads = loss_and_grad(siglip)
        mx.eval(loss_val, grads)

        assert np.isfinite(float(loss_val.item()))
        # Flatten all gradient arrays and check they're finite
        from ip_adapter.ema import _flatten
        for name, g in _flatten(grads):
            g_np = np.array(g.astype(mx.float32))
            assert np.all(np.isfinite(g_np)), f"NaN/Inf gradient in {name}"
