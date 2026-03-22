"""
train/tests/test_loss.py — Unit tests for flow matching loss functions.

Tests correctness of:
  - get_schedule_values: linear schedule boundary and midpoint values
  - fused_flow_noise: v-prediction formula (noisy, target)
  - flow_matching_loss: MSE scalar, zero when velocity == target
  - per-sample timestep broadcast over [B, C, H, W]
"""

import numpy as np
import pytest

import mlx.core as mx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import get_schedule_values, fused_flow_noise, flow_matching_loss


# ---------------------------------------------------------------------------
# get_schedule_values
# ---------------------------------------------------------------------------

class TestGetScheduleValues:
    def test_t0_pure_signal(self):
        """t=0: alpha=1.0, sigma=0.0 — pure clean signal."""
        alpha, sigma = get_schedule_values(mx.array([0]))
        mx.eval(alpha, sigma)
        assert abs(float(alpha[0]) - 1.0) < 1e-6
        assert abs(float(sigma[0]) - 0.0) < 1e-6

    def test_t1000_pure_noise(self):
        """t=1000: alpha=0.0, sigma=1.0 — pure noise."""
        alpha, sigma = get_schedule_values(mx.array([1000]))
        mx.eval(alpha, sigma)
        assert abs(float(alpha[0]) - 0.0) < 1e-6
        assert abs(float(sigma[0]) - 1.0) < 1e-6

    def test_t500_midpoint(self):
        """t=500: alpha=0.5, sigma=0.5 — midpoint."""
        alpha, sigma = get_schedule_values(mx.array([500]))
        mx.eval(alpha, sigma)
        assert abs(float(alpha[0]) - 0.5) < 1e-5
        assert abs(float(sigma[0]) - 0.5) < 1e-5

    def test_alpha_plus_sigma_is_one(self):
        """alpha + sigma = 1 for all t (linear schedule identity)."""
        for t in [0, 100, 250, 500, 750, 999, 1000]:
            alpha, sigma = get_schedule_values(mx.array([t]))
            mx.eval(alpha, sigma)
            total = float(alpha[0]) + float(sigma[0])
            assert abs(total - 1.0) < 1e-5, f"alpha+sigma={total} at t={t}"

    def test_batch_timesteps(self):
        """Batch of timesteps returns correct shapes."""
        t = mx.array([0, 250, 500, 750, 1000])
        alpha, sigma = get_schedule_values(t)
        mx.eval(alpha, sigma)
        assert alpha.shape == (5,)
        assert sigma.shape == (5,)

    def test_output_dtype_float32(self):
        alpha, sigma = get_schedule_values(mx.array([500]))
        mx.eval(alpha, sigma)
        assert alpha.dtype == mx.float32
        assert sigma.dtype == mx.float32


# ---------------------------------------------------------------------------
# fused_flow_noise
# ---------------------------------------------------------------------------

class TestFusedFlowNoise:
    def _make_batch(self, B=2, C=32, H=4, W=4):
        rng = np.random.default_rng(42)
        latent = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        return latent, noise

    def test_noisy_formula(self):
        """noisy = alpha*latent + sigma*noise."""
        latent, noise = self._make_batch(B=1)
        alpha = mx.array([0.7])
        sigma = mx.array([0.3])
        noisy, _ = fused_flow_noise(latent, noise, alpha, sigma)
        expected = 0.7 * latent + 0.3 * noise
        mx.eval(noisy, expected)
        diff = mx.abs(noisy - expected)
        mx.eval(diff)
        assert float(mx.max(diff).item()) < 1e-5

    def test_target_formula(self):
        """target = alpha*noise - sigma*latent (v-prediction)."""
        latent, noise = self._make_batch(B=1)
        alpha = mx.array([0.7])
        sigma = mx.array([0.3])
        _, target = fused_flow_noise(latent, noise, alpha, sigma)
        expected = 0.7 * noise - 0.3 * latent
        mx.eval(target, expected)
        diff = mx.abs(target - expected)
        mx.eval(diff)
        assert float(mx.max(diff).item()) < 1e-5

    def test_t0_noisy_equals_latent(self):
        """At t=0: alpha=1, sigma=0 → noisy=latent, target=noise."""
        latent, noise = self._make_batch(B=1)
        alpha = mx.array([1.0])
        sigma = mx.array([0.0])
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        mx.eval(noisy, target, latent, noise)
        assert float(mx.max(mx.abs(noisy - latent)).item()) < 1e-5
        assert float(mx.max(mx.abs(target - noise)).item()) < 1e-5

    def test_t1000_noisy_equals_noise(self):
        """At t=1000: alpha=0, sigma=1 → noisy=noise, target=-latent."""
        latent, noise = self._make_batch(B=1)
        alpha = mx.array([0.0])
        sigma = mx.array([1.0])
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        mx.eval(noisy, target, latent, noise)
        assert float(mx.max(mx.abs(noisy - noise)).item()) < 1e-5
        assert float(mx.max(mx.abs(target + latent)).item()) < 1e-5

    def test_per_sample_broadcast(self):
        """Per-sample timesteps [B] broadcast correctly over [B, C, H, W]."""
        B, C, H, W = 4, 32, 4, 4
        rng = np.random.default_rng(7)
        latent = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))

        t = mx.array([0, 250, 500, 1000])
        alpha, sigma = get_schedule_values(t)
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        mx.eval(noisy, target)

        assert noisy.shape == (B, C, H, W)
        assert target.shape == (B, C, H, W)

        # Verify each sample independently
        for i, ti in enumerate([0, 250, 500, 1000]):
            a, s = 1 - ti / 1000, ti / 1000
            expected_noisy = a * latent[i] + s * noise[i]
            mx.eval(expected_noisy)
            diff = float(mx.max(mx.abs(noisy[i] - expected_noisy)).item())
            assert diff < 1e-5, f"sample {i} (t={ti}): max diff {diff}"

    def test_output_shapes_preserved(self):
        B, C, H, W = 3, 32, 8, 8
        rng = np.random.default_rng(0)
        latent = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        noisy, target = fused_flow_noise(
            latent, noise, mx.array([0.6]), mx.array([0.4])
        )
        mx.eval(noisy, target)
        assert noisy.shape == (B, C, H, W)
        assert target.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# flow_matching_loss
# ---------------------------------------------------------------------------

class TestFlowMatchingLoss:
    def test_scalar_output(self):
        """Loss is a scalar."""
        rng = np.random.default_rng(1)
        vel    = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        latent = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([500, 500]))
        loss = flow_matching_loss(vel, latent, noise, alpha, sigma)
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) >= 0.0

    def test_zero_when_velocity_equals_target(self):
        """Loss = 0 when model predicts the exact v-prediction target."""
        rng = np.random.default_rng(2)
        latent = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([300, 700]))
        _, target = fused_flow_noise(latent, noise, alpha, sigma)
        # Perfect prediction
        loss = flow_matching_loss(target, latent, noise, alpha, sigma)
        mx.eval(loss)
        assert float(loss.item()) < 1e-8

    def test_loss_increases_with_error(self):
        """Loss increases when the prediction is further from target."""
        rng = np.random.default_rng(3)
        latent = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([500, 500]))
        _, target = fused_flow_noise(latent, noise, alpha, sigma)

        small_err = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32)) * 0.01
        large_err = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32)) * 10.0

        loss_small = flow_matching_loss(target + small_err, latent, noise, alpha, sigma)
        loss_large = flow_matching_loss(target + large_err, latent, noise, alpha, sigma)
        mx.eval(loss_small, loss_large)
        assert float(loss_large.item()) > float(loss_small.item())

    def test_gradient_computable(self):
        """Loss.backward() does not crash — gradient flows through."""
        import mlx.core as mx

        rng = np.random.default_rng(4)
        vel    = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        latent = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([500]))

        def loss_fn(v):
            return flow_matching_loss(v, latent, noise, alpha, sigma)

        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(vel)
        mx.eval(grad)
        assert grad.shape == vel.shape
        # Gradient should not be all zeros
        assert float(mx.max(mx.abs(grad)).item()) > 0
