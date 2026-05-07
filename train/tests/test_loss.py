"""
train/tests/test_loss.py — Unit tests for flow matching loss functions.

Tests correctness of:
  - get_schedule_values: linear schedule boundary and midpoint values
  - fused_flow_noise: v-prediction formula (noisy, target)
  - flow_matching_loss: MSE scalar, zero when velocity == target
  - per-sample timestep broadcast over [B, C, H, W]
  - gram_matrix / gram_style_loss: shape, symmetry, zero-input, gradient
  - Metal kernel path vs MLX fallback: parity within bf16 tolerance
  - Numerical stability: finite outputs at boundary timesteps and extreme inputs
"""

import numpy as np
import pytest

import mlx.core as mx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import (
    get_schedule_values,
    fused_flow_noise,
    flow_matching_loss,
    gram_matrix,
    gram_style_loss,
    _HAS_METAL_KERNEL,
)


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


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------

class TestGramMatrix:
    def test_output_shape(self):
        """[B, C, H, W] → [B, C, C]."""
        x = mx.random.normal((2, 8, 4, 4))
        g = gram_matrix(x)
        mx.eval(g)
        assert g.shape == (2, 8, 8)

    def test_is_symmetric(self):
        """G = G^T for every element in the batch."""
        rng = np.random.default_rng(10)
        x = mx.array(rng.standard_normal((2, 8, 4, 4)).astype(np.float32))
        g = gram_matrix(x)
        mx.eval(g)
        g_np = np.array(g)
        assert np.allclose(g_np, g_np.transpose(0, 2, 1), atol=1e-5)

    def test_diagonal_nonnegative(self):
        """Diagonal entries of the Gram matrix are always >= 0 (squared L2 norms)."""
        rng = np.random.default_rng(11)
        x = mx.array(rng.standard_normal((2, 8, 4, 4)).astype(np.float32))
        g = gram_matrix(x)
        mx.eval(g)
        g_np = np.array(g)
        for b in range(g_np.shape[0]):
            assert np.all(np.diag(g_np[b]) >= -1e-5)

    def test_zero_input_gives_zero_gram(self):
        x = mx.zeros((1, 4, 4, 4))
        g = gram_matrix(x)
        mx.eval(g)
        assert float(mx.max(mx.abs(g)).item()) < 1e-7

    def test_normalization_scale(self):
        """gram_matrix divides by C*H*W so magnitude scales with 1/(C*H*W)."""
        rng = np.random.default_rng(12)
        B, C, H, W = 1, 4, 3, 3
        x = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
        g = gram_matrix(x)
        # Manual: f = x.reshape(B, C, HW); G = f @ f.T / (C*H*W)
        x_np = np.array(x)
        f = x_np.reshape(B, C, H * W)
        expected = np.matmul(f, f.transpose(0, 2, 1)) / (C * H * W)
        mx.eval(g)
        assert np.allclose(np.array(g), expected, atol=1e-5)

    def test_batch_independence(self):
        """Each element in the batch is processed independently."""
        rng = np.random.default_rng(13)
        x = mx.array(rng.standard_normal((3, 4, 4, 4)).astype(np.float32))
        g_batch = gram_matrix(x)
        g_solo = gram_matrix(x[1:2])
        mx.eval(g_batch, g_solo)
        assert np.allclose(np.array(g_batch[1]), np.array(g_solo[0]), atol=1e-5)


# ---------------------------------------------------------------------------
# gram_style_loss
# ---------------------------------------------------------------------------

class TestGramStyleLoss:
    def test_scalar_output(self):
        x = mx.random.normal((1, 4, 4, 4))
        y = mx.random.normal((1, 4, 4, 4))
        loss = gram_style_loss(x, y)
        mx.eval(loss)
        assert loss.shape == ()

    def test_same_input_gives_zero_loss(self):
        """gram_style_loss(x, x) must be exactly zero."""
        rng = np.random.default_rng(20)
        x = mx.array(rng.standard_normal((2, 8, 4, 4)).astype(np.float32))
        loss = gram_style_loss(x, x)
        mx.eval(loss)
        assert float(loss.item()) < 1e-8

    def test_different_inputs_give_positive_loss(self):
        rng = np.random.default_rng(21)
        x = mx.array(rng.standard_normal((2, 8, 4, 4)).astype(np.float32))
        y = mx.array(rng.standard_normal((2, 8, 4, 4)).astype(np.float32))
        loss = gram_style_loss(x, y)
        mx.eval(loss)
        assert float(loss.item()) > 0.0

    def test_output_finite(self):
        """No NaN or Inf for typical latent-scale inputs."""
        rng = np.random.default_rng(22)
        x = mx.array(rng.standard_normal((2, 32, 8, 8)).astype(np.float32))
        y = mx.array(rng.standard_normal((2, 32, 8, 8)).astype(np.float32))
        loss = gram_style_loss(x, y)
        mx.eval(loss)
        assert np.isfinite(float(loss.item()))

    def test_gradient_computable(self):
        """Loss gradient with respect to x0_pred flows without NaN."""
        rng = np.random.default_rng(23)
        x = mx.array(rng.standard_normal((1, 8, 4, 4)).astype(np.float32))
        ref = mx.array(rng.standard_normal((1, 8, 4, 4)).astype(np.float32))

        def fn(v):
            return gram_style_loss(v, ref)

        grad = mx.grad(fn)(x)
        mx.eval(grad)
        assert grad.shape == x.shape
        assert np.isfinite(float(mx.max(mx.abs(grad)).item()))
        assert float(mx.max(mx.abs(grad)).item()) > 0

    def test_x0_reconstruction_math(self):
        """
        Verify the reconstruction identity used in the style loss:
          x0_pred = alpha * x_t - sigma * v_pred
        = alpha*(alpha*x0+sigma*eps) - sigma*(alpha*eps-sigma*x0)
        = (alpha^2 + sigma^2) * x0
        For the linear schedule alpha=1-t/1000, sigma=t/1000, the factor is
        (1-t/1000)^2 + (t/1000)^2, not 1.  The style loss accepts approximate
        x0 (gradient signal is scale-invariant for the Gram MSE).
        """
        rng = np.random.default_rng(24)
        x0 = mx.array(rng.standard_normal((1, 8, 4, 4)).astype(np.float32))
        noise = mx.array(rng.standard_normal((1, 8, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([600]))
        noisy, target = fused_flow_noise(x0, noise, alpha, sigma)

        a = float(alpha[0])
        s = float(sigma[0])
        x0_rec = a * noisy - s * target
        mx.eval(x0_rec)
        # Reconstruction gives (a^2 + s^2) * x0, not x0 itself.
        scale = a ** 2 + s ** 2
        expected = scale * np.array(x0)
        assert np.allclose(np.array(x0_rec), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Metal kernel path vs MLX fallback
# ---------------------------------------------------------------------------

class TestFusedFlowNoiseKernelPaths:
    """
    Explicitly exercise both dispatch paths in fused_flow_noise:
      Metal path : B=1, scalar alpha/sigma (alpha_t.size == 1), _HAS_METAL_KERNEL=True
      MLX path   : B>1 always uses the MLX element-wise fallback
    """

    def test_b1_path_correct_formula(self):
        """B=1 (potential Metal path) gives the exact v-prediction formula."""
        rng = np.random.default_rng(30)
        latent = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha  = mx.array([0.6])
        sigma  = mx.array([0.4])
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        exp_noisy  = 0.6 * latent + 0.4 * noise
        exp_target = 0.6 * noise  - 0.4 * latent
        mx.eval(noisy, target, exp_noisy, exp_target)
        # bf16 FMA vs separate mul+add: up to ~2 × bf16 eps tolerance
        BF16_TOL = 0.016
        assert float(mx.max(mx.abs(noisy  - exp_noisy)).item())  < BF16_TOL
        assert float(mx.max(mx.abs(target - exp_target)).item()) < BF16_TOL

    def test_b_gt1_path_correct_formula(self):
        """B>1 forces MLX fallback — formula must still be correct."""
        rng = np.random.default_rng(31)
        B = 4
        latent = mx.array(rng.standard_normal((B, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((B, 32, 4, 4)).astype(np.float32))
        t = mx.array([100, 300, 600, 900])
        alpha, sigma = get_schedule_values(t)
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        mx.eval(noisy, target)
        # Verify each sample
        for i in range(B):
            a, s = float(alpha[i]), float(sigma[i])
            exp_n = a * np.array(latent[i]) + s * np.array(noise[i])
            exp_t = a * np.array(noise[i])  - s * np.array(latent[i])
            assert np.allclose(np.array(noisy[i]),  exp_n, atol=1e-5)
            assert np.allclose(np.array(target[i]), exp_t, atol=1e-5)

    def test_b1_b2_parity(self):
        """
        B=1 (Metal path when available) and B=2 fallback must agree within bf16 tol.
        Run the same timestep twice via B=2 to match what two independent B=1 calls give.
        """
        rng = np.random.default_rng(32)
        lat = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noi = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha_s = mx.array([0.7])
        sigma_s = mx.array([0.3])

        # B=1 call (may use Metal kernel)
        noisy1, target1 = fused_flow_noise(lat, noi, alpha_s, sigma_s)

        # B=2 call (always MLX fallback): duplicate batch
        lat2   = mx.concatenate([lat, lat], axis=0)
        noi2   = mx.concatenate([noi, noi], axis=0)
        alpha2 = mx.concatenate([alpha_s, alpha_s], axis=0)
        sigma2 = mx.concatenate([sigma_s, sigma_s], axis=0)
        noisy2, target2 = fused_flow_noise(lat2, noi2, alpha2, sigma2)

        mx.eval(noisy1, target1, noisy2, target2)
        BF16_TOL = 0.016
        assert float(mx.max(mx.abs(noisy1  - noisy2[0])).item())  < BF16_TOL
        assert float(mx.max(mx.abs(target1 - target2[0])).item()) < BF16_TOL

    def test_metal_kernel_availability_reported(self):
        """_HAS_METAL_KERNEL is a bool (True on Apple Silicon + macOS 13.2+)."""
        assert isinstance(_HAS_METAL_KERNEL, bool)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestLossNumericalStability:
    def _run_fused(self, latent, noise, alpha, sigma):
        noisy, target = fused_flow_noise(latent, noise, alpha, sigma)
        mx.eval(noisy, target)
        return noisy, target

    def test_near_zero_latents_no_nan(self):
        rng = np.random.default_rng(40)
        latent = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32)) * 1e-7
        noise  = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32)) * 1e-7
        alpha, sigma = get_schedule_values(mx.array([500, 500]))
        noisy, target = self._run_fused(latent, noise, alpha, sigma)
        assert np.all(np.isfinite(np.array(noisy.astype(mx.float32))))
        assert np.all(np.isfinite(np.array(target.astype(mx.float32))))

    def test_moderate_latents_no_nan(self):
        rng = np.random.default_rng(41)
        latent = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32)) * 3.0
        noise  = mx.array(rng.standard_normal((2, 32, 4, 4)).astype(np.float32))
        alpha, sigma = get_schedule_values(mx.array([250, 750]))
        noisy, target = self._run_fused(latent, noise, alpha, sigma)
        assert np.all(np.isfinite(np.array(noisy.astype(mx.float32))))
        assert np.all(np.isfinite(np.array(target.astype(mx.float32))))

    def test_boundary_t0_no_nan(self):
        rng = np.random.default_rng(42)
        latent = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha = mx.array([1.0])
        sigma = mx.array([0.0])
        noisy, target = self._run_fused(latent, noise, alpha, sigma)
        assert np.all(np.isfinite(np.array(noisy.astype(mx.float32))))
        assert np.all(np.isfinite(np.array(target.astype(mx.float32))))

    def test_boundary_t1000_no_nan(self):
        rng = np.random.default_rng(43)
        latent = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha = mx.array([0.0])
        sigma = mx.array([1.0])
        noisy, target = self._run_fused(latent, noise, alpha, sigma)
        assert np.all(np.isfinite(np.array(noisy.astype(mx.float32))))
        assert np.all(np.isfinite(np.array(target.astype(mx.float32))))

    def test_flow_loss_gradient_finite_at_t0(self):
        """Gradient through loss is finite even at t=0 (sigma=0, pure signal)."""
        rng = np.random.default_rng(44)
        vel    = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        latent = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        noise  = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32))
        alpha  = mx.array([1.0])
        sigma  = mx.array([0.0])
        grad = mx.grad(lambda v: flow_matching_loss(v, latent, noise, alpha, sigma))(vel)
        mx.eval(grad)
        assert np.all(np.isfinite(np.array(grad)))

    def test_gram_style_loss_no_nan_small_inputs(self):
        rng = np.random.default_rng(45)
        x = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32)) * 1e-3
        y = mx.array(rng.standard_normal((1, 32, 4, 4)).astype(np.float32)) * 1e-3
        loss = gram_style_loss(x, y)
        mx.eval(loss)
        assert np.isfinite(float(loss.item()))
