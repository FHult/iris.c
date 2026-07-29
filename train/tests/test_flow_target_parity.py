"""
train/tests/test_flow_target_parity.py — golden-fixture guard for the flow-matching
TARGET, anchored to the mflux reference (the authoritative Flux.2 Klein teacher).

Why this exists: test_loss.py only checks reconstruct_x0 ∘ fused_flow_noise self-
consistency, which passes under EITHER the correct rectified-flow target or the wrong
v-prediction target that shipped until 2026-07-30. This fixture pins the target to the
reference objective so the v-prediction regression cannot silently return.

Reference (installed mflux, verified 2026-07-30):
  - interpolation: latent_creator.add_noise_by_interpolation = (1-sigma)*clean + sigma*noise
  - objective:     trainer.py:98  error = (clean + predicted - noise)^2
                   ⇒ the model output is trained to equal  noise - clean  (= noise - x0)
  - predict_noise (flux2_training_adapter.py:38-46) returns the RAW transformer output,
    and iris_sample.c integrates it as a velocity (z += dt*v) — so the training target
    MUST be the constant rectified velocity  noise - x0.
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.requires_mps  # exercises MLX arrays / Metal GPU runtime

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import fused_flow_noise, reconstruct_x0, get_schedule_values


def _batch(seed=0, B=1, C=32, H=8, W=8):
    rng = np.random.default_rng(seed)
    x0 = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
    noise = mx.array(rng.standard_normal((B, C, H, W)).astype(np.float32))
    return x0, noise


@pytest.mark.parametrize("t_int", [1, 100, 250, 500, 750, 900, 999])
def test_target_matches_mflux_rectified_velocity(t_int):
    """target == noise - x0 (constant), and mflux's error is minimized by it."""
    x0, noise = _batch()
    a, s = get_schedule_values(mx.array([t_int]))
    noisy, target = fused_flow_noise(x0, noise, a, s)

    noisy_ref = (1.0 - s) * x0 + s * noise          # mflux add_noise_by_interpolation
    target_ref = noise - x0                          # mflux objective (trainer.py:98)

    mx.eval(noisy, target, noisy_ref, target_ref)
    assert float(mx.max(mx.abs(noisy - noisy_ref))) < 1e-4, f"noisy mismatch @t={t_int}"
    assert float(mx.max(mx.abs(target - target_ref))) < 1e-4, f"target mismatch @t={t_int}"
    # mflux error = (clean + predicted - noise)^2 is exactly zero when predicted == target
    resid = x0 + target - noise
    mx.eval(resid)
    assert float(mx.max(mx.abs(resid))) < 1e-4, f"mflux residual nonzero @t={t_int}"


@pytest.mark.parametrize("t_int", [1, 250, 500, 750, 999])
def test_reconstruct_x0_exact(t_int):
    """x0 = noisy - sigma*v recovers the clean latent from the true velocity."""
    x0, noise = _batch(seed=1)
    a, s = get_schedule_values(mx.array([t_int]))
    noisy, target = fused_flow_noise(x0, noise, a, s)
    x0_rec = reconstruct_x0(noisy, target, a, s)
    mx.eval(x0_rec, x0)
    assert float(mx.max(mx.abs(x0_rec - x0))) < 1e-4, f"x0 mismatch @t={t_int}"


def test_target_is_constant_in_t():
    """The rectified velocity is independent of t — the v-pred bug made it t-dependent."""
    x0, noise = _batch(seed=2)
    tgts = []
    for t_int in [1, 500, 999]:
        a, s = get_schedule_values(mx.array([t_int]))
        _, target = fused_flow_noise(x0, noise, a, s)
        mx.eval(target)
        tgts.append(np.array(target))
    assert np.allclose(tgts[0], tgts[1], atol=1e-4)
    assert np.allclose(tgts[1], tgts[2], atol=1e-4)
