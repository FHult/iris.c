"""
train/ip_adapter/loss.py — Flow matching loss for IP-Adapter training.

Uses v-prediction flow matching (matching Flux Klein's training objective):
  noisy  = alpha_t * latent + sigma_t * noise
  target = alpha_t * noise  - sigma_t * latent   (v-prediction)
  loss   = MSE(model_velocity, target)

The fused Metal kernel computes both noisy and target in one pass,
saving one Metal dispatch and one temporary allocation vs two separate ops.

Based on the kernel specification in plans/ip-adapter-training.md §3.2.
"""

import mlx.core as mx


# ---------------------------------------------------------------------------
# Fused Metal kernel — v-prediction noise scheduler
# Matches plans/ip-adapter-training.md §3.2 exactly.
# Inputs: latent, noise, alpha (scalar arr), sigma (scalar arr)
# Output: noisy = alpha*latent + sigma*noise, target = alpha*noise - sigma*latent
# ---------------------------------------------------------------------------
_FUSED_FLOW_NOISE_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= latent.size) return;
    float l = latent[i], n = noise[i];
    float a = alpha[0],  s = sigma[0];
    noisy[i]  = a * l + s * n;
    target[i] = a * n - s * l;
"""

try:
    _fused_flow_kernel = mx.fast.metal_kernel(
        name="fused_flow_noise",
        input_names=["latent", "noise", "alpha", "sigma"],
        output_names=["noisy", "target"],
        source=_FUSED_FLOW_NOISE_SOURCE,
    )
    _HAS_METAL_KERNEL = True
except Exception:
    _HAS_METAL_KERNEL = False


def fused_flow_noise(
    latent: mx.array,
    noise: mx.array,
    alpha_t: float,
    sigma_t: float,
):
    """
    Compute (noisy, target) for v-prediction flow matching in one Metal pass.

    latent:  clean latent [B, C, H, W] or any shape, float32
    noise:   Gaussian noise, same shape as latent
    alpha_t: noise schedule alpha value (e.g. 1-t for linear flow)
    sigma_t: noise schedule sigma value (e.g. t for linear flow)

    Returns:
      noisy:  alpha_t * latent + sigma_t * noise
      target: alpha_t * noise  - sigma_t * latent  (v-prediction target)
    """
    if _HAS_METAL_KERNEL:
        flat = latent.reshape(-1).astype(mx.float32)
        flat_n = noise.reshape(-1).astype(mx.float32)
        alpha_arr = mx.array([alpha_t], dtype=mx.float32)
        sigma_arr = mx.array([sigma_t], dtype=mx.float32)
        n = flat.shape[0]

        noisy_flat, target_flat = _fused_flow_kernel(
            inputs=[flat, flat_n, alpha_arr, sigma_arr],
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
        )
        noisy = noisy_flat.reshape(latent.shape).astype(latent.dtype)
        target = target_flat.reshape(latent.shape).astype(latent.dtype)
    else:
        noisy = alpha_t * latent + sigma_t * noise
        target = alpha_t * noise - sigma_t * latent

    return noisy, target


def get_schedule_values(t_int: mx.array):
    """
    Convert integer timestep t in [0, 1000] to (alpha_t, sigma_t).
    Linear flow matching: alpha = 1 - t/1000, sigma = t/1000.
    """
    t_frac = t_int.astype(mx.float32) / 1000.0
    alpha_t = 1.0 - t_frac
    sigma_t = t_frac
    return float(alpha_t.item()), float(sigma_t.item())


def flow_matching_loss(
    model_velocity: mx.array,
    latent: mx.array,
    noise: mx.array,
    alpha_t: float,
    sigma_t: float,
) -> mx.array:
    """
    Scalar MSE loss between predicted velocity and v-prediction target.
    """
    _, target = fused_flow_noise(latent, noise, alpha_t, sigma_t)
    diff = model_velocity - target
    return mx.mean(diff * diff)
