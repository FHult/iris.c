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
# Gram matrix style loss
# ---------------------------------------------------------------------------

def gram_matrix(x: mx.array) -> mx.array:
    """Normalized Gram matrix. x: float32 [B, C, H, W] → [B, C, C]"""
    B, C, H, W = x.shape
    f = x.reshape(B, C, H * W)
    return mx.matmul(f, f.transpose(0, 2, 1)) / (C * H * W)


def gram_style_loss(x0_pred: mx.array, x0_ref: mx.array) -> mx.array:
    """
    MSE between normalized Gram matrices of predicted and reference clean latents.
    Both inputs: float32 [B, C, H, W].

    Reconstruct x0_pred from velocity v_pred and noisy latent x_t:
      x0_pred = alpha_t * x_t - sigma_t * v_pred
    (exact because alpha_t^2 + sigma_t^2 = 1 in flow matching)
    """
    return mx.mean((gram_matrix(x0_pred) - gram_matrix(x0_ref)) ** 2)


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
    alpha_t: mx.array,
    sigma_t: mx.array,
):
    """
    Compute (noisy, target) for v-prediction flow matching.

    latent:  clean latent [B, C, H, W], bfloat16 or float32
    noise:   Gaussian noise, same shape as latent
    alpha_t: [B] or scalar mx.array — noise schedule alpha
    sigma_t: [B] or scalar mx.array — noise schedule sigma

    For per-sample timesteps (alpha_t shape [B]), broadcasts over [B, C, H, W].

    Returns:
      noisy:  alpha_t * latent + sigma_t * noise
      target: alpha_t * noise  - sigma_t * latent  (v-prediction target)
    """
    # Reshape scalar [B] to [B,1,1,1] for broadcast over [B,C,H,W]
    while alpha_t.ndim < latent.ndim:
        alpha_t = alpha_t[..., None]
    while sigma_t.ndim < latent.ndim:
        sigma_t = sigma_t[..., None]

    noisy = alpha_t * latent + sigma_t * noise
    target = alpha_t * noise - sigma_t * latent
    return noisy, target


def get_schedule_values(t_int: mx.array):
    """
    Convert integer timestep t in [0, 1000] to (alpha_t, sigma_t).
    Linear flow matching: alpha = 1 - t/1000, sigma = t/1000.
    Returns mx.arrays of the same shape as t_int (no GPU→CPU sync).
    """
    t_frac = t_int.astype(mx.float32) / 1000.0
    return 1.0 - t_frac, t_frac


def flow_matching_loss(
    model_velocity: mx.array,
    latent: mx.array,
    noise: mx.array,
    alpha_t: mx.array,
    sigma_t: mx.array,
) -> mx.array:
    """
    Scalar MSE loss between predicted velocity and v-prediction target.
    """
    _, target = fused_flow_noise(latent, noise, alpha_t, sigma_t)
    diff = model_velocity - target
    return mx.mean(diff * diff)
