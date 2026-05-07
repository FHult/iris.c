"""
train/ip_adapter/loss.py — Flow matching loss for IP-Adapter training.

Uses v-prediction flow matching (matching Flux Klein's training objective):
  noisy  = alpha_t * latent + sigma_t * noise
  target = alpha_t * noise  - sigma_t * latent   (v-prediction)
  loss   = MSE(model_velocity, target)

fused_flow_noise has two paths:
  Metal kernel  — single GPU dispatch; used when B=1 (standard training batch).
  MLX fallback  — element-wise MLX ops; used for B>1 or non-Metal platforms.

Both paths produce numerically equivalent results within bfloat16 precision.
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
# Inputs: latent, noise, alpha (scalar [1]), sigma (scalar [1]), count (uint32 [1])
# Output: noisy = alpha*latent + sigma*noise, target = alpha*noise - sigma*latent
#
# count is passed explicitly because the Metal device pointer type does not
# expose a .size member; bounds-checking via a separate uint32 is the portable
# approach across MLX versions.
#
# Constraint: alpha[0] / sigma[0] are read as scalars, so this kernel only
# supports a single timestep per call (B=1). The MLX fallback handles B>1.
# ---------------------------------------------------------------------------
_FUSED_FLOW_NOISE_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float l = latent[i], nv = noise[i];
    float a = alpha[0],  s = sigma[0];
    noisy[i]  = a * l + s * nv;
    target[i] = a * nv - s * l;
"""

_fused_flow_kernel = None
_HAS_METAL_KERNEL = False
_kernel_compile_error: str = ""

try:
    _fused_flow_kernel = mx.fast.metal_kernel(
        name="fused_flow_noise",
        input_names=["latent", "noise", "alpha", "sigma", "count"],
        output_names=["noisy", "target"],
        source=_FUSED_FLOW_NOISE_SOURCE,
    )
    _HAS_METAL_KERNEL = True
    print("[loss] Metal fused kernel compiled: flow matching noise (1-pass noisy+target).")
except Exception as _e:
    _kernel_compile_error = str(_e)
    print(
        f"[loss] WARNING: Metal fused kernel compilation failed — using MLX fallback.\n"
        f"[loss]   Reason: {_kernel_compile_error}\n"
        f"[loss]   Training is numerically correct but uses two GPU dispatches per step.\n"
        f"[loss]   Requires Apple Silicon + macOS 13.2+. Check your MLX version if unexpected."
    )


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

    Uses the Metal kernel when B=1 (single scalar alpha/sigma); falls back to
    MLX element-wise ops for B>1 or when the kernel is unavailable.

    Returns:
      noisy:  alpha_t * latent + sigma_t * noise
      target: alpha_t * noise  - sigma_t * latent  (v-prediction target)
    """
    if _HAS_METAL_KERNEL and alpha_t.size == 1:
        # Metal path: single GPU kernel dispatch, float32 in/out.
        n = latent.size
        orig_shape = latent.shape
        orig_dtype = latent.dtype
        count = mx.array([n], dtype=mx.uint32)
        noisy_flat, target_flat = _fused_flow_kernel(
            inputs=[
                latent.reshape(-1).astype(mx.float32),
                noise.reshape(-1).astype(mx.float32),
                alpha_t.reshape(-1).astype(mx.float32),  # [1], kernel reads [0]
                sigma_t.reshape(-1).astype(mx.float32),
                count,
            ],
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(n, 1, 1),
            threadgroup=(min(256, n), 1, 1),
        )
        return (
            noisy_flat.reshape(orig_shape).astype(orig_dtype),
            target_flat.reshape(orig_shape).astype(orig_dtype),
        )

    # MLX fallback: correct for any batch size, two element-wise ops.
    while alpha_t.ndim < latent.ndim:
        alpha_t = alpha_t[..., None]
    while sigma_t.ndim < latent.ndim:
        sigma_t = sigma_t[..., None]
    noisy  = alpha_t * latent + sigma_t * noise
    target = alpha_t * noise  - sigma_t * latent
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
