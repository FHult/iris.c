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
    """
    Centered, normalized Gram matrix. x: float32 [B, C, H, W] → [B, C, C]

    Each channel is spatially centred before computing cross-correlation so the
    result captures covariance (local texture / style) rather than correlation
    (which is biased by the per-channel DC mean and encodes content).
    """
    B, C, H, W = x.shape
    f = x.reshape(B, C, H * W)
    f = f - f.mean(axis=-1, keepdims=True)   # centre each channel over spatial dim
    # Normalise by H*W only (not C*H*W) so magnitude is bucket-resolution-invariant.
    # C is fixed in latent space; dividing by C would make the Gram entry scale 1/C²
    # smaller than the MSE term, requiring a ~C²× style_loss_weight adjustment.
    return mx.matmul(f, f.transpose(0, 2, 1)) / (H * W)


def gram_style_loss(x0_pred: mx.array, x0_ref: mx.array) -> mx.array:
    """
    MSE between centered, normalized Gram matrices of two float32 [B, C, H, W] latents.

    x0_pred should be an unbiased clean-latent estimate — use reconstruct_x0()
    rather than the raw alpha·x_t − sigma·v_pred, which is biased by
    (alpha² + sigma²) under the linear schedule.
    x0_ref is the ground-truth clean latent.
    """
    return mx.mean((gram_matrix(x0_pred) - gram_matrix(x0_ref)) ** 2)


# ---------------------------------------------------------------------------
# Content-leak loss (SREF-COMBINE-1 leak penalty) — the complement of Gram style
# ---------------------------------------------------------------------------

def instance_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Per-(B,C) spatial normalization. x: float32 [B, C, H, W] → same shape.

    Subtracts each channel's spatial mean and divides by its spatial std, removing the
    per-channel statistics (mean, std) that ENCODE STYLE (the same stats Gram captures as
    covariance) and leaving the normalized spatial pattern that encodes CONTENT. This is the
    AdaIN decomposition: style = per-channel (mean, std); content = instance-normalized map.
    """
    mu  = x.mean(axis=(2, 3), keepdims=True)
    var = ((x - mu) ** 2).mean(axis=(2, 3), keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps)


def content_leak_loss(x0_cond: mx.array, x0_null: mx.array) -> mx.array:
    """Penalize CONTENT drift between the conditioned and null (prompt-only) predictions.

    Both are unbiased clean-latent estimates (reconstruct_x0) of the SAME noisy latent — one
    with the style reference injected (x0_cond), one with conditioning zeroed (x0_null). MSE on
    their instance-normalized maps penalizes the style reference for CHANGING WHAT IS DEPICTED
    (content/spatial structure) while leaving per-channel style stats free to change. Minimizing
    it trains the adapter to inject STYLE WITHOUT leaking the reference's CONTENT — attacking the
    style/leak entanglement directly (vs the V-gate, which only scaled the entangled signal and
    capped at ~0.65). Both float32 [B, C, H, W]. Naturally ~0 on null steps (x0_cond == x0_null).
    """
    return mx.mean((instance_norm(x0_cond) - instance_norm(x0_null)) ** 2)


# ---------------------------------------------------------------------------
# Reference-discrimination signal (SREF mode-collapse fix — BACKLOG STEP 1A)
#
# Step-0 diagnostic: every trained adapter is REFERENCE-INERT (different refs -> ~identical
# output, cross-ref corr ~0.99). Step-1A.1 pinned the mechanism: to_v_ip is rank ~6, so the
# injected V is near-constant across references. The loss never rewards reference-specific
# output, so a low-rank generic "make-it-stylish" injection is the easy minimum. These two
# terms add that missing pressure — one on the OUTPUT (cause), one on the WEIGHT (symptom).
# ---------------------------------------------------------------------------

def style_stats(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Per-channel STYLE descriptor = (spatial mean, spatial std) — the AdaIN style component
    (complement of instance_norm's content). x: float32 [B, C, H, W] -> [B, 2C] (means then
    stds). These are exactly the per-channel stats Gram captures as covariance and the ones
    content_leak_loss leaves free; here they are the quantity references must DIFFER on."""
    mu  = x.mean(axis=(2, 3))                                      # [B, C]
    var = ((x - mu[:, :, None, None]) ** 2).mean(axis=(2, 3))      # [B, C]
    sd  = mx.sqrt(var + eps)                                       # [B, C]
    return mx.concatenate([mu, sd], axis=1)                        # [B, 2C]


def style_repulsion_loss(x0_a: mx.array, x0_b: mx.array, margin: float = 1.0) -> mx.array:
    """Discrimination-aware signal: two DIFFERENT style references, predicted at the SAME
    prompt / noise / timestep, MUST produce different STYLE stats — else the adapter is
    reference-blind (the to_v_ip collapse). Hinge on the per-channel style descriptor:
    penalize ONLY when the two references' styles are too similar (d2 < margin); no reward for
    pushing them apart beyond the margin (keeps it from fighting the reconstruction term).
    Both float32 [B, C, H, W]; the caller supplies x0_b from a DIFFERENT-style reference at the
    same noisy latent. Returns a scalar; ~margin on collapse, 0 once styles differ enough."""
    d2 = mx.mean((style_stats(x0_a) - style_stats(x0_b)) ** 2)
    return mx.maximum(0.0, margin - d2)


def vproj_rank_penalty(W: mx.array, u: mx.array, eps: float = 1e-8):
    """Rank-promoting spectral penalty on the stacked V-injection weights to_v_ip — the direct
    symptom fix for the Step-1A.1 finding (to_v_ip stable_rank ~6 -> V near-constant across
    refs). Penalizes the top-1 singular-energy fraction sigma1^2 / ||W||_F^2 per block (scale-
    invariant; minimizing it flattens the spectrum, RAISING stable rank = sum(sigma^2)/sigma1^2).

    sigma1 is estimated by ONE power-iteration step warm-started from u (cheap: two batched
    mat-vecs, negligible vs a Flux forward). u is PERSISTENT power-iteration state the caller
    threads across steps; the returned u_next is stop_gradient'd. Across steps the warm start
    converges u to the top singular vector, so the single-step estimate tracks sigma1 closely.

    W: [N, d, e] (to_v_ip_stacked, applied as out_e = in_d @ W[d,e]); u: [N, e] (unit).
    Returns (penalty_scalar, u_next [N, e])."""
    v = mx.einsum("nde,ne->nd", W, u)                             # W u  -> [N, d]
    v = v * mx.rsqrt(mx.sum(v * v, axis=1, keepdims=True) + eps)  # normalize (left vec est)
    un = mx.einsum("nde,nd->ne", W, v)                            # W^T v -> [N, e]
    s2 = mx.sum(un * un, axis=1)                                  # [N] ~ sigma1^2
    frob2 = mx.sum(W * W, axis=(1, 2)) + eps                      # [N] ||W_block||_F^2
    penalty = mx.mean(s2 / frob2)
    u_next = un * mx.rsqrt(s2[:, None] + eps)                     # warm start for next step
    return penalty, mx.stop_gradient(u_next)


def reconstruct_x0(
    noisy:   mx.array,
    v_pred:  mx.array,
    alpha_t: mx.array,
    sigma_t: mx.array,
) -> mx.array:
    """
    Unbiased clean-latent estimate from the predicted velocity.

    In v-prediction flow matching:
        noisy  = alpha_t · x0 + sigma_t · noise
        v_pred ≈ alpha_t · noise − sigma_t · x0
    Solving for x0 exactly:
        x0 = (alpha_t · noisy − sigma_t · v_pred) / (alpha_t² + sigma_t²)

    The raw formula alpha·noisy − sigma·v_pred gives (alpha²+sigma²)·x0 — biased
    at mid-timesteps.  For the linear schedule (alpha = 1−t/1000, sigma = t/1000)
    the denominator lies in [0.5, 1.0] and is always safe to divide by; the clamp
    at 1e-6 is a defensive guard for non-standard schedules.

    All inputs are cast to float32 internally; output shape matches noisy.
    """
    a = alpha_t.reshape(-1, 1, 1, 1).astype(mx.float32)
    s = sigma_t.reshape(-1, 1, 1, 1).astype(mx.float32)
    denom = mx.maximum(a * a + s * s, mx.array(1e-6, dtype=mx.float32))
    return (a * noisy.astype(mx.float32) - s * v_pred.astype(mx.float32)) / denom


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
