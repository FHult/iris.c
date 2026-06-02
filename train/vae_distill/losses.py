"""
train/vae_distill/losses.py — Loss functions for proxy VAE distillation.

Loss composition (all weights configurable):
  1. channel_norm_mse  — per-channel normalised MSE on latents (primary)
  2. decoded_mse       — pixel-level MSE after VAE decode (perceptual proxy)
  3. freq_weighted_mse — Fourier-domain MSE, up-weights high spatial freqs
  4. dist_match        — per-batch channel mean/std alignment
  5. feature_match     — optional: projected intermediate feature MSE

All inputs are in [B, C, H, W] float32.  Losses return scalar mx.array.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Channel statistics  (computed once on training corpus, used for normalisation)
# ---------------------------------------------------------------------------

def compute_channel_stats(latents: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from a list of [32, H, W] latent arrays.
    Used once on the training corpus to populate the channel_mean/channel_std
    fields stored in the proxy checkpoint.

    Returns mean [32], std [32].
    """
    all_latents = np.concatenate([l.reshape(32, -1) for l in latents], axis=1)
    return all_latents.mean(axis=1), all_latents.std(axis=1) + 1e-6


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def channel_norm_mse(
    pred: mx.array,
    target: mx.array,
    channel_std: mx.array,
) -> mx.array:
    """
    Per-channel normalised MSE.  Each channel is divided by its empirical std
    before computing the squared error, giving all channels equal weight.

    pred, target: [B, 32, H, W]
    channel_std:  [1, 32, 1, 1] — from training corpus statistics
    """
    diff = (pred - target) / channel_std
    return mx.mean(diff ** 2)


def decoded_pixel_mse(
    proxy_latent: mx.array,
    teacher_latent: mx.array,
    teacher_decoder,
    scaling_factor: float = 1.0,
    shift_factor: float = 0.0,
) -> mx.array:
    """
    Decode proxy and teacher latents, compute pixel-level MSE.

    The teacher_decoder should accept a latent in the same format that the
    teacher's encode() returns (pre-scaled mean) and produce [B, 3, H, W].

    Gradient flows through proxy_latent only; teacher_latent is stopped.
    """
    # Teacher decode is reference only — stop gradients
    teacher_dec = mx.stop_gradient(teacher_decoder(teacher_latent))
    proxy_dec   = teacher_decoder(proxy_latent)
    diff = proxy_dec - teacher_dec
    return mx.mean(diff ** 2)


def freq_weighted_mse(
    pred: mx.array,
    target: mx.array,
    alpha: float = 0.5,
) -> mx.array:
    """
    MSE weighted toward higher spatial frequencies to counter the smoothing
    bias of plain MSE.

    alpha=0.0 → plain MSE; alpha=1.0 → frequency-magnitude-proportional weight.

    Operates channel-wise: 2D FFT per channel, weighted L2, then sum over freqs.
    """
    # [B, C, H, W] → compute 2D FFT over H, W
    pred_f   = mx.fft.fft2(pred.astype(mx.float32))
    target_f = mx.fft.fft2(target.astype(mx.float32))

    diff_sq = mx.abs(pred_f - target_f) ** 2  # [B, C, H, W]

    if alpha > 0.0:
        # Frequency magnitude weights: higher frequency → higher weight
        B, C, H, W = pred.shape
        fy = mx.array(np.fft.fftfreq(H).reshape(1, 1, H, 1).astype(np.float32))
        fx = mx.array(np.fft.fftfreq(W).reshape(1, 1, 1, W).astype(np.float32))
        freq_mag  = mx.sqrt(fy ** 2 + fx ** 2)        # [1, 1, H, W]
        freq_mag  = freq_mag / (mx.max(freq_mag) + 1e-8)
        weight    = 1.0 + alpha * freq_mag
        diff_sq   = diff_sq * weight

    # Normalise by number of frequency bins
    H, W = pred.shape[-2], pred.shape[-1]
    return mx.mean(diff_sq) / (H * W)


def dist_match(
    pred: mx.array,
    target: mx.array,
) -> mx.array:
    """
    Per-batch channel distribution matching.

    Penalises differences in per-channel mean and std between proxy and teacher
    outputs within the batch.  Cheap regulariser that prevents systematic
    channel-level bias without requiring distribution statistics from the corpus.

    pred, target: [B, 32, H, W]
    """
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)    # [B, 32, HW]
    t = target.reshape(target.shape[0], target.shape[1], -1)

    # Compute mean/std per channel over batch × spatial dims
    p_mean = mx.mean(p, axis=(0, 2))   # [32]
    t_mean = mx.mean(t, axis=(0, 2))
    p_std  = mx.sqrt(mx.var(p, axis=(0, 2)) + 1e-6)
    t_std  = mx.sqrt(mx.var(t, axis=(0, 2)) + 1e-6)

    mean_err = mx.mean((p_mean - mx.stop_gradient(t_mean)) ** 2)
    std_err  = mx.mean((p_std  - mx.stop_gradient(t_std))  ** 2)
    return mean_err + std_err


def feature_match(
    student_feats: list[mx.array],
    teacher_feats: list[mx.array],
    adapters: list,
) -> mx.array:
    """
    Intermediate feature matching with learned 1×1 projection adapters.

    adapters: list of nn.Conv2d(student_ch, teacher_ch, 1) — one per feature level.
    Gradient flows through student_feats and adapters; teacher_feats are stopped.
    """
    loss = mx.array(0.0)
    for s_feat, t_feat, adapter in zip(student_feats, teacher_feats, adapters):
        # Project student to teacher channel count
        s = mx.transpose(s_feat, (0, 2, 3, 1))
        s = adapter(s)
        s = mx.transpose(s, (0, 3, 1, 2))
        t = mx.stop_gradient(t_feat)
        # Normalise spatially before MSE to decouple scale differences
        s_n = s / (mx.sqrt(mx.mean(s ** 2, axis=(1, 2, 3), keepdims=True)) + 1e-8)
        t_n = t / (mx.sqrt(mx.mean(t ** 2, axis=(1, 2, 3), keepdims=True)) + 1e-8)
        loss = loss + mx.mean((s_n - t_n) ** 2)
    return loss / len(adapters)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class DistillationLoss:
    """
    Composite distillation loss with configurable term weights.

    Weights from config (all default to values in plans/precomp2-proxy-vae-design.md):
      latent_mse_weight  : 1.0   channel-normalised MSE on latents
      decoded_mse_weight : 0.1   pixel MSE after VAE decode
      freq_alpha         : 0.5   frequency weighting strength (0=plain MSE)
      dist_weight        : 0.01  per-batch channel distribution matching
      feature_weight     : 0.05  intermediate feature matching (0=disabled)
    """

    def __init__(
        self,
        channel_std: np.ndarray,
        latent_mse_weight:  float = 1.0,
        decoded_mse_weight: float = 0.1,
        freq_alpha:         float = 0.5,
        dist_weight:        float = 0.01,
        feature_weight:     float = 0.0,
    ):
        # Store channel stats as MLX array for fast normalisation
        self._channel_std = mx.array(
            channel_std.reshape(1, -1, 1, 1).astype(np.float32)
        )
        self.w_latent   = latent_mse_weight
        self.w_decoded  = decoded_mse_weight
        self.freq_alpha = freq_alpha
        self.w_dist     = dist_weight
        self.w_feat     = feature_weight

    def __call__(
        self,
        proxy_latent:   mx.array,
        teacher_latent: mx.array,
        teacher_decoder = None,
        scaling_factor: float = 1.0,
        shift_factor:   float = 0.0,
        student_feats:  Optional[list[mx.array]] = None,
        teacher_feats:  Optional[list[mx.array]] = None,
        feat_adapters:  Optional[list] = None,
    ) -> tuple[mx.array, dict[str, float]]:
        """
        Compute composite distillation loss.

        Returns (total_loss, components_dict) where components_dict maps
        term names to their unweighted scalar values for logging.
        """
        components: dict[str, mx.array] = {}

        # Primary: channel-normalised MSE on latents
        l_norm = channel_norm_mse(proxy_latent, teacher_latent, self._channel_std)
        components["latent_mse"] = l_norm

        # Frequency-weighted MSE (adds high-freq penalty on top of plain MSE)
        if self.freq_alpha > 0.0:
            l_freq = freq_weighted_mse(proxy_latent, teacher_latent, self.freq_alpha)
            components["freq_mse"] = l_freq
        else:
            l_freq = mx.array(0.0)

        # Per-batch distribution matching
        l_dist = dist_match(proxy_latent, teacher_latent)
        components["dist_match"] = l_dist

        # Decoded pixel MSE (requires teacher decoder)
        if self.w_decoded > 0.0 and teacher_decoder is not None:
            l_dec = decoded_pixel_mse(
                proxy_latent, teacher_latent, teacher_decoder,
                scaling_factor, shift_factor,
            )
            components["decoded_mse"] = l_dec
        else:
            l_dec = mx.array(0.0)

        # Intermediate feature matching (requires adapters)
        if (self.w_feat > 0.0 and student_feats is not None
                and teacher_feats is not None and feat_adapters is not None):
            l_feat = feature_match(student_feats, teacher_feats, feat_adapters)
            components["feat_match"] = l_feat
        else:
            l_feat = mx.array(0.0)

        total = (
            self.w_latent  * l_norm
            + self.w_latent * l_freq       # freq is additive on top of latent MSE
            + self.w_dist   * l_dist
            + self.w_decoded * l_dec
            + self.w_feat   * l_feat
        )

        # Return float values for logging (after mx.eval)
        return total, components
