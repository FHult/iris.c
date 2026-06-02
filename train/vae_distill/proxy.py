"""
train/vae_distill/proxy.py — Inference wrapper for the proxy VAE encoder.

Drop-in replacement for the Flux2VAE.encode() call in precompute_all.py.
Provides:
  - Fast proxy forward pass
  - Confidence scoring via per-channel z-score against training distribution
  - Automatic fallback to real teacher if confidence < threshold

Checkpoint format (safetensors):
  student.*             — StudentEncoder parameters
  meta.channel_mean     — [32] per-channel mean from training corpus
  meta.channel_std      — [32] per-channel std from training corpus
  meta.scaling_factor   — teacher VAE scaling_factor (float)
  meta.shift_factor     — teacher VAE shift_factor (float)
  meta.version          — semantic version string
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .student import StudentEncoder, build_student


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _confidence_score(
    latent: mx.array,
    channel_mean: mx.array,
    channel_std: mx.array,
    outlier_threshold: float = 3.0,
) -> float:
    """
    Compute a confidence score in [0, 1] for a proxy latent.

    For each of the 32 channels, compute the z-score of the batch mean
    against the training distribution.  The confidence is 1 minus the
    fraction of channels whose z-score exceeds outlier_threshold.

    Low confidence (<0.8) suggests the proxy is extrapolating outside
    its training distribution and the teacher should be used instead.
    """
    # lat: [B, 32, H, W] — compute per-channel batch mean
    lat = latent.astype(mx.float32)
    batch_mean = mx.mean(lat, axis=(0, 2, 3))   # [32]
    z = mx.abs(batch_mean - channel_mean) / (channel_std + 1e-6)
    outlier_frac = float(mx.mean((z > outlier_threshold).astype(mx.float32)))
    return 1.0 - outlier_frac


# ---------------------------------------------------------------------------
# Proxy wrapper
# ---------------------------------------------------------------------------

class ProxyVAE:
    """
    Inference-time proxy for the Flux VAE encoder.

    Usage:
        proxy = ProxyVAE.load("path/to/proxy_checkpoint.safetensors")

        # Fast path (no teacher loaded):
        latent = proxy.encode(image_bchw)

        # With fallback (teacher must be provided at construction):
        proxy = ProxyVAE.load(ckpt_path, teacher=real_vae)
        latent = proxy.encode(image_bchw)  # falls back if confidence < threshold
    """

    def __init__(
        self,
        student: StudentEncoder,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        scaling_factor: float,
        shift_factor: float,
        confidence_threshold: float = 0.75,
        teacher = None,
    ):
        self._student  = student
        self._ch_mean  = mx.array(channel_mean.astype(np.float32))
        self._ch_std   = mx.array(channel_std.astype(np.float32))
        self._scale    = scaling_factor
        self._shift    = shift_factor
        self._threshold = confidence_threshold
        self._teacher  = teacher
        self._fallback_count = 0
        self._total_count    = 0

    @staticmethod
    def load(
        ckpt_path: str,
        teacher = None,
        confidence_threshold: float = 0.75,
        device_cfg: Optional[dict] = None,
    ) -> "ProxyVAE":
        """Load proxy from a safetensors checkpoint."""
        weights = mx.load(ckpt_path)

        # Reconstruct student architecture from metadata
        meta_json = weights.get("meta.config_json")
        if meta_json is not None:
            # Stored as uint8 byte array via np.frombuffer(json_str.encode())
            cfg = json.loads(bytes(np.array(meta_json).tolist()).decode())
        else:
            cfg = {}  # fall back to default architecture

        student = build_student(cfg)
        # Filter to student.* keys, strip prefix
        student_weights = {
            k[len("student."):]: v
            for k, v in weights.items()
            if k.startswith("student.")
        }
        student.load_weights(list(student_weights.items()))
        student.eval()
        mx.eval(student.parameters())

        channel_mean = np.array(weights["meta.channel_mean"])
        channel_std  = np.array(weights["meta.channel_std"])
        scaling_factor = float(np.array(weights["meta.scaling_factor"]))
        shift_factor   = float(np.array(weights["meta.shift_factor"]))

        return ProxyVAE(
            student=student,
            channel_mean=channel_mean,
            channel_std=channel_std,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            confidence_threshold=confidence_threshold,
            teacher=teacher,
        )

    def encode(
        self,
        image: mx.array,
        check_confidence: bool = True,
    ) -> mx.array:
        """
        Encode image to latent space.

        If a teacher is set and confidence_check is enabled, falls back to
        the teacher when the confidence score is below the threshold.

        image: [B, 3, H, W] float32 in [-1, 1]
        Returns: [B, 32, H/8, W/8] latent (same format as Flux2VAE.encode)
        """
        self._total_count += 1

        image_f = image.astype(mx.float32)
        proxy_latent = self._student(image_f)
        # Apply teacher scaling to match Flux2VAE.encode() output convention
        proxy_latent = (proxy_latent - self._shift) * self._scale
        mx.eval(proxy_latent)

        if not check_confidence or self._teacher is None:
            return proxy_latent

        conf = _confidence_score(proxy_latent, self._ch_mean, self._ch_std)
        if conf >= self._threshold:
            return proxy_latent

        # Low confidence — fall back to teacher
        self._fallback_count += 1
        return self._teacher.encode(image)

    @property
    def fallback_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count

    def stats(self) -> dict:
        return {
            "total":         self._total_count,
            "fallbacks":     self._fallback_count,
            "fallback_rate": self.fallback_rate,
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_proxy_checkpoint(
    path: str,
    student: StudentEncoder,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    scaling_factor: float,
    shift_factor: float,
    cfg: dict,
    version: str = "1.0.0",
) -> None:
    """Save a complete proxy checkpoint to safetensors."""
    payload = {}

    # Student weights
    flat = dict(mx.utils.tree_flatten(student.parameters()))
    for k, v in flat.items():
        payload[f"student.{k}"] = v

    # Metadata as arrays for safetensors compatibility
    payload["meta.channel_mean"]    = mx.array(channel_mean.astype(np.float32))
    payload["meta.channel_std"]     = mx.array(channel_std.astype(np.float32))
    payload["meta.scaling_factor"]  = mx.array(np.float32(scaling_factor))
    payload["meta.shift_factor"]    = mx.array(np.float32(shift_factor))
    payload["meta.config_json"]     = mx.array(
        np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    )

    mx.eval(list(payload.values()))
    mx.save_safetensors(path, payload)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  [proxy] saved {path} ({size_mb:.1f} MB)")
