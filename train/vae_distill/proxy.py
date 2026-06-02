"""
train/vae_distill/proxy.py — Production proxy VAE encoder with quality assurance.

Drop-in replacement for Flux2VAE.encode() in precompute_all.py.

Quality modes (set via quality_mode parameter):
  "speed"         — trust proxy unconditionally, maximum throughput
  "balanced"      — diagonal Mahalanobis distance check on channel statistics,
                    no decoder overhead (default)
  "high_fidelity" — Mahalanobis + quick reconstruction check via VAE decoder on
                    one image per batch; falls back if decode MSE is too high

Confidence scoring:
  Balanced: RMS of per-channel z-scores against training corpus distribution.
  A score < threshold triggers fallback to real VAE.

Regression monitoring:
  Tracks a rolling window of per-batch confidence scores.  If the window mean
  drops below regression_threshold for regression_window consecutive batches,
  proxy is auto-disabled and a warning is raised.  Call reset_regression() to
  re-enable after investigating.

Checkpoint format (safetensors):
  student.*             — StudentEncoder parameters
  meta.channel_mean     — [32] per-channel mean from training corpus
  meta.channel_std      — [32] per-channel std from training corpus
  meta.scaling_factor   — teacher VAE scaling_factor (float)
  meta.shift_factor     — teacher VAE shift_factor (float)
  meta.config_json      — student architecture config (uint8 bytes)
  meta.expected_decode_mse — baseline reconstruction MSE (optional, float32)
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .student import StudentEncoder, build_student


# ---------------------------------------------------------------------------
# Quality mode constants
# ---------------------------------------------------------------------------

SPEED         = "speed"
BALANCED      = "balanced"
HIGH_FIDELITY = "high_fidelity"
_VALID_MODES  = (SPEED, BALANCED, HIGH_FIDELITY)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _mahalanobis_confidence(
    latent: mx.array,
    ch_mean: mx.array,
    ch_std: mx.array,
) -> tuple[float, float]:
    """
    Diagonal Mahalanobis distance for a batch of latents.

    Computes per-channel z-scores of the batch mean against the training
    corpus distribution, then takes the RMS across channels.

    Returns (confidence, mahal_distance) where:
      - mahal_distance is the RMS z-score (expected ~1.0 for in-distribution)
      - confidence = exp(-max(0, mahal_distance - 1.0)) clamped to [0, 1]
        (in-distribution = 1.0, degrading smoothly as distance grows)
    """
    lat = latent.astype(mx.float32)
    batch_mean = mx.mean(lat, axis=(0, 2, 3))          # [32]
    z = (batch_mean - ch_mean) / (ch_std + 1e-6)       # [32]
    dist = float(mx.sqrt(mx.mean(z ** 2)))              # scalar RMS
    conf = float(mx.exp(mx.array(-max(0.0, dist - 1.0))))
    return min(1.0, conf), dist


def _reconstruction_confidence(
    image: mx.array,
    proxy_latent: mx.array,
    decoder,
    expected_mse: float,
    mse_ratio_threshold: float = 2.5,
) -> tuple[float, float]:
    """
    Check reconstruction quality by decoding the proxy latent for the first
    image in the batch and comparing to the original input.

    Returns (confidence, mse_ratio) where mse_ratio = actual_mse / expected_mse.
    Confidence is 0.0 when mse_ratio >= mse_ratio_threshold, else 1.0.
    """
    img0    = image[:1]                              # [1, 3, H, W]
    lat0    = proxy_latent[:1]
    decoded = decoder(lat0)                          # [1, 3, H, W]
    mx.eval(decoded)
    mse     = float(mx.mean((decoded - img0) ** 2))
    ratio   = mse / (expected_mse + 1e-8)
    return (1.0 if ratio < mse_ratio_threshold else 0.0), ratio


# ---------------------------------------------------------------------------
# Regression monitor
# ---------------------------------------------------------------------------

class RegressionMonitor:
    """
    Rolling window confidence tracker.

    When the window mean drops below regression_threshold for
    regression_window consecutive samples, sets regression_detected=True
    and emits a warning.  Call reset() to re-enable after investigation.
    """

    def __init__(
        self,
        window: int = 100,
        regression_threshold: float = 0.60,
    ):
        self._window    = deque(maxlen=window)
        self._threshold = regression_threshold
        self.regression_detected = False

    def update(self, confidence: float) -> None:
        if self.regression_detected:
            return
        self._window.append(confidence)
        if len(self._window) == self._window.maxlen:
            mean = sum(self._window) / len(self._window)
            if mean < self._threshold:
                self.regression_detected = True
                warnings.warn(
                    f"[ProxyVAE] Regression detected: rolling mean confidence "
                    f"{mean:.3f} < threshold {self._threshold:.3f} over "
                    f"{len(self._window)} batches.  Proxy auto-disabled. "
                    f"Call proxy.reset_regression() after investigating.",
                    stacklevel=3,
                )

    def reset(self) -> None:
        self._window.clear()
        self.regression_detected = False

    @property
    def mean_confidence(self) -> Optional[float]:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)


# ---------------------------------------------------------------------------
# ProxyVAE
# ---------------------------------------------------------------------------

class ProxyVAE:
    """
    Production inference proxy for the Flux VAE encoder.

    Parameters
    ----------
    student              : trained StudentEncoder
    channel_mean         : [32] per-channel latent mean from training corpus
    channel_std          : [32] per-channel latent std from training corpus
    scaling_factor       : teacher VAE scaling_factor (float)
    shift_factor         : teacher VAE shift_factor (float)
    quality_mode         : "speed" | "balanced" | "high_fidelity"
    fallback_threshold   : confidence below this → use teacher fallback
    teacher              : real VAE encoder for fallback (optional)
    regression_window    : rolling window size for regression detection
    regression_threshold : rolling mean confidence below this → auto-disable
    expected_decode_mse  : baseline reconstruction MSE for high_fidelity mode
    """

    def __init__(
        self,
        student: StudentEncoder,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        scaling_factor: float,
        shift_factor: float,
        quality_mode: str = BALANCED,
        fallback_threshold: float = 0.75,
        teacher = None,
        regression_window: int = 100,
        regression_threshold: float = 0.60,
        expected_decode_mse: Optional[float] = None,
    ):
        if quality_mode not in _VALID_MODES:
            raise ValueError(f"quality_mode must be one of {_VALID_MODES}, got {quality_mode!r}")

        self._student             = student
        self._ch_mean             = mx.array(channel_mean.astype(np.float32))
        self._ch_std              = mx.array(channel_std.astype(np.float32))
        self._scale               = scaling_factor
        self._shift               = shift_factor
        self.quality_mode         = quality_mode
        self._threshold           = fallback_threshold
        self._teacher             = teacher
        self._expected_decode_mse = expected_decode_mse
        self._regression          = RegressionMonitor(regression_window, regression_threshold)

        self._fallback_count = 0
        self._total_count    = 0
        self._disabled       = False   # set True when regression detected

    # ── Loading ─────────────────────────────────────────────────────────────

    @staticmethod
    def load(
        ckpt_path: str,
        teacher = None,
        quality_mode: str = BALANCED,
        fallback_threshold: float = 0.75,
        regression_window: int = 100,
        regression_threshold: float = 0.60,
    ) -> "ProxyVAE":
        """Load proxy from a safetensors checkpoint."""
        weights = mx.load(ckpt_path)

        meta_json = weights.get("meta.config_json")
        cfg = (json.loads(bytes(np.array(meta_json).tolist()).decode())
               if meta_json is not None else {})

        student = build_student(cfg)
        student_w = {k[len("student."):]: v
                     for k, v in weights.items() if k.startswith("student.")}
        student.load_weights(list(student_w.items()))
        student.eval()
        mx.eval(student.parameters())

        channel_mean       = np.array(weights["meta.channel_mean"])
        channel_std        = np.array(weights["meta.channel_std"])
        scaling_factor     = float(np.array(weights["meta.scaling_factor"]))
        shift_factor       = float(np.array(weights["meta.shift_factor"]))
        expected_decode_mse = (
            float(np.array(weights["meta.expected_decode_mse"]))
            if "meta.expected_decode_mse" in weights else None
        )

        return ProxyVAE(
            student=student,
            channel_mean=channel_mean,
            channel_std=channel_std,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            quality_mode=quality_mode,
            fallback_threshold=fallback_threshold,
            teacher=teacher,
            regression_window=regression_window,
            regression_threshold=regression_threshold,
            expected_decode_mse=expected_decode_mse,
        )

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(
        self,
        image: mx.array,
        check_confidence: bool = True,
    ) -> mx.array:
        """
        Encode image to latent space.

        image: [B, 3, H, W] float32 in [-1, 1]
        Returns: [B, 32, H/8, W/8] latent matching Flux2VAE.encode() format.
        """
        self._total_count += 1

        # If regression has been detected, bypass proxy entirely.
        if self._disabled or self._regression.regression_detected:
            self._disabled = True
            if self._teacher is not None:
                self._fallback_count += 1
                return self._teacher.encode(image)
            # No teacher — return proxy output with warning (best effort)
            warnings.warn("[ProxyVAE] Regression detected but no teacher available; "
                          "using proxy with degraded confidence.", stacklevel=2)

        # Speed mode: unconditional proxy
        if self.quality_mode == SPEED or not check_confidence:
            return self._proxy_encode(image)

        # Balanced / high-fidelity: compute confidence
        proxy_latent = self._proxy_encode(image)
        conf, dist   = _mahalanobis_confidence(proxy_latent, self._ch_mean, self._ch_std)

        # High-fidelity: add reconstruction check if decoder + baseline available
        if (self.quality_mode == HIGH_FIDELITY
                and self._teacher is not None
                and self._expected_decode_mse is not None):
            recon_conf, mse_ratio = _reconstruction_confidence(
                image, proxy_latent, self._teacher.decode,
                self._expected_decode_mse,
            )
            # Take the minimum of the two confidence signals
            conf = min(conf, recon_conf)

        self._regression.update(conf)

        if conf >= self._threshold:
            return proxy_latent

        # Below threshold — fall back to teacher
        self._fallback_count += 1
        if self._teacher is not None:
            return self._teacher.encode(image)
        return proxy_latent   # no teacher — return proxy best-effort

    def _proxy_encode(self, image: mx.array) -> mx.array:
        lat = self._student(image.astype(mx.float32))
        lat = (lat - self._shift) * self._scale
        mx.eval(lat)
        return lat

    # ── Regression control ───────────────────────────────────────────────────

    def reset_regression(self) -> None:
        """Re-enable proxy after investigating a regression event."""
        self._regression.reset()
        self._disabled = False

    # ── Statistics ───────────────────────────────────────────────────────────

    @property
    def fallback_rate(self) -> float:
        return self._fallback_count / max(1, self._total_count)

    @property
    def regression_detected(self) -> bool:
        return self._regression.regression_detected

    def stats(self) -> dict:
        return {
            "total":              self._total_count,
            "fallbacks":          self._fallback_count,
            "fallback_rate":      round(self.fallback_rate, 4),
            "quality_mode":       self.quality_mode,
            "regression_detected": self.regression_detected,
            "mean_confidence":    round(self._regression.mean_confidence or 0.0, 4),
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
    expected_decode_mse: Optional[float] = None,
) -> None:
    """Save inference-ready proxy checkpoint to safetensors."""
    import mlx.utils as mx_utils
    payload: dict[str, mx.array] = {}

    for k, v in mx_utils.tree_flatten(student.parameters()):
        payload[f"student.{k}"] = v

    payload["meta.channel_mean"]   = mx.array(channel_mean.astype(np.float32))
    payload["meta.channel_std"]    = mx.array(channel_std.astype(np.float32))
    payload["meta.scaling_factor"] = mx.array(np.float32(scaling_factor))
    payload["meta.shift_factor"]   = mx.array(np.float32(shift_factor))
    payload["meta.config_json"]    = mx.array(
        np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    )
    if expected_decode_mse is not None:
        payload["meta.expected_decode_mse"] = mx.array(np.float32(expected_decode_mse))

    mx.eval(list(payload.values()))
    # mx.save_safetensors forces a .safetensors extension, so the temp path must
    # already end in .safetensors or os.replace will miss the actual file written.
    import os
    tmp = str(path) + ".tmp.safetensors"
    mx.save_safetensors(tmp, payload)
    os.replace(tmp, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  [proxy] saved {path} ({size_mb:.1f} MB)")
