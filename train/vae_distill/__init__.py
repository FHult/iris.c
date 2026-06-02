"""
train/vae_distill — Proxy VAE distillation for Flux Klein 4B.

See plans/precomp2-proxy-vae-design.md for architecture rationale.

Public API:
    TeacherEncoder     — Flux VAE encoder wrapper with intermediate feature hooks
    StudentEncoder     — Lightweight ~3-9M-param proxy encoder
    ProxyVAE           — Inference wrapper: quality modes, confidence, regression detect
    build_student      — Construct StudentEncoder from a config dict
    build_student_small   — ~3.3M-param fast variant
    build_student_medium  — ~9.1M-param high-fidelity variant

Quality modes (proxy.py):
    SPEED         — unconditional proxy, maximum throughput
    BALANCED      — Mahalanobis distance confidence check (default)
    HIGH_FIDELITY — Mahalanobis + reconstruction check via VAE decoder
"""

from .teacher import TeacherEncoder
from .student import StudentEncoder, build_student, build_student_small, build_student_medium
from .proxy   import ProxyVAE, SPEED, BALANCED, HIGH_FIDELITY

__all__ = [
    "TeacherEncoder",
    "StudentEncoder", "build_student", "build_student_small", "build_student_medium",
    "ProxyVAE", "SPEED", "BALANCED", "HIGH_FIDELITY",
]
