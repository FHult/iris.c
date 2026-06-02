"""
train/vae_distill — Proxy VAE distillation for Flux Klein 4B.

See plans/precomp2-proxy-vae-design.md for architecture rationale.

Public API (all imported from here):
    TeacherEncoder   — Flux VAE encoder wrapper that exposes intermediate features
    StudentEncoder   — Lightweight ~6-10M-param proxy encoder
    ProxyVAE         — Inference wrapper with confidence scoring and teacher fallback
    build_student    — Construct StudentEncoder from a config dict
"""

from .teacher import TeacherEncoder
from .student import StudentEncoder, build_student
from .proxy import ProxyVAE

__all__ = ["TeacherEncoder", "StudentEncoder", "build_student", "ProxyVAE"]
