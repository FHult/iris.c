"""
train/experiments — experiment tracking for the v3.21.0 quality loop.

ExperimentRegistry: durable per-model record (campaign, proxy settings + fallback
rate, hyperparameters, golden-set results, weights path) with Champion/Challenger
ranking. tracker: orchestration (create from a finished campaign) + CLI reports.
"""

from .registry import ExperimentRegistry, GOLDEN_METRICS, metric_is_better
from .preferences import PreferenceStore, blend_preference, apply_preferences

__all__ = [
    "ExperimentRegistry", "GOLDEN_METRICS", "metric_is_better",
    "PreferenceStore", "blend_preference", "apply_preferences",
]
