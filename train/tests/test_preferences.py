"""
train/tests/test_preferences.py — preference + synthetic signals (v3.21.0 phase 4).

The data-flywheel-closure store: human/self/auto preference signals on dataset
items, synthetic-generation provenance, and the pure blending that lets those
signals nudge future unified scores. (Image generation is GPU-gated; this is the
durable store + score math.)

Isolated: explicit db_path → tempdir. Pure stdlib sqlite3, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.preferences import (
    PreferenceStore, blend_preference, apply_preferences, SOURCE_WEIGHTS,
)


@pytest.fixture
def store(tmp_path):
    return PreferenceStore(db_path=tmp_path / "pref.db")


# ---------------------------------------------------------------------------
# Preference recording + aggregation
# ---------------------------------------------------------------------------

class TestPreferences:
    def test_record_and_aggregate_single_source(self, store):
        store.record_preference("000042", "human", 0.8)
        store.record_preference("000042", "human", 0.6)
        agg = store.aggregate("000042")
        assert agg["n"] == 2
        assert agg["by_source"]["human"] == pytest.approx(0.7)
        assert agg["score"] == pytest.approx(0.7)

    def test_source_weighting(self, store):
        # human (w=1.0) says +1.0, auto (w=0.3) says -1.0.
        store.record_preference("x", "human", 1.0)
        store.record_preference("x", "auto", -1.0)
        agg = store.aggregate("x")
        # weighted = (1.0*1.0 + 0.3*(-1.0)) / (1.0 + 0.3) = 0.7/1.3 ≈ 0.538
        assert agg["score"] == pytest.approx(0.5385, abs=1e-3)

    def test_value_clamped(self, store):
        store.record_preference("x", "human", 5.0)     # clamps to 1.0
        store.record_preference("y", "human", -9.0)    # clamps to -1.0
        assert store.aggregate("x")["score"] == 1.0
        assert store.aggregate("y")["score"] == -1.0

    def test_no_signal_is_neutral(self, store):
        agg = store.aggregate("never_rated")
        assert agg["n"] == 0 and agg["score"] == 0.0

    def test_invalid_source_raises(self, store):
        with pytest.raises(ValueError):
            store.record_preference("x", "bogus", 0.5)

    def test_all_items(self, store):
        store.record_preference("a", "human", 0.5)
        store.record_preference("b", "self", 0.2)
        store.record_preference("a", "auto", 0.1)
        assert store.all_items() == ["a", "b"]


# ---------------------------------------------------------------------------
# Synthetic provenance
# ---------------------------------------------------------------------------

class TestSynthetic:
    def test_record_and_query(self, store):
        store.record_synthetic("gen_0001", experiment="exp_0007",
                               prompt="a fox in snow", quality=0.82)
        assert store.is_synthetic("gen_0001") is True
        assert store.is_synthetic("real_image") is False
        assert store.synthetic_count() == 1

    def test_upsert_idempotent(self, store):
        store.record_synthetic("gen_0001", "exp_1", "p", 0.5)
        store.record_synthetic("gen_0001", "exp_1", "p", 0.9)   # same id → replace
        assert store.synthetic_count() == 1


# ---------------------------------------------------------------------------
# Score blending (pure)
# ---------------------------------------------------------------------------

class TestBlend:
    def test_neutral_preference_leaves_score(self):
        assert blend_preference(0.50, 0.0, pref_weight=0.10) == 0.50

    def test_positive_preference_raises_score(self):
        # +1.0 preference × 0.10 weight → +0.10.
        assert blend_preference(0.50, 1.0, pref_weight=0.10) == pytest.approx(0.60)

    def test_negative_preference_lowers_score(self):
        assert blend_preference(0.50, -1.0, pref_weight=0.10) == pytest.approx(0.40)

    def test_clamped_to_unit_interval(self):
        assert blend_preference(0.95, 1.0, pref_weight=0.10) == 1.0   # clamps at 1
        assert blend_preference(0.05, -1.0, pref_weight=0.10) == 0.0  # clamps at 0

    def test_weight_controls_magnitude(self):
        assert blend_preference(0.50, 1.0, pref_weight=0.20) == pytest.approx(0.70)

    def test_apply_preferences_only_touches_rated_items(self, store):
        store.record_preference("rated", "human", 1.0)
        base = {"rated": 0.50, "unrated": 0.50}
        out = apply_preferences(base, store, pref_weight=0.10)
        assert out["rated"] == pytest.approx(0.60)
        assert out["unrated"] == 0.50          # untouched (no signal)
