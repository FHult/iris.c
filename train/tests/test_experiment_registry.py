"""
train/tests/test_experiment_registry.py — experiment tracking (v3.21.0).

The ExperimentRegistry is the backbone of the v3.21.0 quality loop: every
trained model's provenance + golden-set results + Champion/Challenger status.
These tests cover registration, golden attachment, ranking (incl. lower-is-better
metrics), Champion promotion with hysteresis, and comparison.

Isolated: explicit db_path → fresh tempdir DB; never touches live experiments.db.
Pure stdlib sqlite3, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.registry import ExperimentRegistry, metric_is_better, GOLDEN_METRICS


@pytest.fixture
def reg(tmp_path):
    return ExperimentRegistry(db_path=tmp_path / "exp.db")


def _golden(clip_i=None, clip_t=None, aesthetic=None, lpips=None, fid=None):
    d = {}
    for k, v in (("clip_i", clip_i), ("clip_t", clip_t), ("aesthetic", aesthetic),
                 ("lpips", lpips), ("fid", fid)):
        if v is not None:
            d[k] = v
    return d


# ---------------------------------------------------------------------------
# Registration + id allocation
# ---------------------------------------------------------------------------

class TestRegister:
    def test_ids_increment(self, reg):
        a = reg.register(campaign="c1")
        b = reg.register(campaign="c1")
        assert a == "exp_0001" and b == "exp_0002"

    def test_record_roundtrip(self, reg):
        eid = reg.register(campaign="warmup", weights_path="/w/step_1000.safetensors",
                           proxy_enabled=True, proxy_mode="balanced",
                           proxy_fallback_rate=0.03,
                           hyperparams={"lr": 1e-4}, cond_gap=0.25, total_steps=1000)
        rec = reg.get(eid)
        assert rec["campaign"] == "warmup"
        assert rec["proxy_enabled"] == 1
        assert rec["proxy_fallback_rate"] == 0.03
        assert rec["hyperparams"] == {"lr": 1e-4}      # JSON decoded
        assert rec["status"] == "registered"

    def test_list_filters(self, reg):
        reg.register(campaign="a")
        reg.register(campaign="b")
        reg.register(campaign="a")
        assert len(reg.list(campaign="a")) == 2
        assert len(reg.list()) == 3


# ---------------------------------------------------------------------------
# Golden attachment
# ---------------------------------------------------------------------------

class TestAttachGolden:
    def test_headline_arm_populates_indexed_columns(self, reg):
        eid = reg.register(campaign="c")
        reg.attach_golden(eid, {
            "real":           _golden(clip_i=0.70, lpips=0.030),
            "proxy_fallback": _golden(clip_i=0.69, lpips=0.032),
            "proxy_forced":   _golden(clip_i=0.66, lpips=0.040),
        })
        rec = reg.get(eid)
        assert rec["golden_clip_i"] == 0.69            # headline = proxy_fallback
        assert rec["golden_lpips"] == 0.032
        assert rec["status"] == "evaluated"
        assert rec["golden_results"]["real"]["clip_i"] == 0.70  # full blob kept

    def test_custom_headline_arm(self, reg):
        eid = reg.register(campaign="c")
        reg.attach_golden(eid, {
            "real":           _golden(clip_i=0.70),
            "proxy_fallback": _golden(clip_i=0.69),
        }, headline_arm="real")
        assert reg.get(eid)["golden_clip_i"] == 0.70


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestRank:
    def test_higher_is_better_default(self, reg):
        for cg in (0.60, 0.80, 0.70):
            eid = reg.register(campaign="c")
            reg.attach_golden(eid, {"proxy_fallback": _golden(clip_i=cg)})
        ranked = reg.rank("clip_i")
        assert [r["golden_clip_i"] for r in ranked] == [0.80, 0.70, 0.60]

    def test_lpips_is_lower_is_better(self, reg):
        for lp in (0.05, 0.02, 0.08):
            eid = reg.register(campaign="c")
            reg.attach_golden(eid, {"proxy_fallback": _golden(lpips=lp)})
        ranked = reg.rank("lpips")
        assert [r["golden_lpips"] for r in ranked] == [0.02, 0.05, 0.08]

    def test_unevaluated_excluded(self, reg):
        reg.register(campaign="c")                      # never evaluated
        eid = reg.register(campaign="c")
        reg.attach_golden(eid, {"proxy_fallback": _golden(clip_i=0.7)})
        assert len(reg.rank("clip_i")) == 1

    def test_invalid_metric_raises(self, reg):
        with pytest.raises(ValueError):
            reg.rank("not_a_metric")


# ---------------------------------------------------------------------------
# Champion / Challenger
# ---------------------------------------------------------------------------

class TestChampion:
    def test_no_evaluated_returns_none(self, reg):
        reg.register(campaign="c")
        assert reg.champion("clip_i") is None
        assert reg.promote_champion("clip_i") is None

    def test_promote_marks_champion_and_challengers(self, reg):
        ids = []
        for cg in (0.60, 0.80, 0.70):
            eid = reg.register(campaign="c")
            reg.attach_golden(eid, {"proxy_fallback": _golden(clip_i=cg)})
            ids.append(eid)
        champ = reg.promote_champion("clip_i")
        assert champ == ids[1]                          # the 0.80 one
        statuses = {r["id"]: r["status"] for r in reg.list()}
        assert statuses[ids[1]] == "champion"
        assert statuses[ids[0]] == "challenger"
        assert statuses[ids[2]] == "challenger"

    def test_hysteresis_keeps_incumbent_within_margin(self, reg):
        a = reg.register(campaign="c")
        reg.attach_golden(a, {"proxy_fallback": _golden(clip_i=0.700)})
        reg.promote_champion("clip_i")                  # a is champion
        # A new experiment beats a by only 0.005 < margin 0.02 → a stays champion.
        b = reg.register(campaign="c")
        reg.attach_golden(b, {"proxy_fallback": _golden(clip_i=0.705)})
        champ = reg.promote_champion("clip_i", min_margin=0.02)
        assert champ == a

    def test_clear_improvement_switches_champion(self, reg):
        a = reg.register(campaign="c")
        reg.attach_golden(a, {"proxy_fallback": _golden(clip_i=0.70)})
        reg.promote_champion("clip_i")
        b = reg.register(campaign="c")
        reg.attach_golden(b, {"proxy_fallback": _golden(clip_i=0.80)})
        champ = reg.promote_champion("clip_i", min_margin=0.02)
        assert champ == b


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_directional_winners(self, reg):
        a = reg.register(campaign="ca", cond_gap=0.20)
        b = reg.register(campaign="cb", cond_gap=0.30)
        reg.attach_golden(a, {"proxy_fallback": _golden(clip_i=0.70, lpips=0.05)})
        reg.attach_golden(b, {"proxy_fallback": _golden(clip_i=0.68, lpips=0.03)})
        cmp = reg.compare(a, b)
        # clip_i higher-is-better: a (0.70) wins.
        assert cmp["winner_by_metric"]["clip_i"] == a
        # lpips lower-is-better: b (0.03) wins.
        assert cmp["winner_by_metric"]["lpips"] == b
        # cond_gap higher-is-better: b (0.30) wins.
        assert cmp["winner_by_metric"]["cond_gap"] == b
        assert cmp["fields"]["golden_clip_i"]["delta"] == pytest.approx(-0.02)

    def test_compare_tie(self, reg):
        a = reg.register(campaign="c")
        b = reg.register(campaign="c")
        reg.attach_golden(a, {"proxy_fallback": _golden(clip_i=0.70)})
        reg.attach_golden(b, {"proxy_fallback": _golden(clip_i=0.70)})
        assert reg.compare(a, b)["winner_by_metric"]["clip_i"] == "tie"

    def test_compare_unknown_id_raises(self, reg):
        a = reg.register(campaign="c")
        with pytest.raises(KeyError):
            reg.compare(a, "exp_9999")


# ---------------------------------------------------------------------------
# metric direction helper
# ---------------------------------------------------------------------------

class TestMetricDirection:
    def test_higher_is_better(self):
        assert metric_is_better("clip_i", 0.8, 0.7) is True
        assert metric_is_better("clip_i", 0.7, 0.8) is False

    def test_lower_is_better(self):
        assert metric_is_better("lpips", 0.03, 0.05) is True
        assert metric_is_better("fid", 20.0, 10.0) is False
