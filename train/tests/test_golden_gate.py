"""
train/tests/test_golden_gate.py — golden-set regression gate (v3.21.0 phase 3).

The regression gate is the trustworthiness decision: does the proxy VAE degrade
final model quality beyond tolerance vs the real VAE? This is the pure core of
evaluate_golden_set.py (the 3-arm training/scoring is GPU-gated and run when the
pipeline is idle). Tests cover the gate logic + the registry/trend recording +
the auto-disable config patch.

Pure: dict math + tempdir DBs + a tempdir config. No GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_golden_set import (
    regression_gate, format_gate, write_results, maybe_disable_proxy,
)
from experiments.registry import ExperimentRegistry
from monitoring.trends import TrendStore


def _arms(real, proxy):
    return {"real": real, "proxy_fallback": proxy}


# ---------------------------------------------------------------------------
# regression_gate
# ---------------------------------------------------------------------------

class TestRegressionGate:
    def test_passes_when_proxy_matches(self):
        g = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.70}), tolerance=0.03)
        assert g["passed"] is True
        assert g["recommend_disable"] is False
        assert g["failures"] == []

    def test_passes_within_tolerance(self):
        # proxy 0.69 vs real 0.70 → 1.4% degradation < 3% tolerance.
        g = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.69}), tolerance=0.03)
        assert g["passed"] is True
        assert g["per_metric"]["clip_i"]["rel_degradation"] == pytest.approx(0.0143, abs=1e-3)

    def test_fails_beyond_tolerance(self):
        # proxy 0.60 vs real 0.70 → ~14% degradation > 3%.
        g = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.60}), tolerance=0.03)
        assert g["passed"] is False
        assert "clip_i" in g["failures"]
        assert g["recommend_disable"] is True

    def test_proxy_better_is_not_a_failure(self):
        # proxy 0.75 > real 0.70 → negative degradation → pass.
        g = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.75}), tolerance=0.03)
        assert g["passed"] is True
        assert g["per_metric"]["clip_i"]["rel_degradation"] < 0

    def test_lpips_lower_is_better_direction(self):
        # lpips: proxy 0.05 vs real 0.03 → proxy WORSE (higher) → ~67% degradation.
        g = regression_gate(_arms({"lpips": 0.03}, {"lpips": 0.05}), tolerance=0.03)
        assert "lpips" in g["failures"]
        # And the reverse: proxy lower lpips is better → pass.
        g2 = regression_gate(_arms({"lpips": 0.05}, {"lpips": 0.03}), tolerance=0.03)
        assert g2["passed"] is True

    def test_any_failing_metric_fails_the_gate(self):
        g = regression_gate(_arms(
            {"clip_i": 0.70, "clip_t": 0.30, "aesthetic": 5.0},
            {"clip_i": 0.70, "clip_t": 0.30, "aesthetic": 4.0},   # aesthetic -20%
        ), tolerance=0.03)
        assert g["passed"] is False
        assert g["failures"] == ["aesthetic"]

    def test_missing_metric_skipped(self):
        # real has clip_t but proxy doesn't → that metric is skipped, not failed.
        g = regression_gate(_arms({"clip_i": 0.70, "clip_t": 0.3}, {"clip_i": 0.70}),
                            tolerance=0.03)
        assert "clip_t" not in g["per_metric"]
        assert g["passed"] is True

    def test_format_gate_renders(self):
        g = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.60}), tolerance=0.03)
        out = format_gate(g)
        assert "FAIL" in out and "clip_i" in out and "RECOMMEND" in out


# ---------------------------------------------------------------------------
# write_results — registry + trend integration
# ---------------------------------------------------------------------------

class TestWriteResults:
    def test_writes_experiment_and_trend(self, tmp_path):
        reg = ExperimentRegistry(db_path=tmp_path / "exp.db")
        trends = TrendStore(db_path=tmp_path / "mon.db")
        arms = _arms({"clip_i": 0.70}, {"clip_i": 0.69})
        gate = regression_gate(arms, tolerance=0.03)
        eid = write_results(arms, gate, campaign="golden-eval",
                            proxy_path="/w/p.safetensors", proxy_fallback_rate=0.04,
                            registry=reg, trends=trends)
        rec = reg.get(eid)
        assert rec["golden_clip_i"] == 0.69          # headline = proxy_fallback arm
        assert rec["proxy_fallback_rate"] == 0.04
        # Trend points recorded.
        assert trends.latest("champion_clip_i")["value"] == 0.69
        assert trends.latest("proxy_fallback_rate")["value"] == 0.04

    def test_write_results_without_stores_is_safe(self):
        arms = _arms({"clip_i": 0.70}, {"clip_i": 0.69})
        gate = regression_gate(arms)
        assert write_results(arms, gate, campaign="c") is None   # no registry → None


# ---------------------------------------------------------------------------
# maybe_disable_proxy — the auto-disable safety action
# ---------------------------------------------------------------------------

class TestMaybeDisableProxy:
    def _cfg(self, tmp_path, enabled=True):
        p = tmp_path / "pipe.yaml"
        p.write_text(yaml.dump({"proxy_vae": {"enabled": enabled, "proxy_path": "/w/p"}}))
        return p

    def test_disables_on_failing_gate(self, tmp_path):
        p = self._cfg(tmp_path, enabled=True)
        gate = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.60}), tolerance=0.03)
        changed = maybe_disable_proxy(gate, str(p))
        assert changed is True
        cfg = yaml.safe_load(p.read_text())
        assert cfg["proxy_vae"]["enabled"] is False
        assert "_auto_disabled_reason" in cfg["proxy_vae"]

    def test_passing_gate_leaves_config_untouched(self, tmp_path):
        p = self._cfg(tmp_path, enabled=True)
        gate = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.70}))
        assert maybe_disable_proxy(gate, str(p)) is False
        assert yaml.safe_load(p.read_text())["proxy_vae"]["enabled"] is True

    def test_no_config_path_is_safe(self):
        gate = regression_gate(_arms({"clip_i": 0.70}, {"clip_i": 0.60}), tolerance=0.03)
        assert maybe_disable_proxy(gate, None) is False
