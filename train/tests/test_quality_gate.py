"""
train/tests/test_quality_gate.py — cross-run quality regression comparison.

quality_gate.compare_quality decides whether a long run's golden-set metrics
improved or regressed vs the previous run — the instrument that validates whether
ablation-chosen params actually paid off at the deployment horizon. Pure: dict in,
verdict out. run_quality_gate's side-effecting steps (GPU eval, registry I/O) are
injectable, so the glue is tested without a model or the live registry.

Metric directions: clip_i/clip_t/aesthetic higher-is-better; lpips/fid lower-is-better.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import quality_gate as qg


# ---------------------------------------------------------------------------
# compare_quality — the verdict core
# ---------------------------------------------------------------------------

class TestCompareQuality:
    def test_all_improved(self):
        cur = {"clip_i": 0.80, "lpips": 0.30}     # clip_i ↑, lpips ↓ = both better
        prev = {"clip_i": 0.70, "lpips": 0.40}
        r = qg.compare_quality(cur, prev)
        assert r["verdict"] == "IMPROVED" and not r["regressed"]
        assert r["per_metric"]["clip_i"]["status"] == "improved"
        assert r["per_metric"]["lpips"]["status"] == "improved"

    def test_higher_is_better_regresses_on_drop(self):
        r = qg.compare_quality({"clip_i": 0.60}, {"clip_i": 0.70})
        assert r["verdict"] == "REGRESSION" and r["regressed"]

    def test_lower_is_better_regresses_on_increase(self):
        # lpips went UP → worse (lower-is-better)
        r = qg.compare_quality({"lpips": 0.50}, {"lpips": 0.40})
        assert r["verdict"] == "REGRESSION"
        assert r["per_metric"]["lpips"]["status"] == "regressed"

    def test_lower_is_better_improves_on_decrease(self):
        r = qg.compare_quality({"fid": 12.0}, {"fid": 18.0})
        assert r["per_metric"]["fid"]["status"] == "improved"
        assert r["verdict"] == "IMPROVED"

    def test_within_threshold_is_neutral(self):
        # 0.5% change, default rel_threshold 1% → neutral
        r = qg.compare_quality({"clip_i": 0.7035}, {"clip_i": 0.70})
        assert r["per_metric"]["clip_i"]["status"] == "neutral"
        assert r["verdict"] == "NEUTRAL"

    def test_any_regression_dominates_verdict(self):
        cur = {"clip_i": 0.80, "fid": 20.0}   # clip_i improved, fid worse (↑)
        prev = {"clip_i": 0.70, "fid": 15.0}
        r = qg.compare_quality(cur, prev)
        assert r["verdict"] == "REGRESSION"   # a single regression wins
        assert r["improved_any"] is True

    def test_no_previous_is_no_baseline(self):
        r = qg.compare_quality({"clip_i": 0.7}, None)
        assert r["verdict"] == "NO_BASELINE"
        assert r["per_metric"]["clip_i"]["status"] == "no_baseline"

    def test_missing_baseline_metric_does_not_regress(self):
        # current has fid, previous doesn't → that metric has no baseline, and the
        # one comparable metric improved
        r = qg.compare_quality({"clip_i": 0.8, "fid": 12.0}, {"clip_i": 0.7})
        assert r["per_metric"]["fid"]["status"] == "no_baseline"
        assert r["verdict"] == "IMPROVED" and not r["regressed"]

    def test_none_values_skipped(self):
        r = qg.compare_quality({"clip_i": None}, {"clip_i": 0.7})
        assert r["per_metric"]["clip_i"]["status"] == "no_baseline"
        assert r["verdict"] == "NO_BASELINE"

    def test_custom_threshold(self):
        # 3% drop; threshold 5% → neutral; threshold 1% → regression
        assert qg.compare_quality({"clip_i": 0.679}, {"clip_i": 0.70},
                                  rel_threshold=0.05)["verdict"] == "NEUTRAL"
        assert qg.compare_quality({"clip_i": 0.679}, {"clip_i": 0.70},
                                  rel_threshold=0.01)["verdict"] == "REGRESSION"


# ---------------------------------------------------------------------------
# _golden_metrics_from_result — extract flat metrics from eval output
# ---------------------------------------------------------------------------

class TestGoldenMetricsExtract:
    KM = ("clip_i", "clip_t", "aesthetic", "lpips", "fid")

    def test_flat_result(self):
        out = qg._golden_metrics_from_result({"clip_i": 0.7, "fid": 12.0, "x": 1}, self.KM)
        assert out == {"clip_i": 0.7, "fid": 12.0}

    def test_champion_section(self):
        out = qg._golden_metrics_from_result({"champion": {"clip_i": 0.8}}, self.KM)
        assert out == {"clip_i": 0.8}

    def test_drops_none_metrics(self):
        out = qg._golden_metrics_from_result({"clip_i": 0.7, "lpips": None}, self.KM)
        assert out == {"clip_i": 0.7}


# ---------------------------------------------------------------------------
# run_quality_gate — glue with injected eval/registry (no GPU, no live registry)
# ---------------------------------------------------------------------------

class TestRunQualityGate:
    def test_end_to_end_with_fakes(self):
        registered = {}

        def fake_eval(ckpt):
            return {"clip_i": 0.80, "lpips": 0.30}

        def fake_prev(campaign):
            return {"clip_i": 0.70, "lpips": 0.40}

        def fake_register(ckpt, campaign, metrics):
            registered.update({"ckpt": ckpt, "campaign": campaign, "metrics": metrics})

        out = qg.run_quality_gate("/ckpt/step_0001000.safetensors", "warmup-run3",
                                  golden_eval=fake_eval, fetch_previous=fake_prev,
                                  register=fake_register)
        assert out["comparison"]["verdict"] == "IMPROVED"
        assert out["metrics"] == {"clip_i": 0.80, "lpips": 0.30}
        assert registered["metrics"] == {"clip_i": 0.80, "lpips": 0.30}
        assert registered["campaign"] == "warmup-run3"

    def test_register_failure_does_not_block_verdict(self):
        def boom(*a):
            raise RuntimeError("registry down")
        out = qg.run_quality_gate("/ckpt/x.safetensors", "c",
                                  golden_eval=lambda c: {"clip_i": 0.5},
                                  fetch_previous=lambda c: {"clip_i": 0.6},
                                  register=boom)
        assert out["comparison"]["verdict"] == "REGRESSION"   # verdict still computed

    def test_first_run_no_baseline(self):
        out = qg.run_quality_gate("/ckpt/x.safetensors", "new",
                                  golden_eval=lambda c: {"clip_i": 0.5},
                                  fetch_previous=lambda c: None,
                                  register=lambda *a: None)
        assert out["comparison"]["verdict"] == "NO_BASELINE"
