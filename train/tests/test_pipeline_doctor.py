"""
train/tests/test_pipeline_doctor.py — black-box tests for doctor _check_* logic.

Covers GROK-TEST-3: the doctor's diagnostic checks were trusted from production
use but never unit-verified against synthetic state. These feed each check
hand-built inputs (cfg dicts, sentinel/eval files in a tempdir) and assert the
exact issues it reports — severity, category, and machine-readable ctx.

Hermetic and flywheel-safe: monkeypatches the doctor's path globals
(DATA_ROOT, SENTINEL_DIR) to a tempdir in THIS process only and clears the
module-level _issues list per test. No GPU, DB, git, or subprocess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import pipeline_doctor as pd


@pytest.fixture
def doctor(tmp_path, monkeypatch):
    """Redirect doctor path globals to a tempdir; reset _issues; yield helpers."""
    monkeypatch.setattr(pd, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pd, "SENTINEL_DIR", tmp_path / "pipeline")
    pd._issues.clear()
    yield pd
    pd._issues.clear()


def _by_category(cat):
    return [i for i in pd._issues if i.category == cat]


# ---------------------------------------------------------------------------
# _check_proxy_vae
# ---------------------------------------------------------------------------

class TestCheckProxyVae:
    def test_disabled_and_no_path_is_silent(self, doctor):
        doctor._check_proxy_vae({"proxy_vae": {"enabled": False, "proxy_path": None}})
        assert _by_category("proxy_vae") == []

    def test_unconfigured_is_silent(self, doctor):
        doctor._check_proxy_vae({})            # no proxy_vae key at all
        assert _by_category("proxy_vae") == []

    def test_enabled_but_no_path_warns(self, doctor):
        doctor._check_proxy_vae({"proxy_vae": {"enabled": True, "proxy_path": None}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "WARNING"
        assert "proxy_path is null" in issues[0].title

    def test_missing_checkpoint_warns(self, doctor):
        doctor._check_proxy_vae({"proxy_vae": {
            "enabled": True, "proxy_path": "/no/such/proxy.safetensors"}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "WARNING"
        assert issues[0].ctx.get("exists") is False

    def test_configured_but_no_eval_report_is_info(self, doctor, tmp_path):
        proxy = tmp_path / "proxy.safetensors"
        proxy.touch()
        doctor._check_proxy_vae({"proxy_vae": {
            "enabled": True, "proxy_path": str(proxy), "default_mode": "balanced"}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "INFO"
        assert "no evaluation report" in issues[0].title

    def test_failed_quality_gates_warn(self, doctor, tmp_path):
        proxy = tmp_path / "proxy.safetensors"
        proxy.touch()
        (tmp_path / "proxy_vae_eval.json").write_text(json.dumps({
            "tier1": {"cosine_sim": 0.80, "ch_std_ratio": 0.70, "fft_corr": 0.90,
                      "pass_cosine": False, "pass_ch_std": False, "pass_fft": False},
            "tier2": {"decoded_psnr_db": 28.0, "pass_psnr": False},
            "tier3": {"proxy_stats": {"fallback_rate": 0.4}},
        }))
        doctor._check_proxy_vae({"proxy_vae": {
            "enabled": True, "proxy_path": str(proxy)}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "WARNING"
        assert "quality gates failed" in issues[0].title
        assert len(issues[0].ctx["failed_gates"]) == 4   # all four gates failed

    def test_healthy_report_is_info(self, doctor, tmp_path):
        proxy = tmp_path / "proxy.safetensors"
        proxy.touch()
        (tmp_path / "proxy_vae_eval.json").write_text(json.dumps({
            "tier1": {"cosine_sim": 0.97, "ch_std_ratio": 0.99, "fft_corr": 0.99,
                      "pass_cosine": True, "pass_ch_std": True, "pass_fft": True},
            "tier2": {"decoded_psnr_db": 38.0, "pass_psnr": True},
            "tier3": {"proxy_stats": {"fallback_rate": 0.02}},
        }))
        doctor._check_proxy_vae({"proxy_vae": {
            "enabled": True, "proxy_path": str(proxy), "default_mode": "high_fidelity"}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "INFO"
        assert "healthy" in issues[0].title
        assert issues[0].ctx["fallback_rate"] == 0.02

    def test_unreadable_report_warns(self, doctor, tmp_path):
        proxy = tmp_path / "proxy.safetensors"
        proxy.touch()
        (tmp_path / "proxy_vae_eval.json").write_text("{ not valid json ")
        doctor._check_proxy_vae({"proxy_vae": {
            "enabled": True, "proxy_path": str(proxy)}})
        issues = _by_category("proxy_vae")
        assert len(issues) == 1
        assert issues[0].severity == "WARNING"
        assert "unreadable" in issues[0].title


# ---------------------------------------------------------------------------
# _check_error_sentinels
# ---------------------------------------------------------------------------

class TestCheckErrorSentinels:
    def _chunk_dir(self, doctor, chunk):
        d = doctor.SENTINEL_DIR / f"chunk{chunk}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_no_sentinel_dir_is_silent(self, doctor):
        doctor._check_error_sentinels([1, 2, 3])
        assert _by_category("error_sentinel") == []

    def test_error_file_raises_critical(self, doctor):
        d = self._chunk_dir(doctor, 2)
        (d / "precompute.error").write_text("2026-06-03\nout of memory\n")
        doctor._check_error_sentinels([2])
        issues = _by_category("error_sentinel")
        assert len(issues) == 1
        assert issues[0].severity == "CRITICAL"
        assert issues[0].chunk == 2
        assert issues[0].ctx["step"] == "precompute"
        assert "out of memory" in issues[0].ctx["error_content"]

    def test_error_with_matching_done_is_skipped(self, doctor):
        # .error AND .done for the same step → handled by the phantom check, not here.
        d = self._chunk_dir(doctor, 1)
        (d / "train.error").write_text("transient\n")
        (d / "train.done").touch()
        doctor._check_error_sentinels([1])
        assert _by_category("error_sentinel") == []

    def test_multiple_errors_across_chunks(self, doctor):
        d1 = self._chunk_dir(doctor, 1)
        d2 = self._chunk_dir(doctor, 2)
        (d1 / "build_shards.error").write_text("disk full\n")
        (d2 / "mine.error").write_text("nan\n")
        doctor._check_error_sentinels([1, 2, 3])
        issues = _by_category("error_sentinel")
        assert len(issues) == 2
        assert {i.chunk for i in issues} == {1, 2}
