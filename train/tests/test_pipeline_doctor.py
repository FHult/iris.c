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
import pipeline_lib


@pytest.fixture
def doctor(tmp_path, monkeypatch):
    """Redirect doctor path globals to a tempdir; reset _issues; yield helpers.

    Patches SENTINEL_DIR in BOTH modules: the doctor's own binding (used by
    checks that read SENTINEL_DIR directly) and pipeline_lib's (used by the
    is_done/has_error helpers the checks call). They must agree.
    """
    sent = tmp_path / "pipeline"
    monkeypatch.setattr(pd, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pd, "SENTINEL_DIR", sent)
    monkeypatch.setattr(pipeline_lib, "SENTINEL_DIR", sent)
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


# ---------------------------------------------------------------------------
# _check_stale_logs — a step's log older than its .done sentinel is from a
# prior run and will mislead. (LOG_DIR lambdas in _STEP_LOGS read the global.)
# ---------------------------------------------------------------------------

class TestCheckStaleLogs:
    @pytest.fixture
    def logs(self, doctor, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(doctor, "LOG_DIR", log_dir)
        (doctor.SENTINEL_DIR / "chunk1").mkdir(parents=True)
        return doctor, log_dir

    def test_log_older_than_sentinel_warns(self, logs):
        import os
        doctor, log_dir = logs
        sent = doctor.SENTINEL_DIR / "chunk1" / "train.done"
        sent.touch()
        log = log_dir / "train_chunk1.log"
        log.touch()
        # Make the log 1h older than the sentinel (well beyond the 300s grace).
        os.utime(log, (sent.stat().st_mtime - 3600, sent.stat().st_mtime - 3600))
        doctor._check_stale_logs([1])
        assert len(_by_category("stale_log")) >= 1

    def test_fresh_log_is_silent(self, logs):
        import os
        doctor, log_dir = logs
        sent = doctor.SENTINEL_DIR / "chunk1" / "train.done"
        sent.touch()
        log = log_dir / "train_chunk1.log"
        log.touch()
        # Log newer than sentinel → not stale.
        os.utime(log, (sent.stat().st_mtime + 10, sent.stat().st_mtime + 10))
        doctor._check_stale_logs([1])
        assert _by_category("stale_log") == []

    def test_no_sentinel_is_silent(self, logs):
        doctor, log_dir = logs
        (log_dir / "train_chunk1.log").touch()   # log but no .done sentinel
        doctor._check_stale_logs([1])
        assert _by_category("stale_log") == []


# ---------------------------------------------------------------------------
# _check_phantom_completions — "promoted.done but the data isn't there".
# The headline phantom detector (same class as iter-10: looks done, data absent).
# ---------------------------------------------------------------------------

class TestCheckPhantomCompletions:
    @pytest.fixture
    def phantom(self, doctor, tmp_path, monkeypatch):
        shards = tmp_path / "shards"
        precomp = tmp_path / "precomputed"
        shards.mkdir()
        precomp.mkdir()
        monkeypatch.setattr(doctor, "SHARDS_DIR", shards)
        monkeypatch.setattr(doctor, "PRECOMP_DIR", precomp)
        (doctor.SENTINEL_DIR / "chunk1").mkdir(parents=True)
        cfg = {"scale": "small", "training": {"steps": {"small": 1000}}}
        return doctor, shards, cfg

    def test_promoted_but_no_shards_is_critical(self, phantom):
        doctor, shards, cfg = phantom
        (doctor.SENTINEL_DIR / "chunk1" / "promoted.done").touch()
        doctor._check_phantom_completions(cfg, [1])
        crit = [i for i in doctor._issues
                if i.category == "phantom" and i.severity == "CRITICAL"]
        assert len(crit) == 1
        assert crit[0].ctx["shard_count"] == 0

    def test_promoted_with_shards_in_range_is_ok(self, phantom):
        doctor, shards, cfg = phantom
        (doctor.SENTINEL_DIR / "chunk1" / "promoted.done").touch()
        (shards / "000005.tar").touch()      # id 5 ∈ chunk 1 range [0, 200000)
        doctor._check_phantom_completions(cfg, [1])
        crit = [i for i in doctor._issues
                if i.category == "phantom" and i.severity == "CRITICAL"]
        assert crit == []

    def test_not_promoted_is_silent(self, phantom):
        doctor, shards, cfg = phantom
        # No promoted.done → nothing to validate.
        doctor._check_phantom_completions(cfg, [1])
        assert [i for i in doctor._issues if i.category == "phantom"] == []
