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


# ---------------------------------------------------------------------------
# Flywheel-phase liveness reconciliation (suppresses staleness false positives
# while a long precompute is demonstrably progressing).
# ---------------------------------------------------------------------------

def _fake_heartbeats(monkeypatch, *, pc=None, pc_age=None, fw=None, fw_age=None):
    """Stub pd.read_heartbeat / pd.heartbeat_age_secs for precompute + flywheel."""
    hbs = {"precompute": pc, "flywheel": fw}
    ages = {"precompute": pc_age, "flywheel": fw_age}
    monkeypatch.setattr(pd, "read_heartbeat", lambda proc, chunk=None: hbs.get(proc))
    monkeypatch.setattr(pd, "heartbeat_age_secs", lambda proc, chunk=None: ages.get(proc))


class TestFlywheelPhaseActive:
    def test_fresh_precompute_is_active_with_progress_label(self, doctor, monkeypatch):
        _fake_heartbeats(
            monkeypatch,
            pc={"process": "precompute", "done": 30, "total": 35,
                "current_phase": "qwen3 290/625", "eta_sec": 11327},
            pc_age=42.0,
            fw={"flywheel_name": "warmup-run1", "iteration": 11, "status": "precomputing"},
            fw_age=65000.0,  # flywheel hb is stale (only rewritten at transitions)
        )
        a = pd._flywheel_phase_active()
        assert a is not None
        assert a["process"] == "precompute"
        # Label merges stale flywheel context with live precompute progress.
        assert "warmup-run1" in a["label"]
        assert "iter 11" in a["label"]
        assert "shard 30/35" in a["label"]
        assert "ETA 3h" in a["label"]

    def test_stale_everything_is_none(self, doctor, monkeypatch):
        _fake_heartbeats(monkeypatch, pc={"done": 1}, pc_age=5000.0,
                         fw={"status": "x"}, fw_age=99999.0)
        assert pd._flywheel_phase_active() is None

    def test_no_heartbeats_is_none(self, doctor, monkeypatch):
        _fake_heartbeats(monkeypatch, pc=None, pc_age=None, fw=None, fw_age=None)
        assert pd._flywheel_phase_active() is None

    def test_fresh_flywheel_only_is_active(self, doctor, monkeypatch):
        # Precompute heartbeat absent/stale but flywheel fresh (e.g. between phases).
        _fake_heartbeats(monkeypatch, pc=None, pc_age=None,
                         fw={"flywheel_name": "run1", "iteration": 3, "status": "training"},
                         fw_age=30.0)
        a = pd._flywheel_phase_active()
        assert a is not None and a["process"] == "flywheel"
        assert "training" in a["label"] and "iter 3" in a["label"]


class TestOrchestratorLogGating:
    """The 'orchestrator may be down' warning must be downgraded to INFO while a
    flywheel phase is demonstrably alive, but still fire on a true stall."""

    def _stale_log(self, doctor, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        f = log_dir / "orchestrator.jsonl"
        f.write_text(json.dumps({"event": "poll", "ts": pd.now_iso(),
                                 "message": "polling"}) + "\n")
        import os as _os
        old = pd.time.time() - 7200  # 2h old mtime -> age > 600
        _os.utime(f, (old, old))
        return log_dir

    def test_downgraded_to_info_when_phase_active(self, doctor, tmp_path, monkeypatch):
        monkeypatch.setattr(pd, "LOG_DIR", self._stale_log(doctor, tmp_path))
        monkeypatch.setattr(pd, "_flywheel_phase_active",
                            lambda: {"process": "precompute", "age_s": 40.0,
                                     "label": "warmup-run1, iter 11, shard 30/35"})
        pd._check_orchestrator_log()
        orch = _by_category("orchestrator")
        assert any(i.severity == "INFO" and "phase active" in i.title for i in orch)
        assert not any(i.severity == "WARNING" and "may be down" in i.title for i in orch)

    def test_warns_when_no_phase_active(self, doctor, tmp_path, monkeypatch):
        monkeypatch.setattr(pd, "LOG_DIR", self._stale_log(doctor, tmp_path))
        monkeypatch.setattr(pd, "_flywheel_phase_active", lambda: None)
        pd._check_orchestrator_log()
        orch = _by_category("orchestrator")
        assert any(i.severity == "WARNING" and "may be down" in i.title for i in orch)


class TestColdStorageVersionGating:
    """Cold-vs-hot precompute version mismatch is expected mid-flywheel; it must
    be INFO while a phase is active and WARNING otherwise."""

    def _symlinks(self, tmp_path, hot_ver, cold_ver):
        precomp = tmp_path / "precomputed"
        cold = tmp_path / "cold"
        for enc in ("qwen3", "vae", "siglip"):
            (precomp / enc / hot_ver).mkdir(parents=True)
            (precomp / enc / "current").symlink_to(precomp / enc / hot_ver)
            (cold / "precomputed" / enc / cold_ver).mkdir(parents=True)
            (cold / "precomputed" / enc / "current").symlink_to(
                cold / "precomputed" / enc / cold_ver)
        return precomp, cold

    def test_mismatch_is_info_when_active(self, doctor, tmp_path, monkeypatch):
        precomp, cold = self._symlinks(tmp_path, "v_hot", "v_cold")
        monkeypatch.setattr(pd, "PRECOMP_DIR", precomp)
        monkeypatch.setattr(pd, "COLD_ROOT", cold)
        monkeypatch.setattr(pd, "_flywheel_phase_active",
                            lambda: {"process": "precompute", "age_s": 10.0, "label": "run1, iter 2"})
        pd._check_cold_storage({"storage": {"cold_root": str(cold)}}, [1])
        cs = _by_category("cold_storage")
        assert len(cs) == 3 and all(i.severity == "INFO" for i in cs)

    def test_mismatch_is_warning_when_idle(self, doctor, tmp_path, monkeypatch):
        precomp, cold = self._symlinks(tmp_path, "v_hot", "v_cold")
        monkeypatch.setattr(pd, "PRECOMP_DIR", precomp)
        monkeypatch.setattr(pd, "COLD_ROOT", cold)
        monkeypatch.setattr(pd, "_flywheel_phase_active", lambda: None)
        pd._check_cold_storage({"storage": {"cold_root": str(cold)}}, [1])
        cs = [i for i in _by_category("cold_storage") if "version mismatch" in i.title]
        assert len(cs) == 3 and all(i.severity == "WARNING" for i in cs)


# ---------------------------------------------------------------------------
# Flywheel failure-loop detection + iter-log forensic fingerprinting.
# The fingerprint tests are driven by REAL historical iter-log tails captured in
# train/tests/fixtures/flywheel_logs/ — mining our own logs as the test corpus.
# ---------------------------------------------------------------------------

import flywheel_lib

_FIXTURES = Path(__file__).parent / "fixtures" / "flywheel_logs"


def _fixture(name):
    return (_FIXTURES / name).read_text(errors="replace")


class TestFingerprint:
    def test_real_shard_cache_empty(self):
        d = pd._fingerprint_log_tail(_fixture("iter0010_shard_cache_empty.log"), 1)
        assert d is not None and d["id"] == "shard_cache_empty"
        assert d["severity"] == "CRITICAL"
        assert "No shards with precomputed" in d["evidence"]

    def test_real_segfault_siglip(self):
        # iter 7 has the mlx_vlm siglip pattern AND exit 139 — the more specific
        # siglip signature must win over the generic segfault one.
        d = pd._fingerprint_log_tail(_fixture("iter0007_segfault_siglip.log"), 139)
        assert d is not None and d["id"] == "siglip_module_missing"
        assert d["severity"] == "CRITICAL"

    def test_real_shape_broadcast(self):
        d = pd._fingerprint_log_tail(_fixture("iter0001_shape_broadcast.log"), 1)
        assert d is not None and d["id"] == "shape_broadcast"

    def test_pure_segfault_without_siglip(self):
        d = pd._fingerprint_log_tail("some native crash\nSegmentation fault\n", 139)
        assert d is not None and d["id"] == "segfault"

    def test_generic_traceback_fallback_carries_exception(self):
        log = ("Traceback (most recent call last):\n"
               "  File 'x.py', line 1\n"
               "KeyError: 'missing_key'\n")
        d = pd._fingerprint_log_tail(log, 1)
        assert d is not None and d["id"] == "generic_traceback"
        assert "KeyError: 'missing_key'" in d["summary"]

    def test_no_match_returns_none(self):
        assert pd._fingerprint_log_tail("everything is fine\nEXIT_CODE=0\n", 0) is None


class TestCountTrailingFailures:
    def test_counts_trailing_and_skips_inflight(self):
        iters = [
            {"iteration": 1, "status": "failed", "exit_code": 1},
            {"iteration": 2, "status": "failed", "exit_code": 1},
            {"iteration": 3, "status": "running", "exit_code": None},  # in-flight tail
        ]
        count, latest, codes = pd._count_trailing_failures(iters)
        assert count == 2 and latest["iteration"] == 2 and codes == [1, 1]

    def test_stops_at_success(self):
        iters = [
            {"iteration": 1, "status": "failed", "exit_code": 1},
            {"iteration": 2, "status": "done", "exit_code": 0},
            {"iteration": 3, "status": "failed", "exit_code": 139},
        ]
        count, latest, codes = pd._count_trailing_failures(iters)
        assert count == 1 and latest["iteration"] == 3 and codes == [139]

    def test_no_failures(self):
        iters = [{"iteration": 1, "status": "done", "exit_code": 0}]
        assert pd._count_trailing_failures(iters) == (0, None, [])


def _fake_db_factory(campaigns, iters_by_name):
    class _FakeDB:
        def __init__(self, path): pass
        def get_non_superseded_campaigns(self): return campaigns
        def get_iterations(self, name): return iters_by_name.get(name, [])
        def close(self): pass
    return _FakeDB


class TestCheckFlywheelFailures:
    def _wire(self, doctor, tmp_path, monkeypatch, campaigns, iters, log_name=None, log_text=None):
        dbfile = tmp_path / "flywheel.db"
        dbfile.touch()
        monkeypatch.setattr(pd, "FLYWHEEL_DB_PATH", dbfile)
        monkeypatch.setattr(flywheel_lib, "FlywheelDB",
                            _fake_db_factory(campaigns, iters))
        logs = tmp_path / "logs"
        logs.mkdir(exist_ok=True)
        monkeypatch.setattr(pd, "LOG_DIR", logs)
        if log_name and log_text is not None:
            (logs / log_name).write_text(log_text)

    def test_fires_critical_with_fingerprint(self, doctor, tmp_path, monkeypatch):
        iters = {"run1": [
            {"iteration": 8, "status": "failed", "exit_code": 1},
            {"iteration": 9, "status": "failed", "exit_code": 139},
            {"iteration": 10, "status": "failed", "exit_code": 1},
        ]}
        self._wire(doctor, tmp_path, monkeypatch,
                   [{"flywheel_name": "run1"}], iters,
                   log_name="flywheel_run1_iter0010.log",
                   log_text=_fixture("iter0010_shard_cache_empty.log"))
        pd._check_flywheel_failures({})
        fw = [i for i in pd._issues
              if i.category == "flywheel" and "consecutive iterations failed" in i.title]
        assert len(fw) == 1
        i = fw[0]
        assert i.severity == "CRITICAL"
        assert i.ctx["signature"] == "shard_cache_empty"
        assert i.ctx["consecutive_failures"] == 3
        assert i.ctx["ever_succeeded"] is False

    def test_silent_when_latest_succeeded(self, doctor, tmp_path, monkeypatch):
        iters = {"run1": [
            {"iteration": 1, "status": "failed", "exit_code": 1},
            {"iteration": 2, "status": "failed", "exit_code": 1},
            {"iteration": 3, "status": "done", "exit_code": 0},
        ]}
        self._wire(doctor, tmp_path, monkeypatch, [{"flywheel_name": "run1"}], iters)
        pd._check_flywheel_failures({})
        assert [i for i in pd._issues
                if i.category == "flywheel" and "consecutive iterations failed" in i.title] == []

    def test_systemic_flag_on_repeated_exit_code(self, doctor, tmp_path, monkeypatch):
        iters = {"run1": [
            {"iteration": 5, "status": "done", "exit_code": 0},
            {"iteration": 6, "status": "failed", "exit_code": 139},
            {"iteration": 7, "status": "failed", "exit_code": 139},
            {"iteration": 8, "status": "failed", "exit_code": 139},
        ]}
        self._wire(doctor, tmp_path, monkeypatch, [{"flywheel_name": "run1"}], iters,
                   log_name="flywheel_run1_iter0008.log",
                   log_text="native crash\nSegmentation fault\nEXIT_CODE=139\n")
        pd._check_flywheel_failures({})
        fw = [i for i in pd._issues if i.category == "flywheel"
              and "consecutive iterations failed" in i.title][0]
        assert fw.ctx["systemic"] is True
        assert "systemic" in fw.detail.lower()

    def test_warning_below_crit_threshold_with_prior_success(self, doctor, tmp_path, monkeypatch):
        # 2 trailing failures but the campaign HAS succeeded before → WARNING not CRITICAL.
        iters = {"run1": [
            {"iteration": 1, "status": "done", "exit_code": 0},
            {"iteration": 2, "status": "failed", "exit_code": 1},
            {"iteration": 3, "status": "failed", "exit_code": 1},
        ]}
        self._wire(doctor, tmp_path, monkeypatch, [{"flywheel_name": "run1"}], iters,
                   log_name="flywheel_run1_iter0003.log",
                   log_text="Traceback (most recent call last):\nValueError: x\nEXIT_CODE=1\n")
        pd._check_flywheel_failures({})
        fw = [i for i in pd._issues if i.category == "flywheel"
              and "consecutive iterations failed" in i.title][0]
        assert fw.severity == "WARNING"


class TestFlywheelCacheCoverage:
    """_flywheel_iter_cache_coverage + the shard_cache_empty auto-confirmation:
    codifies the manual forensics that distinguished a cache-dir RESOLUTION bug
    (npz present, trainer read the wrong dir) from genuinely-missing precompute."""

    def _wire_cov(self, tmp_path, monkeypatch, ids, present_ids, iteration=5):
        dbfile = tmp_path / "fw.db"
        dbfile.touch()
        monkeypatch.setattr(pd, "FLYWHEEL_DB_PATH", dbfile)
        iters = {"run1": [{"iteration": iteration, "status": "failed", "exit_code": 1,
                           "selected_shards": json.dumps(ids)}]}
        monkeypatch.setattr(flywheel_lib, "FlywheelDB",
                            _fake_db_factory([{"flywheel_name": "run1"}], iters))
        cold = tmp_path / "cold"
        for enc in ("qwen3", "vae"):
            cur = cold / "precomputed" / enc / "current"
            cur.mkdir(parents=True)
            for sid in present_ids:
                for i in (0, 49):
                    (cur / f"{sid}_{i:04d}.npz").write_bytes(b"x")
        monkeypatch.setattr(pd, "COLD_ROOT", cold)
        monkeypatch.setattr(pd, "PRECOMP_DIR", tmp_path / "hot_precomp")

    def test_all_present(self, doctor, tmp_path, monkeypatch):
        ids = ["000000", "000001", "000002"]
        self._wire_cov(tmp_path, monkeypatch, ids, ids)
        assert pd._flywheel_iter_cache_coverage("run1", 5) == (3, 3)

    def test_partial_coverage(self, doctor, tmp_path, monkeypatch):
        ids = ["000000", "000001", "000002", "000003"]
        self._wire_cov(tmp_path, monkeypatch, ids, ids[:1])
        assert pd._flywheel_iter_cache_coverage("run1", 5) == (1, 4)

    def test_no_cold_returns_none(self, doctor, tmp_path, monkeypatch):
        dbfile = tmp_path / "fw.db"; dbfile.touch()
        monkeypatch.setattr(pd, "FLYWHEEL_DB_PATH", dbfile)
        monkeypatch.setattr(flywheel_lib, "FlywheelDB", _fake_db_factory(
            [{"flywheel_name": "run1"}],
            {"run1": [{"iteration": 5, "selected_shards": json.dumps(["000000"])}]}))
        monkeypatch.setattr(pd, "COLD_ROOT", tmp_path / "nope")
        monkeypatch.setattr(pd, "PRECOMP_DIR", tmp_path / "nope2")
        assert pd._flywheel_iter_cache_coverage("run1", 5) == (None, None)

    def test_failure_check_confirms_resolution_bug(self, doctor, tmp_path, monkeypatch):
        ids = ["000000", "000001", "000002"]
        dbfile = tmp_path / "fw.db"; dbfile.touch()
        monkeypatch.setattr(pd, "FLYWHEEL_DB_PATH", dbfile)
        iters = {"run1": [
            {"iteration": 9,  "status": "failed", "exit_code": 1, "selected_shards": json.dumps(ids)},
            {"iteration": 10, "status": "failed", "exit_code": 1, "selected_shards": json.dumps(ids)},
        ]}
        monkeypatch.setattr(flywheel_lib, "FlywheelDB",
                            _fake_db_factory([{"flywheel_name": "run1"}], iters))
        logs = tmp_path / "logs"; logs.mkdir()
        (logs / "flywheel_run1_iter0010.log").write_text(_fixture("iter0010_shard_cache_empty.log"))
        monkeypatch.setattr(pd, "LOG_DIR", logs)
        cold = tmp_path / "cold"
        for enc in ("qwen3", "vae"):
            cur = cold / "precomputed" / enc / "current"; cur.mkdir(parents=True)
            for sid in ids:
                for i in (0, 49):
                    (cur / f"{sid}_{i:04d}.npz").write_bytes(b"x")
        monkeypatch.setattr(pd, "COLD_ROOT", cold)
        monkeypatch.setattr(pd, "PRECOMP_DIR", tmp_path / "hot")

        pd._check_flywheel_failures({})
        fw = [i for i in pd._issues if i.category == "flywheel"
              and "consecutive iterations failed" in i.title][0]
        assert fw.ctx["cache_coverage"] == {"present": 3, "total": 3}
        assert "cache-dir resolution bug" in fw.detail.lower()
        assert "confirmed" in fw.detail.lower()


class TestFlywheelTrainerAnomalies:
    """The flywheel trainer writes a chunkless `trainer.json`; it must get the same
    in-flight anomaly checks the chunked per-chunk trainers do (previously absent)."""

    def _stub(self, monkeypatch, trainer_hb, trainer_age=30.0):
        def _rh(proc, chunk=None):
            return trainer_hb if (proc == "trainer" and chunk is None) else None
        def _age(proc, chunk=None):
            return trainer_age if (proc == "trainer" and chunk is None) else None
        monkeypatch.setattr(pd, "read_heartbeat", _rh)
        monkeypatch.setattr(pd, "heartbeat_age_secs", _age)

    def test_high_loss_fires_with_flywheel_label(self, doctor, monkeypatch):
        self._stub(monkeypatch, {"step": 500, "loss": 5.0})
        pd._check_training_anomalies({"anomaly": {"loss_threshold": 2.0},
                                      "training": {"siglip": False}}, [])
        a = _by_category("anomaly")
        assert any(i.severity == "WARNING" and i.title.startswith("Flywheel trainer: loss")
                   for i in a)

    def test_booting_heartbeat_is_silent(self, doctor, monkeypatch):
        # step 0 (booting) must not raise anomalies even with a bad-looking loss.
        self._stub(monkeypatch, {"status": "booting", "step": 0, "loss": 99.0})
        pd._check_training_anomalies({"anomaly": {}, "training": {}}, [])
        assert _by_category("anomaly") == []

    def test_stale_heartbeat_is_silent(self, doctor, monkeypatch):
        self._stub(monkeypatch, {"step": 500, "loss": 5.0}, trainer_age=5000.0)
        pd._check_training_anomalies({"anomaly": {"loss_threshold": 2.0}, "training": {}}, [])
        assert _by_category("anomaly") == []

    def test_ip_adapter_not_learning_flywheel(self, doctor, monkeypatch):
        self._stub(monkeypatch, {"step": 1500, "loss_cond": 0.500, "loss_null": 0.505})
        pd._check_training_anomalies({"anomaly": {}, "training": {}}, [])
        a = _by_category("anomaly")
        assert any("IP adapter not learning" in i.title and i.title.startswith("Flywheel trainer")
                   for i in a)


class TestCampaignEta:
    """_check_campaign_eta estimates campaign wall-clock from live precompute s/shard
    × configured n_shards × remaining iterations."""

    def _wire(self, tmp_path, monkeypatch, *, pc, pc_age, fw, max_iters=15, n_shards=40):
        cfgp = tmp_path / "fw.yaml"
        cfgp.write_text(f"flywheel:\n  max_iterations: {max_iters}\n  n_shards: {n_shards}\n")
        dbfile = tmp_path / "fw.db"; dbfile.touch()
        monkeypatch.setattr(pd, "FLYWHEEL_DB_PATH", dbfile)
        monkeypatch.setattr(flywheel_lib, "FlywheelDB", _fake_db_factory(
            [{"flywheel_name": "run1", "config_path": str(cfgp)}], {}))
        def _rh(proc, chunk=None):
            return {"precompute": pc, "flywheel": fw}.get(proc)
        def _age(proc, chunk=None):
            return {"precompute": pc_age, "flywheel": 30.0}.get(proc)
        monkeypatch.setattr(pd, "read_heartbeat", _rh)
        monkeypatch.setattr(pd, "heartbeat_age_secs", _age)

    def test_eta_uses_n_shards_not_heartbeat_total(self, doctor, tmp_path, monkeypatch):
        # heartbeat total=5 (a pass batch), but per-iter should use n_shards=40.
        self._wire(tmp_path, monkeypatch,
                   pc={"done": 4, "total": 5, "eta_sec": 1800}, pc_age=40.0,
                   fw={"flywheel_name": "run1", "iteration": 12})
        pd._check_campaign_eta({})
        e = [i for i in pd._issues if i.category == "flywheel" and "ETA" in i.title][0]
        assert e.ctx["s_per_shard"] == 1800.0
        assert e.ctx["per_iter_hours"] == 20.0      # 1800*40/3600, NOT 1800*5
        assert e.severity == "INFO"                  # ~2.5d < 7d default

    def test_warns_past_budget(self, doctor, tmp_path, monkeypatch):
        self._wire(tmp_path, monkeypatch,
                   pc={"done": 4, "total": 5, "eta_sec": 1800}, pc_age=40.0,
                   fw={"flywheel_name": "run1", "iteration": 12})
        pd._check_campaign_eta({"flywheel_health": {"max_campaign_days": 1}})
        e = [i for i in pd._issues if i.category == "flywheel" and "ETA" in i.title][0]
        assert e.severity == "WARNING"

    def test_silent_without_active_precompute(self, doctor, tmp_path, monkeypatch):
        self._wire(tmp_path, monkeypatch, pc=None, pc_age=None,
                   fw={"flywheel_name": "run1", "iteration": 12})
        pd._check_campaign_eta({})
        assert [i for i in pd._issues if "ETA" in i.title] == []
