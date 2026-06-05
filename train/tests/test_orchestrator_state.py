"""
train/tests/test_orchestrator_state.py — orchestrator pure state-machine.

Covers the highest-priority untested seam flagged by the testing-suite review
(GROK-TEST-2): derive_chunk_state, the CHUNK_STEPS/_STEP_TO_STATE contract, and
ResourceManager token semantics — the logic that decides what the pipeline does
next from on-disk sentinels.

Fully hermetic and flywheel-safe: monkeypatches pipeline_lib.SENTINEL_DIR to a
tempdir in THIS test process only. The live orchestrator runs in a separate
interpreter with its own SENTINEL_DIR, so these tests cannot touch real state.
No GPU, no tmux, no subprocess — only sentinel files + in-memory dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# conftest.py puts train/ on sys.path; train/scripts is needed for orchestrator.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pipeline_lib
import orchestrator as orch
from orchestrator import ChunkState, CHUNK_STEPS, _STEP_TO_STATE, derive_chunk_state


# ---------------------------------------------------------------------------
# Redirect sentinels to a tempdir for every test (auto-restored by monkeypatch)
# ---------------------------------------------------------------------------

@pytest.fixture
def sentinels(tmp_path, monkeypatch):
    """Point pipeline_lib.SENTINEL_DIR at a tempdir; yield mark helpers."""
    monkeypatch.setattr(pipeline_lib, "SENTINEL_DIR", tmp_path)
    return pipeline_lib


# ---------------------------------------------------------------------------
# CHUNK_STEPS / _STEP_TO_STATE contract
# ---------------------------------------------------------------------------

class TestStepStateContract:
    def test_maps_are_in_sync(self):
        # The orchestrator asserts this at import; make the invariant explicit so
        # a future edit to one list without the other fails here with a clear name.
        assert set(_STEP_TO_STATE) == set(CHUNK_STEPS)

    def test_every_state_value_is_known(self):
        valid = {v for k, v in vars(ChunkState).items() if not k.startswith("_")}
        for step, state in _STEP_TO_STATE.items():
            assert state in valid, f"{step} → unknown state {state}"


# ---------------------------------------------------------------------------
# derive_chunk_state
# ---------------------------------------------------------------------------

class TestDeriveChunkState:
    def test_no_sentinels_is_idle(self, sentinels):
        assert derive_chunk_state(1) == ChunkState.IDLE

    def test_each_step_maps_to_next_state(self, sentinels):
        # Marking steps done in order should walk the state machine exactly per
        # _STEP_TO_STATE (last-done step decides the state).
        for step in CHUNK_STEPS:
            sentinels.mark_done(3, step)
            assert derive_chunk_state(3) == _STEP_TO_STATE[step], \
                f"after {step} done, expected {_STEP_TO_STATE[step]}"

    def test_all_done_is_done(self, sentinels):
        for step in CHUNK_STEPS:
            sentinels.mark_done(2, step)
        assert derive_chunk_state(2) == ChunkState.DONE

    def test_last_done_wins_regardless_of_mark_order(self, sentinels):
        # Mark a later step first, then an earlier one — state follows the latest
        # step in CHUNK_STEPS order, not the mark order.
        sentinels.mark_done(1, "precompute")     # later in the list
        sentinels.mark_done(1, "convert")        # earlier in the list
        assert derive_chunk_state(1) == _STEP_TO_STATE["precompute"]

    def test_error_takes_precedence_over_done(self, sentinels):
        sentinels.mark_done(1, "build_shards")
        sentinels.mark_error(1, "precompute", "boom")
        assert derive_chunk_state(1) == ChunkState.ERROR

    def test_error_on_any_step_is_error(self, sentinels):
        # Even an error on the very first step, with nothing done, is ERROR.
        sentinels.mark_error(4, "download", "net fail")
        assert derive_chunk_state(4) == ChunkState.ERROR

    def test_chunks_are_independent(self, sentinels):
        sentinels.mark_done(1, "train")
        sentinels.mark_error(2, "precompute", "x")
        assert derive_chunk_state(1) == _STEP_TO_STATE["train"]
        assert derive_chunk_state(2) == ChunkState.ERROR
        assert derive_chunk_state(3) == ChunkState.IDLE


# ---------------------------------------------------------------------------
# ResourceManager (non-GPU tokens only — GPU_TOKEN touches the real lock)
# ---------------------------------------------------------------------------

class TestResourceManager:
    def test_request_grants_and_records_holder(self):
        rm = orch.ResourceManager()
        assert rm.request("DISK_WRITE_HIGH", "stager") is True
        assert rm.holder("DISK_WRITE_HIGH") == "stager"

    def test_double_request_is_refused(self):
        rm = orch.ResourceManager()
        assert rm.request("DISK_WRITE_HIGH", "a") is True
        assert rm.request("DISK_WRITE_HIGH", "b") is False
        assert rm.holder("DISK_WRITE_HIGH") == "a"   # original holder unchanged

    def test_release_frees_token(self):
        rm = orch.ResourceManager()
        rm.request("DISK_WRITE_HIGH", "a")
        rm.release("DISK_WRITE_HIGH")
        assert rm.holder("DISK_WRITE_HIGH") is None
        assert rm.request("DISK_WRITE_HIGH", "b") is True

    def test_distinct_tokens_independent(self):
        rm = orch.ResourceManager()
        assert rm.request("DISK_WRITE_HIGH", "a") is True
        assert rm.request("OTHER_TOKEN", "b") is True
        assert rm.holder("DISK_WRITE_HIGH") == "a"
        assert rm.holder("OTHER_TOKEN") == "b"

    def test_release_unheld_token_is_noop(self):
        rm = orch.ResourceManager()
        rm.release("DISK_WRITE_HIGH")   # never requested — must not raise
        assert rm.holder("DISK_WRITE_HIGH") is None


# ---------------------------------------------------------------------------
# _resolve_proxy_vae_args — config → precompute CLI flags (+ campaign overrides)
# ---------------------------------------------------------------------------

class TestResolveProxyVaeArgs:
    @pytest.fixture(autouse=True)
    def _silence_log(self, monkeypatch):
        # The missing-path branch calls log_orch; stub it so tests never write
        # to the live orchestrator log.
        monkeypatch.setattr(orch, "log_orch", lambda *a, **k: None)

    def test_disabled_returns_empty(self, tmp_path):
        p = tmp_path / "proxy.safetensors"; p.touch()
        cfg = {"proxy_vae": {"enabled": False, "proxy_path": str(p)}}
        assert orch._resolve_proxy_vae_args(cfg) == ""

    def test_no_path_returns_empty(self):
        cfg = {"proxy_vae": {"enabled": True, "proxy_path": None}}
        assert orch._resolve_proxy_vae_args(cfg) == ""

    def test_missing_file_returns_empty(self):
        cfg = {"proxy_vae": {"enabled": True, "proxy_path": "/no/such.safetensors"}}
        assert orch._resolve_proxy_vae_args(cfg) == ""

    def test_enabled_emits_default_flags(self, tmp_path):
        p = tmp_path / "proxy.safetensors"; p.touch()
        cfg = {"proxy_vae": {"enabled": True, "proxy_path": str(p)}}
        out = orch._resolve_proxy_vae_args(cfg)
        assert f"--proxy-vae '{p}'" in out
        assert "--proxy-mode balanced" in out
        assert "--proxy-vae-threshold 0.75" in out

    def test_campaign_override_applies(self, tmp_path):
        p = tmp_path / "proxy.safetensors"; p.touch()
        cfg = {"proxy_vae": {
            "enabled": True, "proxy_path": str(p),
            "default_mode": "balanced", "fallback_threshold": 0.75,
            "campaigns": {"wikiart": {"mode": "high_fidelity",
                                      "fallback_threshold": 0.9}}}}
        out = orch._resolve_proxy_vae_args(cfg, campaign="wikiart")
        assert "--proxy-mode high_fidelity" in out
        assert "--proxy-vae-threshold 0.9" in out

    def test_unknown_campaign_uses_defaults(self, tmp_path):
        p = tmp_path / "proxy.safetensors"; p.touch()
        cfg = {"proxy_vae": {
            "enabled": True, "proxy_path": str(p), "default_mode": "speed",
            "campaigns": {"wikiart": {"mode": "high_fidelity"}}}}
        out = orch._resolve_proxy_vae_args(cfg, campaign="not-listed")
        assert "--proxy-mode speed" in out


# ---------------------------------------------------------------------------
# Crash diagnosis + retry/backoff policy (GROK-TEST-2: jetsam vs code-error).
# ---------------------------------------------------------------------------

class TestExitCodeParse:
    def test_extracts_code(self):
        assert orch._parse_exit_code_from_msg("Training exited 137; jetsam") == 137

    def test_missing_or_empty_returns_minus1(self):
        assert orch._parse_exit_code_from_msg("no code here") == -1
        assert orch._parse_exit_code_from_msg("") == -1
        assert orch._parse_exit_code_from_msg(None) == -1


class TestRetryPolicy:
    def test_jetsam_retries_with_backoff(self):
        should, mx, delay = orch._retry_policy("jetsam_oom", 0)
        assert should is True
        assert mx == orch.JETSAM_MAX_RETRIES and delay == orch.JETSAM_RETRY_DELAY_S

    def test_jetsam_stops_at_limit(self):
        should, _, _ = orch._retry_policy("jetsam_oom", orch.JETSAM_MAX_RETRIES)
        assert should is False

    def test_code_error_single_retry_no_delay(self):
        should, mx, delay = orch._retry_policy("code_error", 0)
        assert should is True and mx == 1 and delay == 0

    def test_code_error_stops_after_one(self):
        should, _, _ = orch._retry_policy("code_error", 1)
        assert should is False


class TestDiagnoseCrash:
    def test_non_137_is_code_error(self, tmp_path):
        log = tmp_path / "t.log"; log.write_text("boom\n")
        reason, detail = orch._diagnose_crash(log, 1)
        assert reason == "code_error" and "exit 1" in detail

    def test_137_jetsam_confirmed(self, tmp_path, monkeypatch):
        log = tmp_path / "t.log"; log.write_text("x\n")
        monkeypatch.setattr(orch, "_query_macos_jetsam_log", lambda *a, **k: True)
        reason, detail = orch._diagnose_crash(log, 137)
        assert reason == "jetsam_oom" and "confirmed" in detail

    def test_137_jetsam_assumed_when_log_silent(self, tmp_path, monkeypatch):
        log = tmp_path / "t.log"; log.write_text("x\n")
        monkeypatch.setattr(orch, "_query_macos_jetsam_log", lambda *a, **k: False)
        reason, detail = orch._diagnose_crash(log, 137)
        assert reason == "jetsam_oom" and "assumed" in detail


class TestParseLastMem:
    def test_extracts_last_mem(self, tmp_path):
        log = tmp_path / "t.log"
        log.write_text("step 1 mem: 10.0 GB used  5.0 GB free\n"
                       "step 2 mem: 12.0 GB used  3.0 GB free\n")
        assert orch._parse_last_mem_from_log(log) == "12.0 GB used  3.0 GB free"

    def test_no_mem_returns_empty(self, tmp_path):
        log = tmp_path / "t.log"; log.write_text("no memory here\n")
        assert orch._parse_last_mem_from_log(log) == ""


# ---------------------------------------------------------------------------
# GROK-TEST-2 remainder: chunk-transition gate, staging gate, dispatch seeding.
# ---------------------------------------------------------------------------

class TestReadyGate:
    def test_waits_for_prev_train(self):
        assert orch._ready_gate(2, prev_train_done=False, gpu_free=True,
                                stager_enabled=False, stage_done=True,
                                stage_error=False) == "wait_prev_train"

    def test_waits_for_gpu(self):
        assert orch._ready_gate(1, prev_train_done=True, gpu_free=False,
                                stager_enabled=False, stage_done=True,
                                stage_error=False) == "wait_gpu"

    def test_waits_for_staging(self):
        assert orch._ready_gate(2, prev_train_done=True, gpu_free=True,
                                stager_enabled=True, stage_done=False,
                                stage_error=False) == "wait_stage"

    def test_proceeds_without_stage_on_stage_error(self):
        assert orch._ready_gate(2, prev_train_done=True, gpu_free=True,
                                stager_enabled=True, stage_done=False,
                                stage_error=True) == "proceed_no_stage"

    def test_proceeds_when_all_clear(self):
        assert orch._ready_gate(2, prev_train_done=True, gpu_free=True,
                                stager_enabled=True, stage_done=True,
                                stage_error=False) == "proceed"

    def test_chunk1_not_gated_on_staging(self):
        # chunk 1 never waits for staging even when the stager is enabled.
        assert orch._ready_gate(1, prev_train_done=True, gpu_free=True,
                                stager_enabled=True, stage_done=False,
                                stage_error=False) == "proceed"


class TestShouldAttemptStage:
    def test_chunk1_never_staged(self):
        assert orch._should_attempt_stage(1, predecessor_promoted=True, stage_done=False) is False

    def test_predecessor_not_promoted(self):
        assert orch._should_attempt_stage(2, predecessor_promoted=False, stage_done=False) is False

    def test_ready_and_unstaged(self):
        assert orch._should_attempt_stage(2, predecessor_promoted=True, stage_done=False) is True

    def test_already_staged(self):
        assert orch._should_attempt_stage(2, predecessor_promoted=True, stage_done=True) is False


class TestLoadOpenDispatchIds:
    def _queue(self, tmp_path, monkeypatch, lines):
        import json as _j
        q = tmp_path / "dispatch_queue.jsonl"
        q.write_text("\n".join(_j.dumps(x) for x in lines) + "\n")
        monkeypatch.setattr(orch, "DISPATCH_QUEUE", q)

    def test_open_ids_returned(self, tmp_path, monkeypatch):
        self._queue(tmp_path, monkeypatch,
                    [{"id": "a", "resolved": False}, {"id": "b"}])
        assert orch._load_open_dispatch_ids() == {"a", "b"}

    def test_resolved_after_open_is_excluded(self, tmp_path, monkeypatch):
        # later 'resolved' entry for the same id wins → not open.
        self._queue(tmp_path, monkeypatch,
                    [{"id": "a", "resolved": False}, {"id": "a", "resolved": True}])
        assert orch._load_open_dispatch_ids() == set()

    def test_missing_file_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(orch, "DISPATCH_QUEUE", tmp_path / "nope.jsonl")
        assert orch._load_open_dispatch_ids() == set()


# ---------------------------------------------------------------------------
# _ablation_warmstart_ckpt — opt-in arm warm-start decision (default off)
# ---------------------------------------------------------------------------

class TestAblationWarmstartCkpt:
    def test_flag_off_returns_none(self, tmp_path):
        (tmp_path / "step_0000300.safetensors").write_bytes(b"x")
        # default (key absent) and explicit-false both cold-start
        assert orch._ablation_warmstart_ckpt({}, tmp_path) is None
        assert orch._ablation_warmstart_ckpt({"ablation_warmstart_arms": False}, tmp_path) is None

    def test_flag_on_returns_latest_ckpt(self, tmp_path):
        for s in ("step_0000300.safetensors", "step_0001000.safetensors",
                  "step_0000500.safetensors"):
            (tmp_path / s).write_bytes(b"x")
        got = orch._ablation_warmstart_ckpt({"ablation_warmstart_arms": True}, tmp_path)
        assert got is not None and got.endswith("step_0001000.safetensors")

    def test_flag_on_no_checkpoint_returns_none(self, tmp_path):
        assert orch._ablation_warmstart_ckpt({"ablation_warmstart_arms": True}, tmp_path) is None

    def test_flag_on_missing_dir_returns_none(self, tmp_path):
        assert orch._ablation_warmstart_ckpt(
            {"ablation_warmstart_arms": True}, tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# _quality_gate_target — opt-in cross-run quality gate at campaign end (default off)
# ---------------------------------------------------------------------------

class TestQualityGateTarget:
    def test_flag_off_returns_none(self, tmp_path):
        ck = tmp_path / "step_0001000.safetensors"
        ck.write_bytes(b"x")
        assert orch._quality_gate_target({}, str(ck)) is None
        assert orch._quality_gate_target({"quality_gate": False}, str(ck)) is None

    def test_flag_on_existing_ckpt_returns_it(self, tmp_path):
        ck = tmp_path / "step_0001000.safetensors"
        ck.write_bytes(b"x")
        assert orch._quality_gate_target({"quality_gate": True}, str(ck)) == str(ck)

    def test_flag_on_missing_ckpt_returns_none(self, tmp_path):
        assert orch._quality_gate_target({"quality_gate": True},
                                         str(tmp_path / "nope.safetensors")) is None

    def test_flag_on_no_ckpt_returns_none(self):
        assert orch._quality_gate_target({"quality_gate": True}, None) is None
