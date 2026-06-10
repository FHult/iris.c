#!/usr/bin/env python3
"""
train/scripts/orchestrator.py — V2 pipeline orchestrator.

Drives the full 4-chunk IP-Adapter training pipeline autonomously:
  - State machine per chunk (IDLE → ... → DONE)
  - Resource token scheduling (GPU_TOKEN, DISK_WRITE_HIGH)
  - tmux-based process isolation (iris-prep window, iris-train window)
  - Heartbeat monitoring with auto-restart on crash
  - Dispatch interface for Claude/CLI/web

Usage:
    python train/scripts/orchestrator.py --config train/configs/v2_pipeline.yaml
    python train/scripts/orchestrator.py --resume          # re-reads pipeline_state.json
    python train/scripts/orchestrator.py --dry-run         # print decisions, no launches
    python train/scripts/orchestrator.py --skip-dedup      # skip CLIP steps
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pipeline_lib import (
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, VENV_PYTHON,
    DISPATCH_QUEUE,
    CONTROL_FILE, SENTINEL_DIR, LOG_DIR, STAGING_DIR,
    SHARDS_DIR, PRECOMP_DIR, HARD_EX_DIR, ANCHOR_SHARDS_DIR, DEDUP_DIR,
    COLD_ROOT,
    RUN_METADATA_FILE,
    TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_STAGE_WIN,
    DISK_WARN_GB, DISK_ABORT_GB, HEARTBEAT_STALE_SECS, SHARD_BLOCK,
    STATE_FILE,
    ABLATION_DB_PATH, FLYWHEEL_CONTROL_FILE, FLYWHEEL_REPORTS_DIR,
    ULTRAHOT_ROOT,
    read_state, write_state, update_state,
    is_done, mark_done, mark_error, has_error, read_error, clear_error,
    log_event, log_orch,
    write_heartbeat, read_heartbeat, heartbeat_age_secs,
    dispatch_issue, gpu_is_free,
    tmux_session_exists, tmux_window_exists, tmux_new_window,
    last_exit_code, free_gb, notify, now_iso, load_config,
    acquire_gpu_lock, release_gpu_lock, gpu_lock_holder,
)
from cache_manager import (
    PrecomputeCache, encoder_config_subset, get_git_sha,
)
from data_stager import DataStager


# ---------------------------------------------------------------------------
# Crash recovery policy
# ---------------------------------------------------------------------------

# macOS jetsam (OOM) sends SIGKILL → EXIT_CODE 137. Jetsam kills are transient:
# the system had a memory spike, but training itself is not broken. Retry more
# aggressively than for real code errors, with a backoff to let pressure settle.
JETSAM_EXIT_CODE      = 137
JETSAM_MAX_RETRIES    = 5   # up to 5 retries for transient OS kills
PREP_HUNG_HOURS       = 6   # dispatch alert if a prep step runs longer than this
JETSAM_RETRY_DELAY_S  = 90  # seconds to wait before relaunching


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to path atomically via a tmp file + os.replace."""
    tmp = path.with_suffix(path.suffix + f".stg{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _parse_exit_code_from_msg(msg: str) -> int:
    """Extract numeric exit code from 'Training exited 137; ...' error message."""
    m = re.search(r"exited (\d+)", msg or "")
    return int(m.group(1)) if m else -1


def _retry_policy(reason: str, restarts: int) -> tuple[bool, int, int]:
    """Pure retry decision for a crashed step.

    Jetsam (transient macOS OOM SIGKILL) is retried up to JETSAM_MAX_RETRIES with a
    JETSAM_RETRY_DELAY_S backoff to let memory pressure settle; a real code error is
    retried at most once with no delay. Returns
    (should_retry, max_retries, retry_delay_secs)."""
    is_jetsam   = reason == "jetsam_oom"
    max_retries = JETSAM_MAX_RETRIES if is_jetsam else 1
    retry_delay = JETSAM_RETRY_DELAY_S if is_jetsam else 0
    return restarts < max_retries, max_retries, retry_delay


def _ready_gate(chunk: int, prev_train_done: bool, gpu_free: bool,
                stager_enabled: bool, stage_done: bool, stage_error: bool) -> str:
    """Pure gating decision for a READY chunk (promote → train).

    Returns one of:
      'wait_prev_train' — chunk N≥2 must wait for chunk N-1 training to finish
      'wait_gpu'        — GPU is held by another step
      'wait_stage'      — predictive staging for N≥2 hasn't completed (and didn't error)
      'proceed_no_stage'— staging errored; proceed without staged data (bad cold drive
                          must not permanently stall the pipeline)
      'proceed'         — all gates clear
    """
    if chunk > 1 and not prev_train_done:
        return "wait_prev_train"
    if not gpu_free:
        return "wait_gpu"
    if stager_enabled and chunk > 1 and not stage_done:
        return "proceed_no_stage" if stage_error else "wait_stage"
    return "proceed"


def _should_attempt_stage(chunk: int, predecessor_promoted: bool,
                          stage_done: bool) -> bool:
    """Pure: stage chunk N from cold only once its predecessor (N-1) is promoted
    (training is imminent), only for N>=2 (chunk 1 is never staged), and only if
    not already staged."""
    predecessor_ready = chunk == 1 or predecessor_promoted
    return predecessor_ready and not stage_done and chunk > 1


def _diagnose_crash(log_file: Path, exit_code: int,
                    mem_log: Optional[Path] = None) -> tuple[str, str]:
    """
    Return (reason, detail) describing why a training run crashed.

    reason is one of: "jetsam_oom" | "code_error" | "unknown"

    Checks:
      1. Exit code — 137 (SIGKILL) almost always means jetsam on macOS.
      2. macOS system log — confirms whether a jetsam/memorystatus event fired.
      3. Last memory reading from the training log — memory state at crash.
      4. memory_pressure.log tail — rolling vm_stat log from the watchdog thread.
    """
    last_mem = _parse_last_mem_from_log(log_file)
    mem_str = f", training_mem={last_mem}" if last_mem else ""

    # Append the last two watchdog readings for context.
    if mem_log and mem_log.exists():
        try:
            recent = mem_log.read_text().splitlines()[-2:]
            if recent:
                mem_str += "  watchdog=[" + " | ".join(ln.split("  ", 1)[-1] for ln in recent) + "]"
        except OSError:
            pass

    if exit_code != JETSAM_EXIT_CODE:
        return "code_error", f"exit {exit_code} (Python/logic error){mem_str}"

    # exit 137 = SIGKILL. Query system log to confirm jetsam.
    jetsam_confirmed = _query_macos_jetsam_log()
    if jetsam_confirmed:
        return "jetsam_oom", f"exit 137, jetsam confirmed in system log{mem_str}"
    # SIGKILL without a logged jetsam event: still treat as jetsam (system log
    # can be delayed or filtered); the exit code is sufficient evidence on macOS.
    return "jetsam_oom", f"exit 137, assumed jetsam (SIGKILL){mem_str}"


def _parse_last_mem_from_log(log_file: Path) -> str:
    """Return the last 'X GB used  Y GB free' string from the training log."""
    try:
        with open(log_file, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", errors="replace")
        matches = re.findall(r"mem:\s+([\d.]+ GB used\s+[\d.]+ GB free)", tail)
        return matches[-1] if matches else ""
    except OSError:
        return ""


def _query_macos_jetsam_log(lookback_secs: int = 600) -> bool:
    """Return True if a jetsam/memorystatus kill event appears in the system log."""
    try:
        r = subprocess.run(
            ["log", "show",
             "--predicate", "eventMessage contains \"jetsam\" OR "
                            "eventMessage contains \"memorystatus\"",
             "--last", f"{lookback_secs}s"],
            capture_output=True, text=True, timeout=15,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Chunk state machine
# ---------------------------------------------------------------------------

class ChunkState(str):
    IDLE              = "IDLE"
    DOWNLOADING       = "DOWNLOADING"
    CONVERTING        = "CONVERTING"
    DEDUP_FILTERING   = "DEDUP_FILTERING"
    BUILDING          = "BUILDING"
    CLIP_EMBED        = "CLIP_EMBED"
    CLIP_INDEX        = "CLIP_INDEX"
    CLIP_DUPS         = "CLIP_DUPS"
    PRECOMPUTING      = "PRECOMPUTING"
    READY             = "READY"
    SHARD_VALIDATING  = "SHARD_VALIDATING"
    WARMING_UP        = "WARMING_UP"
    TRAINING          = "TRAINING"
    MINING            = "MINING"
    VALIDATING        = "VALIDATING"
    DONE              = "DONE"
    ERROR             = "ERROR"



CHUNK_STEPS = [
    "download",
    "convert",
    "dedupe_filter",
    "build_shards",
    "clip_embed",
    "clip_index",
    "clip_dups",
    "precompute",
    "promoted",
    "validate_shards",
    "training_warmup",
    "train",
    "mine",
    "validate",
]

_STEP_TO_STATE = {
    "download":         ChunkState.CONVERTING,
    "convert":          ChunkState.DEDUP_FILTERING,
    "dedupe_filter":    ChunkState.BUILDING,
    "build_shards":     ChunkState.CLIP_EMBED,
    "clip_embed":       ChunkState.CLIP_INDEX,
    "clip_index":       ChunkState.CLIP_DUPS,
    "clip_dups":        ChunkState.PRECOMPUTING,
    "precompute":       ChunkState.READY,
    "promoted":         ChunkState.SHARD_VALIDATING,
    "validate_shards":  ChunkState.WARMING_UP,
    "training_warmup":  ChunkState.TRAINING,
    "train":            ChunkState.MINING,
    "mine":             ChunkState.VALIDATING,
    "validate":         ChunkState.DONE,
}


def derive_chunk_state(chunk: int) -> str:
    """Derive state from sentinel files — stateless, safe to call any time."""
    for step in CHUNK_STEPS:
        if has_error(chunk, step):
            return ChunkState.ERROR
    last_done = None
    for step in CHUNK_STEPS:
        if is_done(chunk, step):
            last_done = step
    if last_done is None:
        return ChunkState.IDLE
    return _STEP_TO_STATE.get(last_done, ChunkState.DONE)

assert set(_STEP_TO_STATE) == set(CHUNK_STEPS), \
    f"CHUNK_STEPS/_STEP_TO_STATE out of sync: {set(CHUNK_STEPS) ^ set(_STEP_TO_STATE)}"


# ---------------------------------------------------------------------------
# Resource token manager
# ---------------------------------------------------------------------------

class ResourceManager:
    """
    Tracks exclusive resource tokens. GPU_TOKEN and DISK_WRITE_HIGH are
    mutually exclusive with themselves. Call request() before launching any
    GPU or disk-intensive process; call release() when it finishes.
    """

    def __init__(self):
        self._holders: dict[str, str] = {}

    def request(self, token: str, holder: str) -> bool:
        if token in self._holders:
            return False
        if token == "GPU_TOKEN":
            lock_info = gpu_lock_holder()
            if lock_info is not None:
                # External manual process holds the file lock — wait.
                return False
            if not acquire_gpu_lock(holder):
                # Another process claimed the lock between check and acquire.
                return False
        self._holders[token] = holder
        return True

    def release(self, token: str) -> None:
        if token == "GPU_TOKEN":
            release_gpu_lock()
        self._holders.pop(token, None)

    def holder(self, token: str) -> Optional[str]:
        return self._holders.get(token)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_REQUIRED_CONFIG_KEYS = [
    ("recipe",),
    ("chunks",),
    ("training_config",),
    ("training", "steps"),
    ("training", "lr"),
]


def _load_open_dispatch_ids() -> set[str]:
    """Return IDs of unresolved dispatch issues from the queue file.

    Used to pre-seed _stager_dispatched_errors on orchestrator restart so that
    issues already in the queue are not re-dispatched with a new ID.
    """
    if not DISPATCH_QUEUE.exists():
        return set()
    by_id: dict[str, bool] = {}
    try:
        for line in DISPATCH_QUEUE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            issue_id = entry.get("id")
            if issue_id:
                by_id[issue_id] = not entry.get("resolved", False)
    except Exception:
        return set()
    return {iid for iid, open_ in by_id.items() if open_}


def _validate_config(cfg: dict) -> None:
    """Abort early if the pipeline config is missing required keys."""
    missing = []
    for path in _REQUIRED_CONFIG_KEYS:
        node = cfg
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(str(k) for k in path))
                break
            node = node[key]
    if missing:
        print(f"ERROR: pipeline config missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, config: dict, dry_run: bool = False, skip_dedup: bool = False):
        self.cfg          = config
        self.dry_run      = dry_run
        self.skip_dedup   = skip_dedup
        self.res          = ResourceManager()
        self.total_chunks = config.get("chunks", 4)
        self.scale        = config.get("scale", "small")

        # Active prep root: ultrahot_root when data_prep_tier=ultrahot, else DATA_ROOT.
        _storage = config.get("storage", {})
        _uh = Path(_storage.get("ultrahot_root", str(ULTRAHOT_ROOT)))
        self.prep_root: Path = (
            _uh if _storage.get("data_prep_tier") == "ultrahot" else DATA_ROOT
        )
        # Checkpoint dirs derive from prep_root so ultrahot mode routes correctly.
        self.ckpt_dir         = self.prep_root / "checkpoints" / "stage1"
        self.ckpt_archive_dir = self.ckpt_dir  / "archive"
        self.issue_counter = self._read_max_issue_counter()
        self._restart_counts: dict[tuple, int] = self._load_restart_counts()
        # Retry backoff: {(chunk, step) → earliest epoch_sec to retry}
        self._retry_after: dict[tuple, float] = {}

        # Track what is currently running in the prep window:
        # {"chunk": N, "step": "build_shards", "log": Path, "token": "DISK_WRITE_HIGH"|None}
        self._active_prep: Optional[dict] = None

        # Anomaly detection counters: {chunk: count_of_consecutive_anomaly_polls}
        self._loss_high_since: dict[int, Optional[int]] = {}  # chunk → first anomaly step
        # Consecutive polls with grad_norm above the warn threshold.  Resets to 0
        # after a pause is triggered so a new episode can be detected on resume.
        self._grad_spike_polls: dict[int, int] = {}           # chunk → consecutive high-grad polls
        self._ema_drift_polls:  dict[int, int] = {}           # chunk → consecutive high-ema_drift polls

        # Crash diagnosis cache: avoid re-running `log show` (15 s subprocess) on
        # every poll while a step is in error state.  Keyed by (chunk, step, exit_code).
        self._crash_diag: dict[tuple, tuple] = {}             # → (reason, detail)

        # Dispatch cooldown: {issue_key: last_dispatch_epoch_secs}
        # Dispatch fire count: {issue_key: number of times dispatched}
        self._dispatch_last:  dict[str, float] = {}
        self._dispatch_count: dict[str, int]   = {}

        # Cache of last-written state per chunk — avoid redundant file writes
        self._last_written_state: dict[int, str] = {}

        # Memory watchdog: background thread polls vm_stat every 30s and writes
        # a rolling log so we have pre-crash memory state when training is killed.
        self._mem_log = LOG_DIR / "memory_pressure.log"
        self._mem_watchdog_stop = False
        self._start_memory_watchdog()

        # Temp training config created by _start_training when hard_mix_ratio_by_chunk
        # is set.  Stored here so _post_step can unlink it after the train step completes.
        self._active_train_tmp_cfg: Optional[Path] = None

        self.stager = DataStager(self.cfg)

        # Issue IDs that have been dispatched for stager errors.  Cleared when
        # the operator resolves the error sentinel so re-dispatch fires on recurrence.
        # Pre-seeded from the dispatch queue so restarts don't re-fire existing alerts.
        self._stager_dispatched_errors: set[str] = _load_open_dispatch_ids()

        _validate_config(config)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        log_orch("Orchestrator starting", scale=self.scale, chunks=self.total_chunks)
        self._ensure_tmux_session()
        self._run_doctor()
        self._init_state()
        self._write_run_metadata_start()
        self._rotate_stale_logs()
        self._recover_prep_window()

        poll_interval = self.cfg.get("poll_interval", 60)
        while True:
            try:
                self._poll_prep_window()    # must come first: updates sentinel state
                self._sync_gpu_token()
                self._check_heartbeats()
                self._check_disk()
                self._check_control_signals()

                for chunk in range(1, self.total_chunks + 1):
                    self._advance_chunk(chunk)

                self._poll_stager()
                write_heartbeat("orchestrator", step="poll")
                if self._all_done():
                    log_orch("All chunks complete — pipeline finished")
                    summary_path = self._write_run_summary()
                    self._write_run_metadata_done(summary_path)
                    notify("iris pipeline", "All chunks complete")
                    break
            except KeyboardInterrupt:
                log_orch("Orchestrator interrupted by operator")
                break
            except Exception as e:
                import traceback
                log_orch(f"Orchestrator exception: {e}", level="error")
                traceback.print_exc()
                dispatch_issue(self._next_issue_id(), "error",
                               f"Orchestrator exception: {e}",
                               suggested_action="check_orchestrator_logs")

            if not self.dry_run:
                time.sleep(poll_interval)
            else:
                break

    # -----------------------------------------------------------------------
    # Prep window tracking — the most important state machine piece
    # -----------------------------------------------------------------------

    def _poll_prep_window(self) -> None:
        """
        Check if the active prep step has finished. This must run before any
        _advance_chunk call so that sentinels are current when state is derived.
        """
        if self._active_prep is None:
            return
        if tmux_window_exists(TMUX_PREP_WIN):
            # Alert if the prep window has been alive far longer than expected.
            elapsed_h = (time.time() - self._active_prep.get("started_at", time.time())) / 3600
            if elapsed_h > PREP_HUNG_HOURS:
                ap = self._active_prep
                self._dispatch_once(
                    f"hung_prep_{ap['chunk']}_{ap['step']}",
                    self._next_issue_id(), "warning",
                    f"{ap['step']} chunk {ap['chunk']} has been running {elapsed_h:.1f}h "
                    f"(threshold {PREP_HUNG_HOURS}h) — may be hung",
                    chunk=ap["chunk"], process=ap["step"],
                    suggested_action="check_prep_log_and_kill_if_stuck",
                    cooldown_secs=3600,
                )
            return  # still running

        # Window is gone — read exit code and mark done/error
        ap   = self._active_prep
        chunk = ap["chunk"]
        step  = ap["step"]
        log   = ap["log"]
        token = ap.get("token")
        extra_marks = ap.get("also_mark", [])

        self._active_prep = None  # clear before any returns below

        if token:
            self.res.release(token)

        code = last_exit_code(log)
        if code == 0:
            log_orch(f"Chunk {chunk}: {step} complete", chunk=chunk)
            mark_done(chunk, step)
            for extra in extra_marks:
                mark_done(chunk, extra)
            self._post_step(chunk, step)
        else:
            msg = f"Exit code {code if code is not None else 'unknown'}; see {log}"
            log_orch(f"Chunk {chunk}: {step} FAILED — {msg}", level="error", chunk=chunk)
            mark_error(chunk, step, msg)
            write_heartbeat(f"prep_{step}", chunk, status="failed", exit_code=code,
                            step=step, log=str(log))
            notify("iris pipeline ERROR", f"{step} chunk {chunk} failed")

    def _post_step(self, chunk: int, step: str) -> None:
        """Side effects after a prep step completes successfully."""
        if step == "build_shards":
            # Free staging raw data to reclaim disk space
            self._delete_staging_raw(chunk)
        if step == "mine":
            notify("iris pipeline", f"Chunk {chunk} hard example mining complete")
            self._run_shard_scorer(chunk)
            if not is_done(chunk, "archive"):
                # Archive now that shards are no longer needed for mining.
                # Chain stage of next chunk so cold→hot transfer starts immediately.
                self._launch_stager(
                    f"archive --chunk {chunk}" +
                    (f" && {self._python_cmd('data_stager.py', f'stage --chunk {chunk + 1}')}"
                     if chunk < self.total_chunks else ""),
                    description=f"archive chunk {chunk}",
                    chunk=chunk,
                    log_name=f"stager_archive_chunk{chunk}",
                )
        if step == "train" and self._active_train_tmp_cfg is not None:
            try:
                self._active_train_tmp_cfg.unlink(missing_ok=True)
            except OSError:
                pass
            self._active_train_tmp_cfg = None
        if step in ("train", "mine"):
            update_state(**{"chunks": {str(chunk): {"completed_at": now_iso()}}})

    def _run_shard_scorer(self, chunk: int) -> None:
        """Run shard_scorer.py in a background subprocess after mine.done (no GPU, <1 min)."""
        if self.dry_run:
            return
        import subprocess
        cmd = [str(VENV_PYTHON), str(SCRIPTS_DIR / "shard_scorer.py"),
               "--config", self._config_path()]
        log_path = LOG_DIR / f"shard_scorer_chunk{chunk}.log"
        try:
            with open(log_path, "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf, close_fds=True)
            log_orch(f"Chunk {chunk}: launched shard_scorer in background", chunk=chunk)
        except OSError as e:
            log_orch(f"Chunk {chunk}: failed to launch shard_scorer: {e}", chunk=chunk,
                     level="warning")

    def _delete_staging_raw(self, chunk: int) -> None:
        """Delete staging raw + converted data once shards are built."""
        for subdir in ("raw", "converted"):
            path = STAGING_DIR / f"chunk{chunk}" / subdir
            if path.exists():
                log_orch(f"Chunk {chunk}: freeing staging {subdir} data ({path})", chunk=chunk)
                if not self.dry_run:
                    shutil.rmtree(path, ignore_errors=True)

    _PREP_LOG_PATTERNS = [
        "download_chunk{c}.log",
        "dedupe_filter_chunk{c}.log",
        "build_chunk{c}.log",
        "clip_embed_chunk{c}.log",
        "clip_index_chunk{c}.log",
        "clip_dups_chunk{c}.log",
        "precompute_chunk{c}.log",
        "validate_shards_chunk{c}.log",
        "training_warmup_chunk{c}.log",
    ]

    def _cleanup_prep_logs(self, chunk: int) -> None:
        """Delete step logs for prep steps now that the chunk is promoted to production."""
        if self.dry_run:
            return
        deleted = []
        for pattern in self._PREP_LOG_PATTERNS:
            log = LOG_DIR / pattern.format(c=chunk)
            if log.exists():
                try:
                    log.unlink()
                    deleted.append(log.name)
                except OSError:
                    pass
        if deleted:
            log_orch(f"Chunk {chunk}: deleted {len(deleted)} prep log(s) after promotion",
                     chunk=chunk)

    # -----------------------------------------------------------------------
    # PIPELINE-6/7: Checkpoint archive on chunk completion
    # -----------------------------------------------------------------------

    def _archive_chunk_checkpoint(self, chunk: int) -> None:
        """Copy the latest checkpoint pair + EMA to archive/chunk{N}_final.* ."""
        if self.dry_run:
            return
        ckpts = sorted(self.ckpt_dir.glob("step_*.safetensors"))
        if not ckpts:
            log_orch(f"Chunk {chunk}: no checkpoint to archive", chunk=chunk)
            return
        latest_st = ckpts[-1]
        step_stem = latest_st.stem  # e.g. "step_0050000"
        self.ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for suffix in [".safetensors", ".json", ".ema.safetensors"]:
            src = self.ckpt_dir / f"{step_stem}{suffix}"
            if not src.exists():
                continue
            dst = self.ckpt_archive_dir / f"chunk{chunk}_final{suffix}"
            tmp = dst.with_suffix(dst.suffix + f".stg{os.getpid()}")
            try:
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                copied.append(dst.name)
            except OSError as e:
                tmp.unlink(missing_ok=True)
                log_orch(f"Chunk {chunk}: archive copy failed for {src.name}: {e}",
                         level="warning", chunk=chunk)
        if copied:
            log_orch(f"Chunk {chunk}: archived checkpoint {step_stem} → "
                     f"{self.ckpt_archive_dir.relative_to(self.prep_root)} ({len(copied)} files)",
                     chunk=chunk)

    def _create_release_bundle(self, chunk: int) -> None:
        """Create a clearly-named release bundle when the final chunk finishes training.

        Writes archive/release_YYYYMMDD_step{N}/ containing:
          adapter_weights.safetensors  — copy of best.safetensors (best val-loss)
          release.json                 — full provenance (git SHA, step, loss, config, scale)
        """
        if self.dry_run:
            return
        from datetime import datetime, timezone

        best = self.ckpt_dir / "best.safetensors"
        best_meta = self.ckpt_dir / "best.json"
        if not best.exists():
            log_orch(f"Chunk {chunk}: no best.safetensors — skipping release bundle",
                     level="warning", chunk=chunk)
            return

        # Derive step and loss from best.json if present.
        step, loss = 0, None
        if best_meta.exists():
            try:
                with open(best_meta) as f:
                    meta = json.load(f)
                step = meta.get("step", 0)
                loss = meta.get("loss")
            except Exception:
                pass

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        bundle_name = f"release_{date_str}_step{step:07d}"
        bundle_dir = self.ckpt_archive_dir / bundle_name
        self.ckpt_archive_dir.mkdir(parents=True, exist_ok=True)

        if bundle_dir.exists():
            log_orch(f"Chunk {chunk}: release bundle already exists at {bundle_name}",
                     chunk=chunk)
            return

        bundle_dir.mkdir(parents=True)

        # Copy best weights with a clear, stable name.
        try:
            shutil.copy2(best, bundle_dir / "adapter_weights.safetensors")
        except OSError as e:
            log_orch(f"Chunk {chunk}: release bundle copy failed: {e}", level="warning",
                     chunk=chunk)
            return

        # Write provenance.
        provenance = {
            "bundle": bundle_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "loss": loss,
            "scale": self.scale,
            "total_chunks": self.total_chunks,
            "source": str(best.relative_to(DATA_ROOT)),
        }
        try:
            import subprocess as _sp
            provenance["git_sha"] = _sp.check_output(
                ["git", "-C", str(TRAIN_DIR.parent), "rev-parse", "--short", "HEAD"],
                stderr=_sp.DEVNULL, text=True
            ).strip()
        except Exception:
            pass
        if best_meta.exists():
            try:
                with open(best_meta) as f:
                    provenance["training_config"] = json.load(f).get("config", {})
            except Exception:
                pass

        try:
            with open(bundle_dir / "release.json", "w") as f:
                json.dump(provenance, f, indent=2)
        except OSError as e:
            log_orch(f"Chunk {chunk}: failed to write release.json: {e}", level="warning",
                     chunk=chunk)

        log_orch(
            f"Chunk {chunk}: release bundle created → "
            f"{bundle_dir.relative_to(DATA_ROOT)}  "
            f"(step={step:,}  loss={loss})",
            chunk=chunk,
        )
        notify("iris pipeline", f"Release bundle ready: {bundle_name}  loss={loss}")

    # -----------------------------------------------------------------------
    # PIPELINE-17: Auto-populate anchor_shards after chunk 1
    # -----------------------------------------------------------------------

    def _auto_populate_anchor_shards(self, chunk: int) -> None:
        """After chunk 1 training, copy every Nth shard to ANCHOR_SHARDS_DIR."""
        if self.dry_run or chunk != 1:
            return
        if ANCHOR_SHARDS_DIR.exists() and any(ANCHOR_SHARDS_DIR.glob("*.tar")):
            log_orch("Anchor shards already populated — skipping auto-populate", chunk=chunk)
            return
        sample_rate = self.cfg.get("training", {}).get("anchor_sample_rate", 10)
        if not sample_rate or sample_rate <= 0:
            log_orch("Anchor shard auto-populate disabled (sample_rate=0)", chunk=chunk)
            return
        shards = sorted(SHARDS_DIR.glob("*.tar"))
        if not shards:
            log_orch("No shards found for anchor sampling", level="warning", chunk=chunk)
            return
        ANCHOR_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        selected = shards[::sample_rate]
        copied = 0
        for shard in selected:
            try:
                shutil.copy2(shard, ANCHOR_SHARDS_DIR / shard.name)
                copied += 1
            except OSError as e:
                log_orch(f"Anchor shard copy failed {shard.name}: {e}",
                         level="warning", chunk=chunk)
        log_orch(f"Chunk {chunk}: auto-populated {copied} anchor shards "
                 f"(1/{sample_rate} sample rate) → {ANCHOR_SHARDS_DIR}", chunk=chunk)

    # -----------------------------------------------------------------------
    # PIPELINE-20: Run metadata
    # -----------------------------------------------------------------------

    def _write_run_metadata_start(self) -> None:
        """Write run_metadata.json at orchestrator start."""
        state = read_state()
        run_id = state.get("run_id", f"run_{now_iso()[:10].replace('-', '')}")
        meta = {
            "run_id":               run_id,
            "scale":                self.scale,
            "total_chunks":         self.total_chunks,
            "started_at":           now_iso(),
            "orchestrator_pid":     os.getpid(),
            "memory_watchdog_log":  str(self._mem_log),
            "config":               {k: v for k, v in self.cfg.items() if k != "_config_path"},
        }
        try:
            _atomic_write_text(RUN_METADATA_FILE, json.dumps(meta, indent=2))
        except OSError as e:
            log_orch(f"Could not write run metadata: {e}", level="warning")

    def _write_run_metadata_done(self, summary_path: Optional[Path] = None) -> None:
        """Append completion info to run_metadata.json."""
        try:
            meta = json.loads(RUN_METADATA_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["completed_at"] = now_iso()
        ckpts = sorted(self.ckpt_dir.glob("step_*.safetensors"))
        if ckpts:
            meta["final_checkpoint"] = str(ckpts[-1])
        if summary_path and summary_path.exists():
            meta["run_summary"] = str(summary_path)
        try:
            _atomic_write_text(RUN_METADATA_FILE, json.dumps(meta, indent=2))
        except OSError as e:
            log_orch(f"Could not update run metadata: {e}", level="warning")

    # -----------------------------------------------------------------------
    # PIPELINE-10: Run summary report
    # -----------------------------------------------------------------------

    def _write_run_summary(self) -> Optional[Path]:
        """Parse orchestrator logs to produce a human-readable run summary."""
        summary_path = LOG_DIR / "run_summary.txt"
        try:
            lines: list[str] = []
            for log_file in sorted(LOG_DIR.glob("orchestrator*.jsonl"),
                                   key=lambda p: p.name):
                if log_file.exists():
                    lines.extend(log_file.read_text(errors="replace").splitlines())

            events_by_chunk: dict[int, list[dict]] = {}
            training_events: dict[int, dict] = {}
            for line in lines:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = ev.get("chunk")
                msg   = ev.get("message", "")
                ts    = ev.get("ts", "")
                if chunk and ("starting training" in msg or "training complete" in msg
                              or "hard example mining complete" in msg):
                    events_by_chunk.setdefault(chunk, []).append(ev)
                # Keep the latest start per chunk — chunk 1 may have many restarts;
                # we want the start of the final successful run, not the first attempt.
                if chunk and "starting training" in msg:
                    training_events.setdefault(chunk, {})["start"] = ts
                if chunk and "training complete" in msg:
                    training_events.setdefault(chunk, {})["end"] = ts

            out: list[str] = ["=== Pipeline Run Summary ===", f"Generated: {now_iso()}", ""]
            for chunk in range(1, self.total_chunks + 1):
                ev = training_events.get(chunk, {})
                start_ts = ev.get("start")
                end_ts   = ev.get("end")
                if start_ts and end_ts:
                    from datetime import datetime, timezone
                    fmt = "%Y-%m-%dT%H:%M:%S+00:00"
                    try:
                        t0 = datetime.fromisoformat(start_ts)
                        t1 = datetime.fromisoformat(end_ts)
                        elapsed_h = (t1 - t0).total_seconds() / 3600
                        out.append(f"Chunk {chunk}: {elapsed_h:.1f}h training")
                    except ValueError:
                        out.append(f"Chunk {chunk}: training timestamps unparseable")
                else:
                    out.append(f"Chunk {chunk}: incomplete")

            out.append("")
            ckpts = sorted(self.ckpt_dir.glob("step_*.safetensors"))
            if ckpts:
                out.append(f"Final checkpoint: {ckpts[-1].name}")
            hard_counts = {}
            if HARD_EX_DIR.exists():
                for d in HARD_EX_DIR.iterdir():
                    if d.is_dir() and d.name.startswith("chunk"):
                        hard_counts[d.name] = sum(1 for f in d.glob("*.tar"))
            if hard_counts:
                out.append("Hard examples: " + "  ".join(
                    f"{k}={v} tars" for k, v in sorted(hard_counts.items())))

            summary_path.write_text("\n".join(out) + "\n")
            log_orch(f"Run summary written → {summary_path}")
            return summary_path
        except Exception as e:
            log_orch(f"Could not write run summary: {e}", level="warning")
            return None

    def _disk_guard(self, step_description: str) -> bool:
        """Return False and pause if disk is below the critical threshold."""
        min_gb = self.cfg.get("min_free_gb", DISK_ABORT_GB)
        gb = free_gb()
        if gb < min_gb:
            log_orch(f"Disk guard: {gb:.1f} GB free < {min_gb} GB — skipping {step_description}",
                     level="warning")
            return False
        return True

    def _launch_prep(self, description: str, cmd: str, log_file: Path,
                     chunk: int, step: str,
                     token: Optional[str] = None,
                     also_mark: Optional[list] = None) -> None:
        """Launch cmd in the iris-prep tmux window and record the active step."""
        if self._active_prep is not None:
            return  # already something running
        if not self._disk_guard(description):
            return
        if self.dry_run:
            log_orch(f"DRY RUN: would launch {description}")
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        activated = (f"export PIPELINE_DATA_ROOT='{DATA_ROOT}' PIPELINE_ORCHESTRATED=1 && "
                     f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}")
        tmux_new_window(TMUX_PREP_WIN, activated, log_file)
        self._active_prep = {
            "chunk": chunk, "step": step, "log": log_file,
            "token": token, "also_mark": also_mark or [],
            "started_at": time.time(),
        }
        log_orch(f"Launched: {description} → {log_file}")

    def _prep_busy(self) -> bool:
        """True when the prep window is occupied."""
        return self._active_prep is not None or tmux_window_exists(TMUX_PREP_WIN)

    def _launch_stager(self, stager_args: str, description: str,
                       chunk: int, log_name: str) -> None:
        """
        Launch data_stager.py in the iris-stage tmux window.

        Uses `nice -n 10 taskpolicy -d throttle` unconditionally — the stager
        is always I/O-heavy background work that must yield to training and
        precompute regardless of what else is running.

        If iris-stage is already occupied (e.g. a previous archive/stage job is
        still running), the launch is skipped and logged; the caller is
        responsible for retry if the operation is mandatory.

        For the no-op single-SSD case (stager.enabled == False) the stager CLI
        exits immediately, so a spurious window is harmless but we skip the
        launch to avoid noise in the tmux session list.
        """
        if not self.stager.enabled:
            return
        if tmux_window_exists(TMUX_STAGE_WIN):
            log_orch(f"iris-stage busy — skipping {description} (will retry next poll)", chunk=chunk)
            return
        if self.dry_run:
            log_orch(f"DRY RUN: would launch {description}")
            return
        cfg_path = self._config_path()
        inner = self._python_cmd("data_stager.py", f"--config '{cfg_path}' {stager_args}")
        # Always throttle; wrap only the inner python command so the shell &&
        # chain itself runs at normal priority.
        cmd = f"nice -n 10 taskpolicy -d throttle {inner}"
        log_file = LOG_DIR / f"{log_name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        activated = (f"export PIPELINE_DATA_ROOT='{DATA_ROOT}' PIPELINE_ORCHESTRATED=1 && "
                     f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}")
        tmux_new_window(TMUX_STAGE_WIN, activated, log_file)
        log_orch(f"Launched: {description} → {log_file}", chunk=chunk)

    def _poll_stager(self) -> None:
        """
        Called every orchestrator tick.  Handles four responsibilities:

        1. Retry pending archive:  if chunk N mining is done but archive is
           not yet done (and no error is set), re-launch archive when iris-stage
           is free.  This recovers from the initial launch being skipped because
           iris-stage was busy, or from an orchestrator restart.

        2. Retry pending stage:  if chunk N is ready to enter training (its
           predecessor has been promoted) but staging is not yet done, re-launch
           stage when iris-stage is free.

        3. Dispatch errors:  write a dispatch_issue for any stage or archive
           error sentinel that has not already been reported.  The dispatch cache
           is cleared when the operator resolves the sentinel so that a subsequent
           failure re-triggers a new alert.

        4. Stale heartbeat:  if iris-stage is running but its most-recently-written
           heartbeat is older than HEARTBEAT_STALE_SECS, dispatch a warning so the
           operator can kill and restart the window.
        """
        if not self.stager.enabled:
            return

        stage_busy = tmux_window_exists(TMUX_STAGE_WIN)

        for chunk in range(1, self.total_chunks + 1):
            # ---- archive retry ------------------------------------------------
            archive_err_id = f"stager_archive_{chunk}"
            if is_done(chunk, "mine") and not is_done(chunk, "archive"):
                if has_error(chunk, "archive"):
                    # Error is set — dispatch if not already, then wait for operator.
                    if archive_err_id not in self._stager_dispatched_errors:
                        dispatch_issue(
                            archive_err_id, "error",
                            f"Chunk {chunk}: archive to cold storage failed",
                            chunk=chunk, process="stager",
                            context={"error": read_error(chunk, "archive")},
                            suggested_action=(
                                f"Resolve the issue then run: "
                                f"pipeline_ctl.py clear-error --chunk {chunk} --step archive"
                            ),
                        )
                        self._stager_dispatched_errors.add(archive_err_id)
                elif not stage_busy:
                    self._launch_stager(
                        f"archive --chunk {chunk}",
                        description=f"archive chunk {chunk} (retry)",
                        chunk=chunk,
                        log_name=f"stager_archive_chunk{chunk}",
                    )
            else:
                # Not in an error state (or already done) — clear dispatch cache so
                # a future error re-triggers an alert.
                self._stager_dispatched_errors.discard(archive_err_id)

            # ---- stage retry --------------------------------------------------
            # Only retry staging for chunk N once chunk N-1 has been promoted
            # (training is imminent).  Chunk 1 is never gated on staging.
            stage_err_id = f"stager_stage_{chunk}"
            if _should_attempt_stage(
                chunk,
                predecessor_promoted=(chunk > 1 and is_done(chunk - 1, "promoted")),
                stage_done=is_done(chunk, "stage"),
            ):
                if has_error(chunk, "stage"):
                    if stage_err_id not in self._stager_dispatched_errors:
                        dispatch_issue(
                            stage_err_id, "error",
                            f"Chunk {chunk}: staging from cold storage failed — "
                            f"training for this chunk is blocked",
                            chunk=chunk, process="stager",
                            context={"error": read_error(chunk, "stage")},
                            suggested_action=(
                                f"Resolve the issue then run: "
                                f"pipeline_ctl.py clear-error --chunk {chunk} --step stage"
                            ),
                        )
                        self._stager_dispatched_errors.add(stage_err_id)
                elif not stage_busy:
                    self._launch_stager(
                        f"stage --chunk {chunk}",
                        description=f"stage chunk {chunk} (retry)",
                        chunk=chunk,
                        log_name=f"stager_stage_chunk{chunk}",
                    )
            else:
                self._stager_dispatched_errors.discard(stage_err_id)

            # ---- stale heartbeat ----------------------------------------------
            stale_id = f"stager_stale_{chunk}"
            if stage_busy:
                age = heartbeat_age_secs("stager", chunk)
                if age is not None and age > HEARTBEAT_STALE_SECS:
                    if stale_id not in self._stager_dispatched_errors:
                        dispatch_issue(
                            stale_id, "warning",
                            f"Stager heartbeat for chunk {chunk} is stale "
                            f"({age:.0f}s > {HEARTBEAT_STALE_SECS}s threshold) — "
                            f"iris-stage may be hung",
                            chunk=chunk, process="stager",
                            suggested_action=(
                                "Kill iris-stage window, then clear the error sentinel "
                                "if one was written; stager will auto-retry on next poll."
                            ),
                        )
                        self._stager_dispatched_errors.add(stale_id)
            else:
                # Window gone — clear stale alert so it can re-fire on the next run.
                self._stager_dispatched_errors.discard(stale_id)

    # Fix 2: recover _active_prep after orchestrator restart mid-step.
    # _active_prep is in-memory; if orchestrator is killed while iris-prep runs,
    # on restart _poll_prep_window() returns early and the step is never marked done.
    # This method reconstructs _active_prep from sentinel state so the completion
    # is detected on the next poll.
    def _recover_prep_window(self) -> None:
        if not tmux_window_exists(TMUX_PREP_WIN):
            return
        _state_to_step = {
            ChunkState.IDLE:             "download",
            ChunkState.CONVERTING:       "download",
            ChunkState.DEDUP_FILTERING:  "dedupe_filter",
            ChunkState.BUILDING:         "build_shards",
            ChunkState.CLIP_EMBED:       "clip_embed",
            ChunkState.CLIP_INDEX:       "clip_index",
            ChunkState.CLIP_DUPS:        "clip_dups",
            ChunkState.PRECOMPUTING:     "precompute",
            ChunkState.SHARD_VALIDATING: "validate_shards",
            ChunkState.WARMING_UP:       "training_warmup",
            ChunkState.MINING:           "mine",
            ChunkState.VALIDATING:       "validate",
        }
        _step_log = {
            "download":         lambda c: LOG_DIR / f"download_chunk{c}.log",
            "dedupe_filter":    lambda c: LOG_DIR / f"dedupe_filter_chunk{c}.log",
            "build_shards":     lambda c: LOG_DIR / f"build_chunk{c}.log",
            "clip_embed":       lambda c: LOG_DIR / f"clip_embed_chunk{c}.log",
            "clip_index":       lambda c: LOG_DIR / f"clip_index_chunk{c}.log",
            "clip_dups":        lambda c: LOG_DIR / f"clip_dups_chunk{c}.log",
            "precompute":       lambda c: LOG_DIR / f"precompute_chunk{c}.log",
            "validate_shards":  lambda c: LOG_DIR / f"validate_shards_chunk{c}.log",
            "training_warmup":  lambda c: LOG_DIR / f"training_warmup_chunk{c}.log",
            "mine":             lambda c: LOG_DIR / f"mine_chunk{c}.log",
            "validate":         lambda c: LOG_DIR / f"validate_chunk{c}.log",
        }
        _GPU_STEPS  = {"precompute", "clip_embed", "training_warmup", "mine", "validate", "dedupe_filter"}
        _DISK_STEPS = {"build_shards"}
        # Collect all chunks in a pending state, then pick the one whose log
        # file was most recently written — that is the step actually running.
        candidates = []
        for chunk in range(1, self.total_chunks + 1):
            state = derive_chunk_state(chunk)
            step = _state_to_step.get(state)
            if step:
                log = _step_log.get(step, lambda c: LOG_DIR / f"{step}_chunk{c}.log")(chunk)
                mtime = log.stat().st_mtime if log.exists() else 0.0
                candidates.append((mtime, chunk, step, log))
        if not candidates:
            log_orch("Startup: iris-prep window exists but no pending step found — monitoring only",
                     level="warning")
            return
        # Most-recently-written log wins.
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, chunk, step, log = candidates[0]
        token = None
        if step in _GPU_STEPS:
            token = "GPU_TOKEN"
        elif step in _DISK_STEPS:
            token = "DISK_WRITE_HIGH"
        if token:
            self.res.request(token, f"recovered: {step} chunk {chunk}")
        # Estimate how long the prep window has been running by reading the
        # heartbeat file mtime.  This preserves the elapsed time across restarts
        # so _poll_prep_window() can fire the hung-prep alert correctly.
        from pipeline_lib import heartbeat_path as _hb_path
        _hb_file = _hb_path(f"prep_{step}", chunk)
        try:
            _started_at = _hb_file.stat().st_mtime
        except OSError:
            _started_at = time.time() - 3600  # conservative: assume 1h already elapsed
        self._active_prep = {
            "chunk": chunk, "step": step, "log": log,
            "token": token, "also_mark": [],
            "started_at": _started_at,
        }
        log_orch(f"Startup: recovered iris-prep tracking ({step} chunk {chunk})",
                 chunk=chunk)

    # Fix 5: archive training logs from a previous session so stale EXIT_CODE values
    # don't mislead _start_training() detection and don't pollute status log tails.
    def _rotate_stale_logs(self) -> None:
        for chunk in range(1, self.total_chunks + 1):
            if is_done(chunk, "train"):
                continue
            log = LOG_DIR / f"train_chunk{chunk}.log"
            if log.exists() and last_exit_code(log) is not None:
                ts = now_iso().replace(":", "").replace("+", "").replace("-", "")[:15]
                archived = log.with_suffix(f".{ts}.log")
                log.rename(archived)
                log_orch(f"Chunk {chunk}: archived stale training log → {archived.name}",
                         chunk=chunk)

    # -----------------------------------------------------------------------
    # Per-chunk advancement
    # -----------------------------------------------------------------------

    def _advance_chunk(self, chunk: int) -> None:
        state = derive_chunk_state(chunk)

        # Sync sentinel-derived state into pipeline_state.json (authoritative source is
        # the sentinel files; state file is a convenience mirror for tooling).
        if self._last_written_state.get(chunk) != state:
            update_state(**{"chunks": {str(chunk): {"state": state}}})
            self._last_written_state[chunk] = state

        # Chunk N+1 only starts prep when chunk N has entered the training phase.
        # This enforces strict one-ahead pipelining: chunk 1 trains → chunk 2 preps,
        # chunk 2 trains → chunk 3 preps, etc.  Prevents disk pressure from multiple
        # future chunks building simultaneously.
        # Chunk N+1 prep is gated on chunk N reaching the training phase.
        # Use is_done(chunk-1, "train") instead of derive_chunk_state so that a
        # post-training error (mine, validate, archive) on chunk N does not prevent
        # chunk N+1 from starting prep — ChunkState.ERROR would otherwise deadlock
        # the pipeline permanently with no doctor alert.
        if chunk > 1 and not is_done(chunk - 1, "train"):
            return

        handlers = {
            ChunkState.IDLE:             self._start_download_convert,
            ChunkState.DOWNLOADING:      self._noop,  # wait for prep window
            ChunkState.CONVERTING:       self._noop,  # wait for prep window
            ChunkState.DEDUP_FILTERING:  self._start_dedupe_filter,
            ChunkState.BUILDING:         self._start_build,
            ChunkState.CLIP_EMBED:       self._start_clip_embed,
            ChunkState.CLIP_INDEX:       self._start_clip_index,
            ChunkState.CLIP_DUPS:        self._start_clip_dups,
            ChunkState.PRECOMPUTING:     self._start_precompute,
            ChunkState.READY:            self._check_ready,
            ChunkState.SHARD_VALIDATING: self._start_shard_validation,
            ChunkState.WARMING_UP:       self._start_training_warmup,
            ChunkState.TRAINING:         self._start_training,
            ChunkState.MINING:           self._start_mining,
            ChunkState.VALIDATING:       self._start_validation,
            ChunkState.DONE:             self._noop,
            ChunkState.ERROR:            self._handle_error,
        }
        handler = handlers.get(state, self._noop)
        handler(chunk)

    def _noop(self, chunk: int) -> None:
        pass

    # -----------------------------------------------------------------------
    # Step handlers
    # -----------------------------------------------------------------------

    def _start_download_convert(self, chunk: int) -> None:
        """Download + convert all sources for this chunk (combined, MLX-19 pattern)."""
        if self._prep_busy():
            return
        training = self._training_active()
        log_orch(f"Chunk {chunk}: starting download+convert"
                 + (" (throttled — training active)" if training else ""), chunk=chunk)
        log_file = LOG_DIR / f"download_chunk{chunk}.log"
        jdb_only = self.cfg.get("download", {}).get("jdb_only", False)
        extra = " --jdb-only" if jdb_only else ""
        cmd = self._python_cmd("download_convert.py",
                               f"--chunk {chunk} --scale {self.scale} --config '{self._config_path()}'{extra}")
        cmd = self._throttle_wrap(cmd)
        # Marks both download.done AND convert.done on exit 0
        self._launch_prep(f"download+convert chunk {chunk}", cmd, log_file,
                          chunk, "download", also_mark=["convert"])

    def _training_active(self) -> bool:
        """True when any chunk is in the training phase (window-close race-free).

        Uses sentinel files directly: "promoted.done exists AND train.done absent"
        means the chunk is assigned to training regardless of whether the window
        is momentarily down or the step is in an error/retry state.
        """
        for c in range(1, self.total_chunks + 1):
            if is_done(c, "promoted") and not is_done(c, "train"):
                return True
        return tmux_window_exists(TMUX_TRAIN_WIN)

    def _start_build(self, chunk: int) -> None:
        if is_done(chunk, "build_shards") or self._prep_busy():
            return
        if not self.res.request("DISK_WRITE_HIGH", f"build chunk {chunk}"):
            return
        training = self._training_active()
        log_orch(f"Chunk {chunk}: building shards"
                 + (" (throttled — training active)" if training else ""), chunk=chunk)
        out      = STAGING_DIR / f"chunk{chunk}" / "shards"
        sources  = self._build_sources(chunk)
        blocklist_arg = ""
        _cold_root = Path(self.cfg.get("storage", {}).get("cold_root", str(COLD_ROOT)))
        _cold_blocklist = _cold_root / "metadata" / "duplicate_ids.txt"
        if _cold_blocklist.exists():
            blocklist_arg = f"--blocklist '{_cold_blocklist}'"
        log_file = LOG_DIR / f"build_chunk{chunk}.log"
        # Reserve a non-overlapping shard ID space per chunk so that internal
        # record IDs (embedded in shard member names, e.g. "200000_0003") are
        # globally unique across all chunks.  This prevents precomputed .npz
        # files from colliding when chunks are promoted to the shared production
        # precomputed/ directory.  200000 shards × 5000 images = 1B images per
        # chunk.  _promote_chunk enforces this ceiling with a hard error.
        start_idx = (chunk - 1) * SHARD_BLOCK
        # Cap build output to the precompute budget so downstream steps (filter,
        # clip_embed, clip_dups, validate_shards) never process shards that won't
        # be trained on.  Reads the same precompute.max_shards config key used by
        # _start_precompute() so the two steps stay in sync automatically.
        max_shards_raw = self.cfg.get("precompute", {}).get("max_shards")
        max_shards_cfg = (max_shards_raw.get(self.scale)
                          if isinstance(max_shards_raw, dict) else max_shards_raw)
        max_shards_arg = f"--max-shards {max_shards_cfg}" if max_shards_cfg else ""
        if max_shards_cfg:
            log_orch(f"Chunk {chunk}: build capped at {max_shards_cfg} shards "
                     f"(precompute.max_shards[{self.scale}])", chunk=chunk)
        cmd = self._python_cmd("build_shards.py",
                               f"--sources {sources} --output '{out}' "
                               f"--start-idx {start_idx} "
                               f"--chunk {chunk} "
                               f"--workers 1 {blocklist_arg} {max_shards_arg}")
        cmd = self._throttle_wrap(cmd)
        self._launch_prep(f"build chunk {chunk}", cmd, log_file,
                          chunk, "build_shards", token="DISK_WRITE_HIGH")

    def _start_dedupe_filter(self, chunk: int) -> None:
        if is_done(chunk, "dedupe_filter"): return
        if not gpu_is_free() or self._prep_busy(): return
        if not self.res.request("GPU_TOKEN", f"dedupe_filter chunk {chunk}"): return
        log_orch(f"Chunk {chunk}: CLIP dedup-filter", chunk=chunk)
        conv_dir  = STAGING_DIR / f"chunk{chunk}" / "converted" / "journeydb"
        cold_root = Path(self.cfg.get("storage", {}).get("cold_root", str(COLD_ROOT)))
        log_file  = LOG_DIR / f"dedupe_filter_chunk{chunk}.log"
        cmd = self._python_cmd("dedupe_filter.py",
                               f"--chunk {chunk} "
                               f"--conv-dir '{conv_dir}' "
                               f"--cold-root '{cold_root}'")
        self._launch_prep(f"dedupe_filter chunk {chunk}", cmd, log_file,
                          chunk, "dedupe_filter", token="GPU_TOKEN")

    def _start_clip_embed(self, chunk: int) -> None:
        if is_done(chunk, "clip_embed"):
            return
        if self.skip_dedup:
            log_orch(f"Chunk {chunk}: --skip-dedup set", chunk=chunk)
            mark_done(chunk, "clip_embed")
            mark_done(chunk, "clip_index")
            mark_done(chunk, "clip_dups")
            return
        if not gpu_is_free() or self._prep_busy():
            return
        if not self.res.request("GPU_TOKEN", f"clip_embed chunk {chunk}"):
            return
        log_orch(f"Chunk {chunk}: CLIP embedding", chunk=chunk)
        shard_dir = STAGING_DIR / f"chunk{chunk}" / "shards"
        embed_dir = STAGING_DIR / f"chunk{chunk}" / "embeddings"
        log_file  = LOG_DIR / f"clip_embed_chunk{chunk}.log"
        cmd = self._python_cmd("clip_dedup.py",
                               f"embed --shards '{shard_dir}' --embeddings '{embed_dir}' --chunk {chunk}")
        self._launch_prep(f"clip_embed chunk {chunk}", cmd, log_file,
                          chunk, "clip_embed", token="GPU_TOKEN")

    def _start_clip_index(self, chunk: int) -> None:
        if is_done(chunk, "clip_index") or self._prep_busy():
            return
        log_orch(f"Chunk {chunk}: building CLIP index", chunk=chunk)
        embed_dir = STAGING_DIR / f"chunk{chunk}" / "embeddings"
        index     = DEDUP_DIR / "dedup_index.faiss"
        log_file  = LOG_DIR / f"clip_index_chunk{chunk}.log"
        cmd = self._python_cmd("clip_dedup.py",
                               f"build-index --embeddings '{embed_dir}' --index '{index}'")
        self._launch_prep(f"clip_index chunk {chunk}", cmd, log_file,
                          chunk, "clip_index")

    def _start_clip_dups(self, chunk: int) -> None:
        if is_done(chunk, "clip_dups") or self._prep_busy():
            return
        log_orch(f"Chunk {chunk}: finding CLIP duplicates", chunk=chunk)
        index    = DEDUP_DIR / "dedup_index.faiss"
        dups     = DEDUP_DIR / "duplicate_ids.txt"
        report   = LOG_DIR / f"clip_dups_report_chunk{chunk}.json"
        log_file = LOG_DIR / f"clip_dups_chunk{chunk}.log"
        cmd = self._python_cmd("clip_dedup.py",
                               f"find-dups --index '{index}' --out '{dups}'"
                               f" --report-out '{report}'")
        self._launch_prep(f"clip_dups chunk {chunk}", cmd, log_file,
                          chunk, "clip_dups")

    def _proxy_vae_args(self, campaign: Optional[str] = None) -> str:
        """Resolve proxy VAE CLI args for precompute_all.py from pipeline config.

        Reads cfg["proxy_vae"]: requires both proxy_path and enabled=true.
        Per-campaign overrides (proxy_vae.campaigns.<name>) take precedence over
        default_mode / fallback_threshold.  Returns "" when proxy is disabled or
        the checkpoint is missing (precompute then uses the real VAE).
        """
        pcfg = self.cfg.get("proxy_vae", {}) or {}
        proxy_path = pcfg.get("proxy_path")
        if not proxy_path or not pcfg.get("enabled", False):
            return ""
        if not Path(proxy_path).exists():
            log_orch(f"proxy_vae.proxy_path missing ({proxy_path}); using real VAE",
                     level="warning")
            return ""

        mode = pcfg.get("default_mode", "balanced")
        thr  = pcfg.get("fallback_threshold", 0.75)
        camp_overrides = (pcfg.get("campaigns") or {}).get(campaign, {}) if campaign else {}
        mode = camp_overrides.get("mode", mode)
        thr  = camp_overrides.get("fallback_threshold", thr)

        return (f"--proxy-vae '{proxy_path}' "
                f"--proxy-vae-threshold {thr} "
                f"--proxy-mode {mode} ")

    def _start_precompute(self, chunk: int) -> None:
        if is_done(chunk, "precompute") or self._prep_busy():
            return
        if not self.res.request("GPU_TOKEN", f"precompute chunk {chunk}"):
            return
        staging    = STAGING_DIR / f"chunk{chunk}"
        shard_dir  = staging / "shards"
        qwen3_out  = staging / "precomputed" / "qwen3"
        vae_out    = staging / "precomputed" / "vae"
        siglip_out = staging / "precomputed" / "siglip"
        training_cfg = self.cfg.get("training", {})
        siglip_flag  = "--siglip" if training_cfg.get("siglip", False) else ""
        flux_model_name = self.cfg.get("model", {}).get("flux_model", "flux-klein-4b")
        flux_model_path = Path(flux_model_name)
        if not flux_model_path.is_absolute():
            flux_model_path = TRAIN_DIR.parent / flux_model_name
        flux_model_arg = f"--flux-model '{flux_model_path}'" if flux_model_path.exists() else ""
        precomp_cfg    = self.cfg.get("precompute", {})
        max_shards_raw = precomp_cfg.get("max_shards", None)
        max_shards = max_shards_raw.get(self.scale) if isinstance(max_shards_raw, dict) else max_shards_raw
        max_shards_arg = f"--max-shards {max_shards}" if max_shards else ""
        vae_batch      = precomp_cfg.get("vae_batch", None)
        vae_batch_arg  = f"--vae-batch {vae_batch}" if vae_batch is not None else ""
        log_file     = LOG_DIR / f"precompute_chunk{chunk}.log"
        log_orch(f"Chunk {chunk}: precomputing Qwen3+VAE embeddings", chunk=chunk)
        proxy_arg = self._proxy_vae_args(campaign=self.cfg.get("run_name"))
        cmd = self._python_cmd("precompute_all.py",
                               f"--shards '{shard_dir}' "
                               f"--qwen3-output '{qwen3_out}' "
                               f"--vae-output '{vae_out}' "
                               f"--siglip-output '{siglip_out}' "
                               f"--chunk {chunk} "
                               f"{flux_model_arg} "
                               f"{max_shards_arg} "
                               f"{vae_batch_arg} "
                               f"{proxy_arg}"
                               f"{siglip_flag}")
        self._launch_prep(f"precompute chunk {chunk}", cmd, log_file,
                          chunk, "precompute", token="GPU_TOKEN")

    def _check_ready(self, chunk: int) -> None:
        """Promote staging to production, then start training."""
        # Chunk 1 is never subject to predictive staging; the staging gate only
        # applies to chunks predictively staged while the predecessor trained.
        # If staging errored, _poll_stager already dispatched an issue; we proceed
        # anyway so a bad cold drive doesn't permanently stall the pipeline.
        gate = _ready_gate(
            chunk,
            prev_train_done=(chunk == 1 or is_done(chunk - 1, "train")),
            gpu_free=gpu_is_free(),
            stager_enabled=self.stager.enabled,
            stage_done=is_done(chunk, "stage"),
            stage_error=has_error(chunk, "stage"),
        )
        if gate in ("wait_prev_train", "wait_gpu"):
            return
        if gate == "wait_stage":
            log_orch(f"Chunk {chunk}: waiting for staging to complete before training",
                     chunk=chunk)
            return
        if gate == "proceed_no_stage":
            log_orch(f"Chunk {chunk}: staging failed — proceeding without staged data "
                     f"(see dispatch queue for details)", chunk=chunk, level="warning")

        if not is_done(chunk, "promoted"):
            self._promote_chunk(chunk)
        if not is_done(chunk, "promoted"):
            return  # promotion failed or incomplete; wait for next poll
        self._start_training(chunk)

    def _promote_chunk(self, chunk: int) -> None:
        """Move staging shards + precomputed to production directories."""
        staging  = STAGING_DIR / f"chunk{chunk}"
        shard_src   = staging / "shards"
        precomp_src = staging / "precomputed"

        # Canary: verify no shard index overflows this chunk's reserved ID space.
        ceiling = chunk * SHARD_BLOCK
        overflow = []
        for tar in shard_src.glob("*.tar"):
            try:
                idx = int(tar.stem)
            except ValueError:
                continue
            if idx >= ceiling:
                overflow.append(tar.name)
        if overflow:
            msg = (
                f"Chunk {chunk}: SHARD ID OVERFLOW — {len(overflow)} shard(s) exceed "
                f"reserved ceiling {ceiling - 1} (block size {SHARD_BLOCK}). "
                f"First offenders: {sorted(overflow)[:5]}. "
                f"Increase SHARD_BLOCK or reduce data volume per chunk."
            )
            log_orch(msg, chunk=chunk)
            mark_error(chunk, "promoted", msg)
            return

        # Coverage check: precomputed dirs must have >= 90% of the records the
        # trainer will actually see.  When precompute.max_shards is configured,
        # only that many shards are precomputed intentionally; use that as the
        # expected count so smoke/limited runs don't trip this check.
        n_shards = sum(1 for _ in shard_src.glob("*.tar"))
        max_shards_raw = self.cfg.get("precompute", {}).get("max_shards")
        max_shards_cfg = max_shards_raw.get(self.scale) if isinstance(max_shards_raw, dict) else max_shards_raw
        n_expected_shards = min(n_shards, max_shards_cfg) if max_shards_cfg else n_shards
        if n_expected_shards > 0:
            shard_size = self.cfg.get("data", {}).get("shard_size", 5000)
            expected = n_expected_shards * shard_size
            min_records = int(expected * 0.90)
            subdirs_required = ["qwen3", "vae"]
            # Fix 4: enforce siglip coverage when siglip is enabled in training config.
            # Previously only qwen3/vae were checked, allowing a silent 0%-coverage
            # siglip cache to pass promotion and corrupt image conditioning in training.
            if self.cfg.get("training", {}).get("siglip", False):
                subdirs_required.append("siglip")
            for subdir in subdirs_required:
                src = precomp_src / subdir
                if not src.exists():
                    msg = (f"Chunk {chunk}: precomputed/{subdir} missing — "
                           f"run precompute_all.py before promotion.")
                    log_orch(msg, chunk=chunk, level="error")
                    mark_error(chunk, "promoted", msg)
                    return
                actual = sum(1 for f in src.iterdir() if f.suffix == ".npz")
                if actual < min_records:
                    flag = " --siglip" if subdir == "siglip" else ""
                    msg = (f"Chunk {chunk}: precomputed/{subdir} coverage too low — "
                           f"{actual:,} records vs {min_records:,} required "
                           f"(90% of {expected:,} expected). "
                           f"Re-run precompute_all.py{flag} to complete the cache.")
                    log_orch(msg, chunk=chunk, level="error")
                    mark_error(chunk, "promoted", msg)
                    return

        # ── CLIP-I regression gate: warn (or block) if val metrics show regression ─
        _val_metrics_path = LOG_DIR / f"val_chunk{chunk}" / "metrics.json"
        _clip_thresh = self.cfg.get("training", {}).get("clip_i_promotion_min_delta", -0.03)
        _clip_gate   = self.cfg.get("training", {}).get("clip_i_gate_promotion", False)
        if _val_metrics_path.exists():
            try:
                _vm = json.loads(_val_metrics_path.read_text())
                _delta = _vm.get("clip_i_delta_vs_prev")
                if _delta is not None and _delta < _clip_thresh:
                    _msg = (f"Chunk {chunk}: CLIP-I delta {_delta:.4f} < {_clip_thresh} "
                            f"— quality regression vs prior checkpoint")
                    log_orch(_msg, chunk=chunk, level="warning")
                    self._dispatch_once(
                        f"clip_regression_{chunk}", self._next_issue_id(), "warning",
                        f"CLIP-I regression chunk {chunk}: delta={_delta:.4f} < {_clip_thresh}",
                        chunk=chunk, suggested_action="review_val_metrics_before_promoting",
                        cooldown_secs=3600,
                    )
                    if _clip_gate:
                        mark_error(chunk, "promoted", _msg)
                        return
            except (ValueError, OSError) as _e:
                log_orch(f"Chunk {chunk}: could not read val metrics for CLIP-I gate: {_e}",
                         chunk=chunk)

        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        count = 0
        for tar in sorted(shard_src.glob("*.tar")):
            # Keep the original staging filename (e.g. "000000.tar" for chunk 1,
            # "200000.tar" for chunk 2).  This preserves the match between shard
            # member names like "200000_0003" and precomputed file "200000_0003.npz".
            tar.rename(SHARDS_DIR / tar.name)
            count += 1
        log_orch(f"Chunk {chunk}: promoted {count} shards to production", chunk=chunk)

        # Move precomputed files to versioned production dirs and update `current` symlink.
        # Version hash derived from the same config fields used by precompute_all.py.
        # IMPORTANT: move precomputed data BEFORE writing the promoted.done sentinel so that
        # a crash here leaves the chunk in a recoverable READY state (sentinel absent).
        _git_sha  = get_git_sha(Path(__file__).parent)
        for subdir in ["qwen3", "vae", "siglip"]:
            src = precomp_src / subdir
            if not src.exists():
                continue
            _cfg_subset = encoder_config_subset(subdir, self.cfg)
            from cache_manager import version_hash as _vh
            _ver = _vh(_cfg_subset, _git_sha)
            dst = PRECOMP_DIR / subdir / _ver
            dst.mkdir(parents=True, exist_ok=True)
            moved = 0
            for f in src.iterdir():
                # Precomputed files keep their staging-derived names (e.g. "000000_0012.npz").
                # The trainer derives the internal record prefix from the production shard name.
                f.rename(dst / f.name)
                moved += 1
            # Atomic symlink: PRECOMP_DIR/{encoder}/current → v_XXXXXX/
            from cache_manager import _atomic_symlink
            _atomic_symlink(PRECOMP_DIR / subdir / "current", _ver)
            log_orch(
                f"Chunk {chunk}: promoted {moved} {subdir} records → "
                f"{PRECOMP_DIR / subdir / _ver} (current → {_ver})",
                chunk=chunk,
            )

        mark_done(chunk, "promoted")
        self._cleanup_prep_logs(chunk)

    def _start_shard_validation(self, chunk: int) -> None:
        """PIPELINE-8: fast tarfile header scan before training."""
        if is_done(chunk, "validate_shards") or self._prep_busy():
            return
        log_orch(f"Chunk {chunk}: starting shard integrity scan", chunk=chunk)
        report = LOG_DIR / f"validate_shards_chunk{chunk}.json"
        log_file = LOG_DIR / f"validate_shards_chunk{chunk}.log"
        cmd = self._python_cmd("validate_shards.py",
                               f"--chunk {chunk} --report '{report}'")
        self._launch_prep(f"validate_shards chunk {chunk}", cmd, log_file,
                          chunk, "validate_shards")

    def _start_training_warmup(self, chunk: int) -> None:
        """Compile Metal PSO training graphs before training starts.

        Runs train_ip_adapter.py --warmup-only, which loads the model, runs one
        forward+backward eval per bucket shape to populate the Metal PSO cache,
        then exits.  The cache is machine-wide and persists across restarts, so
        subsequent chunks skip warmup immediately.
        """
        if is_done(chunk, "training_warmup") or self._prep_busy():
            return

        # Metal PSO cache is machine-wide — if any prior chunk completed warmup,
        # the compiled kernels are already cached.  Mark done and move on.
        if chunk > 1 and is_done(chunk - 1, "training_warmup"):
            log_orch(
                f"Chunk {chunk}: Metal PSO cache warm (chunk {chunk - 1} warmup done)"
                f" — skipping training_warmup",
                chunk=chunk,
            )
            mark_done(chunk, "training_warmup")
            return

        if not gpu_is_free():
            return
        if not self.res.request("GPU_TOKEN", f"training_warmup chunk {chunk}"):
            return

        log_orch(f"Chunk {chunk}: warming up IP Adapter Metal training graphs", chunk=chunk)
        cfg_path = self.cfg.get("training_config", str(TRAIN_DIR / "configs" / "stage1_512px.yaml"))
        log_file = LOG_DIR / f"training_warmup_chunk{chunk}.log"
        cmd = f"{VENV_PYTHON} -u '{TRAIN_DIR}/train_ip_adapter.py' --config '{cfg_path}' --warmup-only --data-root '{self.prep_root}'"
        self._launch_prep(
            f"training_warmup chunk {chunk}", cmd, log_file,
            chunk, "training_warmup", token="GPU_TOKEN",
        )

    def _start_training(self, chunk: int) -> None:
        log_file = LOG_DIR / f"train_chunk{chunk}.log"

        if tmux_window_exists(TMUX_TRAIN_WIN):
            return  # still running

        # Window is gone — if a previous attempt left a log, check outcome.
        if log_file.exists():
            code = last_exit_code(log_file)
            _train_token = f"train chunk {chunk}"
            if code == 0:
                log_orch(f"Chunk {chunk}: training complete", chunk=chunk)
                mark_done(chunk, "train")
                # Archive fires after mine.done (see _post_step / _poll_stager) so that
                # shards remain hot and readable during hard-example scoring.
                self._archive_chunk_checkpoint(chunk)
                self._auto_populate_anchor_shards(chunk)
                if chunk >= self.total_chunks:
                    self._create_release_bundle(chunk)
                notify("iris pipeline", f"Chunk {chunk} training complete")
                if self.res.holder("GPU_TOKEN") == _train_token:
                    self.res.release("GPU_TOKEN")
                return
            if code is not None:
                msg = f"Training exited {code}; see {log_file}"
                log_orch(f"Chunk {chunk}: training FAILED — {msg}", level="error", chunk=chunk)
                mark_error(chunk, "train", msg)
                # Only release if training holds the token — do NOT release a token
                # held by another step (e.g. precompute on another chunk), which would
                # let training start concurrently with that step on the same GPU.
                if self.res.holder("GPU_TOKEN") == _train_token:
                    self.res.release("GPU_TOKEN")
                notify("iris pipeline ERROR", f"Training chunk {chunk} failed")
                return
            # code is None: window closed but EXIT_CODE not yet written (narrow race
            # between process exit and the shell appending EXIT_CODE= to the log).
            # Wait for the next poll rather than acquiring GPU_TOKEN prematurely.
            log_orch(f"Chunk {chunk}: training window gone, exit code not yet written — waiting",
                     chunk=chunk)
            return

        if not self.res.request("GPU_TOKEN", f"train chunk {chunk}"):
            return

        if not self._disk_guard(f"train chunk {chunk}"):
            self.res.release("GPU_TOKEN")
            return

        training_cfg = self.cfg.get("training", {})
        steps_map    = training_cfg.get("steps", {}).get(self.scale, {})
        steps = steps_map.get(chunk, steps_map.get(str(chunk), 15000))
        lr_map = training_cfg.get("lr", {})
        lr = lr_map.get(chunk, lr_map.get(str(chunk), 1e-5))

        # chunk_base_step: sum of all prior chunks' steps.  Passed to the
        # training script so it can compute the correct end step regardless of
        # whether it's a fresh chunk start or a mid-chunk crash resume.
        chunk_base_step = sum(
            steps_map.get(c, steps_map.get(str(c), 0))
            for c in range(1, chunk)
        )

        log_orch(f"Chunk {chunk}: starting training ({steps} steps, lr={lr}, base_step={chunk_base_step})", chunk=chunk)
        # Predictive staging: stage chunk N+1's data while chunk N trains, so it
        # is ready on hot storage before training finishes.
        if chunk < self.total_chunks:
            self._launch_stager(
                f"stage --chunk {chunk + 1}",
                description=f"predictive stage chunk {chunk + 1}",
                chunk=chunk + 1,
                log_name=f"stager_stage_chunk{chunk + 1}",
            )

        resume_arg = ""
        # Resume from the latest intermediate checkpoint if one exists.
        # This handles both mid-run restarts (any chunk) and chunk>1 warm-starts.
        _ckpts = sorted(self.ckpt_dir.glob("step_*.safetensors"))
        if _ckpts:
            resume_arg = f"--resume '{_ckpts[-1]}'"
        elif chunk > 1:
            best = self.ckpt_dir / "best.safetensors"
            if best.exists():
                resume_arg = f"--resume '{best}'"

        hard_arg = ""
        if chunk > 1:
            hard_chunk = HARD_EX_DIR / f"chunk{chunk-1}"
            if hard_chunk.exists() and any(hard_chunk.glob("*.tar")):
                hard_arg = f"--hard-examples '{hard_chunk}'"

        anchor_arg = ""
        if chunk > 1:
            if ANCHOR_SHARDS_DIR.exists() and any(ANCHOR_SHARDS_DIR.glob("*.tar")):
                anchor_arg = f"--anchor-shards '{ANCHOR_SHARDS_DIR}'"
            else:
                log_orch(
                    f"Chunk {chunk}: {ANCHOR_SHARDS_DIR} missing or empty — "
                    "starting without anchor mixing (populate dir to enable)",
                    chunk=chunk,
                )

        config_file = self.cfg.get("training_config",
                                   str(TRAIN_DIR / "configs" / "stage1_512px.yaml"))

        # PIPELINE-14: apply per-chunk hard_mix_ratio override via a temp config
        # so we don't need to touch the training script.
        _by_chunk = (self.cfg.get("training", {})
                         .get("hard_mix_ratio_by_chunk", {}))
        _override = _by_chunk.get(chunk, _by_chunk.get(str(chunk)))
        if _override is not None:
            import yaml as _yaml
            with open(config_file) as _f:
                _train_yaml = _yaml.safe_load(_f)
            _default = _train_yaml.get("hard_mix_ratio", 0.05)
            _train_yaml["hard_mix_ratio"] = float(_override)
            _tmp_cfg = Path(f"/tmp/iris_train_chunk{chunk}_config.yaml")
            _tmp_cfg.write_text(_yaml.dump(_train_yaml, default_flow_style=False))
            self._active_train_tmp_cfg = _tmp_cfg
            log_orch(
                f"Chunk {chunk}: hard_mix_ratio override {_default} → {_override} "
                f"(temp config: {_tmp_cfg})",
                chunk=chunk)
            config_file = str(_tmp_cfg)

        cmd = (
            f"caffeinate -dim python -u '{TRAIN_DIR}/train_ip_adapter.py' "
            f"--config '{config_file}' "
            f"--max-steps {steps} --lr {lr} "
            f"--chunk-base-step {chunk_base_step} "
            f"--data-root '{self.prep_root}' "
            f"--chunk {chunk} "
            f"{resume_arg} {hard_arg} {anchor_arg}"
        )

        if not self.dry_run:
            activated = (f"export PIPELINE_DATA_ROOT='{DATA_ROOT}' && "
                         f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            tmux_new_window(TMUX_TRAIN_WIN, activated, log_file)
            notify("iris pipeline", f"Chunk {chunk} training started ({steps} steps, lr={lr:.0e})")
            # Reset the trainer heartbeat timestamp so _check_heartbeats doesn't
            # immediately flag the just-launched process as stale (old heartbeat
            # from a previous crashed session can be hours old, triggering a kill
            # within 60s before the new process has a chance to write its own).
            write_heartbeat("trainer", chunk, status="booting", step=0)

        update_state(**{"chunks": {str(chunk): {
            "state": "TRAINING", "started_at": now_iso(),
            "steps": steps, "lr": lr,
        }}})

    def _start_mining(self, chunk: int) -> None:
        if is_done(chunk, "mine") or self._prep_busy():
            return

        training_cfg = self.cfg.get("training", {})

        # Hard examples feed the next chunk's training — skip for the final chunk.
        if chunk >= self.total_chunks:
            log_orch(f"Chunk {chunk}: final chunk — skipping mining (no next chunk)", chunk=chunk)
            mark_done(chunk, "mine")
            return

        # Allow operators to disable mining entirely (e.g. dev/integration runs).
        if not training_cfg.get("mine", True):
            log_orch(f"Chunk {chunk}: mining disabled in config — skipping", chunk=chunk)
            mark_done(chunk, "mine")
            return

        if not gpu_is_free():
            return
        if not self.res.request("GPU_TOKEN", f"mine chunk {chunk}"):
            return

        best = self.ckpt_dir / "best.safetensors"
        if not best.exists():
            log_orch(f"Chunk {chunk}: no checkpoint — skipping mining", chunk=chunk)
            mark_done(chunk, "mine")
            self.res.release("GPU_TOKEN")
            return

        log_orch(f"Chunk {chunk}: mining hard examples", chunk=chunk)
        out = HARD_EX_DIR / f"chunk{chunk}"
        out.mkdir(parents=True, exist_ok=True)
        flux_model  = self.cfg.get("model", {}).get("flux_model", "flux-klein-4b")
        # Resolve versioned cache dirs; fall back to flat layout for backwards compat.
        qwen3_cache = (PrecomputeCache.effective_dir(PRECOMP_DIR, "qwen3")
                       or PRECOMP_DIR / "qwen3")
        vae_cache   = (PrecomputeCache.effective_dir(PRECOMP_DIR, "vae")
                       or PRECOMP_DIR / "vae")
        log_file    = LOG_DIR / f"mine_chunk{chunk}.log"
        use_ema_flag = "--use-ema " if training_cfg.get("mine_use_ema", False) else ""

        # Use precomputed SigLIP features when available — aligns mining loss with
        # training which conditions on real visual features.  Fall back to
        # null-siglip only when siglip precompute is disabled or cache is absent.
        siglip_eff   = PrecomputeCache.effective_dir(PRECOMP_DIR, "siglip")
        siglip_cache = siglip_eff or PRECOMP_DIR / "siglip"
        if training_cfg.get("siglip", False) and siglip_eff:
            siglip_arg = f"--siglip-cache '{siglip_cache}' "
        else:
            siglip_arg = "--null-siglip "

        # Eval budget: look up by scale (dict) or use flat value; 5000 default.
        _ev = training_cfg.get("mine_eval_records", 5000)
        eval_records = (
            _ev.get(self.scale, _ev.get("default", 5000)) if isinstance(_ev, dict) else int(_ev)
        )
        _tk = training_cfg.get("mine_top_k", 2000)
        top_k = (
            _tk.get(self.scale, _tk.get("default", 2000)) if isinstance(_tk, dict) else int(_tk)
        )

        cmd = self._python_cmd("mine_hard_examples.py",
                               f"--checkpoint '{best}' "
                               f"--shards '{SHARDS_DIR}' "
                               f"--qwen3-cache '{qwen3_cache}' "
                               f"--vae-cache '{vae_cache}' "
                               f"--flux-model {flux_model} "
                               f"{siglip_arg}"
                               f"--chunk {chunk} "
                               f"{use_ema_flag}"
                               f"--eval-records {eval_records} --top-k {top_k} "
                               f"--output '{out}'")
        self._launch_prep(f"mine chunk {chunk}", cmd, log_file,
                          chunk, "mine", token="GPU_TOKEN")

    def _start_validation(self, chunk: int) -> None:
        if is_done(chunk, "validate") or self._prep_busy():
            return
        if not gpu_is_free():
            return
        if not self.res.request("GPU_TOKEN", f"validate chunk {chunk}"):
            return
        best = self.ckpt_dir / "best.safetensors"
        if not best.exists():
            log_orch(f"Chunk {chunk}: no checkpoint — skipping validation", chunk=chunk)
            self.res.release("GPU_TOKEN")
            mark_done(chunk, "validate")
            return
        log_orch(f"Chunk {chunk}: starting validation", chunk=chunk)
        prev_val_arg = ""
        if chunk > 1:
            prev_dir = LOG_DIR / f"val_chunk{chunk - 1}"
            if prev_dir.exists():
                prev_val_arg = f"--prev-val-dir '{prev_dir}'"
        log_file = LOG_DIR / f"validate_chunk{chunk}.log"
        cmd = self._python_cmd("validator.py",
                               f"--chunk {chunk} --checkpoint '{best}' "
                               f"--config '{self._config_path()}' {prev_val_arg}")
        self._launch_prep(f"validate chunk {chunk}", cmd, log_file,
                          chunk, "validate", token="GPU_TOKEN")

    def _handle_error(self, chunk: int) -> None:
        for step in CHUNK_STEPS:
            if has_error(chunk, step):
                msg = read_error(chunk, step)
                key = (chunk, step)
                restarts = self._restart_counts.get(key, 0)

                # Diagnose crash reason to decide retry policy.  Cache the result
                # so _query_macos_jetsam_log() (15 s subprocess) only runs once per
                # error episode rather than on every orchestrator poll.
                exit_code = _parse_exit_code_from_msg(msg)
                log_file = LOG_DIR / f"{step}_chunk{chunk}.log"
                _diag_key = (chunk, step, exit_code)
                if _diag_key not in self._crash_diag:
                    self._crash_diag[_diag_key] = _diagnose_crash(log_file, exit_code, self._mem_log)
                reason, detail = self._crash_diag[_diag_key]
                should_retry, max_retries, retry_delay = _retry_policy(reason, restarts)

                if should_retry:
                    # Honour backoff window: don't relaunch until pressure settles.
                    retry_at = self._retry_after.get(key, 0.0)
                    if time.time() < retry_at:
                        remaining = int(retry_at - time.time())
                        log_orch(
                            f"Chunk {chunk}: {step} retry #{restarts + 1}/{max_retries} "
                            f"pending — {remaining}s backoff remaining ({reason})",
                            chunk=chunk,
                        )
                        break

                    log_orch(
                        f"Chunk {chunk}: auto-retrying {step} "
                        f"({detail}, attempt {restarts + 1}/{max_retries})",
                        chunk=chunk,
                    )
                    self._restart_counts[key] = restarts + 1
                    self._persist_restart_counts()
                    if retry_delay > 0:
                        self._retry_after[key] = time.time() + retry_delay
                    clear_error(chunk, step)
                    self._crash_diag.pop(_diag_key, None)  # invalidate for next episode
                    # Rename the old step log with a timestamp so _start_training
                    # (and similar detection loops) don't re-read a stale EXIT_CODE,
                    # but the crash evidence is preserved for post-mortem debugging.
                    old_log = LOG_DIR / f"{step}_chunk{chunk}.log"
                    if old_log.exists():
                        ts = now_iso().replace(":", "").replace("+", "").replace("-", "")[:15]
                        old_log.rename(old_log.with_suffix(f".{ts}.log"))
                else:
                    issue_id = self._next_issue_id()
                    log_orch(
                        f"Chunk {chunk}: {step} failed {restarts + 1} times "
                        f"({reason}) — escalating",
                        level="error", chunk=chunk,
                    )
                    dispatch_issue(
                        issue_id, "error",
                        f"{step} chunk {chunk} failed {restarts + 1} times "
                        f"({reason}: {detail}) — manual intervention needed",
                        chunk=chunk, process=step, context={"error": msg},
                        suggested_action="investigate_logs_and_clear_error",
                    )
                    notify("iris pipeline ERROR",
                           f"{step} chunk {chunk} needs manual intervention ({reason})")
                break

    # -----------------------------------------------------------------------
    # Source specification for build_shards
    # -----------------------------------------------------------------------

    def _effective_chunk_denom(self) -> int:
        """Effective denominator for LAION/COYO slice based on scale data_volume fraction."""
        fractions = self.cfg.get("data_volume", {
            "small": 0.10, "medium": 0.25, "large": 0.50, "all-in": 1.00,
        })
        frac = fractions.get(self.scale, 1.0)
        return max(self.total_chunks, round(self.total_chunks / frac))

    def _build_sources(self, chunk: int) -> str:
        """Space-separated source arguments, with :chunk/total slice for global sources."""
        parts = []
        # download_convert outputs extracted JDB images here
        jdb_dir  = STAGING_DIR / f"chunk{chunk}" / "converted" / "journeydb"
        # WikiArt is saved as a HuggingFace dataset directory
        wiki_dir = STAGING_DIR / f"chunk{chunk}" / "raw" / "wikiart"
        if jdb_dir.exists():
            parts.append(f"'{jdb_dir}'")
        if wiki_dir.exists():
            parts.append(f"'{wiki_dir}'")
        laion = DATA_ROOT / "converted" / "laion"
        coyo  = DATA_ROOT / "converted" / "coyo"
        eff_denom = self._effective_chunk_denom()
        if laion.exists() and any(laion.glob("*.tar")):
            parts.append(f"'{laion}:{chunk}/{eff_denom}'")
        if coyo.exists() and any(coyo.glob("*.tar")):
            parts.append(f"'{coyo}:{chunk}/{eff_denom}'")
        return " ".join(parts) if parts else "''"

    # -----------------------------------------------------------------------
    # GPU token sync — detect when training window disappears
    # -----------------------------------------------------------------------

    def _sync_gpu_token(self) -> None:
        holder = self.res.holder("GPU_TOKEN")
        if not holder:
            return
        # Training runs in TMUX_TRAIN_WIN; all other GPU steps (precompute, clip_embed,
        # training_warmup, mine, validate) run in TMUX_PREP_WIN via _launch_prep.
        window = TMUX_TRAIN_WIN if holder.startswith("train chunk") else TMUX_PREP_WIN
        if not tmux_window_exists(window):
            self.res.release("GPU_TOKEN")

    # -----------------------------------------------------------------------
    # Heartbeat monitoring
    # -----------------------------------------------------------------------

    def _check_heartbeats(self) -> None:
        # Fix 3: check prep worker heartbeat for hung processes (no crash, no progress).
        # Threshold is generous (1800s) to avoid false positives during slow operations
        # like siglip precompute (~24 min/shard) where the heartbeat thread still writes
        # every 60s; a 30-min stale gap means the process is truly hung or dead.
        if self._active_prep is not None:
            self._check_prep_heartbeat()
        if not tmux_window_exists(TMUX_TRAIN_WIN):
            return
        for chunk in range(1, self.total_chunks + 1):
            if derive_chunk_state(chunk) == ChunkState.TRAINING:
                self._check_training_anomalies(chunk)

    def _check_prep_heartbeat(self) -> None:
        ap = self._active_prep
        if ap is None:
            return
        chunk = ap["chunk"]
        step  = ap["step"]
        _step_process = {
            "download":      "download_convert",
            "dedupe_filter": "dedupe_filter",
            "build_shards":  "build_shards",
            "clip_embed":      "clip_dedup",
            "clip_index":      "clip_dedup",
            "clip_dups":       "clip_dedup",
            "precompute":      "precompute",
            "validate_shards": "validate_shards",
            "mine":            "mine_hard_examples",
            "validate":        "validator",
        }
        process = _step_process.get(step)
        if not process:
            return
        age = heartbeat_age_secs(process, chunk)
        if age is None or age < 1800:
            return
        log_orch(f"Chunk {chunk}: {step} heartbeat stale ({age:.0f}s) — may be hung",
                 level="warning", chunk=chunk)
        self._dispatch_once(
            f"prep_stale_{chunk}_{step}", self._next_issue_id(), "warning",
            f"{step} chunk {chunk} heartbeat stale for {age:.0f}s — may be hung. "
            f"Check: tail -20 {ap['log']}",
            chunk=chunk, process=step,
            suggested_action="check_prep_log_and_kill_if_hung",
        )

    def _dispatch_once(self, key: str, issue_id: str, severity: str, message: str,
                       cooldown_secs: float = 3600.0, **kwargs) -> None:
        """dispatch_issue with per-key cooldown and back-off to prevent alert floods.

        After 5 fires of the same key the cooldown doubles to 7200s, reducing
        log noise during prolonged stuck states (e.g. 200+ escalations/hour).
        """
        now = time.time()
        count = self._dispatch_count.get(key, 0)
        effective_cooldown = cooldown_secs * 2 if count >= 5 else cooldown_secs
        if now - self._dispatch_last.get(key, 0.0) < effective_cooldown:
            return
        self._dispatch_last[key]  = now
        self._dispatch_count[key] = count + 1
        dispatch_issue(issue_id, severity, message, **kwargs)

    def _check_training_anomalies(self, chunk: int) -> None:
        """Section 5.1 anomaly detection rules for a training chunk."""
        import math

        hb = read_heartbeat("trainer", chunk)
        age = heartbeat_age_secs("trainer", chunk)

        # ── Heartbeat staleness → crash detection + restart ──────────────────
        if age is not None and age > HEARTBEAT_STALE_SECS:
            hb_step = (hb or {}).get("step", 0)
            hb_status = (hb or {}).get("status", "")
            # Boot-phase grace: model load + graph compile can take 20–30 min.
            # If the training window is still alive and the heartbeat shows step=0
            # (or status="booting"), the process is loading — do not restart it.
            # A genuine hang during load would require manual intervention anyway.
            if (hb_step == 0 or hb_status == "booting") and tmux_window_exists(TMUX_TRAIN_WIN):
                log_orch(
                    f"Chunk {chunk}: heartbeat stale ({age:.0f}s) but training still "
                    f"in boot phase (step=0, window alive) — waiting",
                    level="warning", chunk=chunk,
                )
                return
            log_orch(f"Chunk {chunk}: trainer heartbeat stale ({age:.0f}s) — restarting",
                     level="error", chunk=chunk)
            self._dispatch_once(f"stale_{chunk}", self._next_issue_id(), "error",
                                f"Trainer heartbeat stale ({age:.0f}s) — restarting process",
                                chunk=chunk, process="trainer",
                                suggested_action="check_training_log")
            self._restart_trainer(chunk)
            return

        if hb is None:
            return

        loss = hb.get("loss")
        step = hb.get("step", 0)

        # Reset stale-restart counter once the training loop is running.
        # Successful boot means any previous restart budget from boot-phase kills
        # is no longer relevant; fresh budget for real mid-training hangs.
        if step > 0:
            if self._restart_counts.pop(("restart_trainer", chunk), None) is not None:
                self._persist_restart_counts()

        # ── Loss NaN/Inf → immediate pause ───────────────────────────────────
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            log_orch(f"Chunk {chunk}: loss NaN/Inf at step {step} — pausing",
                     level="error", chunk=chunk)
            self._dispatch_once(f"nan_{chunk}", self._next_issue_id(), "critical",
                                f"Training loss NaN/Inf at step {step}",
                                chunk=chunk, suggested_action="investigate_training")
            self._write_pause()
            return

        _a = self.cfg.get("anomaly", {})
        _loss_thresh   = _a.get("loss_threshold",    2.0)
        _loss_steps    = _a.get("loss_high_steps",   100)
        _grad_pause    = _a.get("grad_norm_pause",   50.0)
        _grad_warn     = _a.get("grad_norm_warn",    10.0)
        _grad_polls    = _a.get("grad_spike_polls",  10)
        _siglip_min    = _a.get("siglip_min_coverage", 90)

        # ── Loss persistently high for loss_high_steps → pause ───────────────
        if loss is not None:
            if loss > _loss_thresh:
                first = self._loss_high_since.get(chunk)
                if first is None:
                    self._loss_high_since[chunk] = step
                elif step - first >= _loss_steps:
                    log_orch(f"Chunk {chunk}: loss {loss:.3f} > {_loss_thresh} for {step-first} steps — pausing",
                             level="error", chunk=chunk)
                    self._dispatch_once(f"loss_high_{chunk}", self._next_issue_id(), "error",
                                        f"Training loss {loss:.3f} persistently > {_loss_thresh} (since step {first})",
                                        chunk=chunk, suggested_action="investigate_training")
                    self._write_pause()
                    self._loss_high_since[chunk] = None
            else:
                self._loss_high_since[chunk] = None

        # ── Grad norm sustained high → pause; mildly high → warn ────────────
        grad_norm = hb.get("grad_norm")
        if grad_norm is not None:
            if grad_norm > _grad_pause:
                self._grad_spike_polls[chunk] = self._grad_spike_polls.get(chunk, 0) + 1
                n_spikes = self._grad_spike_polls[chunk]
                if n_spikes >= _grad_polls:
                    log_orch(f"Chunk {chunk}: grad_norm {grad_norm:.1f} > {_grad_pause} for {n_spikes} polls — pausing",
                             level="error", chunk=chunk)
                    self._dispatch_once(f"grad_{chunk}", self._next_issue_id(), "error",
                                        f"Grad norm {grad_norm:.1f} sustained > {_grad_pause} ({n_spikes} polls)",
                                        chunk=chunk, suggested_action="investigate_training")
                    self._write_pause()
                    self._grad_spike_polls[chunk] = 0
                elif grad_norm > _grad_warn:
                    log_orch(f"Chunk {chunk}: grad_norm warning {grad_norm:.1f} > {_grad_warn}",
                             level="warning", chunk=chunk)
            else:
                self._grad_spike_polls[chunk] = 0

        # ── SigLIP coverage below minimum → log warning ───────────────────────
        siglip_cov = hb.get("siglip_coverage_pct")
        if siglip_cov is not None and siglip_cov < _siglip_min:
            log_orch(f"Chunk {chunk}: SigLIP coverage {siglip_cov:.0f}% < {_siglip_min}%",
                     level="warning", chunk=chunk)

        # ── Low system memory → pre-OOM dispatch warning ─────────────────────
        mem_avail = hb.get("mem_available_gb")
        if mem_avail is not None and mem_avail < 3.0:
            self._dispatch_once(f"mem_low_{chunk}", self._next_issue_id(), "warning",
                                f"Trainer mem_available_gb={mem_avail:.1f} < 3.0 — OOM risk",
                                chunk=chunk, process="trainer",
                                suggested_action="investigate_memory")

        # ── EMA drift sustained high → dispatch instability warning ──────────
        ema_drift = hb.get("ema_drift")
        _drift_warn  = _a.get("ema_drift_warn",  0.05)
        _drift_polls = _a.get("ema_drift_polls",  3)
        if ema_drift is not None and ema_drift > _drift_warn:
            self._ema_drift_polls[chunk] = self._ema_drift_polls.get(chunk, 0) + 1
            if self._ema_drift_polls[chunk] >= _drift_polls:
                self._dispatch_once(
                    f"ema_drift_{chunk}", self._next_issue_id(), "warning",
                    f"EMA drift {ema_drift:.4f} > {_drift_warn} for {_drift_polls}+ polls"
                    f" — adapter weights drifting from EMA; check LR or gradient scale",
                    chunk=chunk, suggested_action="inspect_training_stability",
                )
        else:
            self._ema_drift_polls[chunk] = 0

    def _restart_trainer(self, chunk: int) -> None:
        """Kill and relaunch the training window for the given chunk."""
        if self.dry_run:
            log_orch(f"[dry-run] would restart trainer for chunk {chunk}")
            return
        key = ("restart_trainer", chunk)
        count = self._restart_counts.get(key, 0) + 1
        self._restart_counts[key] = count
        self._persist_restart_counts()
        if count > JETSAM_MAX_RETRIES:
            log_orch(f"Chunk {chunk}: trainer restart limit ({JETSAM_MAX_RETRIES}) exceeded — aborting",
                     level="error", chunk=chunk)
            self._dispatch_once(
                f"restart_limit_{chunk}", self._next_issue_id(), "critical",
                f"Trainer restart limit exceeded for chunk {chunk}",
                chunk=chunk, suggested_action="manual_intervention",
                cooldown_secs=3600,
            )
            return
        log_orch(f"Chunk {chunk}: killing iris-train window (restart #{count})", chunk=chunk)
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_TRAIN_WIN}"],
                       capture_output=True)
        # Delete the stale heartbeat so the next poll doesn't immediately re-trigger
        # another staleness alarm before the relaunched trainer has a chance to write one.
        from pipeline_lib import heartbeat_path as _hb_path
        _hbf = _hb_path("trainer", chunk)
        _hbf.unlink(missing_ok=True)
        # Let orchestrator's normal _check_training path relaunch on next poll
        # by clearing the train.done sentinel (training was not complete)
        clear_error(chunk, "train")

    def _write_pause(self) -> None:
        """Write a pause control signal so the operator can investigate."""
        try:
            _tmp = str(CONTROL_FILE) + ".tmp"
            with open(_tmp, "w") as _f:
                json.dump({"action": "pause"}, _f)
            os.replace(_tmp, CONTROL_FILE)
        except OSError:
            pass
        notify("iris pipeline WARNING", "Training paused by anomaly — check dispatch queue")

    # -----------------------------------------------------------------------
    # Memory watchdog
    # -----------------------------------------------------------------------

    def _start_memory_watchdog(self) -> None:
        """
        Launch a daemon thread that polls vm_stat every 30 seconds and appends
        a timestamped one-liner to memory_pressure.log.  On macOS, SIGKILL from
        jetsam may not produce a JetsamEvent report for every kill.  This rolling
        log gives us a pre-crash memory time-series so we can confirm memory
        pressure after the fact.

        Kept at 30-second granularity: low overhead, fine enough to bracket a
        crash within one poll interval.
        """
        import threading

        def _poll() -> None:
            import subprocess as _sp
            KEEP_LINES = 2880  # 24 h at 30 s per sample
            while not self._mem_watchdog_stop:
                try:
                    vm = _sp.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                    lines = vm.stdout.splitlines()
                    # Extract the three most useful numbers: free, inactive, compressor
                    stats: dict[str, int] = {}
                    for ln in lines:
                        for key, label in [
                            ("Pages free", "free"),
                            ("Pages inactive", "inactive"),
                            ("Pages stored in compressor", "compressor"),
                            ("Pages active", "active"),
                            ("Pages wired down", "wired"),
                        ]:
                            if ln.startswith(key):
                                try:
                                    stats[label] = int(ln.split(":")[1].strip().rstrip("."))
                                except ValueError:
                                    pass
                    page_bytes = 16384  # 16 KiB pages on Apple Silicon
                    def gb(pages: int) -> float:
                        return pages * page_bytes / 1e9
                    entry = (
                        f"{now_iso()}  "
                        f"free={gb(stats.get('free', 0)):.2f}GB  "
                        f"active={gb(stats.get('active', 0)):.2f}GB  "
                        f"inactive={gb(stats.get('inactive', 0)):.2f}GB  "
                        f"wired={gb(stats.get('wired', 0)):.2f}GB  "
                        f"compressor={gb(stats.get('compressor', 0)):.2f}GB"
                    )
                    with open(self._mem_log, "a") as fh:
                        fh.write(entry + "\n")
                    # Trim to last KEEP_LINES to prevent unbounded growth.
                    try:
                        existing = self._mem_log.read_text().splitlines()
                        if len(existing) > KEEP_LINES:
                            self._mem_log.write_text("\n".join(existing[-KEEP_LINES:]) + "\n")
                    except OSError:
                        pass
                except Exception:
                    pass
                # Sleep in short increments so _mem_watchdog_stop is checked promptly.
                for _ in range(30):
                    if self._mem_watchdog_stop:
                        return
                    time.sleep(1)

        t = threading.Thread(target=_poll, name="mem-watchdog", daemon=True)
        t.start()

    # -----------------------------------------------------------------------
    # Disk check
    # -----------------------------------------------------------------------

    def _check_disk(self) -> None:
        gb = free_gb()
        if gb < DISK_ABORT_GB:
            log_orch(f"CRITICAL: {gb:.1f} GB free — pausing pipeline", level="error")
            self._dispatch_once(
                "disk_critical", self._next_issue_id(), "error",
                f"Disk critically low: {gb:.1f} GB free (threshold {DISK_ABORT_GB} GB). "
                f"Pipeline paused until disk is freed.",
                suggested_action="free_disk_then_resume",
                cooldown_secs=300,
            )
            notify("iris pipeline WARNING", f"Disk critically low: {gb:.1f} GB — pipeline paused")
            self._write_pause()
        elif gb < DISK_WARN_GB:
            log_orch(f"WARNING: {gb:.1f} GB free", level="warning")

    # -----------------------------------------------------------------------
    # Control signals
    # -----------------------------------------------------------------------

    def _check_control_signals(self) -> None:
        if not CONTROL_FILE.exists():
            return
        try:
            with open(CONTROL_FILE) as f:
                ctrl = json.load(f)
            action = ctrl.get("action")
            if action == "pause":
                log_orch("Operator: pause — sleeping up to 300s (re-checks every 5s)")
                deadline = time.time() + 300
                while time.time() < deadline:
                    time.sleep(5)
                    try:
                        with open(CONTROL_FILE) as _f2:
                            if json.load(_f2).get("action") != "pause":
                                break
                    except Exception:
                        break  # file removed or replaced — exit pause immediately
            elif action == "abort":
                log_orch("Operator: abort")
                sys.exit(0)
            elif action == "force-next-chunk":
                chunk = ctrl.get("chunk")
                if chunk:
                    log_orch(f"Operator: force-next-chunk {chunk}")
                    mark_done(chunk, "validate")
            elif action == "retry":
                chunk = ctrl.get("chunk")
                step  = ctrl.get("step")
                if chunk and step:
                    key = (chunk, step)
                    self._restart_counts.pop(key, None)
                    # Also reset the stale-restart counter so a retry after manual
                    # intervention doesn't immediately hit the inherited limit.
                    if step == "train":
                        self._restart_counts.pop(("restart_trainer", chunk), None)
                    self._persist_restart_counts()
                    # Clear cached crash diagnosis so next episode gets a fresh read.
                    for k in list(self._crash_diag):
                        if k[:2] == (chunk, step):
                            del self._crash_diag[k]
                    clear_error(chunk, step)
                    # Archive any stale log that still has EXIT_CODE at the end.
                    # _start_training() reads the log to detect completion/failure;
                    # without archiving, it would re-write the error sentinel and
                    # never actually launch the new run.
                    if step == "train":
                        log_file = LOG_DIR / f"train_chunk{chunk}.log"
                        if log_file.exists() and last_exit_code(log_file) is not None:
                            ts = now_iso().replace(":", "").replace("+", "").replace("-", "")[:15]
                            archived = log_file.with_suffix(f".{ts}.log")
                            log_file.rename(archived)
                            log_orch(f"Operator: archived stale train log → {archived.name}",
                                     chunk=chunk)
                    log_orch(f"Operator: retry chunk {chunk} step {step} — counter reset", chunk=chunk)
            CONTROL_FILE.unlink(missing_ok=True)
        except Exception as e:
            log_orch(f"Control file error: {e}", level="warning")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _python_cmd(self, script: str, args: str) -> str:
        return f"python -u '{SCRIPTS_DIR}/{script}' {args}"

    def _throttle_wrap(self, cmd: str) -> str:
        """Wrap cmd with background I/O + CPU deprioritisation when training is active.

        Uses macOS taskpolicy(1) to set IOPOL_THROTTLE: the kernel services
        the wrapped process's disk I/O only when the device queue is otherwise
        idle, so training reads always take precedence.  nice -n 10 gives a
        matching CPU yield.  No-op when training is not active.
        """
        if not self._training_active():
            return cmd
        return f"nice -n 10 taskpolicy -d throttle {cmd}"

    def _config_path(self) -> str:
        return self.cfg.get("_config_path", str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))

    def _all_done(self) -> bool:
        return all(derive_chunk_state(c) == ChunkState.DONE
                   for c in range(1, self.total_chunks + 1))

    @staticmethod
    def _read_max_issue_counter() -> int:
        """Scan dispatch queue for the highest issue number so restarts don't reuse IDs."""
        max_n = 0
        try:
            for line in DISPATCH_QUEUE.read_text().splitlines():
                m = re.search(r'"id"\s*:\s*"I-(\d+)"', line)
                if m:
                    max_n = max(max_n, int(m.group(1)))
        except Exception:
            pass
        return max_n

    def _load_restart_counts(self) -> dict:
        """Load _restart_counts from pipeline_state.json on startup."""
        try:
            raw = read_state().get("restart_counts", {})
            return {tuple(k.split(":", 1)) if ":" in k else (k,): v
                    for k, v in raw.items()}
        except Exception:
            return {}

    def _persist_restart_counts(self) -> None:
        """Write _restart_counts to pipeline_state.json so restarts survive."""
        try:
            serialised = {f"{k[0]}:{k[1]}" if len(k) == 2 else k[0]: v
                          for k, v in self._restart_counts.items()}
            update_state(restart_counts=serialised)
        except Exception:
            pass

    def _next_issue_id(self) -> str:
        self.issue_counter += 1
        return f"I-{self.issue_counter:03d}"

    def _ensure_tmux_session(self) -> None:
        if not tmux_session_exists():
            if not self.dry_run:
                subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION], check=True)
            log_orch(f"Created tmux session: {TMUX_SESSION}")

    def _init_state(self) -> None:
        try:
            with open(STATE_FILE) as f:
                if json.load(f):
                    return  # already initialised
        except Exception:
            pass
        write_state({
            "schema_version": 2,
            "run_id": f"run_{now_iso()[:10].replace('-', '')}",
            "recipe": self.cfg.get("recipe", "ip_adapter_flux4b"),
            "scale": self.scale,
            "chunks": {},
            "issues": [],
        })

    # -----------------------------------------------------------------------
    # Doctor check
    # -----------------------------------------------------------------------

    def _run_doctor(self) -> None:
        import importlib.util, shutil as sh
        checks = [
            ("tmux available",     lambda: sh.which("tmux") is not None),
            ("DATA_ROOT exists",   lambda: DATA_ROOT.exists()),
            ("DATA_ROOT writable", lambda: os.access(DATA_ROOT, os.W_OK)),
            ("venv python exists", lambda: VENV_PYTHON.exists()),
            ("disk >= 40 GB",      lambda: free_gb() >= DISK_ABORT_GB),
            ("numpy importable",   lambda: importlib.util.find_spec("numpy") is not None),
            ("yaml importable",    lambda: importlib.util.find_spec("yaml") is not None),
        ]
        fatal = []
        for name, check in checks:
            try:
                ok = check()
            except Exception:
                ok = False
            print(f"  {'✅' if ok else '❌'} {name}")
            if not ok and any(k in name for k in ("disk", "DATA_ROOT", "python")):
                fatal.append(name)
        if fatal:
            log_orch(f"Doctor: fatal failures: {fatal}", level="error")
            sys.exit(1)
        else:
            log_orch("Doctor: all checks passed")


# ---------------------------------------------------------------------------
# Flywheel loop
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(TRAIN_DIR.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


class _RestartIteration(Exception):
    """Raised by _check_flywheel_control on `pause --free-gpu` after the GPU work has been
    killed and the campaign resumed — unwinds to the iteration loop, which re-runs the
    interrupted iteration (precompute resumes from cache; training re-runs cleanly)."""


# Holds the currently-running flywheel GPU subprocess (precompute Popen) so a pause can
# kill it to free the device. Training runs in TMUX_TRAIN_WIN (killed by window name).
_FLYWHEEL_GPU_PROC: list = [None]


def _kill_flywheel_gpu(name: str) -> None:
    """Terminate whatever flywheel GPU work is running — the registered precompute
    subprocess and/or the training window — so the GPU frees within seconds."""
    proc = _FLYWHEEL_GPU_PROC[0]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        log_orch(f"[flywheel:{name}] pause --free-gpu: killed precompute subprocess (GPU freed)")
    if tmux_window_exists(TMUX_TRAIN_WIN):
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_TRAIN_WIN}"],
                       check=False)
        log_orch(f"[flywheel:{name}] pause --free-gpu: killed training window (GPU freed)")


def _interruptible_proc_wait(proc, name: str, fw_db: Optional[object] = None,
                             poll: int = 10) -> None:
    """Wait for a precompute subprocess while honoring pause/stop signals (so a
    `pause --free-gpu` can interrupt it). Registers the proc for killing for the
    duration. Propagates _RestartIteration on free-gpu pause."""
    _FLYWHEEL_GPU_PROC[0] = proc
    try:
        while proc.poll() is None:
            _check_flywheel_control(name, fw_db)   # may raise _RestartIteration
            time.sleep(poll)
    finally:
        _FLYWHEEL_GPU_PROC[0] = None


def _check_flywheel_control(name: str, fw_db: Optional[object] = None,
                            allow_restart: bool = True) -> None:
    """Handle pause/stop signals from FLYWHEEL_CONTROL_FILE.

    allow_restart=False at call sites BEFORE the per-iteration GPU-work try (top of loop,
    window/lock waits): there a `pause --free-gpu` just waits in place and then proceeds —
    no iteration to re-run, so it must not raise _RestartIteration (which would escape
    uncaught)."""
    if not FLYWHEEL_CONTROL_FILE.exists():
        return
    try:
        ctrl = json.loads(FLYWHEEL_CONTROL_FILE.read_text())
        action = ctrl.get("action")
        if action == "stop":
            log_orch(f"[flywheel:{name}] stop signal — exiting")
            FLYWHEEL_CONTROL_FILE.unlink(missing_ok=True)
            release_gpu_lock()
            sys.exit(0)
        elif action == "pause":
            free_gpu = bool(ctrl.get("free_gpu"))
            if free_gpu:
                # Kill the running GPU work now so the device frees within seconds; on
                # resume we raise _RestartIteration so the iteration re-runs from cache.
                _kill_flywheel_gpu(name)
                log_orch(f"[flywheel:{name}] paused (GPU freed) — waiting for resume")
            else:
                log_orch(f"[flywheel:{name}] paused — waiting for resume")
            if fw_db is not None:
                fw_db.set_campaign_status(name, "paused")
            while True:
                time.sleep(30)
                resumed = not FLYWHEEL_CONTROL_FILE.exists()
                if not resumed:
                    ctrl2 = json.loads(FLYWHEEL_CONTROL_FILE.read_text())
                    if ctrl2.get("action") == "stop":
                        log_orch(f"[flywheel:{name}] stop signal while paused — exiting")
                        FLYWHEEL_CONTROL_FILE.unlink(missing_ok=True)
                        release_gpu_lock()
                        sys.exit(0)
                    resumed = ctrl2.get("action") in ("run", "resume")
                    if resumed:
                        FLYWHEEL_CONTROL_FILE.unlink(missing_ok=True)
                if resumed:
                    log_orch(f"[flywheel:{name}] resumed")
                    if fw_db is not None:
                        fw_db.set_campaign_status(name, "active")
                    if free_gpu and allow_restart:
                        raise _RestartIteration()
                    break
        else:
            FLYWHEEL_CONTROL_FILE.unlink(missing_ok=True)
    except _RestartIteration:
        raise   # not an error — must propagate to the iteration loop
    except Exception as e:
        log_orch(f"[flywheel:{name}] control file error: {e}", level="warning")


def _build_flywheel_train_config(
    fw_cfg: dict,
    staging_dir: Path,
    steps: int,
    hyperparams: dict,
    iteration: int,
    name: str,
) -> Path:
    """Write a temp training config YAML for this flywheel iteration."""
    import yaml
    base_config = fw_cfg.get("training_config",
                             str(TRAIN_DIR / "configs" / "stage1_512px.yaml"))
    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    # Override shard path with the staged symlink dir (absolute — not prefixed by --data-root)
    cfg.setdefault("data", {})["shard_path"] = str(staging_dir)

    # Scale warmup_steps to at most 10% of the iteration steps so we never hit
    # decay_steps=0 when steps_per_iteration is small (e.g. 1000 with warmup=1000).
    cfg.setdefault("training", {})
    base_warmup = cfg["training"].get("warmup_steps", 1000)
    cfg["training"]["warmup_steps"] = max(1, min(base_warmup, steps // 10))

    # Apply hyperparams into training section
    for k, v in hyperparams.items():
        cfg["training"][k] = v

    # Always save checkpoints in flywheel mode
    cfg["training"]["skip_checkpoint_save"] = False

    tmp_path = Path(f"/tmp/flywheel_{name}_iter{iteration:04d}_train.yaml")
    tmp_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return tmp_path


def _ablation_warmstart_ckpt(fw_cfg: dict, ckpt_dir) -> Optional[str]:
    """Opt-in warm-start checkpoint for ablation arms.

    When `ablation_warmstart_arms` is truthy in the flywheel config, return the
    campaign's latest `step_*.safetensors` so each arm warm-starts its weights
    (with a FRESH schedule — see train_ip_adapter --warmstart-weights) instead of
    cold-starting. Returns None when the flag is off (default) or no checkpoint
    exists yet.

    Default-off on purpose: warm-starting arms matches the flywheel's warm-started
    per-iter regime (higher fidelity) but reduces how much the arms can diverge in
    1000 steps (lower discrimination). Which is better is empirical; keep the
    today's cold-start default until the golden-set cross-run comparison can settle it.
    """
    if not fw_cfg.get("ablation_warmstart_arms"):
        return None
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("step_*.safetensors"))
    return str(ckpts[-1]) if ckpts else None


def _quality_gate_target(fw_cfg: dict, best_checkpoint: Optional[str]) -> Optional[str]:
    """Opt-in champion checkpoint for the cross-run quality gate.

    When `quality_gate` is truthy in the flywheel config, return the campaign's
    champion checkpoint so quality_gate.py can golden-set-eval it and compare to
    the previous campaign's champion. Returns None when the flag is off (default)
    or the checkpoint is missing.

    Default-off: the gate runs a GPU golden-set eval, so it only fires at campaign
    end when explicitly enabled. See quality_gate.py / BACKLOG ABL-FIDELITY.
    """
    if not fw_cfg.get("quality_gate"):
        return None
    if best_checkpoint and os.path.isfile(best_checkpoint):
        return best_checkpoint
    return None


def _run_flywheel_ablation(
    fw_cfg: dict,
    name: str,
    iteration: int,
    warmstart_ckpt: Optional[str] = None,
) -> Optional[str]:
    """
    Run a capped ablation burst to tune hyperparams.
    Blocks until the ablation subprocess exits.
    Returns the ablation run_name on success, None on failure.

    warmstart_ckpt: if set (opt-in), arms warm-start their weights from this
    checkpoint with a fresh schedule instead of cold-starting.
    """
    import yaml
    abl_cfg_path = fw_cfg.get("ablation_config")
    if not abl_cfg_path:
        log_orch(f"[flywheel:{name}] no ablation_config in flywheel config — skipping")
        return None
    abl_cfg_path = Path(abl_cfg_path)
    if not abl_cfg_path.exists():
        log_orch(f"[flywheel:{name}] ablation_config not found: {abl_cfg_path}", level="warning")
        return None

    with open(abl_cfg_path) as f:
        abl_cfg = yaml.safe_load(f)

    # _load_harness_config strips the top-level "ablation:" key, so we must
    # write the override into the inner dict to ensure it is honoured.
    inner = abl_cfg.get("ablation", abl_cfg)
    max_runs = int(fw_cfg.get("ablation_max_runs", 10))
    inner["max_total_runs"] = max_runs
    run_name = inner.get("name", abl_cfg_path.stem)

    tmp_cfg = Path(f"/tmp/flywheel_{name}_ablation_{iteration:04d}.yaml")
    tmp_cfg.write_text(yaml.dump(abl_cfg, default_flow_style=False))

    # data_root: optional fw_cfg key for Ultrahot-tier ablation (PIPELINE-31).
    _fw_data_root = Path(fw_cfg.get("data_root", str(DATA_ROOT)))
    output_dir = _fw_data_root / "ablation_long" / f"{name}_iter{iteration:04d}"
    log_file   = LOG_DIR / f"flywheel_{name}_ablation_iter{iteration:04d}.log"
    shards_dir = fw_cfg.get("shards_dir", str(_fw_data_root / "shards"))

    log_orch(f"[flywheel:{name}] iter {iteration}: ablation burst ({max_runs} runs) → {log_file}")

    cmd = (
        f"export PIPELINE_DATA_ROOT='{_fw_data_root}' && "
        f"source '{TRAIN_DIR}/.venv/bin/activate' && "
        f"python -u '{SCRIPTS_DIR}/ablation_harness.py' "
        f"--config '{tmp_cfg}' "
        f"--output-dir '{output_dir}' "
        f"--db '{ABLATION_DB_PATH}' "
        f"--shards '{shards_dir}' "
        f"--qwen3-cache '{_fw_data_root / 'precomputed' / 'qwen3'}' "
        f"--vae-cache '{_fw_data_root / 'precomputed' / 'vae'}' "
        f"--siglip-cache '{_fw_data_root / 'precomputed' / 'siglip'}'"
    )
    if warmstart_ckpt:
        cmd += f" --warmstart-weights '{warmstart_ckpt}'"
        log_orch(f"[flywheel:{name}] iter {iteration}: ablation arms warm-start from "
                 f"{os.path.basename(warmstart_ckpt)} (fresh schedule)")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_cmd = f"({cmd}) >> '{log_file}' 2>&1; echo EXIT_CODE=$? >> '{log_file}'"
    try:
        subprocess.run(["bash", "-c", full_cmd])
    finally:
        tmp_cfg.unlink(missing_ok=True)

    exit_code = last_exit_code(log_file)
    if exit_code == 0:
        log_orch(f"[flywheel:{name}] ablation complete — run_name={run_name}")
        return run_name
    log_orch(f"[flywheel:{name}] ablation failed (exit {exit_code})", level="warning")
    return None


def _read_ablation_best(run_name: str) -> Optional[dict]:
    """Return the best hyperparams for a given ablation run_name, or None."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from ablation_harness import AblationDB
    db = AblationDB(ABLATION_DB_PATH)
    best_list = db.get_best(run_name, 1)
    db.close()
    if not best_list:
        return None
    return best_list[0].get("params") or None


def _resolve_proxy_vae_args(pipeline_cfg: dict, campaign: Optional[str] = None) -> str:
    """Resolve proxy VAE CLI args for precompute_all.py (flywheel + standalone use).

    Mirrors Orchestrator._proxy_vae_args but operates on a plain config dict so
    the module-level flywheel loop can use it.  Returns "" when the proxy is
    disabled, unconfigured, or the checkpoint file is missing.
    """
    pcfg = (pipeline_cfg or {}).get("proxy_vae", {}) or {}
    proxy_path = pcfg.get("proxy_path")
    if not proxy_path or not pcfg.get("enabled", False):
        return ""
    if not Path(proxy_path).exists():
        log_orch(f"[flywheel] proxy_vae.proxy_path missing ({proxy_path}); using real VAE",
                 level="warning")
        return ""
    mode = pcfg.get("default_mode", "balanced")
    thr  = pcfg.get("fallback_threshold", 0.75)
    camp = (pcfg.get("campaigns") or {}).get(campaign, {}) if campaign else {}
    mode = camp.get("mode", mode)
    thr  = camp.get("fallback_threshold", thr)
    return (f"--proxy-vae '{proxy_path}' "
            f"--proxy-vae-threshold {thr} "
            f"--proxy-mode {mode} ")


def _flywheel_report_only(fw_cfg: dict) -> None:
    """Regenerate the HTML report from existing DB data and exit."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from flywheel_lib import FlywheelDB, render_flywheel_index
    from shard_selector import ShardScoreDB

    name = fw_cfg["name"]
    fw_db    = FlywheelDB()
    score_db = ShardScoreDB()

    iterations_data = fw_db.get_iterations(name)
    shard_stats     = score_db.get_stats()
    top_shards      = score_db.get_top_shards(20)
    hyperparams     = dict(fw_cfg.get("hyperparams", {}))

    reports_dir = FLYWHEEL_REPORTS_DIR / name
    reports_dir.mkdir(parents=True, exist_ok=True)

    html = render_flywheel_index(name, iterations_data, shard_stats, top_shards, hyperparams)
    report_path = reports_dir / "index.html"
    report_path.write_text(html)
    print(f"Report written → {report_path}")
    fw_db.close()
    score_db.close()


def _run_flywheel_loop(fw_cfg: dict, fw_cfg_path: Optional[str] = None) -> None:
    """
    Self-improving sref optimization flywheel.

    Each iteration: select shards → stage as symlinks → train N steps →
    collect metrics → update shard scores → (every N iters) run ablation →
    regenerate HTML report → repeat.

    Reuses pipeline infrastructure: GPU lock, heartbeat, disk checks,
    log_orch, dispatch_issue, tmux.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    from flywheel_lib import (
        FlywheelDB, collect_metrics_from_log, render_flywheel_index,
        _checkpoint_hash, check_plateau, write_campaign_summary_json,
    )

    from shard_selector import (
        ShardScoreDB, scan_shard_pool, select_shards,
        stage_shards_for_iteration, render_shard_report,
    )

    name           = fw_cfg["name"]
    max_iters      = int(fw_cfg.get("max_iterations", 20))
    steps_per_iter = int(fw_cfg.get("steps_per_iteration", 5000))
    # resume_from_champion: warm-start every iteration from the campaign CHAMPION
    # (best cond_gap) with a FRESH schedule (--warmstart-weights), instead of
    # continuing the LATEST checkpoint with --resume. The default (latest + --resume)
    # accumulates steps across iterations, so a regressed iteration compounds and the
    # adapter over-trains (cond_gap falls while train_loss falls). Champion warm-start
    # makes each iteration an independent re-roll on fresh data that cannot drag the
    # working checkpoint below the champion. See plans/warmup-campaign-runbook.md §3.
    resume_from_champion = bool(fw_cfg.get("resume_from_champion", False))
    # from_scratch_each_iter: train every iteration from random init (no --resume,
    # no --warmstart-weights). Turns the flywheel into a DATA-SELECTION loop: each
    # iteration's cond_gap measures the quality of its selected shard mix (like a
    # fresh 1000-step run), and the bandit/attribution optimize WHICH data — without
    # the over-training that warm-start-and-keep-training causes on a saturating
    # adapter (warmup-run2/3: warm-start drove cond_gap +0.0273 -> ~-0.005 regardless
    # of mechanism; only from-scratch produced a positive gap). Takes precedence over
    # resume_from_champion. Output is shard_scores.db, not a shippable checkpoint.
    from_scratch_each_iter = bool(fw_cfg.get("from_scratch_each_iter", False))
    n_shards       = int(fw_cfg.get("n_shards", 20))
    poll_interval  = int(fw_cfg.get("poll_interval", 60))
    ablation_every = int(fw_cfg.get("ablation_every_n", 0))
    # data_root: optional fw_cfg key for Ultrahot-tier flywheel (PIPELINE-31).
    _fw_data_root = Path(fw_cfg.get("data_root", str(DATA_ROOT)))
    _fw_ckpt_dir  = _fw_data_root / "checkpoints" / "stage1"
    shards_dir     = Path(fw_cfg.get("shards_dir", str(_fw_data_root / "shards")))
    manifest_path  = Path(fw_cfg["shard_manifest"]) if fw_cfg.get("shard_manifest") else None

    # DataStager handles cold→hot staging of shards (and eventually precomputed
    # caches) for each iteration.  pipeline_config defaults to v2_pipeline.yaml
    # auto-detection so the storage.cold_root / hot_root settings are respected.
    pipeline_cfg_path = fw_cfg.get("pipeline_config")
    _pipeline_cfg = load_config(pipeline_cfg_path)
    _stager = DataStager(_pipeline_cfg)

    fw_db    = FlywheelDB()
    score_db = ShardScoreDB()

    try:
        reports_dir = FLYWHEEL_REPORTS_DIR / name
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Record config path once so the doctor can find ablation_config without
        # parsing logs.
        if fw_cfg_path:
            fw_db.set_config_path(name, fw_cfg_path)

        # Resume: start after the last recorded iteration
        prior     = fw_db.get_iterations(name)
        iteration = (max(r["iteration"] for r in prior) + 1) if prior else 1
        hyperparams = dict(fw_cfg.get("hyperparams", {}))

        # On restart, merge previously persisted ablation-tuned hyperparams so we
        # don't regress back to config defaults after a pause/resume cycle.
        saved_hp = fw_db.get_best_hyperparams(name)
        if saved_hp:
            hyperparams.update(saved_hp)
            log_orch(f"[flywheel:{name}] restored ablation hyperparams on restart: {saved_hp}")

        # Determine starting checkpoint.
        # Prefer the chronologically latest step_*.safetensors from CKPT_DIR so that
        # restarts always continue the cumulative training chain.  get_best() returns
        # best-by-ref_gap which may be an older checkpoint (FLYWHEEL-BUG-1).
        #
        # base_checkpoint semantics:
        #   key absent or non-null value  → auto-detect latest ckpt, then use the value
        #   key explicitly set to null    → fresh start; do NOT auto-detect from disk
        #     (use this when starting a new flywheel campaign from scratch so that
        #      leftover checkpoints from prior campaigns are not accidentally loaded)
        _base_ckpt_explicit_null = ("base_checkpoint" in fw_cfg
                                    and fw_cfg["base_checkpoint"] is None)
        resume_ckpt: Optional[str] = None

        def _step_num(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0

        if from_scratch_each_iter:
            # Data-selection mode: every iteration starts from random init.
            resume_ckpt = None
        elif resume_from_champion:
            # Always warm-start from the campaign champion (never the latest step
            # checkpoint), so a regressed iteration can't compound. iter-1 (no best
            # yet) falls back to base_checkpoint.
            best = fw_db.get_best(name)
            if best and best.get("checkpoint") and Path(best["checkpoint"]).exists():
                resume_ckpt = best["checkpoint"]
            else:
                resume_ckpt = fw_cfg.get("base_checkpoint")
        elif not _base_ckpt_explicit_null:
            # Auto-detect: use latest step checkpoint, falling back to DB best.
            ckpts = sorted(_fw_ckpt_dir.glob("step_*.safetensors"), key=_step_num)
            if ckpts:
                resume_ckpt = str(ckpts[-1])
            else:
                done_iters = [r for r in prior if r["status"] == "done"]
                if done_iters:
                    best = fw_db.get_best(name)
                    if best and best.get("checkpoint"):
                        resume_ckpt = best["checkpoint"]
            if resume_ckpt is None:
                resume_ckpt = fw_cfg.get("base_checkpoint")

        log_orch(f"[flywheel:{name}] starting — iter={iteration}/{max_iters}  "
                 f"n_shards={n_shards}  steps_per_iter={steps_per_iter}  "
                 f"resume={Path(resume_ckpt).name if resume_ckpt else 'none'}")

        # Record the starting checkpoint once so later campaigns know what to warm-start from.
        fw_db.set_base_checkpoint(name, resume_ckpt)

        plateau_patience         = int(fw_cfg.get("plateau_patience",       0))
        plateau_threshold        = float(fw_cfg.get("plateau_threshold",    0.02))
        # Max ablation runs to fire when plateau is detected. 0 = disabled.
        # Allows one targeted hyperparameter search before the campaign pauses,
        # so the operator resumes with fresh tuning rather than stale config defaults.
        plateau_ablation_runs    = int(fw_cfg.get("plateau_ablation_runs",  0))

        while iteration <= max_iters:
            _restart = False
            _check_flywheel_control(name, fw_db, allow_restart=False)

            # Disk guard
            gb     = free_gb()
            min_gb = float(fw_cfg.get("min_free_gb", DISK_ABORT_GB))
            if gb < min_gb:
                log_orch(f"[flywheel:{name}] disk low ({gb:.1f} GB < {min_gb} GB) — waiting 300s",
                         level="warning")
                write_heartbeat("flywheel", status="disk_wait",
                                flywheel_name=name, iteration=iteration)
                time.sleep(300)
                continue

            log_orch(f"[flywheel:{name}] === iteration {iteration}/{max_iters} ===")
            write_heartbeat("flywheel", status="selecting",
                            flywheel_name=name, iteration=iteration)

            # Register any new shards (with SigLIP embeddings and manifest if configured)
            scan_shard_pool(score_db, shards_dir, manifest_path=manifest_path, verbose=False)

            # Select shards
            shard_cfg = dict(fw_cfg.get("shard_selection", {}))
            selected_paths = select_shards(score_db, n_shards, shard_cfg, name, iteration)

            if len(selected_paths) < 2:
                msg = f"only {len(selected_paths)} shards available in {shards_dir}"
                log_orch(f"[flywheel:{name}] {msg} — waiting 300s", level="warning")
                dispatch_issue(f"fw_{name}_no_shards", "warning",
                               f"Flywheel {name}: {msg}",
                               suggested_action="populate_shards_dir")
                time.sleep(300)
                continue

            shard_ids = [Path(p).stem for p in selected_paths]

            # Stage selected shards via DataStager (copy→hot when cross-device,
            # symlinks when same device — same logic as the V2 pipeline's chunk staging).
            # Cross-device: run in a background thread so precompute can start as soon as
            # the first few shards land (masking model-load time and early encode work).
            # Symlink mode is instant; no pipelining needed — stage inline.
            staging_dir = DATA_ROOT / "flywheel_staging" / name / f"iter{iteration:04d}"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

            _stage_result: list = []

            def _do_stage():
                try:
                    s = _stager.stage_iteration_shards(selected_paths, staging_dir)
                    _stage_result.append({"ok": True, "summary": s})
                except Exception as _se:
                    _stage_result.append({"ok": False, "error": _se})

            if _stager.enabled:
                import threading as _threading
                _stage_thread = _threading.Thread(target=_do_stage, daemon=True)
                _stage_thread.start()
            else:
                _do_stage()
                _stage_thread = None

            # Per-iteration shard report
            shard_html = render_shard_report(score_db, shard_ids, iteration, name)
            (reports_dir / f"shard_selection_iter{iteration:04d}.html").write_text(shard_html)

            # Checkpoint hash for this iteration (derived from the checkpoint we're resuming from)
            ckpt_hash = _checkpoint_hash(resume_ckpt)

            # Insert DB record (status=running)
            row_id = fw_db.insert_iteration(
                name=name, iteration=iteration,
                n_shards=len(shard_ids), shard_ids=shard_ids,
                hyperparams=hyperparams, steps=steps_per_iter,
                git_commit=_git_sha(),
                checkpoint_hash=ckpt_hash,
            )

            # Build temp training config pointing at staged shards
            train_cfg_path = _build_flywheel_train_config(
                fw_cfg, staging_dir, steps_per_iter, hyperparams,
                iteration, name,
            )

            # =====================================================================
            # FLYWHEEL PER-ITER PRECOMPUTE + CACHED TRAINING + VERSIONED PUBLISH
            # (Claude/AI-readable architecture summary)
            #
            # PROBLEM (historical):
            #   Early flywheel warmup used "online" training config (all *_cache_dir: null)
            #   for cold shard pools (no precomp yet). This meant inside every training
            #   step the loader yielded no np arrays, so train_ip_adapter did live
            #   _encode_text (Qwen3), _vae_encode, siglip() for the batch.
            #   Even with the per-step _ensure_live_encoders / _release_live_encoders
            #   (unload after ~100-400ms encode window, before Flux fwd), this was
            #   inefficient: repeated encode work (if data replayed in 1000-5000 step
            #   iters), extra latency serialized with training steps, high memory
            #   pressure (hence 0.55 mlx_memory_pct + torch clear + our debug prints
            #   with [online-encode] prefix).
            #
            #   The "actions" (encode the selected shards) were the same as normal
            #   pipeline, but done in the wrong place/order.
            #
            # SOLUTION (this code):
            #   After select_shards + stage_iteration_shards (shards now in staging_dir
            #   as .tar, possibly some npz from cold via stager):
            #   1. ALWAYS run precompute_all.py --shards <staging_dir> (with --siglip)
            #      targeting per-iter flat tree: staging_dir/precomputed/{qwen3,vae,siglip}/
            #      (models loaded once in precomp process with its 14GB cap + releases;
            #       one tar read per shard + prefetch; skips already-present via scan;
            #       under the GPU lock acquired for the whole iter).
            #   2. Override the temp train config (even if base was online.yaml) to point
            #      the three cache_dirs at the per-iter tree. Training now runs the
            #      EFFICIENT CACHED PATH (only Flux+adapter, fast npz loads in loader,
            #      no live encoders, lower memory).
            #   3. AFTER training (inside the lock try, before unlink), call
            #      _stager.publish_precomp_from_flywheel_iter(..., training_cfg=...)
            #   4. In stager: if training_cfg, use encoder_config_subset + version_hash
            #      + PrecomputeCache to target the EXACT v_XXXXXX/ dir on COLD for this
            #      config. Copy .npz there, write_manifest_incomplete + mark_complete
            #      (which does atomic current symlink update). This makes data first-class:
            #      - participates in versioned invalidation story
            #      - PrecomputeCache.is_complete / all_records will see manifests
            #      - future stagings (which follow "current") will bring it
            #      - effective_dir etc. work
            #      Then ALWAYS rmtree the per-iter precomp_base (data now durable on cold).
            #   Fallback (no cfg) does flat copy to existing current target + rmtree.
            #
            # INTEGRATION WITH PIPELINE:
            #   - Matches exactly what _promote_chunk + precompute_all do for chunks
            #     (version dirs, manifests, current update, move out of staging).
            #   - DataStager.stage_iteration_shards will pick up published npz from cold
            #     "current" on future iters (see its cold_ver logic).
            #   - PrecomputeCache / list_versions / cache_manager tools see proper
            #     versions + manifests.
            #   - Main pipeline (chunks, mine, eval, etc.) benefits automatically.
            #   - Incremental: re-selected shards are cheap (precomp skips).
            #   - The original per-step unload logic + [online-encode] debug prints in
            #     train_ip_adapter.py remain for manual online runs or as fallback when
            #     precomp produces partial coverage.
            #
            # WHY THIS (efficiency + reusability):
            #   "Actions same, efficiency vastly different" -- user observation was correct.
            #   Precomp once (good batching, dedicated process) + cached train (fast,
            #   low-mem) + publish to versioned cold (reusable, no waste) vs. live
            #   per-step inside training.
            #
            # See also: plans/warmup-campaign-runbook.md (flywheel goals), data_stager.py
            # (stager + publish impl), cache_manager.py (PrecomputeCache, version_hash,
            # encoder_config_subset, mark_complete), train_ip_adapter.py (old online path
            # + unload for compatibility).
            # =====================================================================

            # === Precompute selected shards before training (the efficient path) ===
            # (see big comment block above for full Claude/AI explanation)
            # Instead of "online" live Q/V/S encoding inside the training loop (with per-step
            # load/unload), we run precompute_all.py on exactly the shards chosen for this iter.
            # Benefits:
            #  - encoders loaded once (precomp process uses its own 14 GB cap + releases between shards)
            #  - shard tars read once
            #  - encoding work done once per record → .npz
            #  - training then runs purely cached (only Flux + adapter in memory), fast steps,
            #    no repeated encode cost even if the iter's data is consumed multiple times
            #  - also incrementally builds persistent caches for the pool (future iters/stager benefit)
            # The precompute_all logic already skips records that have outputs, so this is
            # incremental and cheap when re-selecting shards or when partial caches were staged.
            precomp_base = staging_dir / "precomputed"
            q_out = precomp_base / "qwen3"
            v_out = precomp_base / "vae"
            s_out = precomp_base / "siglip"
            for dd in (q_out, v_out, s_out):
                dd.mkdir(parents=True, exist_ok=True)

            log_pre = LOG_DIR / f"flywheel_{name}_precompute_iter{iteration:04d}.log"
            log_pre.parent.mkdir(parents=True, exist_ok=True)

            # Inspect draft config for flux model and siglip intent
            import yaml
            with open(train_cfg_path) as _f:
                _draft = yaml.safe_load(_f)
            _mcfg = _draft.get("model", {}) or {}
            _flux_dir = _mcfg.get("flux_model_dir", "flux-klein-model")
            _flux_p = Path(_flux_dir)
            if not _flux_p.is_absolute():
                _flux_p = _fw_data_root.parent / _flux_dir
            _flux_arg = f"--flux-model '{_flux_p}'" if _flux_p.exists() else ""
            _sig_flag = "--siglip"

            _proxy_arg = _resolve_proxy_vae_args(_pipeline_cfg, campaign=name)
            pre_cmd = (
                f"caffeinate -dim python -u '{SCRIPTS_DIR}/precompute_all.py' "
                f"--shards '{staging_dir}' "
                f"--qwen3-output '{q_out}' --vae-output '{v_out}' --siglip-output '{s_out}' "
                f"{_flux_arg} {_proxy_arg}{_sig_flag}"
            )
            # PIPELINE_ORCHESTRATED=1: precompute_all skips GPU lock acquisition because
            # the orchestrator already holds the lock for the whole iteration.
            full_pre = (
                f"export PIPELINE_DATA_ROOT='{_fw_data_root}' && "
                f"export PIPELINE_ORCHESTRATED=1 && "
                f"source '{TRAIN_DIR}/.venv/bin/activate' && {pre_cmd}"
            )

            # Launch training in TMUX_TRAIN_WIN
            log_file = LOG_DIR / f"flywheel_{name}_iter{iteration:04d}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Wait for any pre-existing training window to finish (crash-restart guard)
            if tmux_window_exists(TMUX_TRAIN_WIN):
                log_orch(f"[flywheel:{name}] {TMUX_TRAIN_WIN} already running — waiting",
                         level="warning")
                while tmux_window_exists(TMUX_TRAIN_WIN):
                    _check_flywheel_control(name, fw_db, allow_restart=False)
                    time.sleep(poll_interval)

            # Acquire the GPU lock *once* for the whole iteration's work (precompute + training).
            # Staging may still be running in a background thread; acquiring the lock here lets
            # precompute pass 1 start as soon as the trigger threshold is reached.
            while True:
                if acquire_gpu_lock(f"flywheel_{name}_iter{iteration}"):
                    break
                holder = gpu_lock_holder()
                log_orch(f"[flywheel:{name}] GPU busy (holder={holder}), waiting 30s",
                         level="warning")
                _check_flywheel_control(name, fw_db, allow_restart=False)
                time.sleep(30)

            try:
                # ── Pipelined staging + precompute ─────────────────────────────────
                # If cross-device staging is still running in the background, start
                # precompute pass 1 as a non-blocking Popen once enough shards have
                # landed.  This masks model-load time (Qwen3/VAE/SigLIP load = several
                # minutes) and processes early shards while the remaining staging IO is
                # in flight.  Pass 2 catches shards that arrived after pass 1's scan.
                _pass1_proc = None
                if _stage_thread is not None and _stage_thread.is_alive():
                    _trigger = int(fw_cfg.get("precomp_trigger_shards", 4))
                    write_heartbeat("flywheel", status="staging",
                                    flywheel_name=name, iteration=iteration)
                    while _stage_thread.is_alive():
                        _n_ready = sum(1 for _p in staging_dir.glob("*.tar"))
                        if _n_ready >= _trigger:
                            break
                        _check_flywheel_control(name, fw_db)
                        time.sleep(5)
                    if _stage_thread.is_alive():
                        _n_ready = sum(1 for _p in staging_dir.glob("*.tar"))
                        _n_inflight = len(selected_paths) - _n_ready
                        log_orch(f"[flywheel:{name}] iter {iteration}: {_n_ready} shards staged — "
                                 f"starting precompute pass 1 while staging continues "
                                 f"({_n_inflight} shards in flight)")
                        write_heartbeat("flywheel", status="precomputing",
                                        flywheel_name=name, iteration=iteration)
                        _pass1_proc = subprocess.Popen(
                            ["bash", "-c", f"({full_pre}) >> '{log_pre}' 2>&1"])

                # Wait for staging to complete, then log the summary.
                if _stage_thread is not None:
                    _stage_thread.join()
                if not _stage_result or not _stage_result[0].get("ok"):
                    _se = (_stage_result[0]["error"] if _stage_result
                           else RuntimeError("staging thread produced no result"))
                    log_orch(f"[flywheel:{name}] staging failed: {_se} — falling back to symlinks",
                             level="warning")
                    if _pass1_proc is not None:
                        _pass1_proc.terminate()
                        _pass1_proc.wait()
                        _pass1_proc = None
                    stage_shards_for_iteration(selected_paths, staging_dir)
                    stage_summary = {"shards_staged": len(selected_paths),
                                     "npz_staged": 0, "bytes_transferred": 0}
                else:
                    stage_summary = _stage_result[0]["summary"]
                log_orch(f"[flywheel:{name}] staged {stage_summary['shards_staged']} shards "
                         f"+ {stage_summary.get('npz_staged', 0)} npz "
                         f"({'copy' if _stager.enabled else 'symlink'}, "
                         f"{stage_summary['bytes_transferred'] / 1e6:.0f} MB) → {staging_dir}")

                # Wait for precompute pass 1 (if started concurrently with staging).
                write_heartbeat("flywheel", status="precomputing",
                                flywheel_name=name, iteration=iteration)
                if _pass1_proc is not None:
                    log_orch(f"[flywheel:{name}] iter {iteration}: waiting for precompute pass 1")
                    _interruptible_proc_wait(_pass1_proc, name, fw_db)
                    if _pass1_proc.returncode == 0:
                        log_orch(f"[flywheel:{name}] iter {iteration}: precompute pass 1 complete")
                    else:
                        log_orch(f"[flywheel:{name}] iter {iteration}: precompute pass 1 exited "
                                 f"{_pass1_proc.returncode}; pass 2 will fill any gaps",
                                 level="warning")

                # Precompute pass 2: catches shards that arrived after pass 1's initial scan,
                # or handles all shards when staging finished before the trigger threshold.
                _pass_label = "pass 2 (catch-up)" if _pass1_proc is not None else "pass 1"
                log_orch(f"[flywheel:{name}] iter {iteration}: precomputing Qwen3/VAE/SigLIP "
                         f"{_pass_label} → {precomp_base}")
                pre_full = f"({full_pre}) >> '{log_pre}' 2>&1 ; echo EXIT_CODE=$? >> '{log_pre}'"
                _pass2_proc = subprocess.Popen(["bash", "-c", pre_full])
                _interruptible_proc_wait(_pass2_proc, name, fw_db)
                pre_exit = last_exit_code(log_pre)
                if pre_exit == 0:
                    log_orch(f"[flywheel:{name}] iter {iteration}: precompute complete")
                else:
                    log_orch(f"[flywheel:{name}] iter {iteration}: precompute exited {pre_exit}; "
                             f"training will use produced caches + live fallback for any misses",
                             level="warning")

                # Override the (possibly "online") config so this training iter uses the caches we just ensured.
                # This forces the efficient cached path (only Flux+adapter resident) for the step loop.
                with open(train_cfg_path) as _f:
                    _tcfg = yaml.safe_load(_f)
                _d = _tcfg.setdefault("data", {})
                _d["qwen3_cache_dir"] = str(q_out)
                _d["vae_cache_dir"] = str(v_out)
                _d["siglip_cache_dir"] = str(s_out)
                train_cfg_path.write_text(yaml.dump(_tcfg, default_flow_style=False))

                # Capture the final config (with model/data for versioning) before launch.
                # We will use it after training to publish with proper PrecomputeCache version dirs,
                # manifests, and current symlink update.
                _final_train_cfg_for_publish = dict(_tcfg)

                # Training launch (still under the same lock)
                write_heartbeat("trainer", status="booting", step=0)

                train_cmd = (
                    f"caffeinate -dim python -u '{TRAIN_DIR}/train_ip_adapter.py' "
                    f"--config '{train_cfg_path}' "
                    f"--max-steps {steps_per_iter} "
                    f"--data-root '{_fw_data_root}'"
                )
                if resume_ckpt and Path(resume_ckpt).exists():
                    # Champion warm-start uses a fresh schedule (start_step=0, fresh
                    # warmup/optimizer) so steps don't accumulate across iterations.
                    flag = "--warmstart-weights" if resume_from_champion else "--resume"
                    train_cmd += f" {flag} '{resume_ckpt}'"

                activated = (
                    f"export PIPELINE_DATA_ROOT='{_fw_data_root}' && "
                    f"source '{TRAIN_DIR}/.venv/bin/activate' && {train_cmd}"
                )
                tmux_new_window(TMUX_TRAIN_WIN, activated, log_file)
                write_heartbeat("flywheel", status="training",
                                flywheel_name=name, iteration=iteration)
                log_orch(f"[flywheel:{name}] iter {iteration}: training started (using precomputed caches) → {log_file.name}")
                t_start = time.time()

                # Monitor training window
                while True:
                    time.sleep(poll_interval)
                    _check_flywheel_control(name, fw_db)
                    write_heartbeat("flywheel", status="training",
                                    flywheel_name=name, iteration=iteration)
                    if not tmux_window_exists(TMUX_TRAIN_WIN):
                        break

                # Post-training publish: use PrecomputeCache + encoder_config_subset + version_hash
                # to put the data into a proper version dir on cold, write manifest, and update
                # the "current" symlink so this iter's precomp participates in the full versioned
                # story (invalidation, is_complete, effective_dir, future stagings that follow current).
                # Then rmtree the per-iter precomp (data is now durable on cold).
                if "precomp_base" in locals() and precomp_base.exists():
                    try:
                        pub = _stager.publish_precomp_from_flywheel_iter(
                            src_precomp_base=precomp_base,
                            selected_shard_ids=shard_ids,
                            training_cfg=_final_train_cfg_for_publish,
                        )
                        n = pub.get("npz_published", 0)
                        if n > 0:
                            mb = pub.get("bytes_transferred", 0) / 1e6
                            log_orch(
                                f"[flywheel:{name}] iter {iteration}: published {n} new npz "
                                f"({mb:.1f} MB) to cold precomputed (versioned dir + current updated + rmtree'd per-iter)"
                            )
                    except Exception as _pub_err:
                        log_orch(
                            f"[flywheel:{name}] iter {iteration}: publish of precomp npz to cold (versioned) failed: {_pub_err}",
                            level="warning",
                        )
            except _RestartIteration:
                # pause --free-gpu: GPU work already killed + campaign resumed; re-run
                # this iteration (no increment). The finally releases the GPU lock.
                _restart = True
            finally:
                release_gpu_lock()
                train_cfg_path.unlink(missing_ok=True)

            if _restart:
                _restart = False
                # The cross-device staging thread may still be copying into
                # staging_dir; let it finish before deleting the tree, or the old
                # thread writes into a removed dir while the re-entered iteration
                # starts a second stager against the same campaign.
                if _stage_thread is not None and _stage_thread.is_alive():
                    log_orch(f"[flywheel:{name}] restart: waiting for in-flight staging "
                             f"to finish before cleanup")
                    _stage_thread.join()
                # Drop the aborted attempt's iterations row — the re-run inserts a
                # fresh one for the same iteration number; without this the stale
                # 'running' row lingers as a duplicate in get_iterations/get_best.
                fw_db.delete_iteration(row_id)
                try:
                    shutil.rmtree(staging_dir)
                except OSError:
                    pass
                log_orch(f"[flywheel:{name}] re-running iter {iteration} after pause --free-gpu")
                continue

            elapsed   = int(time.time() - t_start)
            exit_code = last_exit_code(log_file)
            status    = "done" if exit_code == 0 else "failed"

            # Collect training metrics
            metrics = collect_metrics_from_log(log_file)
            log_orch(
                f"[flywheel:{name}] iter {iteration}: {status}  "
                f"ref_gap={metrics.get('ref_gap')}  loss={metrics.get('loss_smooth')}  "
                f"elapsed={elapsed}s"
            )

            # Locate the checkpoint produced by this iteration
            ckpt_path: Optional[str] = None
            ckpts = sorted(_fw_ckpt_dir.glob("step_*.safetensors"))
            if ckpts:
                ckpt_path = str(ckpts[-1])
                # FLYWHEEL-CKPT-1: start_step=0 modes (from_scratch / resume_from_champion)
                # save the same step_000{budget}.safetensors every iteration, so the next
                # iteration would clobber it — and get_best()'s recorded path would then
                # point at the wrong (latest, not best) weights, reintroducing compounding.
                # Archive this iteration's file under a unique iter-tagged name and record
                # THAT, so the champion file is always preserved. (Default --resume mode
                # accumulates step numbers, so names are already unique — left untouched.)
                if (from_scratch_each_iter or resume_from_champion) and status == "done":
                    iter_ckpt = _fw_ckpt_dir / f"iter{iteration:04d}_{Path(ckpt_path).name}"
                    try:
                        os.replace(ckpt_path, iter_ckpt)   # atomic, same filesystem
                        ckpt_path = str(iter_ckpt)
                    except OSError as _arch_err:
                        log_orch(f"[flywheel:{name}] iter {iteration}: checkpoint archive "
                                 f"failed ({_arch_err}); recording live path", level="warning")
                    # Also archive the EMA weights (best.safetensors): typically the
                    # better-generalizing weights, but the trainer overwrites the file
                    # every iteration — without this the champion's EMA is lost.
                    _ema_live = _fw_ckpt_dir / "best.safetensors"
                    if _ema_live.exists():
                        _ema_arch = _fw_ckpt_dir / f"iter{iteration:04d}_best.safetensors"
                        try:
                            os.replace(_ema_live, _ema_arch)
                        except OSError as _arch_err:
                            log_orch(f"[flywheel:{name}] iter {iteration}: EMA archive "
                                     f"failed ({_arch_err})", level="warning")
            new_ckpt_hash = _checkpoint_hash(ckpt_path)

            # Snapshot best-so-far BEFORE writing the current iteration's ref_gap.
            # get_best() queries all iterations; snapshotting here avoids comparing
            # new_ref against itself after update_iteration() writes it to the DB.
            prior_best = fw_db.get_best(name)

            # Update iteration record
            fw_db.update_iteration(
                row_id=row_id, status=status,
                exit_code=exit_code if exit_code is not None else -1,
                elapsed_secs=elapsed,
                train_loss=metrics.get("loss_smooth"),
                ref_gap=metrics.get("ref_gap"),
                cond_gap=metrics.get("cond_gap"),
                checkpoint=ckpt_path,
                checkpoint_hash=new_ckpt_hash,
            )

            fw_db.refresh_campaign_summary(name)

            # Record checkpoint to checkpoint_log (successful iterations only)
            if status == "done" and ckpt_path:
                fw_db.upsert_checkpoint(
                    name=name, iteration=iteration,
                    checkpoint_path=ckpt_path,
                    checkpoint_hash=new_ckpt_hash,
                    ref_gap=metrics.get("ref_gap"),
                    cond_gap=metrics.get("cond_gap"),
                    train_loss=metrics.get("loss_smooth"),
                )

            # Update per-shard included EMA scores
            has_metrics = (metrics.get("ref_gap") is not None
                           or metrics.get("cond_gap") is not None)
            if has_metrics:
                # Snapshot which selected shards are currently unscored before the update
                # so we can count first-contact discoveries for this iteration.
                pre_zero = {s["shard_id"] for s in score_db.get_all_shards()
                            if s.get("n_scored", 0) == 0}
                _temporal_decay = fw_cfg.get("temporal_decay") or None
                for sid in shard_ids:
                    score_db.update_scores(
                        shard_id=sid,
                        ref_gap=metrics.get("ref_gap"),
                        cond_gap=metrics.get("cond_gap"),
                        loss=metrics.get("loss_smooth"),
                        flywheel_name=name,
                        iteration=iteration,
                        checkpoint_hash=new_ckpt_hash,
                        checkpoint_iter=iteration,
                        n_in_batch=len(shard_ids),
                        temporal_decay=_temporal_decay,
                    )
                # Update excluded EMA for all other shards (contrastive attribution)
                all_pool_ids = [s["shard_id"] for s in score_db.get_all_shards()]
                score_db.update_excluded_scores(
                    all_shard_ids=all_pool_ids,
                    selected_ids=set(shard_ids),
                    ref_gap=metrics.get("ref_gap"),
                    cond_gap=metrics.get("cond_gap"),
                    flywheel_name=name,
                    iteration=iteration,
                    checkpoint_hash=new_ckpt_hash,
                    checkpoint_iter=iteration,
                    n_in_batch=len(shard_ids),
                    temporal_decay=_temporal_decay,
                )

                # Count shards that just received their first-ever score observation.
                n_first_contact = sum(1 for sid in shard_ids if sid in pre_zero)
                fw_db.set_n_first_contact(row_id, n_first_contact)
                log_orch(f"[flywheel:{name}] iter {iteration}: "
                         f"first_contact={n_first_contact}/{len(shard_ids)}")

            # Promote checkpoint for the next iteration; mark as best if quality improved.
            # cond_gap is the primary criterion: stable and informative at 1000-step budgets.
            if status == "done" and ckpt_path:
                prior_cond = prior_best.get("cond_gap") if prior_best else None
                new_cond   = metrics.get("cond_gap")
                if new_cond is not None and (prior_cond is None or new_cond > prior_cond):
                    fw_db.mark_best_checkpoint(name, iteration)
                    log_orch(f"[flywheel:{name}] new best checkpoint  "
                             f"cond_gap={new_cond:.4f}  hash={new_ckpt_hash}")
                if from_scratch_each_iter:
                    resume_ckpt = None          # every iteration is independent
                elif resume_from_champion:
                    # Next iteration warm-starts from the (possibly just-updated)
                    # champion, not this iteration's output — no compounding.
                    best = fw_db.get_best(name)
                    resume_ckpt = (best.get("checkpoint") if best else None) or resume_ckpt
                else:
                    resume_ckpt = ckpt_path

                # FLYWHEEL-CKPT-1: prune archived iter-tagged checkpoints, keeping only the
                # champion's (needed by get_best / resume_from_champion) and this
                # iteration's — both the raw step weights and the EMA archive
                # (iterNNNN_best.safetensors, ~2 GB each; unpruned they accumulate).
                if from_scratch_each_iter or resume_from_champion:
                    _best = fw_db.get_best(name)
                    _keep = {ckpt_path}
                    _keep_iters = {iteration}
                    if _best and _best.get("checkpoint"):
                        _keep.add(_best["checkpoint"])
                        if _best.get("iteration") is not None:
                            _keep_iters.add(int(_best["iteration"]))
                    for _old in _fw_ckpt_dir.glob("iter*_step_*.safetensors"):
                        if str(_old) not in _keep:
                            try:
                                _old.unlink()
                            except OSError:
                                pass
                    for _old in _fw_ckpt_dir.glob("iter*_best.safetensors"):
                        try:
                            _it = int(_old.name[4:8])
                        except ValueError:
                            continue
                        if _it not in _keep_iters:
                            try:
                                _old.unlink()
                            except OSError:
                                pass

            # Clean up staging symlinks
            try:
                shutil.rmtree(staging_dir)
            except OSError:
                pass

            # Ablation burst (every N iterations, only on successful runs)
            ablation_run: Optional[str] = None
            if ablation_every > 0 and iteration % ablation_every == 0 and status == "done":
                ablation_run = _run_flywheel_ablation(
                    fw_cfg, name, iteration,
                    warmstart_ckpt=_ablation_warmstart_ckpt(fw_cfg, _fw_ckpt_dir))
                if ablation_run:
                    new_hp = _read_ablation_best(ablation_run)
                    if new_hp:
                        hyperparams.update(new_hp)
                        log_orch(f"[flywheel:{name}] hyperparams updated → {hyperparams}")
                        fw_db.set_best_hyperparams(name, hyperparams)
                    fw_db.update_iteration(
                        row_id=row_id, status=status,
                        exit_code=exit_code if exit_code is not None else -1,
                        elapsed_secs=elapsed,
                        train_loss=metrics.get("loss_smooth"),
                        ref_gap=metrics.get("ref_gap"),
                        cond_gap=metrics.get("cond_gap"),
                        checkpoint=ckpt_path,
                        checkpoint_hash=new_ckpt_hash,
                        ablation_run=ablation_run,
                    )
                    fw_db.refresh_campaign_summary(name)

            # Plateau detection
            plateau_reason: Optional[str] = None
            if plateau_patience > 0:
                done_iters = [r for r in fw_db.get_iterations(name) if r["status"] == "done"]
                plateau_reason = check_plateau(done_iters, plateau_patience, plateau_threshold)
                if plateau_reason:
                    log_orch(f"[flywheel:{name}] plateau detected — pausing: {plateau_reason}",
                             level="warning")

                    # Fire an ablation burst before pausing so the operator resumes
                    # with fresh hyperparameter tuning rather than stale config defaults.
                    # Only fires when plateau_ablation_runs > 0 and ablation_config is set,
                    # and only if this iteration hasn't already run a scheduled ablation.
                    if plateau_ablation_runs > 0 and not ablation_run:
                        # Temporarily override max_runs for this targeted burst
                        _saved_max = fw_cfg.get("ablation_max_runs")
                        fw_cfg["ablation_max_runs"] = plateau_ablation_runs
                        plateau_abl_run = _run_flywheel_ablation(
                            fw_cfg, name, iteration,
                            warmstart_ckpt=_ablation_warmstart_ckpt(fw_cfg, _fw_ckpt_dir))
                        if _saved_max is None:
                            fw_cfg.pop("ablation_max_runs", None)
                        else:
                            fw_cfg["ablation_max_runs"] = _saved_max
                        if plateau_abl_run:
                            new_hp = _read_ablation_best(plateau_abl_run)
                            if new_hp:
                                hyperparams.update(new_hp)
                                fw_db.set_best_hyperparams(name, hyperparams)
                                log_orch(f"[flywheel:{name}] plateau ablation updated hyperparams "
                                         f"→ {hyperparams}")
                            # Record the plateau ablation on the current iteration row
                            fw_db.update_iteration(
                                row_id=row_id, status=status,
                                exit_code=exit_code if exit_code is not None else -1,
                                elapsed_secs=elapsed,
                                train_loss=metrics.get("loss_smooth"),
                                ref_gap=metrics.get("ref_gap"),
                                cond_gap=metrics.get("cond_gap"),
                                checkpoint=ckpt_path,
                                checkpoint_hash=new_ckpt_hash,
                                ablation_run=plateau_abl_run,
                            )
                            fw_db.refresh_campaign_summary(name)

                    fw_db.set_campaign_status(name, "plateau")
                    try:
                        FLYWHEEL_CONTROL_FILE.write_text(
                            json.dumps({"action": "pause",
                                        "reason": plateau_reason,
                                        "auto":   True})
                        )
                    except Exception as _e:
                        log_orch(f"[flywheel:{name}] could not write plateau pause: {_e}",
                                 level="warning")

            # Regenerate HTML report
            iterations_data = fw_db.get_iterations(name)
            shard_stats     = score_db.get_stats()
            top_shards      = score_db.get_top_shards(20)
            html = render_flywheel_index(name, iterations_data, shard_stats, top_shards,
                                         hyperparams, plateau_reason=plateau_reason)
            (reports_dir / "index.html").write_text(html)

            notify("iris flywheel",
                   f"Iter {iteration}/{max_iters}: {status}  ref_gap={metrics.get('ref_gap')}")

            iteration += 1

        log_orch(f"[flywheel:{name}] loop complete — {max_iters} iterations done")
        fw_db.refresh_campaign_summary(name)
        fw_db.set_campaign_status(name, "completed")
        superseded = fw_db.mark_superseded_by(name)
        if superseded:
            log_orch(f"[flywheel:{name}] superseded campaigns: {superseded}")
        cold_root = Path(fw_cfg.get("storage", {}).get("cold_root", str(
            __import__("pipeline_lib").COLD_ROOT
        )))
        written = write_campaign_summary_json(name, fw_db, cold_root)
        if written:
            log_orch(f"[flywheel:{name}] summary written: {written}")

        # Opt-in cross-run quality gate: golden-set eval the champion checkpoint and
        # compare to the previous campaign's champion (clip_i/lpips/fid). Default off
        # (needs `quality_gate: true` + a golden set + GPU); only runs at campaign end.
        try:
            _qg_ckpt = _quality_gate_target(
                fw_cfg, (fw_db.get_best(name) or {}).get("checkpoint"))
            if _qg_ckpt:
                _qg_log = LOG_DIR / f"flywheel_{name}_quality_gate.log"
                _qg_cmd = (f"source '{TRAIN_DIR}/.venv/bin/activate' && "
                           f"python -u '{SCRIPTS_DIR}/quality_gate.py' "
                           f"--checkpoint '{_qg_ckpt}' --campaign '{name}'")
                log_orch(f"[flywheel:{name}] cross-run quality gate on "
                         f"{os.path.basename(_qg_ckpt)} → {_qg_log.name}")
                subprocess.run(["bash", "-c",
                                f"({_qg_cmd}) >> '{_qg_log}' 2>&1; "
                                f"echo EXIT_CODE=$? >> '{_qg_log}'"])
        except Exception as _qg_err:
            log_orch(f"[flywheel:{name}] quality gate error: {_qg_err}", level="warning")

        notify("iris flywheel", f"{name} complete ({max_iters} iterations)")
    finally:
        fw_db.close()
        score_db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _orchestrator_snapshot() -> dict:
    """Read-only state snapshot for --ai mode. Does not start or modify anything."""
    import json as _json
    from datetime import datetime, timezone

    state = read_state()
    active_chunk = None
    best_age = float("inf")
    for c in range(1, 9):
        age = heartbeat_age_secs("trainer", c)
        if age is not None and age < best_age:
            best_age = age
            active_chunk = c

    orch_age: int | None = None
    try:
        updated = state.get("last_updated", "")
        if updated:
            ts = datetime.fromisoformat(updated)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            orch_age = round((datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:
        pass

    # Pending action: read control file if present
    pending_action = None
    if CONTROL_FILE.exists():
        try:
            ctl = _json.loads(CONTROL_FILE.read_text())
            pending_action = ctl.get("action")
        except Exception:
            pass

    # Derive overall state string
    if active_chunk is not None and best_age <= HEARTBEAT_STALE_SECS:
        overall_state = "training"
    elif pending_action in ("pause", "abort"):
        overall_state = pending_action
    elif state:
        overall_state = "idle"
    else:
        overall_state = "unknown"

    return {
        "state": overall_state,
        "active_chunk": active_chunk,
        "last_poll_age_sec": orch_age,
        "pending_action": pending_action,
        "run_id": state.get("run_id"),
        "scale": state.get("scale"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 pipeline orchestrator")
    ap.add_argument("--config",          default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--scale",           default=None)
    ap.add_argument("--resume",          action="store_true")
    ap.add_argument("--dry-run",         action="store_true")
    ap.add_argument("--skip-dedup",      action="store_true")
    ap.add_argument("--flywheel-config", default=None, metavar="PATH",
                    help="YAML flywheel config — enables self-improving sref optimization loop")
    ap.add_argument("--report-only",     action="store_true",
                    help="With --flywheel-config: regenerate HTML report from DB and exit")
    ap.add_argument("--ai",              action="store_true",
                    help="Emit read-only JSON snapshot to stdout and exit")
    args = ap.parse_args()

    if args.ai:
        import json as _json
        print(_json.dumps(_orchestrator_snapshot()))
        sys.exit(0)

    if args.flywheel_config:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from flywheel_lib import load_flywheel_config
        fw_cfg = load_flywheel_config(Path(args.flywheel_config))
        if args.report_only:
            _flywheel_report_only(fw_cfg)
        else:
            _run_flywheel_loop(fw_cfg, str(Path(args.flywheel_config).resolve()))
        sys.exit(0)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config not found: {args.config}", file=sys.stderr)
        cfg = {}
    cfg["_config_path"] = args.config

    if args.scale:
        cfg["scale"] = args.scale

    Orchestrator(cfg, dry_run=args.dry_run, skip_dedup=args.skip_dedup).run()


if __name__ == "__main__":
    main()
