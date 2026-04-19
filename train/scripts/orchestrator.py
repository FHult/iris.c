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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pipeline_lib import (
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, VENV_PYTHON,
    CONTROL_FILE, SENTINEL_DIR, LOG_DIR, STAGING_DIR,
    SHARDS_DIR, PRECOMP_DIR, HARD_EX_DIR, DEDUP_DIR, CKPT_DIR,
    TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_ORCH_WIN,
    DISK_WARN_GB, DISK_ABORT_GB, HEARTBEAT_STALE_SECS,
    STATE_FILE,
    read_state, write_state, update_state,
    is_done, mark_done, mark_error, has_error, read_error, clear_error,
    log_event, log_orch,
    write_heartbeat, heartbeat_age_secs,
    dispatch_issue, gpu_is_free,
    tmux_session_exists, tmux_window_exists, tmux_new_window,
    last_exit_code, free_gb, notify, now_iso, load_config,
)


# ---------------------------------------------------------------------------
# Chunk state machine
# ---------------------------------------------------------------------------

class ChunkState(str):
    IDLE         = "IDLE"
    DOWNLOADING  = "DOWNLOADING"
    CONVERTING   = "CONVERTING"
    BUILDING     = "BUILDING"
    FILTERING    = "FILTERING"
    CLIP_EMBED   = "CLIP_EMBED"
    CLIP_INDEX   = "CLIP_INDEX"
    CLIP_DUPS    = "CLIP_DUPS"
    PRECOMPUTING = "PRECOMPUTING"
    READY        = "READY"
    TRAINING     = "TRAINING"
    MINING       = "MINING"
    VALIDATING   = "VALIDATING"
    DONE         = "DONE"
    ERROR        = "ERROR"


CHUNK_STEPS = [
    "download",
    "convert",
    "build_shards",
    "filter_shards",
    "clip_embed",
    "clip_index",
    "clip_dups",
    "precompute",
    "train",
    "mine",
    "validate",
]

_STEP_TO_STATE = {
    "download":     ChunkState.CONVERTING,
    "convert":      ChunkState.BUILDING,
    "build_shards": ChunkState.FILTERING,
    "filter_shards":ChunkState.CLIP_EMBED,
    "clip_embed":   ChunkState.CLIP_INDEX,
    "clip_index":   ChunkState.CLIP_DUPS,
    "clip_dups":    ChunkState.PRECOMPUTING,
    "precompute":   ChunkState.READY,
    "train":        ChunkState.MINING,
    "mine":         ChunkState.VALIDATING,
    "validate":     ChunkState.DONE,
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
        self._holders[token] = holder
        return True

    def release(self, token: str) -> None:
        self._holders.pop(token, None)

    def holder(self, token: str) -> Optional[str]:
        return self._holders.get(token)


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
        self.issue_counter = 0
        self._restart_counts: dict[tuple, int] = {}

        # Track what is currently running in the prep window:
        # {"chunk": N, "step": "build_shards", "log": Path, "token": "DISK_WRITE_HIGH"|None}
        self._active_prep: Optional[dict] = None

        # Anomaly detection counters: {chunk: count_of_consecutive_anomaly_polls}
        self._loss_high_since: dict[int, Optional[int]] = {}  # chunk → first anomaly step
        self._grad_spike_polls: dict[int, int] = {}           # chunk → consecutive high-grad polls

        # Cache of last-written state per chunk — avoid redundant file writes
        self._last_written_state: dict[int, str] = {}

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        log_orch("Orchestrator starting", scale=self.scale, chunks=self.total_chunks)
        self._ensure_tmux_session()
        self._run_doctor()
        self._init_state()

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

                write_heartbeat("orchestrator", step="poll")
                if self._all_done():
                    log_orch("All chunks complete — pipeline finished")
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
            notify("iris pipeline ERROR", f"{step} chunk {chunk} failed")

    def _post_step(self, chunk: int, step: str) -> None:
        """Side effects after a prep step completes successfully."""
        if step == "build_shards":
            # Free staging raw data to reclaim disk space
            self._delete_staging_raw(chunk)
        if step in ("train", "mine"):
            update_state(**{"chunks": {str(chunk): {"completed_at": now_iso()}}})

    def _delete_staging_raw(self, chunk: int) -> None:
        """Delete staging raw data once shards are built (V2 Section 6 lifecycle)."""
        raw = STAGING_DIR / f"chunk{chunk}" / "raw"
        if raw.exists():
            log_orch(f"Chunk {chunk}: freeing staging raw data ({raw})", chunk=chunk)
            if not self.dry_run:
                shutil.rmtree(raw, ignore_errors=True)

    def _launch_prep(self, description: str, cmd: str, log_file: Path,
                     chunk: int, step: str,
                     token: Optional[str] = None,
                     also_mark: Optional[list] = None) -> None:
        """Launch cmd in the iris-prep tmux window and record the active step."""
        if self._active_prep is not None:
            return  # already something running
        if self.dry_run:
            log_orch(f"DRY RUN: would launch {description}")
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        activated = (f"export PIPELINE_DATA_ROOT='{DATA_ROOT}' && "
                     f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}")
        tmux_new_window(TMUX_PREP_WIN, activated, log_file)
        self._active_prep = {
            "chunk": chunk, "step": step, "log": log_file,
            "token": token, "also_mark": also_mark or [],
        }
        log_orch(f"Launched: {description} → {log_file}")

    def _prep_busy(self) -> bool:
        """True when the prep window is occupied."""
        return self._active_prep is not None or tmux_window_exists(TMUX_PREP_WIN)

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

        # Chunk N+1 doesn't start until chunk N is at least BUILDING
        if chunk > 1 and derive_chunk_state(chunk - 1) == ChunkState.IDLE:
            return

        handlers = {
            ChunkState.IDLE:        self._start_download_convert,
            ChunkState.DOWNLOADING: self._noop,  # wait for prep window
            ChunkState.CONVERTING:  self._noop,  # wait for prep window
            ChunkState.BUILDING:    self._start_build,
            ChunkState.FILTERING:   self._start_filter,
            ChunkState.CLIP_EMBED:  self._start_clip_embed,
            ChunkState.CLIP_INDEX:  self._start_clip_index,
            ChunkState.CLIP_DUPS:   self._start_clip_dups,
            ChunkState.PRECOMPUTING:self._start_precompute,
            ChunkState.READY:       self._check_ready,
            ChunkState.MINING:      self._start_mining,
            ChunkState.VALIDATING:  self._start_validation,
            ChunkState.DONE:        self._noop,
            ChunkState.ERROR:       self._handle_error,
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
        log_orch(f"Chunk {chunk}: starting download+convert", chunk=chunk)
        log_file = LOG_DIR / f"download_chunk{chunk}.log"
        jdb_only = self.cfg.get("download", {}).get("jdb_only", False)
        extra = " --jdb-only" if jdb_only else ""
        cmd = self._python_cmd("download_convert.py",
                               f"--chunk {chunk} --scale {self.scale} --config '{self._config_path()}'{extra}")
        # Marks both download.done AND convert.done on exit 0
        self._launch_prep(f"download+convert chunk {chunk}", cmd, log_file,
                          chunk, "download", also_mark=["convert"])

    def _start_build(self, chunk: int) -> None:
        if is_done(chunk, "build_shards") or self._prep_busy():
            return
        if not self.res.request("DISK_WRITE_HIGH", f"build chunk {chunk}"):
            return
        log_orch(f"Chunk {chunk}: building shards", chunk=chunk)
        out      = STAGING_DIR / f"chunk{chunk}" / "shards"
        sources  = self._build_sources(chunk)
        blocklist_arg = ""
        blocklist = DEDUP_DIR / "duplicate_ids.txt"
        if blocklist.exists():
            blocklist_arg = f"--blocklist '{blocklist}'"
        log_file = LOG_DIR / f"build_chunk{chunk}.log"
        cmd = self._python_cmd("build_shards.py",
                               f"--sources {sources} --output '{out}' "
                               f"--workers 1 {blocklist_arg}")
        self._launch_prep(f"build chunk {chunk}", cmd, log_file,
                          chunk, "build_shards", token="DISK_WRITE_HIGH")

    def _start_filter(self, chunk: int) -> None:
        if is_done(chunk, "filter_shards") or self._prep_busy():
            return
        log_orch(f"Chunk {chunk}: filtering shards", chunk=chunk)
        shard_dir = STAGING_DIR / f"chunk{chunk}" / "shards"
        log_file  = LOG_DIR / f"filter_chunk{chunk}.log"
        cmd = self._python_cmd("filter_shards.py",
                               f"--shards '{shard_dir}' --workers 1")
        self._launch_prep(f"filter chunk {chunk}", cmd, log_file,
                          chunk, "filter_shards")

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
        log_orch(f"Chunk {chunk}: CLIP embedding", chunk=chunk)
        shard_dir = STAGING_DIR / f"chunk{chunk}" / "shards"
        embed_dir = STAGING_DIR / f"chunk{chunk}" / "embeddings"
        log_file  = LOG_DIR / f"clip_embed_chunk{chunk}.log"
        cmd = self._python_cmd("clip_dedup.py",
                               f"embed --shards '{shard_dir}' --embeddings '{embed_dir}'")
        self._launch_prep(f"clip_embed chunk {chunk}", cmd, log_file,
                          chunk, "clip_embed")

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
        log_file = LOG_DIR / f"clip_dups_chunk{chunk}.log"
        cmd = self._python_cmd("clip_dedup.py",
                               f"find-dups --index '{index}' --out '{dups}'")
        self._launch_prep(f"clip_dups chunk {chunk}", cmd, log_file,
                          chunk, "clip_dups")

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
        max_shards = self.cfg.get("precompute", {}).get("max_shards", None)
        max_shards_arg = f"--max-shards {max_shards}" if max_shards else ""
        log_file     = LOG_DIR / f"precompute_chunk{chunk}.log"
        log_orch(f"Chunk {chunk}: precomputing Qwen3+VAE embeddings", chunk=chunk)
        cmd = self._python_cmd("precompute_all.py",
                               f"--shards '{shard_dir}' "
                               f"--qwen3-output '{qwen3_out}' "
                               f"--vae-output '{vae_out}' "
                               f"--siglip-output '{siglip_out}' "
                               f"{flux_model_arg} "
                               f"{max_shards_arg} "
                               f"{siglip_flag}")
        self._launch_prep(f"precompute chunk {chunk}", cmd, log_file,
                          chunk, "precompute", token="GPU_TOKEN")

    def _check_ready(self, chunk: int) -> None:
        """Promote staging to production, then start training."""
        if chunk > 1 and not is_done(chunk - 1, "train"):
            return  # wait for previous chunk's training
        if not gpu_is_free():
            return
        self._promote_chunk(chunk)
        self._start_training(chunk)

    def _promote_chunk(self, chunk: int) -> None:
        """Move staging shards + precomputed to production directories."""
        staging  = STAGING_DIR / f"chunk{chunk}"
        shard_src   = staging / "shards"
        precomp_src = staging / "precomputed"

        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        count = 0
        for tar in sorted(shard_src.glob("*.tar")):
            dest = SHARDS_DIR / f"chunk{chunk}_{count:04d}.tar"
            tar.rename(dest)
            count += 1
        log_orch(f"Chunk {chunk}: promoted {count} shards to production", chunk=chunk)

        for subdir in ["qwen3", "vae", "siglip"]:
            src = precomp_src / subdir
            if not src.exists():
                continue
            dst = PRECOMP_DIR / subdir
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                f.rename(dst / f.name)

    def _start_training(self, chunk: int) -> None:
        log_file = LOG_DIR / f"train_chunk{chunk}.log"

        if tmux_window_exists(TMUX_TRAIN_WIN):
            return  # still running

        # Window is gone — if a previous attempt left a log, check outcome.
        if log_file.exists():
            code = last_exit_code(log_file)
            if code == 0:
                log_orch(f"Chunk {chunk}: training complete", chunk=chunk)
                mark_done(chunk, "train")
                self.res.release("GPU_TOKEN")
                return
            if code is not None:
                msg = f"Training exited {code}; see {log_file}"
                log_orch(f"Chunk {chunk}: training FAILED — {msg}", level="error", chunk=chunk)
                mark_error(chunk, "train", msg)
                self.res.release("GPU_TOKEN")
                notify("iris pipeline ERROR", f"Training chunk {chunk} failed")
                return

        if not self.res.request("GPU_TOKEN", f"train chunk {chunk}"):
            return

        training_cfg = self.cfg.get("training", {})
        steps_map    = training_cfg.get("steps", {}).get(self.scale, {})
        steps = steps_map.get(chunk, steps_map.get(str(chunk), 15000))
        lr_map = training_cfg.get("lr", {})
        lr = lr_map.get(chunk, lr_map.get(str(chunk), 1e-5))

        log_orch(f"Chunk {chunk}: starting training ({steps} steps, lr={lr})", chunk=chunk)

        resume_arg = ""
        best = CKPT_DIR / "best.safetensors"
        if chunk > 1 and best.exists():
            resume_arg = f"--resume '{best}'"

        hard_arg = ""
        if chunk > 1:
            hard_chunk = HARD_EX_DIR / f"chunk{chunk-1}"
            if hard_chunk.exists() and any(hard_chunk.glob("*.tar")):
                hard_arg = f"--hard-examples '{hard_chunk}'"

        config_file = self.cfg.get("training_config",
                                   str(TRAIN_DIR / "configs" / "stage1_512px.yaml"))
        cmd = (
            f"caffeinate -i python -u '{TRAIN_DIR}/train_ip_adapter.py' "
            f"--config '{config_file}' "
            f"--max-steps {steps} --lr {lr} "
            f"--data-root '{DATA_ROOT}' "
            f"{resume_arg} {hard_arg}"
        )

        if not self.dry_run:
            activated = (f"export PIPELINE_DATA_ROOT='{DATA_ROOT}' && "
                         f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            tmux_new_window(TMUX_TRAIN_WIN, activated, log_file)

        update_state(**{"chunks": {str(chunk): {
            "state": "TRAINING", "started_at": now_iso(),
            "steps": steps, "lr": lr,
        }}})

    def _start_mining(self, chunk: int) -> None:
        if is_done(chunk, "mine") or self._prep_busy():
            return
        if not gpu_is_free():
            return
        if not self.res.request("GPU_TOKEN", f"mine chunk {chunk}"):
            return

        best = CKPT_DIR / "best.safetensors"
        if not best.exists():
            log_orch(f"Chunk {chunk}: no checkpoint — skipping mining", chunk=chunk)
            mark_done(chunk, "mine")
            self.res.release("GPU_TOKEN")
            return

        log_orch(f"Chunk {chunk}: mining hard examples", chunk=chunk)
        out = HARD_EX_DIR / f"chunk{chunk}"
        out.mkdir(parents=True, exist_ok=True)
        flux_model  = self.cfg.get("model", {}).get("flux_model", "flux-klein-4b")
        qwen3_cache = PRECOMP_DIR / "qwen3"
        vae_cache   = PRECOMP_DIR / "vae"
        log_file    = LOG_DIR / f"mine_chunk{chunk}.log"
        cmd = self._python_cmd("mine_hard_examples.py",
                               f"--checkpoint '{best}' "
                               f"--shards '{SHARDS_DIR}' "
                               f"--qwen3-cache '{qwen3_cache}' "
                               f"--vae-cache '{vae_cache}' "
                               f"--flux-model {flux_model} --null-siglip "
                               f"--output '{out}'")
        self._launch_prep(f"mine chunk {chunk}", cmd, log_file,
                          chunk, "mine", token="GPU_TOKEN")

    def _start_validation(self, chunk: int) -> None:
        if is_done(chunk, "validate") or self._prep_busy():
            return
        best = CKPT_DIR / "best.safetensors"
        if not best.exists():
            log_orch(f"Chunk {chunk}: no checkpoint — skipping validation", chunk=chunk)
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
                          chunk, "validate")

    def _handle_error(self, chunk: int) -> None:
        for step in CHUNK_STEPS:
            if has_error(chunk, step):
                msg = read_error(chunk, step)
                key = (chunk, step)
                restarts = self._restart_counts.get(key, 0)
                if restarts < 1:
                    log_orch(f"Chunk {chunk}: auto-retrying {step}", chunk=chunk)
                    self._restart_counts[key] = restarts + 1
                    clear_error(chunk, step)
                else:
                    issue_id = self._next_issue_id()
                    log_orch(f"Chunk {chunk}: {step} failed twice — escalating", level="error")
                    dispatch_issue(issue_id, "error",
                                   f"{step} chunk {chunk} failed twice — manual intervention needed",
                                   chunk=chunk, process=step, context={"error": msg},
                                   suggested_action="investigate_logs_and_clear_error")
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
        laion = DATA_ROOT / "raw" / "laion"
        coyo  = DATA_ROOT / "raw" / "coyo"
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
        if holder and "train" in holder and not tmux_window_exists(TMUX_TRAIN_WIN):
            self.res.release("GPU_TOKEN")

    # -----------------------------------------------------------------------
    # Heartbeat monitoring
    # -----------------------------------------------------------------------

    def _check_heartbeats(self) -> None:
        if not tmux_window_exists(TMUX_TRAIN_WIN):
            return
        for chunk in range(1, self.total_chunks + 1):
            if derive_chunk_state(chunk) == ChunkState.TRAINING:
                self._check_training_anomalies(chunk)

    def _check_training_anomalies(self, chunk: int) -> None:
        """Section 5.1 anomaly detection rules for a training chunk."""
        import math

        hb = read_heartbeat("trainer", chunk)
        age = heartbeat_age_secs("trainer", chunk)

        # ── Heartbeat staleness → crash detection + restart ──────────────────
        if age is not None and age > HEARTBEAT_STALE_SECS:
            log_orch(f"Chunk {chunk}: trainer heartbeat stale ({age:.0f}s) — restarting",
                     level="error", chunk=chunk)
            dispatch_issue(self._next_issue_id(), "error",
                           f"Trainer heartbeat stale ({age:.0f}s) — restarting process",
                           chunk=chunk, process="trainer",
                           suggested_action="check_training_log")
            self._restart_trainer(chunk)
            return

        if hb is None:
            return

        loss = hb.get("loss")
        step = hb.get("step", 0)

        # ── Loss NaN/Inf → immediate pause ───────────────────────────────────
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            log_orch(f"Chunk {chunk}: loss NaN/Inf at step {step} — pausing",
                     level="error", chunk=chunk)
            dispatch_issue(self._next_issue_id(), "critical",
                           f"Training loss NaN/Inf at step {step}",
                           chunk=chunk, suggested_action="investigate_training")
            self._write_pause()
            return

        # ── Loss persistently > 2.0 for 100 steps → pause ───────────────────
        if loss is not None:
            if loss > 2.0:
                first = self._loss_high_since.get(chunk)
                if first is None:
                    self._loss_high_since[chunk] = step
                elif step - first >= 100:
                    log_orch(f"Chunk {chunk}: loss {loss:.3f} > 2.0 for {step-first} steps — pausing",
                             level="error", chunk=chunk)
                    dispatch_issue(self._next_issue_id(), "error",
                                   f"Training loss {loss:.3f} persistently > 2.0 (since step {first})",
                                   chunk=chunk, suggested_action="investigate_training")
                    self._write_pause()
                    self._loss_high_since[chunk] = None
            else:
                self._loss_high_since[chunk] = None

        # ── Grad norm > 50.0 for 10 polls → pause; > 10.0 → warn ────────────
        grad_norm = hb.get("grad_norm")
        if grad_norm is not None:
            if grad_norm > 50.0:
                self._grad_spike_polls[chunk] = self._grad_spike_polls.get(chunk, 0) + 1
                n_spikes = self._grad_spike_polls[chunk]
                if n_spikes >= 10:
                    log_orch(f"Chunk {chunk}: grad_norm {grad_norm:.1f} > 50 for {n_spikes} polls — pausing",
                             level="error", chunk=chunk)
                    dispatch_issue(self._next_issue_id(), "error",
                                   f"Grad norm {grad_norm:.1f} sustained > 50 ({n_spikes} polls)",
                                   chunk=chunk, suggested_action="investigate_training")
                    self._write_pause()
                    self._grad_spike_polls[chunk] = 0
                elif grad_norm > 10.0:
                    log_orch(f"Chunk {chunk}: grad_norm warning {grad_norm:.1f} > 10",
                             level="warning", chunk=chunk)
            else:
                self._grad_spike_polls[chunk] = 0

        # ── SigLIP coverage < 90% → log warning ──────────────────────────────
        siglip_cov = hb.get("siglip_coverage_pct")
        if siglip_cov is not None and siglip_cov < 90:
            log_orch(f"Chunk {chunk}: SigLIP coverage {siglip_cov:.0f}% < 90%",
                     level="warning", chunk=chunk)

    def _restart_trainer(self, chunk: int) -> None:
        """Kill and relaunch the training window for the given chunk."""
        if self.dry_run:
            log_orch(f"[dry-run] would restart trainer for chunk {chunk}")
            return
        import subprocess
        key = ("restart_trainer", chunk)
        count = self._restart_counts.get(key, 0) + 1
        self._restart_counts[key] = count
        if count > 3:
            log_orch(f"Chunk {chunk}: trainer restart limit (3) exceeded — aborting",
                     level="error", chunk=chunk)
            dispatch_issue(self._next_issue_id(), "critical",
                           f"Trainer restart limit exceeded for chunk {chunk}",
                           chunk=chunk, suggested_action="manual_intervention")
            return
        log_orch(f"Chunk {chunk}: killing iris-train window (restart #{count})", chunk=chunk)
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_TRAIN_WIN}"],
                       capture_output=True)
        # Let orchestrator's normal _check_training path relaunch on next poll
        # by clearing the train.done sentinel (training was not complete)
        clear_error(chunk, "train")

    def _write_pause(self) -> None:
        """Write a pause control signal so the operator can investigate."""
        try:
            import json as _json
            with open(CONTROL_FILE, "w") as _f:
                _json.dump({"action": "pause"}, _f)
        except OSError:
            pass

    # -----------------------------------------------------------------------
    # Disk check
    # -----------------------------------------------------------------------

    def _check_disk(self) -> None:
        gb = free_gb()
        if gb < DISK_ABORT_GB:
            log_orch(f"CRITICAL: {gb:.1f} GB free — pausing", level="error")
            dispatch_issue(self._next_issue_id(), "error",
                           f"Disk critically low: {gb:.1f} GB", suggested_action="free_disk")
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
                log_orch("Operator: pause — sleeping 300s")
                time.sleep(300)
            elif action == "abort":
                log_orch("Operator: abort")
                sys.exit(0)
            elif action == "force-next-chunk":
                chunk = ctrl.get("chunk")
                if chunk:
                    log_orch(f"Operator: force-next-chunk {chunk}")
                    mark_done(chunk, "validate")
            CONTROL_FILE.unlink(missing_ok=True)
        except Exception as e:
            log_orch(f"Control file error: {e}", level="warning")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _python_cmd(self, script: str, args: str) -> str:
        return f"python -u '{SCRIPTS_DIR}/{script}' {args}"

    def _config_path(self) -> str:
        return self.cfg.get("_config_path", str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))

    def _all_done(self) -> bool:
        return all(derive_chunk_state(c) == ChunkState.DONE
                   for c in range(1, self.total_chunks + 1))

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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V2 pipeline orchestrator")
    ap.add_argument("--config",     default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--scale",      default=None)
    ap.add_argument("--resume",     action="store_true")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()

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
