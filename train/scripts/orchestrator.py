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
    # Start orchestrating from chunk 1:
    python train/scripts/orchestrator.py --config train/configs/v2_pipeline.yaml

    # Resume (re-reads pipeline_state.json to find current position):
    python train/scripts/orchestrator.py --resume

    # Dry-run (print decisions, don't launch processes):
    python train/scripts/orchestrator.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from pipeline_lib import (
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, VENV_PYTHON,
    STATE_FILE, CONTROL_FILE, SENTINEL_DIR, LOG_DIR, STAGING_DIR,
    SHARDS_DIR, PRECOMP_DIR, HARD_EX_DIR, DEDUP_DIR, CKPT_DIR,
    TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_ORCH_WIN,
    DISK_WARN_GB, DISK_ABORT_GB, HEARTBEAT_STALE_SECS,
    read_state, write_state, update_state,
    is_done, mark_done, mark_error, has_error, read_error, clear_error,
    sentinel_path, log_event, log_orch,
    write_heartbeat, read_heartbeat, is_heartbeat_stale,
    dispatch_issue, gpu_is_free,
    tmux_session_exists, tmux_window_exists, tmux_new_window,
    last_exit_code, free_gb, notify, now_iso, load_config,
)


# ---------------------------------------------------------------------------
# Chunk state machine
# ---------------------------------------------------------------------------

class ChunkState(str, Enum):
    IDLE              = "IDLE"
    DOWNLOADING       = "DOWNLOADING"
    CONVERTING        = "CONVERTING"
    BUILDING          = "BUILDING"
    FILTERING         = "FILTERING"
    CLIP_EMBED        = "CLIP_EMBED"
    CLIP_INDEX        = "CLIP_INDEX"
    CLIP_DUPS         = "CLIP_DUPS"
    PRECOMPUTING      = "PRECOMPUTING"
    READY             = "READY"         # precompute done; waiting for prev chunk training
    PROMOTING         = "PROMOTING"     # staging → production (brief, atomic)
    TRAINING          = "TRAINING"
    MINING            = "MINING"
    VALIDATING        = "VALIDATING"
    DONE              = "DONE"
    ERROR             = "ERROR"


# Steps in dependency order — used to derive state from sentinels
CHUNK_STEPS = [
    "download",
    "convert",
    "build_shards",
    "filter_shards",
    "clip_embed",
    "clip_index",
    "clip_dups",
    "precompute",
    "promoted",
    "train",
    "mine",
    "validate",
]

STEP_TO_STATE = {
    "download":    ChunkState.CONVERTING,
    "convert":     ChunkState.BUILDING,
    "build_shards": ChunkState.FILTERING,
    "filter_shards": ChunkState.CLIP_EMBED,
    "clip_embed":  ChunkState.CLIP_INDEX,
    "clip_index":  ChunkState.CLIP_DUPS,
    "clip_dups":   ChunkState.PRECOMPUTING,
    "precompute":  ChunkState.READY,
    "promoted":    ChunkState.TRAINING,
    "train":       ChunkState.MINING,
    "mine":        ChunkState.VALIDATING,
    "validate":    ChunkState.DONE,
}


def derive_chunk_state(chunk: int) -> ChunkState:
    """Derive current state from sentinel files — no in-memory state needed."""
    if has_error(chunk, "orchestrator"):
        return ChunkState.ERROR
    last_done = None
    for step in CHUNK_STEPS:
        if is_done(chunk, step):
            last_done = step
        elif has_error(chunk, step):
            return ChunkState.ERROR
    if last_done is None:
        return ChunkState.IDLE
    next_state = STEP_TO_STATE.get(last_done, ChunkState.DONE)
    return next_state


# ---------------------------------------------------------------------------
# Resource token manager
# ---------------------------------------------------------------------------

class ResourceManager:
    """
    Tracks resource tokens. The orchestrator consults this before launching
    any process. GPU_TOKEN is exclusive — only one holder at a time.
    """

    def __init__(self):
        self._holders: dict[str, str] = {}  # token → holder description

    def request(self, token: str, holder: str) -> bool:
        if token == "GPU_TOKEN" and "GPU_TOKEN" in self._holders:
            return False
        if token == "DISK_WRITE_HIGH" and "DISK_WRITE_HIGH" in self._holders:
            return False
        self._holders[token] = holder
        return True

    def release(self, token: str) -> None:
        self._holders.pop(token, None)

    def holder(self, token: str) -> Optional[str]:
        return self._holders.get(token)

    def is_free(self, token: str) -> bool:
        return token not in self._holders


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, config: dict, dry_run: bool = False, skip_dedup: bool = False):
        self.cfg     = config
        self.dry_run = dry_run
        self.skip_dedup = skip_dedup
        self.res     = ResourceManager()
        self.total_chunks = config.get("chunks", 4)
        self.scale   = config.get("scale", "small")
        self.issue_counter = 0
        self._restart_counts: dict[tuple, int] = {}

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
                self._sync_resources()
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
                log_orch(f"Orchestrator error: {e}", level="error")
                dispatch_issue(self._next_issue_id(), "error",
                               f"Orchestrator exception: {e}",
                               suggested_action="check_orchestrator_logs")

            if not self.dry_run:
                time.sleep(poll_interval)
            else:
                break  # dry-run: one pass only

    # -----------------------------------------------------------------------
    # Per-chunk advancement
    # -----------------------------------------------------------------------

    def _advance_chunk(self, chunk: int) -> None:
        state = derive_chunk_state(chunk)
        log_event("orchestrator", "chunk_state", chunk=chunk, state=state.value)

        # Don't advance beyond chunk 1 until chunk 1 is at least BUILDING
        if chunk > 1:
            prev_state = derive_chunk_state(chunk - 1)
            if prev_state == ChunkState.IDLE:
                return  # previous chunk hasn't started yet

        handlers = {
            ChunkState.IDLE:         self._start_download,
            ChunkState.DOWNLOADING:  self._check_download,
            ChunkState.CONVERTING:   self._start_or_check_convert,
            ChunkState.BUILDING:     self._start_or_check_build,
            ChunkState.FILTERING:    self._start_or_check_filter,
            ChunkState.CLIP_EMBED:   self._start_or_check_clip_embed,
            ChunkState.CLIP_INDEX:   self._start_or_check_clip_index,
            ChunkState.CLIP_DUPS:    self._start_or_check_clip_dups,
            ChunkState.PRECOMPUTING: self._start_or_check_precompute,
            ChunkState.READY:        self._check_ready_to_promote,
            ChunkState.PROMOTING:    self._run_promote,
            ChunkState.TRAINING:     self._check_training,
            ChunkState.MINING:       self._start_or_check_mining,
            ChunkState.VALIDATING:   self._start_or_check_validation,
            ChunkState.DONE:         lambda c: None,
            ChunkState.ERROR:        self._handle_error,
        }
        handler = handlers.get(state)
        if handler:
            handler(chunk)

    # -----------------------------------------------------------------------
    # Step handlers
    # -----------------------------------------------------------------------

    def _start_download(self, chunk: int) -> None:
        if self._prep_busy():
            return
        log_orch(f"Chunk {chunk}: starting download", chunk=chunk)
        cmd = self._python_cmd("downloader.py",
                               f"--chunk {chunk} --config '{self._config_path()}'")
        log_file = LOG_DIR / f"download_chunk{chunk}.log"
        self._launch_prep(f"download chunk {chunk}", cmd, log_file, chunk, "download")

    def _check_download(self, chunk: int) -> None:
        self._check_prep_step(chunk, "download")

    def _start_or_check_convert(self, chunk: int) -> None:
        if is_done(chunk, "convert"):
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "convert")
            return
        log_orch(f"Chunk {chunk}: starting conversion", chunk=chunk)
        jdb_input  = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb"
        jdb_output = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb_wds"
        cmd = self._python_cmd("convert_journeydb.py",
                               f"--input '{jdb_input}' --output '{jdb_output}' --workers 1")
        log_file = LOG_DIR / f"convert_chunk{chunk}.log"
        self._launch_prep(f"convert chunk {chunk}", cmd, log_file, chunk, "convert")

    def _start_or_check_build(self, chunk: int) -> None:
        if is_done(chunk, "build_shards"):
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "build_shards")
            return
        if not self.res.request("DISK_WRITE_HIGH", f"build chunk {chunk}"):
            return
        log_orch(f"Chunk {chunk}: building shards", chunk=chunk)
        out = STAGING_DIR / f"chunk{chunk}" / "shards"
        blocklist = DEDUP_DIR / "duplicate_ids.txt"
        sources = self._build_sources(chunk)
        blocklist_arg = f"--blocklist '{blocklist}'" if blocklist.exists() else ""
        cmd = self._python_cmd("build_shards.py",
                               f"--sources {sources} --output '{out}' "
                               f"--workers 1 {blocklist_arg}")
        log_file = LOG_DIR / f"build_chunk{chunk}.log"
        self._launch_prep(f"build chunk {chunk}", cmd, log_file, chunk, "build_shards",
                          on_done=lambda: self.res.release("DISK_WRITE_HIGH"))

    def _start_or_check_filter(self, chunk: int) -> None:
        if is_done(chunk, "filter_shards"):
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "filter_shards")
            return
        log_orch(f"Chunk {chunk}: filtering shards", chunk=chunk)
        shard_dir = STAGING_DIR / f"chunk{chunk}" / "shards"
        cmd = self._python_cmd("filter_shards.py", f"--shards '{shard_dir}' --workers 1")
        log_file = LOG_DIR / f"filter_chunk{chunk}.log"
        self._launch_prep(f"filter chunk {chunk}", cmd, log_file, chunk, "filter_shards")

    def _start_or_check_clip_embed(self, chunk: int) -> None:
        if is_done(chunk, "clip_embed"):
            return
        if self.skip_dedup:
            log_orch(f"Chunk {chunk}: --skip-dedup set, skipping CLIP embed", chunk=chunk)
            mark_done(chunk, "clip_embed")
            mark_done(chunk, "clip_index")
            mark_done(chunk, "clip_dups")
            return
        if not gpu_is_free():
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "clip_embed")
            return
        log_orch(f"Chunk {chunk}: CLIP embedding", chunk=chunk)
        shard_dir  = STAGING_DIR / f"chunk{chunk}" / "shards"
        embed_dir  = STAGING_DIR / f"chunk{chunk}" / "embeddings"
        cmd = self._python_cmd("clip_dedup.py",
                               f"embed --shards '{shard_dir}' --embeddings '{embed_dir}'")
        log_file = LOG_DIR / f"clip_embed_chunk{chunk}.log"
        self._launch_prep(f"clip_embed chunk {chunk}", cmd, log_file, chunk, "clip_embed")

    def _start_or_check_clip_index(self, chunk: int) -> None:
        if is_done(chunk, "clip_index"):
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "clip_index")
            return
        log_orch(f"Chunk {chunk}: building CLIP index", chunk=chunk)
        embed_dir = STAGING_DIR / f"chunk{chunk}" / "embeddings"
        index     = DEDUP_DIR / "dedup_index.faiss"
        cmd = self._python_cmd("clip_dedup.py",
                               f"build-index --embeddings '{embed_dir}' --index '{index}'")
        log_file = LOG_DIR / f"clip_index_chunk{chunk}.log"
        self._launch_prep(f"clip_index chunk {chunk}", cmd, log_file, chunk, "clip_index")

    def _start_or_check_clip_dups(self, chunk: int) -> None:
        if is_done(chunk, "clip_dups"):
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "clip_dups")
            return
        log_orch(f"Chunk {chunk}: finding CLIP duplicates", chunk=chunk)
        index    = DEDUP_DIR / "dedup_index.faiss"
        dups     = DEDUP_DIR / "duplicate_ids.txt"
        cmd = self._python_cmd("clip_dedup.py",
                               f"find-dups --index '{index}' --out '{dups}'")
        log_file = LOG_DIR / f"clip_dups_chunk{chunk}.log"
        self._launch_prep(f"clip_dups chunk {chunk}", cmd, log_file, chunk, "clip_dups")

    def _start_or_check_precompute(self, chunk: int) -> None:
        if is_done(chunk, "precompute"):
            return
        if not gpu_is_free():
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "precompute")
            return
        if not self.res.request("GPU_TOKEN", f"precompute chunk {chunk}"):
            return
        log_orch(f"Chunk {chunk}: precomputing", chunk=chunk)
        shard_dir = STAGING_DIR / f"chunk{chunk}" / "shards"
        qwen3_out = STAGING_DIR / f"chunk{chunk}" / "precomputed" / "qwen3"
        vae_out   = STAGING_DIR / f"chunk{chunk}" / "precomputed" / "vae"
        flux_model = self.cfg.get("model", {}).get("flux_model", "flux-klein-4b")
        cmd = self._python_cmd("precompute_all.py",
                               f"--shards '{shard_dir}' "
                               f"--qwen3-output '{qwen3_out}' "
                               f"--vae-output '{vae_out}' "
                               f"--flux-model {flux_model} --workers 1")
        log_file = LOG_DIR / f"precompute_chunk{chunk}.log"
        self._launch_prep(f"precompute chunk {chunk}", cmd, log_file, chunk, "precompute",
                          on_done=lambda: self.res.release("GPU_TOKEN"))

    def _check_ready_to_promote(self, chunk: int) -> None:
        """Promote when: prev chunk's training is done (or chunk==1) and GPU is free."""
        if chunk > 1 and not is_done(chunk - 1, "train"):
            return  # wait for previous chunk's training
        if not gpu_is_free():
            return
        log_orch(f"Chunk {chunk}: promoting staging to production", chunk=chunk)
        self._promote_chunk(chunk)

    def _run_promote(self, chunk: int) -> None:
        pass  # promotion is synchronous in _check_ready_to_promote

    def _promote_chunk(self, chunk: int) -> None:
        """Move staging shards and precomputed cache to production."""
        staging = STAGING_DIR / f"chunk{chunk}"
        shard_src   = staging / "shards"
        precomp_src = staging / "precomputed"

        # Move shards with chunk prefix
        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        idx = 0
        for tar in sorted(shard_src.glob("*.tar")):
            dest = SHARDS_DIR / f"chunk{chunk}_{idx:04d}.tar"
            tar.rename(dest)
            idx += 1
        log_orch(f"Chunk {chunk}: promoted {idx} shards", chunk=chunk)

        # Merge precomputed caches
        for subdir in ["qwen3", "vae", "siglip"]:
            src = precomp_src / subdir
            if not src.exists():
                continue
            dst = PRECOMP_DIR / subdir
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                f.rename(dst / f.name)

        mark_done(chunk, "promoted")
        self._start_training(chunk)

    def _start_training(self, chunk: int) -> None:
        if tmux_window_exists(TMUX_TRAIN_WIN):
            log_orch(f"Chunk {chunk}: training window already running", chunk=chunk)
            return
        if not self.res.request("GPU_TOKEN", f"train chunk {chunk}"):
            return

        log_orch(f"Chunk {chunk}: starting training", chunk=chunk)
        training_cfg = self.cfg.get("training", {})
        steps_map = training_cfg.get("steps", {}).get(self.scale, {})
        steps = steps_map.get(chunk, steps_map.get(str(chunk), 15000))
        lr_map = training_cfg.get("lr", {})
        lr = lr_map.get(chunk, lr_map.get(str(chunk), 1e-5))

        ckpt_dir = CKPT_DIR
        resume_ckpt = ""
        if chunk > 1:
            prev_ckpt = ckpt_dir / "best.safetensors"
            if prev_ckpt.exists():
                resume_ckpt = f"--resume '{prev_ckpt}'"

        hard_ex_arg = ""
        if chunk > 1 and HARD_EX_DIR.exists() and any(HARD_EX_DIR.glob("*.tar")):
            hard_mix = training_cfg.get("hard_mix_ratio", 0.05)
            hard_ex_arg = f"--hard-example-dir '{HARD_EX_DIR}' --hard-mix-ratio {hard_mix}"

        config_file = self.cfg.get("training_config",
                                   str(TRAIN_DIR / "configs" / "stage1_512px.yaml"))
        cmd = (
            f"source '{TRAIN_DIR}/.venv/bin/activate' && "
            f"caffeinate -i python -u '{TRAIN_DIR}/train_ip_adapter.py' "
            f"--config '{config_file}' "
            f"--steps {steps} --lr {lr} "
            f"--shard-dir '{SHARDS_DIR}' "
            f"--precomputed-dir '{PRECOMP_DIR}' "
            f"--checkpoint-dir '{ckpt_dir}' "
            f"{resume_ckpt} {hard_ex_arg}"
        )
        log_file = LOG_DIR / f"train_chunk{chunk}.log"

        if not self.dry_run:
            tmux_new_window(TMUX_TRAIN_WIN, cmd, log_file)
        log_orch(f"Chunk {chunk}: training launched ({steps} steps, lr={lr})", chunk=chunk)
        update_state(**{
            "chunks": {str(chunk): {
                "state": "TRAINING",
                "started_at": now_iso(),
                "steps": steps,
                "lr": lr,
            }}
        })

    def _check_training(self, chunk: int) -> None:
        if tmux_window_exists(TMUX_TRAIN_WIN):
            return  # still running
        # Window gone — training finished or crashed
        log_file = LOG_DIR / f"train_chunk{chunk}.log"
        code = last_exit_code(log_file)
        self.res.release("GPU_TOKEN")
        if code == 0:
            log_orch(f"Chunk {chunk}: training complete", chunk=chunk)
            mark_done(chunk, "train")
            update_state(**{"chunks": {str(chunk): {"state": "MINING",
                                                     "completed_at": now_iso()}}})
        else:
            msg = f"Training chunk {chunk} exited with code {code}"
            self._handle_step_failure(chunk, "train", msg)

    def _start_or_check_mining(self, chunk: int) -> None:
        if is_done(chunk, "mine"):
            return
        if not gpu_is_free():
            self._check_prep_step(chunk, "mine")
            return
        if self._prep_busy():
            self._check_prep_step(chunk, "mine")
            return
        if not self.res.request("GPU_TOKEN", f"mine chunk {chunk}"):
            return

        log_orch(f"Chunk {chunk}: mining hard examples", chunk=chunk)
        best_ckpt = CKPT_DIR / "best.safetensors"
        if not best_ckpt.exists():
            log_orch(f"Chunk {chunk}: no best.safetensors — skipping mining", chunk=chunk)
            mark_done(chunk, "mine")
            self.res.release("GPU_TOKEN")
            return

        out = HARD_EX_DIR / f"chunk{chunk}"
        out.mkdir(parents=True, exist_ok=True)
        flux_model = self.cfg.get("model", {}).get("flux_model", "flux-klein-4b")
        qwen3_cache = PRECOMP_DIR / "qwen3"
        vae_cache   = PRECOMP_DIR / "vae"
        cmd = self._python_cmd("mine_hard_examples.py",
                               f"--checkpoint '{best_ckpt}' "
                               f"--shards '{SHARDS_DIR}' "
                               f"--qwen3-cache '{qwen3_cache}' "
                               f"--vae-cache '{vae_cache}' "
                               f"--flux-model {flux_model} --null-siglip "
                               f"--output '{out}'")
        log_file = LOG_DIR / f"mine_chunk{chunk}.log"
        self._launch_prep(f"mine chunk {chunk}", cmd, log_file, chunk, "mine",
                          on_done=lambda: self.res.release("GPU_TOKEN"))

    def _start_or_check_validation(self, chunk: int) -> None:
        if is_done(chunk, "validate"):
            return
        # Validation not yet implemented — auto-pass for now
        log_orch(f"Chunk {chunk}: validation (auto-pass — validator not yet implemented)",
                 chunk=chunk)
        mark_done(chunk, "validate")

    def _handle_error(self, chunk: int) -> None:
        # Find which step errored
        for step in CHUNK_STEPS:
            if has_error(chunk, step):
                msg = read_error(chunk, step)
                log_orch(f"Chunk {chunk}: ERROR in {step}: {msg}", level="error", chunk=chunk)
                key = (chunk, step)
                restarts = self._restart_counts.get(key, 0)
                if restarts < 1:
                    log_orch(f"Chunk {chunk}: auto-restarting {step} (attempt {restarts+1})")
                    self._restart_counts[key] = restarts + 1
                    clear_error(chunk, step)
                else:
                    issue_id = self._next_issue_id()
                    dispatch_issue(issue_id, "error",
                                   f"Step {step} chunk {chunk} failed twice — manual intervention required",
                                   chunk=chunk, process=step, context={"error": msg},
                                   suggested_action="investigate_logs")
                break

    # -----------------------------------------------------------------------
    # Prep window management
    # -----------------------------------------------------------------------

    def _prep_busy(self) -> bool:
        return tmux_window_exists(TMUX_PREP_WIN)

    def _launch_prep(self, description: str, cmd: str, log_file: Path,
                     chunk: int, step: str, on_done=None) -> None:
        if self.dry_run:
            log_orch(f"DRY RUN: would launch {description}")
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        activated = f"source '{TRAIN_DIR}/.venv/bin/activate' && {cmd}"
        tmux_new_window(TMUX_PREP_WIN, activated, log_file)
        log_orch(f"Launched: {description} → {log_file}")

    def _check_prep_step(self, chunk: int, step: str) -> None:
        """Check if the prep window just finished and record success/failure."""
        if tmux_window_exists(TMUX_PREP_WIN):
            return  # still running

        # Window gone — check exit code
        log_file = LOG_DIR / f"{step}_chunk{chunk}.log"
        code = last_exit_code(log_file)
        if code == 0:
            log_orch(f"Chunk {chunk}: {step} complete", chunk=chunk)
            mark_done(chunk, step)
        elif code is not None:
            msg = f"Exit code {code}; see {log_file}"
            log_orch(f"Chunk {chunk}: {step} FAILED — {msg}", level="error", chunk=chunk)
            self._handle_step_failure(chunk, step, msg)

    def _handle_step_failure(self, chunk: int, step: str, message: str) -> None:
        self.res.release("GPU_TOKEN")
        self.res.release("DISK_WRITE_HIGH")
        mark_error(chunk, step, message)
        notify("iris pipeline ERROR", f"{step} chunk {chunk} failed")

    # -----------------------------------------------------------------------
    # Source specification for build_shards
    # -----------------------------------------------------------------------

    def _build_sources(self, chunk: int) -> str:
        """Return space-separated source arguments for build_shards.py."""
        parts = []
        # Pre-staged sources (already chunk-specific)
        jdb_wds  = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb_wds"
        wiki_wds = STAGING_DIR / f"chunk{chunk}" / "raw" / "wikiart_wds"
        if jdb_wds.exists():
            parts.append(f"'{jdb_wds}'")
        if wiki_wds.exists():
            parts.append(f"'{wiki_wds}'")
        # Global sources — sliced per chunk via :chunk/total notation
        laion = DATA_ROOT / "raw" / "laion"
        coyo  = DATA_ROOT / "raw" / "coyo"
        if laion.exists():
            parts.append(f"'{laion}:{chunk}/{self.total_chunks}'")
        if coyo.exists():
            parts.append(f"'{coyo}:{chunk}/{self.total_chunks}'")
        return " ".join(parts)

    # -----------------------------------------------------------------------
    # Resource sync — detect GPU release when training window closes
    # -----------------------------------------------------------------------

    def _sync_resources(self) -> None:
        if not tmux_window_exists(TMUX_TRAIN_WIN):
            if self.res.holder("GPU_TOKEN") and "train" in (self.res.holder("GPU_TOKEN") or ""):
                self.res.release("GPU_TOKEN")

    # -----------------------------------------------------------------------
    # Heartbeat monitoring
    # -----------------------------------------------------------------------

    def _check_heartbeats(self) -> None:
        # Check training heartbeat when training is active
        if tmux_window_exists(TMUX_TRAIN_WIN):
            for chunk in range(1, self.total_chunks + 1):
                if derive_chunk_state(chunk) == ChunkState.TRAINING:
                    if is_heartbeat_stale("trainer", chunk):
                        age = "unknown"
                        log_orch(f"Chunk {chunk}: trainer heartbeat stale ({age}s)",
                                 level="warning", chunk=chunk)
                        dispatch_issue(self._next_issue_id(), "warning",
                                       f"Trainer heartbeat stale — process may be hung",
                                       chunk=chunk, process="trainer",
                                       suggested_action="check_training_log")

    # -----------------------------------------------------------------------
    # Disk check
    # -----------------------------------------------------------------------

    def _check_disk(self) -> None:
        gb = free_gb()
        if gb < DISK_ABORT_GB:
            log_orch(f"CRITICAL: only {gb:.1f} GB free — pausing pipeline", level="error")
            dispatch_issue(self._next_issue_id(), "error",
                           f"Disk critically low: {gb:.1f} GB free",
                           suggested_action="free_disk_space")
        elif gb < DISK_WARN_GB:
            log_orch(f"WARNING: {gb:.1f} GB free (warn threshold {DISK_WARN_GB} GB)",
                     level="warning")

    # -----------------------------------------------------------------------
    # Control signals
    # -----------------------------------------------------------------------

    def _check_control_signals(self) -> None:
        if not CONTROL_FILE.exists():
            return
        try:
            import json
            with open(CONTROL_FILE) as f:
                ctrl = json.load(f)
            action = ctrl.get("action")
            if action == "pause":
                log_orch("Operator signal: pause — orchestrator sleeping 300s")
                time.sleep(300)
            elif action == "abort":
                log_orch("Operator signal: abort — orchestrator stopping")
                sys.exit(0)
            elif action == "force-next-chunk":
                chunk = ctrl.get("chunk")
                if chunk:
                    log_orch(f"Operator signal: force-next-chunk {chunk}")
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
                subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION],
                               check=True)
            log_orch(f"Created tmux session: {TMUX_SESSION}")

    def _init_state(self) -> None:
        """Write initial state file if it doesn't exist."""
        import json
        state = {
            "schema_version": 2,
            "run_id": f"run_{now_iso()[:10].replace('-', '')}",
            "recipe": self.cfg.get("recipe", "ip_adapter_flux4b"),
            "scale": self.scale,
            "chunks": {},
            "issues": [],
        }
        existing = {}
        try:
            with open(STATE_FILE) as f:
                existing = json.load(f)
        except (FileNotFoundError, Exception):
            pass
        if not existing:
            write_state(state)

    # -----------------------------------------------------------------------
    # Doctor check
    # -----------------------------------------------------------------------

    def _run_doctor(self) -> None:
        import shutil
        import importlib.util
        issues = []

        checks = [
            ("tmux available",       lambda: shutil.which("tmux") is not None),
            ("DATA_ROOT exists",     lambda: DATA_ROOT.exists()),
            ("DATA_ROOT writable",   lambda: os.access(DATA_ROOT, os.W_OK)),
            ("venv python exists",   lambda: VENV_PYTHON.exists()),
            ("disk >= 40 GB",        lambda: free_gb() >= DISK_ABORT_GB),
            ("numpy importable",     lambda: importlib.util.find_spec("numpy") is not None),
            ("yaml importable",      lambda: importlib.util.find_spec("yaml") is not None),
        ]

        all_pass = True
        for name, check in checks:
            try:
                ok = check()
            except Exception:
                ok = False
            status = "✅" if ok else "❌"
            print(f"  {status} {name}")
            if not ok:
                issues.append(name)
                all_pass = False

        if not all_pass:
            fatal = [i for i in issues if "disk" in i or "DATA_ROOT" in i or "python" in i]
            if fatal:
                log_orch(f"Doctor: fatal checks failed: {fatal}", level="error")
                sys.exit(1)
            else:
                log_orch(f"Doctor: non-fatal warnings: {issues}", level="warning")
        else:
            log_orch("Doctor: all checks passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V2 pipeline orchestrator")
    ap.add_argument("--config",      default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--scale",       default=None, help="Override scale from config")
    ap.add_argument("--resume",      action="store_true", help="Resume from pipeline_state.json")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--skip-dedup",  action="store_true", help="Skip CLIP dedup steps")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config not found: {args.config}", file=sys.stderr)
        cfg = {}
    cfg["_config_path"] = args.config

    if args.scale:
        cfg["scale"] = args.scale

    orch = Orchestrator(cfg, dry_run=args.dry_run, skip_dedup=args.skip_dedup)
    orch.run()


if __name__ == "__main__":
    main()
