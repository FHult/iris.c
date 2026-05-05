#!/usr/bin/env python3
"""
train/scripts/pipeline_doctor.py — Deep pipeline diagnostic tool.

Unlike pipeline_status.py (passive status reader), this script actively
investigates each claim made by sentinels, logs, and heartbeats — and flags
discrepancies. Every CRITICAL issue includes a remediation command.

Usage:
    python train/scripts/pipeline_doctor.py
    python train/scripts/pipeline_doctor.py --chunk 2
    python train/scripts/pipeline_doctor.py --json
    python train/scripts/pipeline_doctor.py --fix       # interactive remediation
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── bootstrap sys.path so we can import pipeline_lib ────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pipeline_lib import (
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, CKPT_DIR,
    SENTINEL_DIR, LOG_DIR, SHARDS_DIR, PRECOMP_DIR,
    HARD_EX_DIR, STAGING_DIR, DEDUP_DIR, DISPATCH_QUEUE,
    HEARTBEAT_STALE_SECS, GPU_LOCK_FILE,
    TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN,
    read_state, load_config,
    is_done, has_error, read_error,
    read_heartbeat, heartbeat_age_secs,
    last_exit_code, free_gb, now_iso,
    tmux_session_exists, tmux_window_exists, gpu_lock_holder,
)

# ── colour codes ─────────────────────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# Chunk N owns shard IDs [(N-1)*SHARD_BLOCK, N*SHARD_BLOCK)
_SHARD_BLOCK = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Issue accumulator
# ─────────────────────────────────────────────────────────────────────────────

class Issue:
    __slots__ = ("severity", "category", "chunk", "title", "detail", "fix")

    def __init__(self, severity: str, category: str, title: str,
                 detail: str = "", fix: str = "", chunk: Optional[int] = None):
        self.severity = severity   # CRITICAL | WARNING | INFO
        self.category = category
        self.chunk    = chunk
        self.title    = title
        self.detail   = detail
        self.fix      = fix        # shell command the user can run to remediate

    def to_dict(self) -> dict:
        d = {"severity": self.severity, "category": self.category,
             "title": self.title, "detail": self.detail, "fix": self.fix}
        if self.chunk is not None:
            d["chunk"] = self.chunk
        return d


_issues: list[Issue] = []


def _add(severity: str, category: str, title: str,
         detail: str = "", fix: str = "", chunk: Optional[int] = None) -> None:
    _issues.append(Issue(severity, category, title, detail, fix, chunk))


# ─────────────────────────────────────────────────────────────────────────────
# Shard / precomputed helpers (flat directory layout)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_shard_range(chunk: int) -> tuple[int, int]:
    """Return (start_id, end_id_exclusive) for a chunk's shard IDs."""
    return (chunk - 1) * _SHARD_BLOCK, chunk * _SHARD_BLOCK


def _count_shards_for_chunk(chunk: int) -> int:
    """Count .tar files in the flat SHARDS_DIR that belong to this chunk."""
    lo, hi = _chunk_shard_range(chunk)
    count = 0
    try:
        for f in os.listdir(SHARDS_DIR):
            if not f.endswith(".tar"):
                continue
            try:
                idx = int(Path(f).stem)
                if lo <= idx < hi:
                    count += 1
            except ValueError:
                pass
    except OSError:
        pass
    return count


def _count_precomp_for_chunk(chunk: int, subdir: str = "qwen3") -> tuple[int, int]:
    """Return (clean_npz, tmp_npz) for precomputed files belonging to this chunk."""
    lo, hi = _chunk_shard_range(chunk)
    clean = tmp = 0
    precomp_dir = PRECOMP_DIR / subdir
    try:
        for f in os.listdir(precomp_dir):
            # Names like "200000_0012.npz"
            stem = f
            is_tmp = False
            if f.endswith(".tmp.npz") or f.endswith(".npz.tmp.npz"):
                is_tmp = True
                stem = f.split(".")[0]
            elif f.endswith(".npz"):
                stem = f[:-4]
            else:
                continue
            try:
                shard_id = int(stem.split("_")[0])
                if lo <= shard_id < hi:
                    if is_tmp:
                        tmp += 1
                    else:
                        clean += 1
            except (ValueError, IndexError):
                pass
    except OSError:
        pass
    return clean, tmp


def _count_hard_examples_for_chunk(chunk: int) -> int:
    """Count hard-example .tar files in HARD_EX_DIR/chunk{N}/."""
    hard_dir = HARD_EX_DIR / f"chunk{chunk}"
    try:
        return sum(1 for f in os.listdir(hard_dir) if f.endswith(".tar"))
    except OSError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_age(secs: Optional[float]) -> str:
    if secs is None:
        return "N/A"
    if secs < 120:
        return f"{secs:.0f}s ago"
    if secs < 3600:
        return f"{secs/60:.1f}m ago"
    return f"{secs/3600:.1f}h ago"


def _read_log(log_file: Path) -> str:
    if not log_file.exists():
        return ""
    try:
        return log_file.read_text(errors="replace")
    except OSError:
        return ""


def _find_checkpoint_steps() -> list[int]:
    """Return sorted list of step numbers that have a checkpoint in CKPT_DIR.

    Checkpoints are files named step_NNNNNNNN.json (or .safetensors).
    We deduplicate by step number.
    """
    if not CKPT_DIR.exists():
        return []
    seen: set[int] = set()
    for f in CKPT_DIR.iterdir():
        m = re.match(r"^step[_-](\d+)\.", f.name)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def _latest_ckpt_step() -> Optional[int]:
    steps = _find_checkpoint_steps()
    return max(steps) if steps else None


def _chunk_base_and_end(chunk: int, steps_map: dict) -> tuple[int, int]:
    expected_steps = int(steps_map.get(chunk, steps_map.get(str(chunk), 0)))
    base = sum(
        int(steps_map.get(c, steps_map.get(str(c), 0)))
        for c in range(1, chunk)
    )
    return base, base + expected_steps


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentinel / artifact cross-check ("phantom completion")
# ─────────────────────────────────────────────────────────────────────────────

def _check_phantom_completions(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    for chunk in chunks:
        # ── promoted.done ─────────────────────────────────────────────────
        if is_done(chunk, "promoted"):
            # Shards must exist in flat dir under chunk's ID range
            n_shards = _count_shards_for_chunk(chunk)
            if n_shards == 0:
                lo, hi = _chunk_shard_range(chunk)
                _add("CRITICAL", "phantom", f"Chunk {chunk} promoted.done but NO shards in production",
                     detail=f"Expected .tar files with IDs {lo:06d}-{hi-1:06d} in {SHARDS_DIR}",
                     fix=f"rm {SENTINEL_DIR}/chunk{chunk}/promoted.done",
                     chunk=chunk)

            # Verify precomputed coverage
            if is_done(chunk, "precompute"):
                clean, tmp = _count_precomp_for_chunk(chunk)
                if clean == 0:
                    _add("CRITICAL", "phantom", f"Chunk {chunk} precompute.done but 0 NPZ files in production",
                         detail=f"Expected NPZ files with chunk-{chunk} shard IDs in {PRECOMP_DIR}/qwen3/",
                         fix=(f"rm {SENTINEL_DIR}/chunk{chunk}/precompute.done"),
                         chunk=chunk)
                elif tmp > 0:
                    _add("WARNING", "phantom", f"Chunk {chunk} has {tmp} leftover .tmp.npz files in precomputed",
                         detail="Crash artifacts from a broken atomic-write run",
                         fix=f"python -c \"import os; [os.unlink(f'{PRECOMP_DIR}/qwen3/'+f) for f in os.listdir('{PRECOMP_DIR}/qwen3/') if f.endswith('.tmp.npz')]\"",
                         chunk=chunk)

            # Check for promoted error residue
            promoted_err = SENTINEL_DIR / f"chunk{chunk}" / "promoted.error"
            if promoted_err.exists():
                _add("WARNING", "phantom", f"Chunk {chunk} has BOTH promoted.done and promoted.error",
                     detail=promoted_err.read_text(errors="replace").strip()[:300],
                     fix=f"rm {promoted_err}",
                     chunk=chunk)

        # ── train.done ────────────────────────────────────────────────────
        if is_done(chunk, "train"):
            chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)
            ckpt_steps = _find_checkpoint_steps()
            near_end = [s for s in ckpt_steps if abs(s - expected_end) <= 1000]
            if not near_end:
                latest = _latest_ckpt_step()
                _add("CRITICAL", "phantom",
                     f"Chunk {chunk} train.done but no checkpoint near step {expected_end:,}",
                     detail=f"Expected end: {expected_end:,}, latest checkpoint: {latest}",
                     fix=f"rm {SENTINEL_DIR}/chunk{chunk}/train.done",
                     chunk=chunk)

        # ── mine.done ─────────────────────────────────────────────────────
        if is_done(chunk, "mine"):
            n_hard = _count_hard_examples_for_chunk(chunk)
            if n_hard == 0:
                hard_dir = HARD_EX_DIR / f"chunk{chunk}"
                _add("CRITICAL", "phantom", f"Chunk {chunk} mine.done but 0 hard-example files",
                     detail=f"Expected .tar files in {hard_dir}",
                     fix=f"rm {SENTINEL_DIR}/chunk{chunk}/mine.done",
                     chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training integrity audit
# ─────────────────────────────────────────────────────────────────────────────

def _check_training_integrity(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    for chunk in chunks:
        log_file = LOG_DIR / f"train_chunk{chunk}.log"
        if not log_file.exists():
            continue

        text = _read_log(log_file)
        lines = text.splitlines()
        chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)
        expected_steps = expected_end - chunk_base

        # ── 0-step exit: resumed at or past the end step ──────────────────
        resume_match = re.search(r"Resuming from step ([\d,]+)", text)
        if resume_match:
            resume_step = int(resume_match.group(1).replace(",", ""))
            if resume_step >= expected_end:
                _add("CRITICAL", "training",
                     f"Chunk {chunk} training ran 0 steps (resumed at/past end)",
                     detail=(f"Resumed at step {resume_step:,}, expected end {expected_end:,}. "
                             f"chunk_base={chunk_base:,}, num_steps={expected_steps:,}. "
                             f"Likely missing --chunk-base-step or wrong base."),
                     fix=(f"rm {SENTINEL_DIR}/chunk{chunk}/train.done"),
                     chunk=chunk)

        # ── very short log + marked done ──────────────────────────────────
        if len(lines) < 20 and is_done(chunk, "train"):
            _add("WARNING", "training",
                 f"Chunk {chunk} training log very short ({len(lines)} lines) but marked done",
                 detail=f"Log: {log_file}  tail: {lines[-3:] if lines else []}",
                 chunk=chunk)

        # ── actual last step in log vs expected end ────────────────────────
        step_lines = [l for l in lines if re.search(r"step\s+\d+[,/]", l, re.I)]
        if step_lines:
            last = step_lines[-1]
            m = re.search(r"step\s+([\d,]+)", last, re.I)
            if m:
                last_step = int(m.group(1).replace(",", ""))
                if is_done(chunk, "train") and last_step < expected_end - 100:
                    _add("WARNING", "training",
                         f"Chunk {chunk} train.done but last logged step {last_step:,} < expected {expected_end:,}",
                         detail=f"Last log line: {last.strip()[:120]}",
                         chunk=chunk)

        # ── NaN loss detection ────────────────────────────────────────────
        nan_lines = [l for l in lines if "nan" in l.lower() and "loss" in l.lower()]
        if nan_lines:
            _add("CRITICAL", "training",
                 f"Chunk {chunk} training log contains NaN loss",
                 detail=f"First occurrence: {nan_lines[0].strip()[:120]}",
                 chunk=chunk)

        # ── non-zero exit code ────────────────────────────────────────────
        exit_code = last_exit_code(log_file)
        if exit_code is not None and exit_code != 0:
            if not is_done(chunk, "train"):
                _add("CRITICAL", "training",
                     f"Chunk {chunk} training exited with code {exit_code}",
                     detail=f"Log tail: {lines[-5:]}",
                     fix=f"python train/scripts/pipeline_ctl.py reset --chunk {chunk} --step train",
                     chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precomputed data forensics
# ─────────────────────────────────────────────────────────────────────────────

def _check_precompute_forensics(cfg: dict, chunks: list[int]) -> None:
    for chunk in chunks:
        n_shards = _count_shards_for_chunk(chunk)
        if n_shards == 0:
            continue  # no shards yet, nothing to check

        clean, tmp = _count_precomp_for_chunk(chunk)

        if is_done(chunk, "precompute") and clean < n_shards * 0.5:
            pct = 100 * clean / n_shards if n_shards else 0
            _add("WARNING", "precompute",
                 f"Chunk {chunk} precompute coverage low: {clean} NPZ / {n_shards} shards ({pct:.0f}%)",
                 detail=(f"Expected roughly ≥1 NPZ per shard. Leftover .tmp files: {tmp}. "
                         f"A partial run or sample-shard optimization bug could cause this."),
                 fix=(f"rm {SENTINEL_DIR}/chunk{chunk}/precompute.done"),
                 chunk=chunk)

        if tmp > 0:
            _add("WARNING", "precompute",
                 f"Chunk {chunk}: {tmp} orphaned .tmp.npz files (broken atomic write)",
                 detail=(f"Created by np.savez during a crash. "
                         f"Training ignores them (wrong extension) but they add noise."),
                 fix=(f"python -c \"import os, re; "
                      f"[os.unlink(p) for p in __import__('glob').glob('{PRECOMP_DIR}/qwen3/*.tmp.npz')]\""),
                 chunk=chunk)

        # ── double-extension crash artifacts (pre-fix atomic write) ───────
        lo, hi = _chunk_shard_range(chunk)
        double_tmp = []
        try:
            for f in os.listdir(PRECOMP_DIR / "qwen3"):
                if f.endswith(".npz.tmp.npz"):
                    try:
                        shard_id = int(f.split("_")[0])
                        if lo <= shard_id < hi:
                            double_tmp.append(f)
                    except (ValueError, IndexError):
                        pass
        except OSError:
            pass
        if double_tmp:
            _add("CRITICAL", "precompute",
                 f"Chunk {chunk}: {len(double_tmp)} .npz.tmp.npz files (pre-fix atomic write bug)",
                 detail=(f"From broken _save_npz_atomic before fix. "
                         f"The real .npz files were never written. Examples: {double_tmp[:3]}"),
                 fix=f"find {PRECOMP_DIR}/qwen3 -name '*.npz.tmp.npz' -delete",
                 chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint continuity
# ─────────────────────────────────────────────────────────────────────────────

def _check_checkpoint_continuity(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    all_steps = _find_checkpoint_steps()

    # ── per-chunk end-step checkpoint check ───────────────────────────────
    for chunk in chunks:
        if not is_done(chunk, "train"):
            continue
        chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)
        near = [s for s in all_steps if abs(s - expected_end) <= 500]
        if not near:
            _add("WARNING", "checkpoint",
                 f"Chunk {chunk}: no checkpoint near expected end step {expected_end:,}",
                 detail=f"Available steps: {all_steps}",
                 chunk=chunk)

    # ── orphaned checkpoints past max done step ───────────────────────────
    max_done_step = 0
    for chunk in chunks:
        if is_done(chunk, "train"):
            _, end = _chunk_base_and_end(chunk, steps_map)
            max_done_step = max(max_done_step, end)

    for s in all_steps:
        if s > max_done_step + 1000:
            _add("INFO", "checkpoint",
                 f"Checkpoint at step {s:,} has no matching train.done",
                 detail="In-progress run or leftover from a reset.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Process liveness — heartbeat cross-check
# ─────────────────────────────────────────────────────────────────────────────

def _check_process_liveness(chunks: list[int]) -> None:
    sess = tmux_session_exists()

    # Find which chunk (if any) currently owns the training window with a fresh heartbeat.
    # This lets us distinguish "window is running chunk X" from "window is a zombie for chunk Y".
    active_training_chunk: Optional[int] = None
    for c in chunks:
        age = heartbeat_age_secs("trainer", c)
        if age is not None and age <= HEARTBEAT_STALE_SECS:
            active_training_chunk = c
            break

    for chunk in chunks:
        for step, process, win in [
            ("precompute", "precompute", TMUX_PREP_WIN),
            ("train",      "trainer",    TMUX_TRAIN_WIN),
            ("mine",       "mining",     TMUX_PREP_WIN),
            ("validate",   "validate",   TMUX_PREP_WIN),
        ]:
            if is_done(chunk, step) or has_error(chunk, step):
                continue

            hb = read_heartbeat(process, chunk)
            age = heartbeat_age_secs(process, chunk)
            win_running = sess and tmux_window_exists(win)

            if hb is not None and age is not None and age > HEARTBEAT_STALE_SECS:
                # For training: window may be serving a *different* chunk.
                # Only call it a zombie if no other chunk has a fresh training heartbeat.
                window_is_this_chunk = (
                    step != "train" or active_training_chunk is None
                    or active_training_chunk == chunk
                )
                detail = (f"Heartbeat age: {_fmt_age(age)} "
                          f"(stale threshold: {HEARTBEAT_STALE_SECS}s). "
                          f"tmux window '{win}' present: {win_running}. "
                          + (f"Window is serving chunk {active_training_chunk}."
                             if step == "train" and active_training_chunk not in (None, chunk)
                             else ""))
                if win_running and window_is_this_chunk:
                    _add("CRITICAL", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat but tmux window alive (zombie?)",
                         detail=detail,
                         fix=f"tmux kill-window -t {TMUX_SESSION}:{win}",
                         chunk=chunk)
                elif win_running and not window_is_this_chunk:
                    # Stale hb + window alive but serving another chunk — just informational
                    _add("INFO", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat from prior run "
                         f"(window currently serving chunk {active_training_chunk})",
                         detail=detail,
                         chunk=chunk)
                elif not win_running and age < HEARTBEAT_STALE_SECS * 5:
                    _add("WARNING", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat, no active tmux window",
                         detail=detail,
                         chunk=chunk)

            elif hb is None and win_running and (step != "train" or active_training_chunk == chunk):
                _add("WARNING", "liveness",
                     f"Chunk {chunk} {step}: tmux window active but no heartbeat file",
                     detail="Process may have just started, or heartbeat path differs.",
                     chunk=chunk)

    # ── GPU lock orphan ────────────────────────────────────────────────────
    holder = gpu_lock_holder()
    if holder is None and GPU_LOCK_FILE.exists():
        _add("WARNING", "liveness",
             "GPU lock file exists but owner process is dead",
             detail=str(GPU_LOCK_FILE),
             fix=f"rm {GPU_LOCK_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Code / commit consistency
# ─────────────────────────────────────────────────────────────────────────────

def _check_code_consistency(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    for chunk in chunks:
        if chunk <= 1 or not is_done(chunk, "train"):
            continue

        log_file = LOG_DIR / f"train_chunk{chunk}.log"
        text = _read_log(log_file)
        if not text:
            continue

        # ── missing --chunk-base-step (0-step exit risk) ──────────────────
        if "--chunk-base-step" not in text:
            _add("WARNING", "code",
                 f"Chunk {chunk} training log lacks --chunk-base-step arg",
                 detail=(f"This arg prevents 0-step exits on cross-chunk warmstarts. "
                         f"If training ran before the fix (commit c526934+), it may have "
                         f"completed with 0 real steps. Verify checkpoint step matches "
                         f"expected end step."),
                 chunk=chunk)

        # ── 0-step exit without the base_step arg ─────────────────────────
        chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)
        resume_m = re.search(r"Resuming from step ([\d,]+)", text)
        if resume_m and "--chunk-base-step" not in text:
            resume_step = int(resume_m.group(1).replace(",", ""))
            if resume_step >= expected_end:
                _add("CRITICAL", "code",
                     f"Chunk {chunk}: confirmed 0-step run (pre-fix, no --chunk-base-step)",
                     detail=(f"Resumed at {resume_step:,} ≥ expected end {expected_end:,}. "
                             f"Training reported success without running any steps."),
                     fix=(f"rm {SENTINEL_DIR}/chunk{chunk}/train.done && "
                          f"# Restart orchestrator — it will pass --chunk-base-step {chunk_base}"),
                     chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dispatch queue analysis
# ─────────────────────────────────────────────────────────────────────────────

def _check_dispatch_queue() -> None:
    if not DISPATCH_QUEUE.exists():
        return

    try:
        lines = DISPATCH_QUEUE.read_text(errors="replace").splitlines()
    except OSError:
        return

    by_id: dict[str, dict] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = entry.get("id", "unknown")
        by_id[eid] = entry  # keep latest per id

    open_issues = [e for e in by_id.values() if not e.get("resolved", False)]
    if not open_issues:
        return

    for entry in open_issues:
        sev = entry.get("severity", "warning").upper()
        msg = entry.get("message", "")
        ctx = entry.get("context", {})
        chunk = entry.get("chunk")
        suggested = entry.get("suggested_action", "")
        ts = entry.get("ts", "")
        sev_out = sev if sev in ("CRITICAL", "WARNING", "INFO") else "WARNING"
        _add(sev_out, "dispatch",
             f"[dispatch:{entry.get('id','?')}] {msg}",
             detail=f"ts={ts}  context={json.dumps(ctx, separators=(',',':'))}  suggested={suggested}",
             fix=suggested,
             chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Data pipeline ordering sanity
# ─────────────────────────────────────────────────────────────────────────────

def _check_ordering_sanity(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    for chunk in chunks:
        if not is_done(chunk, "train") or not is_done(chunk, "mine"):
            continue

        chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)

        # Find the checkpoint file closest to expected_end
        ckpt_steps = _find_checkpoint_steps()
        best_step: Optional[int] = None
        best_delta = 10**9
        for s in ckpt_steps:
            delta = abs(s - expected_end)
            if delta < best_delta:
                best_delta = delta
                best_step = s

        # mtime of the closest checkpoint file (use .json sidecar)
        best_ckpt_mtime: Optional[float] = None
        if best_step is not None:
            ckpt_json = CKPT_DIR / f"step_{best_step:07d}.json"
            if not ckpt_json.exists():
                ckpt_json = CKPT_DIR / f"step_{best_step:010d}.json"
            try:
                best_ckpt_mtime = ckpt_json.stat().st_mtime
            except OSError:
                # Try any file matching the step
                for f in CKPT_DIR.iterdir():
                    if f"step_{best_step}" in f.name:
                        try:
                            best_ckpt_mtime = f.stat().st_mtime
                        except OSError:
                            pass
                        break

        # Hard examples mtime for this chunk
        hard_dir = HARD_EX_DIR / f"chunk{chunk}"
        hard_mtime: Optional[float] = None
        try:
            mtimes = [p.stat().st_mtime for p in hard_dir.iterdir()]
            hard_mtime = max(mtimes) if mtimes else None
        except OSError:
            pass

        if best_ckpt_mtime is not None and hard_mtime is not None:
            if hard_mtime < best_ckpt_mtime - 60:  # 60s grace
                age_diff = best_ckpt_mtime - hard_mtime
                _add("CRITICAL", "ordering",
                     f"Chunk {chunk} hard examples are OLDER than the training checkpoint",
                     detail=(f"Hard examples mtime: {_fmt_age(time.time() - hard_mtime)} old. "
                             f"Checkpoint (step_{best_step}) mtime: {_fmt_age(time.time() - best_ckpt_mtime)} old. "
                             f"Diff: {age_diff/3600:.1f}h. Hard examples mined from a stale model."),
                     fix=(f"rm {SENTINEL_DIR}/chunk{chunk}/mine.done && "
                          f"rm -rf {hard_dir}  # orchestrator will re-run mine step"),
                     chunk=chunk)

    # ── next chunk's hard examples must postdate current chunk's training ──
    for chunk in chunks:
        next_chunk = chunk + 1
        if next_chunk not in chunks:
            continue
        if not is_done(chunk, "train") or not is_done(next_chunk, "mine"):
            continue

        hard_next = HARD_EX_DIR / f"chunk{next_chunk}"
        hard_mtime: Optional[float] = None
        try:
            mtimes = [p.stat().st_mtime for p in hard_next.iterdir()]
            hard_mtime = max(mtimes) if mtimes else None
        except OSError:
            pass

        train_done_sent = SENTINEL_DIR / f"chunk{chunk}" / "train.done"
        if train_done_sent.exists() and hard_mtime is not None:
            train_done_mtime = train_done_sent.stat().st_mtime
            if hard_mtime < train_done_mtime - 60:
                _add("WARNING", "ordering",
                     f"Chunk {next_chunk} hard examples predate chunk {chunk} training completion",
                     detail=(f"Hard examples for chunk {next_chunk} were mined before chunk {chunk} "
                             f"training finished. They reflect stale weights."),
                     fix=(f"rm {SENTINEL_DIR}/chunk{next_chunk}/mine.done && "
                          f"rm -rf {hard_next}"),
                     chunk=next_chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Error sentinel analysis
# ─────────────────────────────────────────────────────────────────────────────

def _check_error_sentinels(chunks: list[int]) -> None:
    for chunk in chunks:
        sentinel_chunk_dir = SENTINEL_DIR / f"chunk{chunk}"
        if not sentinel_chunk_dir.exists():
            continue
        for err_file in sentinel_chunk_dir.glob("*.error"):
            step = err_file.stem
            content = ""
            try:
                content = err_file.read_text(errors="replace").strip()
            except OSError:
                pass
            # Skip if .done also exists — that's already flagged in phantom check
            done_file = sentinel_chunk_dir / f"{step}.done"
            if done_file.exists():
                continue
            _add("CRITICAL", "error_sentinel",
                 f"Chunk {chunk} step '{step}' has .error sentinel (no .done)",
                 detail=content[:300] if content else "(empty)",
                 fix=(f"cat {err_file}  # then: rm {err_file} to clear and retry"),
                 chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Environment and disk checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_environment() -> None:
    # ── disk space ────────────────────────────────────────────────────────
    try:
        gb = free_gb(DATA_ROOT)
        if gb < 40:
            _add("CRITICAL", "environment",
                 f"Disk space critically low: {gb:.1f} GB free",
                 detail=f"Pipeline aborts below 40 GB. Path: {DATA_ROOT}",
                 fix="du -sh /Volumes/2TBSSD/* | sort -h")
        elif gb < 80:
            _add("WARNING", "environment",
                 f"Disk space low: {gb:.1f} GB free",
                 detail=f"Pipeline warns below 80 GB. Path: {DATA_ROOT}")
    except OSError:
        _add("WARNING", "environment",
             f"Could not read disk space for {DATA_ROOT}",
             detail="DATA_ROOT may not be mounted.")

    # ── venv ──────────────────────────────────────────────────────────────
    venv_python = TRAIN_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        _add("WARNING", "environment",
             "train/.venv/bin/python not found",
             detail="Pipeline scripts require the venv. cd train && python -m venv .venv && pip install -r requirements.txt")

    # ── pipeline_state.json staleness ─────────────────────────────────────
    state_file = DATA_ROOT / "pipeline_state.json"
    if not state_file.exists():
        _add("INFO", "environment", "No pipeline_state.json — orchestrator not yet run")
    else:
        try:
            state = read_state()
            updated = state.get("last_updated", "")
            if updated:
                ts = datetime.fromisoformat(updated)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > 3600:
                    _add("INFO", "environment",
                         f"pipeline_state.json not updated in {_fmt_age(age)}",
                         detail="Orchestrator may not be running.")
        except Exception:
            _add("WARNING", "environment",
                 "pipeline_state.json unreadable or corrupt",
                 fix=f"rm {DATA_ROOT}/pipeline_state.json")


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_pipeline_state() -> str:
    if not tmux_session_exists():
        return "no tmux session"
    parts = []
    for win, label in [(TMUX_TRAIN_WIN, "training"), (TMUX_PREP_WIN, "prep")]:
        if tmux_window_exists(win):
            parts.append(label)
    return "running: " + ", ".join(parts) if parts else "idle (no active windows)"


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

_SEV_COLOR = {"CRITICAL": _RED, "WARNING": _YELLOW, "INFO": _CYAN}
_SEV_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


def print_report(args_chunk: Optional[int]) -> None:
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]

    criticals = [i for i in issues if i.severity == "CRITICAL"]
    warnings   = [i for i in issues if i.severity == "WARNING"]
    infos      = [i for i in issues if i.severity == "INFO"]

    pipeline_state = _current_pipeline_state()

    print(f"\n{_BOLD}Pipeline Doctor Report{_RESET}  —  {now_iso()}")
    print(f"Pipeline: {pipeline_state}")
    print(f"Issues: {_RED}{len(criticals)} CRITICAL{_RESET}  "
          f"{_YELLOW}{len(warnings)} WARNING{_RESET}  "
          f"{_CYAN}{len(infos)} INFO{_RESET}\n")

    if not issues:
        print(f"{_GREEN}No issues found.{_RESET}")
        return

    for sev in ("CRITICAL", "WARNING", "INFO"):
        group = [i for i in issues if i.severity == sev]
        if not group:
            continue
        col = _SEV_COLOR[sev]
        bar = "─" * max(1, 60 - len(sev))
        print(f"{_BOLD}{col}── {sev} {bar}{_RESET}")
        for issue in group:
            chunk_str = f"[chunk {issue.chunk}] " if issue.chunk is not None else ""
            print(f"\n  {col}{_BOLD}{chunk_str}{issue.title}{_RESET}")
            if issue.detail:
                for line in issue.detail.splitlines():
                    print(f"    {line}")
            if issue.fix:
                print(f"    {_CYAN}Fix:{_RESET}")
                for line in issue.fix.splitlines():
                    print(f"      $ {line}")
        print()


def print_json_output(args_chunk: Optional[int]) -> None:
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]
    print(json.dumps([i.to_dict() for i in issues], indent=2))


def run_fix_mode(args_chunk: Optional[int]) -> None:
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]
    actionable = [i for i in issues if i.fix and i.severity in ("CRITICAL", "WARNING")]
    if not actionable:
        print("No actionable issues with fix commands.")
        return

    for issue in actionable:
        col = _SEV_COLOR[issue.severity]
        chunk_str = f"[chunk {issue.chunk}] " if issue.chunk is not None else ""
        print(f"\n{col}{_BOLD}{issue.severity}: {chunk_str}{issue.title}{_RESET}")
        if issue.detail:
            print(f"  {issue.detail[:200]}")
        print(f"\n  Proposed fix:")
        for line in issue.fix.splitlines():
            print(f"    $ {line}")
        ans = input("\n  Run this fix? [y/N] ").strip().lower()
        if ans == "y":
            for cmd in issue.fix.splitlines():
                cmd = cmd.strip()
                if not cmd or cmd.startswith("#"):
                    continue
                print(f"  Running: {cmd}")
                ret = subprocess.run(cmd, shell=True)
                if ret.returncode != 0:
                    print(f"  Command exited {ret.returncode}")
        else:
            print("  Skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep pipeline diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--chunk", type=int, default=None, metavar="N",
                        help="Restrict diagnosis to a single chunk")
    parser.add_argument("--json", action="store_true",
                        help="Output issues as JSON array")
    parser.add_argument("--fix", action="store_true",
                        help="Interactively offer to run remediation commands")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Path to v2_pipeline.yaml (auto-detected if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_chunks = cfg.get("chunks", 4)
    chunks = list(range(1, total_chunks + 1))
    if args.chunk is not None:
        chunks = [args.chunk]

    if not args.json:
        print(f"Diagnosing chunks: {chunks}")

    # ── run all checks ────────────────────────────────────────────────────
    _check_environment()
    _check_error_sentinels(chunks)
    _check_phantom_completions(cfg, chunks)
    _check_training_integrity(cfg, chunks)
    _check_precompute_forensics(cfg, chunks)
    _check_checkpoint_continuity(cfg, chunks)
    _check_process_liveness(chunks)
    _check_code_consistency(cfg, chunks)
    _check_dispatch_queue()
    _check_ordering_sanity(cfg, chunks)

    # Sort: CRITICAL first, then WARNING, INFO; within each group by chunk
    _issues.sort(key=lambda i: (_SEV_ORDER.get(i.severity, 9), i.chunk or 0, i.category))

    # ── output ────────────────────────────────────────────────────────────
    if args.json:
        print_json_output(args.chunk)
    elif args.fix:
        print_report(args.chunk)
        run_fix_mode(args.chunk)
    else:
        print_report(args.chunk)

    has_critical = any(i.severity == "CRITICAL" for i in _issues
                       if args.chunk is None or i.chunk is None or i.chunk == args.chunk)
    sys.exit(1 if has_critical else 0)


if __name__ == "__main__":
    main()
