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
    python train/scripts/pipeline_doctor.py --ai        # compact JSON for AI
    python train/scripts/pipeline_doctor.py --fix       # interactive remediation
    python train/scripts/pipeline_doctor.py --watch     # re-run every 60s on change
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
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, CKPT_DIR, CKPT_ARCHIVE_DIR,
    SENTINEL_DIR, LOG_DIR, SHARDS_DIR, PRECOMP_DIR,
    HARD_EX_DIR, STAGING_DIR, DEDUP_DIR, DISPATCH_QUEUE,
    HEARTBEAT_STALE_SECS, GPU_LOCK_FILE, SHARD_BLOCK,
    COLD_ROOT, COLD_PRECOMPUTE_DIR, COLD_WEIGHTS_DIR, COLD_METADATA_DIR,
    TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_STAGE_WIN,
    TMUX_ABLATION_WIN,
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


# ─────────────────────────────────────────────────────────────────────────────
# Issue accumulator
# ─────────────────────────────────────────────────────────────────────────────

class Issue:
    __slots__ = ("severity", "category", "chunk", "title", "detail", "fix", "ctx")

    def __init__(self, severity: str, category: str, title: str,
                 detail: str = "", fix: str = "", chunk: Optional[int] = None,
                 ctx: Optional[dict] = None):
        self.severity = severity   # CRITICAL | WARNING | INFO
        self.category = category
        self.chunk    = chunk
        self.title    = title
        self.detail   = detail
        self.fix      = fix        # shell command the user can run to remediate
        self.ctx      = ctx or {}  # machine-readable key-value context for --ai mode

    def to_dict(self) -> dict:
        d = {"severity": self.severity, "category": self.category,
             "title": self.title, "detail": self.detail, "fix": self.fix}
        if self.chunk is not None:
            d["chunk"] = self.chunk
        if self.ctx:
            d["context"] = self.ctx
        return d


_issues: list[Issue] = []


def _add(severity: str, category: str, title: str,
         detail: str = "", fix: str = "", chunk: Optional[int] = None,
         ctx: Optional[dict] = None) -> None:
    _issues.append(Issue(severity, category, title, detail, fix, chunk, ctx))


# ─────────────────────────────────────────────────────────────────────────────
# Shard / precomputed helpers (flat directory layout)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_shard_range(chunk: int) -> tuple[int, int]:
    """Return (start_id, end_id_exclusive) for a chunk's shard IDs."""
    return (chunk - 1) * SHARD_BLOCK, chunk * SHARD_BLOCK


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
                     chunk=chunk,
                     ctx={"shard_count": 0, "shard_id_lo": lo, "shard_id_hi": hi - 1})

            # Verify precomputed coverage for all required subdirs
            if is_done(chunk, "precompute"):
                siglip_on = cfg.get("training", {}).get("siglip", False)
                subdirs = ["qwen3", "vae"] + (["siglip"] if siglip_on else [])
                for subdir in subdirs:
                    clean, tmp = _count_precomp_for_chunk(chunk, subdir)
                    if clean == 0:
                        _add("CRITICAL", "phantom",
                             f"Chunk {chunk} precompute.done but 0 {subdir} NPZ files in production",
                             detail=f"Expected NPZ files with chunk-{chunk} shard IDs in {PRECOMP_DIR}/{subdir}/",
                             fix=f"rm {SENTINEL_DIR}/chunk{chunk}/precompute.done",
                             chunk=chunk,
                             ctx={"subdir": subdir, "clean_npz": 0, "tmp_npz": tmp})
                    elif tmp > 0:
                        _add("WARNING", "phantom",
                             f"Chunk {chunk} has {tmp} leftover .tmp.npz in precomputed/{subdir}",
                             detail="Crash artifacts from a broken atomic-write run.",
                             fix=f"find {PRECOMP_DIR}/{subdir} -name '*.tmp.npz' -delete",
                             chunk=chunk,
                             ctx={"subdir": subdir, "tmp_npz": tmp})

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
                next_chunk = chunk + 1
                # If the next chunk is already training/done, the transition checkpoint
                # was consumed successfully and may have been rotated by keep_last_n.
                # That is expected — don't raise CRITICAL.
                next_already_started = (is_done(next_chunk, "train") or
                                        heartbeat_age_secs("trainer", next_chunk) is not None)
                if next_already_started:
                    _add("INFO", "phantom",
                         f"Chunk {chunk} final checkpoint (step {expected_end:,}) rotated by keep_last_n",
                         detail=(f"Chunk {next_chunk} already started training, confirming the "
                                 f"transition was successful. Latest checkpoint: {latest:,}."),
                         chunk=chunk,
                         ctx={"expected_end_step": expected_end, "latest_ckpt_step": latest})
                else:
                    _add("CRITICAL", "phantom",
                         f"Chunk {chunk} train.done but no checkpoint near step {expected_end:,}",
                         detail=f"Expected end: {expected_end:,}, latest checkpoint: {latest}",
                         fix=f"rm {SENTINEL_DIR}/chunk{chunk}/train.done",
                         chunk=chunk,
                         ctx={"expected_end_step": expected_end, "latest_ckpt_step": latest,
                              "available_ckpt_steps": ckpt_steps})

        # ── mine.done ─────────────────────────────────────────────────────
        if is_done(chunk, "mine"):
            n_hard = _count_hard_examples_for_chunk(chunk)
            if n_hard == 0:
                hard_dir = HARD_EX_DIR / f"chunk{chunk}"
                # Hard examples from chunk N feed into chunk N+1 training.
                # If N+1 training is already done or in progress they were
                # consumed (or deliberately skipped on restart) — downgrade
                # to INFO rather than CRITICAL.
                next_chunk = chunk + 1
                next_train_done    = is_done(next_chunk, "train")
                next_train_active  = heartbeat_age_secs("trainer", next_chunk) is not None
                if next_train_done or next_train_active:
                    _add("INFO", "phantom",
                         f"Chunk {chunk} mine.done but 0 hard-example files "
                         f"(chunk {next_chunk} training {'complete' if next_train_done else 'in progress'} — no longer needed)",
                         detail=f"Hard examples in {hard_dir} are absent but chunk {next_chunk} "
                                f"training has already started so they cannot be mixed in. "
                                f"This is expected after a restart-from-chunk or manual sentinel backfill.",
                         chunk=chunk,
                         ctx={"hard_ex_count": 0, "next_chunk_trained": True})
                else:
                    _add("CRITICAL", "phantom",
                         f"Chunk {chunk} mine.done but 0 hard-example files",
                         detail=f"Expected .tar files in {hard_dir}. "
                                f"mine.done was written but the extraction produced nothing, "
                                f"or the directory was deleted after mining completed.",
                         fix=f"rm {SENTINEL_DIR}/chunk{chunk}/mine.done",
                         chunk=chunk,
                         ctx={"hard_ex_count": 0, "next_chunk_trained": False})


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
                     chunk=chunk,
                     ctx={"resume_step": resume_step, "expected_end": expected_end,
                          "chunk_base": chunk_base, "num_steps": expected_steps})

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
                     chunk=chunk,
                     ctx={"exit_code": exit_code,
                          "is_jetsam": exit_code == 137,
                          "log_tail": lines[-3:] if lines else []})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precomputed data forensics
# ─────────────────────────────────────────────────────────────────────────────

def _check_precompute_forensics(cfg: dict, chunks: list[int]) -> None:
    siglip_on = cfg.get("training", {}).get("siglip", False)
    subdirs = ["qwen3", "vae"] + (["siglip"] if siglip_on else [])

    for chunk in chunks:
        n_shards = _count_shards_for_chunk(chunk)
        if n_shards == 0:
            continue  # no shards yet, nothing to check

        lo, hi = _chunk_shard_range(chunk)

        for subdir in subdirs:
            clean, tmp = _count_precomp_for_chunk(chunk, subdir)

            if is_done(chunk, "precompute") and clean < n_shards * 0.5:
                pct = 100 * clean / n_shards if n_shards else 0
                _add("WARNING", "precompute",
                     f"Chunk {chunk} {subdir} precompute coverage low: {clean} NPZ / {n_shards} shards ({pct:.0f}%)",
                     detail=(f"Expected roughly ≥1 NPZ per shard. Leftover .tmp files: {tmp}. "
                             f"A partial run or sample-shard optimization bug could cause this."),
                     fix=f"rm {SENTINEL_DIR}/chunk{chunk}/precompute.done",
                     chunk=chunk,
                     ctx={"subdir": subdir, "clean_npz": clean, "n_shards": n_shards,
                          "coverage_pct": round(pct, 1), "tmp_npz": tmp})

            if tmp > 0:
                _add("WARNING", "precompute",
                     f"Chunk {chunk}: {tmp} orphaned .tmp.npz files in {subdir} (broken atomic write)",
                     detail="Created by np.savez during a crash. Training ignores them but they add noise.",
                     fix=f"find {PRECOMP_DIR}/{subdir} -name '*.tmp.npz' -delete",
                     chunk=chunk,
                     ctx={"subdir": subdir, "tmp_npz": tmp})

            # ── double-extension crash artifacts (pre-fix atomic write) ──
            double_tmp = []
            try:
                for f in os.listdir(PRECOMP_DIR / subdir):
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
                     f"Chunk {chunk}: {len(double_tmp)} .npz.tmp.npz files in {subdir} (pre-fix atomic write bug)",
                     detail=(f"From broken _save_npz_atomic before fix. "
                             f"The real .npz files were never written. Examples: {double_tmp[:3]}"),
                     fix=f"find {PRECOMP_DIR}/{subdir} -name '*.npz.tmp.npz' -delete",
                     chunk=chunk)


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Versioned precompute cache staleness
# ─────────────────────────────────────────────────────────────────────────────

def _check_precompute_cache_staleness() -> None:
    """Warn if any encoder's versioned cache is stale/incomplete, or if no cache exists."""
    try:
        from cache_manager import PrecomputeCache
    except ImportError:
        return

    encoders = ("qwen3", "vae", "siglip")
    for enc in encoders:
        try:
            versions = PrecomputeCache.list_versions(PRECOMP_DIR, enc)
        except Exception:
            continue
        if not versions:
            _add("INFO", "precompute",
                 f"Precompute: no versioned cache for encoder '{enc}'",
                 detail=f"No version directories found under {PRECOMP_DIR}/{enc}. "
                        "Run precompute step to populate.",
                 fix=f"pipeline_ctl start (or run precompute step manually)")
            continue
        current = [v for v in versions if v.get("current")]
        if not current:
            _add("WARNING", "precompute",
                 f"Precompute: no 'current' symlink for encoder '{enc}'",
                 detail=f"{len(versions)} version dir(s) exist under {PRECOMP_DIR}/{enc} "
                        "but none is marked current. Training will fall back to legacy flat layout.",
                 fix=f"Re-run precompute for '{enc}' to update the current symlink.",
                 ctx={"encoder": enc, "n_versions": len(versions)})
            continue
        cur = current[0]
        if not cur.get("complete"):
            _add("WARNING", "precompute",
                 f"Precompute: current version for '{enc}' is incomplete",
                 detail=f"Version {cur.get('version','?')} has complete=False in its manifest. "
                        "Training may use partial embeddings.",
                 fix=f"Re-run precompute for '{enc}' to complete the cache.",
                 ctx={"encoder": enc, "version": cur.get("version"), "records": cur.get("record_count", 0)})
        elif cur.get("record_count", 0) == 0:
            _add("WARNING", "precompute",
                 f"Precompute: current '{enc}' cache reports 0 records",
                 detail=f"Version {cur.get('version','?')} is complete but has 0 records. "
                        "Manifest may be stale.",
                 fix=f"Re-run precompute for '{enc}'.",
                 ctx={"encoder": enc, "version": cur.get("version")})


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
            next_chunk = chunk + 1
            next_already_started = (is_done(next_chunk, "train") or
                                    heartbeat_age_secs("trainer", next_chunk) is not None)
            if not next_already_started:
                _add("WARNING", "checkpoint",
                     f"Chunk {chunk}: no checkpoint near expected end step {expected_end:,}",
                     detail=f"Available steps: {all_steps}",
                     chunk=chunk)

    # ── checkpoint archive check ──────────────────────────────────────────
    # After each chunk's training completes, the orchestrator archives the final
    # checkpoint to CKPT_ARCHIVE_DIR/chunk{N}_final.{safetensors,json}.
    # If train.done exists but the archive is absent, recovery via
    # restart-from-chunk is impossible without re-training.
    for chunk in chunks:
        if not is_done(chunk, "train"):
            continue
        arch_st  = CKPT_ARCHIVE_DIR / f"chunk{chunk}_final.safetensors"
        arch_js  = CKPT_ARCHIVE_DIR / f"chunk{chunk}_final.json"
        arch_ema = CKPT_ARCHIVE_DIR / f"chunk{chunk}_final.ema.safetensors"
        missing = [f.name for f in [arch_st, arch_js] if not f.exists()]
        if missing:
            # Distinguish: archive never created (old run pre-PIPELINE-7) vs partial write
            if not arch_st.exists() and not arch_js.exists():
                _add("WARNING", "checkpoint",
                     f"Chunk {chunk}: no archive checkpoint (chunk{chunk}_final.*) in {CKPT_ARCHIVE_DIR.relative_to(DATA_ROOT)}",
                     detail="Orchestrator archives on train.done since PIPELINE-7. "
                            "This chunk pre-dates that change or the archive was deleted. "
                            "Recovery via restart-from-chunk requires manual checkpoint copy.",
                     fix=(f"# To archive manually (if step_{{}}_final checkpoint is still in CKPT_DIR):\n"
                          f"cp {CKPT_DIR}/step_*.safetensors {CKPT_ARCHIVE_DIR}/chunk{chunk}_final.safetensors\n"
                          f"cp {CKPT_DIR}/step_*.json        {CKPT_ARCHIVE_DIR}/chunk{chunk}_final.json"),
                     chunk=chunk,
                     ctx={"archive_dir": str(CKPT_ARCHIVE_DIR), "chunk": chunk})
            else:
                # Partial archive — one file present, one missing
                _add("WARNING", "checkpoint",
                     f"Chunk {chunk}: archive incomplete — missing {', '.join(missing)}",
                     detail="A partial write may have left an unusable archive. "
                            "Re-copy from CKPT_DIR or from a prior run.",
                     chunk=chunk,
                     ctx={"missing": missing})
        else:
            # Both required files present; optionally note if EMA is absent
            if not arch_ema.exists():
                _add("INFO", "checkpoint",
                     f"Chunk {chunk}: archive present but no EMA file (chunk{chunk}_final.ema.safetensors)",
                     detail="EMA weights improve mining quality. Absent if training didn't produce one.",
                     chunk=chunk)

    # ── orphaned checkpoints past max done step ───────────────────────────
    # Skip when any chunk is actively training — checkpoints beyond train.done
    # are just live snapshots from the running chunk, not orphans.
    any_training = any(
        heartbeat_age_secs("trainer", c) is not None
        and (heartbeat_age_secs("trainer", c) or float("inf")) <= HEARTBEAT_STALE_SECS
        for c in chunks
    )
    if not any_training:
        max_done_step = 0
        for chunk in chunks:
            if is_done(chunk, "train"):
                _, end = _chunk_base_and_end(chunk, steps_map)
                max_done_step = max(max_done_step, end)

        for s in all_steps:
            if s > max_done_step + 1000:
                _add("INFO", "checkpoint",
                     f"Checkpoint at step {s:,} has no matching train.done",
                     detail="Leftover from a reset or aborted run.")

    # ── checkpoint pair completeness (.json + .safetensors must both exist) ──
    # Build a map: step → extensions present (excluding EMA files)
    pair_map: dict[int, set[str]] = {}
    if CKPT_DIR.exists():
        for f in CKPT_DIR.iterdir():
            m = re.match(r"^step[_-](\d+)\.", f.name)
            if not m or ".ema." in f.name:
                continue
            step = int(m.group(1))
            ext = f.suffix  # ".json" or ".safetensors"
            pair_map.setdefault(step, set()).add(ext)

    for step, exts in pair_map.items():
        has_json = ".json" in exts
        has_st   = ".safetensors" in exts
        if has_json and not has_st:
            _add("WARNING", "checkpoint",
                 f"Checkpoint step {step:,}: .json present but .safetensors missing (incomplete write?)",
                 detail="Resume from this step will fail. Delete or let the next checkpoint overwrite it.",
                 fix=f"rm {CKPT_DIR}/step_{step:07d}.json",
                 ctx={"step": step, "has_json": True, "has_safetensors": False})
        elif has_st and not has_json:
            _add("WARNING", "checkpoint",
                 f"Checkpoint step {step:,}: .safetensors present but .json metadata missing",
                 detail="Step metadata required for resume. Delete the orphaned weights file.",
                 fix=f"rm {CKPT_DIR}/step_{step:07d}.safetensors",
                 ctx={"step": step, "has_json": False, "has_safetensors": True})


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
                liveness_ctx = {"hb_age_s": round(age), "stale_threshold_s": HEARTBEAT_STALE_SECS,
                                "win_alive": win_running}
                if win_running and window_is_this_chunk:
                    _add("CRITICAL", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat but tmux window alive (zombie?)",
                         detail=detail,
                         fix=f"tmux kill-window -t {TMUX_SESSION}:{win}",
                         chunk=chunk, ctx=liveness_ctx)
                elif win_running and not window_is_this_chunk:
                    _add("INFO", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat from prior run "
                         f"(window currently serving chunk {active_training_chunk})",
                         detail=detail,
                         chunk=chunk,
                         ctx={**liveness_ctx, "active_training_chunk": active_training_chunk})
                elif not win_running and age < HEARTBEAT_STALE_SECS * 5:
                    _add("WARNING", "liveness",
                         f"Chunk {chunk} {step}: stale heartbeat, no active tmux window",
                         detail=detail,
                         chunk=chunk, ctx=liveness_ctx)

            elif hb is None and win_running and step == "train" and active_training_chunk == chunk:
                # Only meaningful for training (window is chunk-dedicated).
                # For prep steps (mine/validate/precompute) the iris-prep window is shared
                # across all chunks, so "window alive, no heartbeat" just means queued.
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

    # ── direct run (smoke/dev) heartbeat stale while iris-train window alive ──
    hb_dir = DATA_ROOT / ".heartbeat"
    train_win_alive = sess and tmux_window_exists(TMUX_TRAIN_WIN)
    if hb_dir.exists():
        for p in hb_dir.glob("trainer_*.json"):
            if p.stem.startswith("trainer_chunk"):
                continue
            run_name = p.stem[len("trainer_"):]
            try:
                hb = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            age = heartbeat_age_secs(p.stem, None)
            if age is None or age <= HEARTBEAT_STALE_SECS:
                continue  # fresh or unreadable — no issue
            if age > 1800:
                continue  # suppress old files from prior sessions
            if train_win_alive:
                _add("WARNING", "liveness",
                     f"Direct run '{run_name}' heartbeat stale ({_fmt_age(age)}) but iris-train window alive (hung?)",
                     detail=(f"trainer_{run_name}.json last updated {_fmt_age(age)} ago. "
                             f"stale threshold: {HEARTBEAT_STALE_SECS}s. "
                             f"Attach with: tmux attach -t {TMUX_SESSION}"),
                     ctx={"run_name": run_name, "hb_age_s": round(age)})


# ─────────────────────────────────────────────────────────────────────────────
# 6a. Stager health checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_stager_health(chunks: list[int]) -> None:
    sess = tmux_session_exists()
    stage_win_alive = sess and tmux_window_exists(TMUX_STAGE_WIN)

    for chunk in chunks:
        # ── mine.done present but archive.done missing and stager not running ──
        # archive.done is written by data_stager.py after archiving training outputs.
        # If mine.done is set (training is complete and mined) but archive.done is
        # absent and the stager window is not running, the archive step was skipped
        # or crashed without leaving a .error sentinel.
        if (is_done(chunk, "mine") and
                not is_done(chunk, "archive") and
                not has_error(chunk, "archive")):
            shb = read_heartbeat("stager", chunk)
            shb_age = heartbeat_age_secs("stager", chunk)
            stager_active = (shb is not None and shb_age is not None
                             and shb_age <= HEARTBEAT_STALE_SECS
                             and shb.get("phase") == "archive")
            if not stager_active:
                _add("WARNING", "stager",
                     f"Chunk {chunk}: mine.done present but archive.done missing and stager not running",
                     detail=(f"Training outputs have not been archived to cold storage. "
                             f"If staging completed silently (no sentinel written), "
                             f"check the stager log or manually verify {CKPT_ARCHIVE_DIR}."),
                     fix=(f"# If archive was already done manually, create the sentinel:\n"
                          f"touch {SENTINEL_DIR}/chunk{chunk}/archive.done\n"
                          f"# Or restart the stager to archive now:\n"
                          f"python train/scripts/data_stager.py archive --chunk {chunk}"),
                     chunk=chunk,
                     ctx={"stager_hb_age_s": round(shb_age) if shb_age is not None else None,
                          "stage_win_alive": stage_win_alive})

        # ── stager heartbeat stale while iris-stage window is alive (hung transfer) ──
        shb = read_heartbeat("stager", chunk)
        shb_age = heartbeat_age_secs("stager", chunk)
        if (shb is not None and shb_age is not None
                and shb_age > 900
                and stage_win_alive
                and shb.get("status") == "running"):
            phase = shb.get("phase", "?")
            bytes_xfer = shb.get("bytes_transferred", 0)
            _add("WARNING", "stager",
                 f"Chunk {chunk}: stager heartbeat stale ({_fmt_age(shb_age)}) but iris-stage window alive (hung transfer?)",
                 detail=(f"Phase: {phase}. Last bytes transferred: {bytes_xfer / 1e9:.1f} GB. "
                         f"Stager heartbeats every ~60s; 900s gap with a live window suggests "
                         f"the transfer is blocked or the process is hung."),
                 fix=(f"# Check stager log, then if hung:\n"
                      f"tmux kill-window -t {TMUX_SESSION}:{TMUX_STAGE_WIN}"),
                 chunk=chunk,
                 ctx={"hb_age_s": round(shb_age), "phase": phase,
                      "bytes_transferred": bytes_xfer, "stage_win_alive": True})

        # ── stage.done gate: chunk in TRAINING state but stage.done absent ──
        # The orchestrator gates training for chunk N+1 on stage.done for chunk N.
        # If a chunk has started training (train heartbeat is fresh) but the prior
        # chunk's stage.done sentinel is missing, the gate was bypassed or the
        # sentinel was deleted — and the stager may not be running.
        if chunk > 1:
            prev = chunk - 1
            thb_age = heartbeat_age_secs("trainer", chunk)
            training_active = (thb_age is not None and thb_age <= HEARTBEAT_STALE_SECS)
            if (training_active
                    and not is_done(prev, "stage")
                    and not has_error(prev, "stage")):
                prev_shb = read_heartbeat("stager", prev)
                prev_shb_age = heartbeat_age_secs("stager", prev)
                stager_staging = (prev_shb is not None and prev_shb_age is not None
                                  and prev_shb_age <= HEARTBEAT_STALE_SECS
                                  and prev_shb.get("phase") == "stage")
                if not stager_staging:
                    _add("CRITICAL", "stager",
                         f"Chunk {chunk} training active but chunk {prev} stage.done is missing and stager not staging",
                         detail=(f"The orchestrator should gate chunk {chunk} training on chunk {prev} "
                                 f"stage.done. Either the gate was skipped or the stager crashed "
                                 f"without writing the sentinel. Precomputed data for chunk {chunk} "
                                 f"may not have been staged to the hot volume."),
                         fix=(f"# Verify hot-volume contents, then if staging was actually done:\n"
                              f"touch {SENTINEL_DIR}/chunk{prev}/stage.done\n"
                              f"# Or restart staging for chunk {prev}:\n"
                              f"python train/scripts/data_stager.py stage --chunk {prev}"),
                         chunk=chunk,
                         ctx={"prev_chunk": prev, "stage_done_missing": True,
                              "stager_hb_age_s": round(prev_shb_age) if prev_shb_age is not None else None})


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
# 7. Training anomaly detection (live heartbeat vs config thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def _check_training_anomalies(cfg: dict, chunks: list[int]) -> None:
    anomaly    = cfg.get("anomaly", {})
    loss_thr   = float(anomaly.get("loss_threshold",   2.0))
    gn_pause   = float(anomaly.get("grad_norm_pause",  50.0))
    gn_warn    = float(anomaly.get("grad_norm_warn",   10.0))
    siglip_min = float(anomaly.get("siglip_min_coverage", 90))
    siglip_on  = cfg.get("training", {}).get("siglip", False)

    for chunk in chunks:
        if is_done(chunk, "train"):
            continue
        hb  = read_heartbeat("trainer", chunk)
        age = heartbeat_age_secs("trainer", chunk)
        if hb is None or age is None or age > HEARTBEAT_STALE_SECS:
            continue

        step         = hb.get("step", 0)
        loss         = hb.get("loss")
        loss_smooth  = hb.get("loss_smooth")
        gn_smooth    = hb.get("grad_norm_smooth")
        siglip_pct   = hb.get("siglip_coverage_pct")

        if loss is not None and loss > loss_thr:
            _add("WARNING", "anomaly",
                 f"Chunk {chunk}: loss {loss:.4f} exceeds threshold {loss_thr} at step {step:,}",
                 detail=(f"loss_smooth={loss_smooth}. Orchestrator pauses after "
                         f"{anomaly.get('loss_high_steps', 100)} sustained steps above threshold."),
                 chunk=chunk,
                 ctx={"step": step, "loss": loss, "loss_smooth": loss_smooth,
                      "threshold": loss_thr})

        if gn_smooth is not None and gn_smooth > gn_pause:
            _add("WARNING", "anomaly",
                 f"Chunk {chunk}: grad_norm_smooth {gn_smooth:.2f} exceeds pause threshold {gn_pause}",
                 detail=(f"Step {step:,}. Orchestrator pauses after "
                         f"{anomaly.get('grad_spike_polls', 10)} consecutive polls above threshold."),
                 chunk=chunk,
                 ctx={"step": step, "grad_norm_smooth": gn_smooth, "pause_threshold": gn_pause})
        elif gn_smooth is not None and gn_smooth > gn_warn:
            _add("INFO", "anomaly",
                 f"Chunk {chunk}: grad_norm_smooth {gn_smooth:.2f} above warn threshold {gn_warn}",
                 chunk=chunk,
                 ctx={"step": step, "grad_norm_smooth": gn_smooth, "warn_threshold": gn_warn})

        if siglip_on and siglip_pct is not None and siglip_pct < siglip_min:
            _add("WARNING", "anomaly",
                 f"Chunk {chunk}: SigLIP coverage {siglip_pct:.1f}% below minimum {siglip_min:.0f}%",
                 detail=(f"Step {step:,}. Some shards lack precomputed SigLIP embeddings — "
                         f"image conditioning quality will be degraded for those samples."),
                 fix=(f"# After training: recheck precompute coverage for chunk {chunk}\n"
                      f"python train/scripts/pipeline_ctl.py clear-error {chunk} precompute"),
                 chunk=chunk,
                 ctx={"step": step, "siglip_pct": siglip_pct, "min_pct": siglip_min})

        # QUALITY-4: adapter conditioning health checks (only after warm-up)
        loss_cond = hb.get("loss_cond")
        loss_null = hb.get("loss_null")
        ip_scale  = hb.get("ip_scale_mean")
        if step > 1000 and loss_cond is not None and loss_null is not None and loss_null > 0:
            _gap_pct = 100 * (loss_null - loss_cond) / loss_null
            if _gap_pct < 1.0:
                _add("WARNING", "anomaly",
                     f"Chunk {chunk}: IP adapter not learning — loss_cond ≈ loss_null at step {step:,}",
                     detail=(f"loss_cond={loss_cond:.4f} loss_null={loss_null:.4f} gap={_gap_pct:+.1f}%. "
                             f"Adapter is not improving conditioned reconstruction vs unconditioned baseline. "
                             f"Check adapter.scale values and learning rate."),
                     chunk=chunk,
                     ctx={"step": step, "loss_cond": loss_cond, "loss_null": loss_null,
                          "gap_pct": round(_gap_pct, 2)})
        if step > 500 and ip_scale is not None and ip_scale < 0.05:
            _add("WARNING", "anomaly",
                 f"Chunk {chunk}: IP adapter scales near zero (mean={ip_scale:.4f}) at step {step:,}",
                 detail="Adapter scale weights have collapsed — IP conditioning has no effect on output.",
                 chunk=chunk,
                 ctx={"step": step, "ip_scale_mean": ip_scale})

        # QUALITY-1/6: cross-ref vs self-ref loss health check (only when permutation training is active)
        loss_self_ref  = hb.get("loss_self_ref")
        loss_cross_ref = hb.get("loss_cross_ref")
        if step > 1000 and loss_self_ref is not None and loss_cross_ref is not None:
            if loss_cross_ref < loss_self_ref - 0.01:
                _add("WARNING", "anomaly",
                     f"Chunk {chunk}: loss_cross_ref < loss_self_ref at step {step:,} — unexpected",
                     detail=(f"loss_self_ref={loss_self_ref:.4f} loss_cross_ref={loss_cross_ref:.4f}. "
                             f"Cross-ref batches (different style reference) should be harder than "
                             f"self-ref; this inversion suggests the model may be ignoring SigLIP "
                             f"conditioning entirely and relying on text only."),
                     chunk=chunk,
                     ctx={"step": step, "loss_self_ref": loss_self_ref,
                          "loss_cross_ref": loss_cross_ref,
                          "gap": round(loss_cross_ref - loss_self_ref, 4)})

        loader_wait = hb.get("loader_wait_pct")
        if loader_wait is not None and loader_wait > 20.0:
            sev = "WARNING" if loader_wait > 40.0 else "INFO"
            _add(sev, "anomaly",
                 f"Chunk {chunk}: loader_wait_pct {loader_wait:.1f}% — training I/O bound",
                 detail=(f"Step {step:,}. Training is spending {loader_wait:.1f}% of wall-clock "
                         f"time waiting for the data loader. Note: download/build/filter steps "
                         f"run throttled (IOPOL_THROTTLE) during training and should not cause "
                         f"this. Likely causes: precomputed cache read contention or slow 2TBSSD."),
                 chunk=chunk,
                 ctx={"step": step, "loader_wait_pct": loader_wait})


# ─────────────────────────────────────────────────────────────────────────────
# 8. Orchestrator log analysis
# ─────────────────────────────────────────────────────────────────────────────

_ORCH_TAIL = 60  # events to scan for stuck-loop detection


def _orch_log_files() -> list[Path]:
    """Return all orchestrator JSONL log files (main + per-chunk), newest-mtime first."""
    files = list(LOG_DIR.glob("orchestrator*.jsonl"))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files


def _check_orchestrator_log() -> None:
    log_files = _orch_log_files()
    if not log_files:
        return

    # Collect up to _ORCH_TAIL events across all orchestrator log files,
    # reading from the most recently modified file first so that chunk-specific
    # events (orchestrator_chunkN.jsonl) are included in staleness detection.
    raw_lines: list[str] = []
    for lf in log_files:
        try:
            raw_lines.extend(lf.read_text(errors="replace").splitlines())
        except OSError:
            pass

    # Parse last _ORCH_TAIL non-empty events, restricted to the last 24 hours.
    # Reading all files means stale per-chunk logs from prior runs would surface
    # ancient errors as if they were current — timestamp filtering prevents this.
    cutoff = time.time() - 86400  # 24h
    tail: list[dict] = []
    for line in reversed(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_str = entry.get("ts", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts.timestamp() < cutoff:
                    continue  # older than 24h — skip
            except ValueError:
                pass
        tail.append(entry)
        if len(tail) >= _ORCH_TAIL:
            break
    tail.reverse()

    if not tail:
        return

    # ── error events (deduplicated by message text) ───────────────────────
    seen_errors: set[str] = set()
    for entry in tail:
        if entry.get("event") == "error":
            msg = entry.get("message", "")[:100]
            if msg in seen_errors:
                continue
            seen_errors.add(msg)
            chunk = entry.get("chunk")
            _add("WARNING", "orchestrator",
                 f"Orchestrator error: {msg}",
                 detail=f"ts={entry.get('ts', '')}",
                 chunk=chunk,
                 ctx={"ts": entry.get("ts", ""), "message": entry.get("message", "")[:200],
                      "chunk": chunk})

    # ── stuck-loop detection: same message ≥5× in tail ────────────────────
    msg_counts: dict[str, int] = {}
    for entry in tail:
        if entry.get("event") in ("heartbeat", "poll"):
            continue
        key = entry.get("message", "")[:80]
        if key:
            msg_counts[key] = msg_counts.get(key, 0) + 1
    for msg, count in msg_counts.items():
        if count >= 15:
            _add("WARNING", "orchestrator",
                 f"Orchestrator repeating same message {count}× in last {len(tail)} events (stuck loop?)",
                 detail=f"Repeated: '{msg}'",
                 ctx={"repeated_message": msg, "count": count, "tail_size": len(tail)})
            break  # one report is enough

    # ── last-event age: use the most recently modified log file's mtime ───
    # Parsing timestamps from combined lines is unreliable when old per-chunk
    # logs are included — use filesystem mtime of the newest log file instead.
    newest_log = log_files[0]  # already sorted by mtime desc
    try:
        newest_mtime = newest_log.stat().st_mtime
        age = time.time() - newest_mtime
        if age > 600:
            _add("WARNING", "orchestrator",
                 f"Orchestrator log last written {_fmt_age(age)} (orchestrator may be down)",
                 detail=f"Most recent log file: {newest_log.name}",
                 ctx={"last_write_age_s": round(age), "log_file": newest_log.name})
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 9. Dispatch queue analysis
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

    # Drop entries whose flagged step is now sentinel-done (stale alerts from
    # a prep step that completed after the alert was fired — e.g. precompute
    # ran >6h legitimately, fired hung alerts, then finished successfully).
    def _step_now_done(entry: dict) -> bool:
        chunk = entry.get("chunk")
        process = entry.get("process", "")
        if chunk is None or not process:
            return False
        return is_done(chunk, process)

    open_issues = [e for e in open_issues if not _step_now_done(e)]

    # Semantic dedup: collapse hourly re-fires of the same alert into one.
    # Each re-fire gets a unique auto-incrementing issue ID, so the by-ID
    # dedup above keeps all N copies. Group by (chunk, process, suggested_action)
    # and keep only the latest by timestamp.
    seen: dict[tuple, dict] = {}
    for e in open_issues:
        key = (e.get("chunk"), e.get("process", ""), e.get("suggested_action", ""))
        if e.get("ts", "") > seen.get(key, {}).get("ts", ""):
            seen[key] = e
    open_issues = list(seen.values())

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
# 10. Data pipeline ordering sanity
# ─────────────────────────────────────────────────────────────────────────────

def _next_chunk_training_active(chunk: int) -> bool:
    """Return True if the chunk immediately after `chunk` is currently training."""
    age = heartbeat_age_secs("trainer", chunk + 1)
    return age is not None and age <= HEARTBEAT_STALE_SECS


# Quality mode set by --quality flag; controls ordering-issue severity and fix strategy.
_quality_mode: str = "strict"  # "strict" | "fast"


def _check_ordering_sanity(cfg: dict, chunks: list[int]) -> None:
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    ckpt_steps = _find_checkpoint_steps()

    for chunk in chunks:
        if not is_done(chunk, "train") or not is_done(chunk, "mine"):
            continue

        chunk_base, expected_end = _chunk_base_and_end(chunk, steps_map)

        # Use train.done sentinel mtime as the authoritative reference for when
        # training finished.  Comparing against checkpoint file mtimes is fragile:
        # checkpoints rotate (keep_last_n) so the chunk N final checkpoint may be
        # gone, and chunk N+1 training creates newer checkpoints that make valid
        # chunk N hard examples look stale by comparison.
        train_sent = SENTINEL_DIR / f"chunk{chunk}" / "train.done"
        train_done_mtime: Optional[float] = None
        try:
            train_done_mtime = train_sent.stat().st_mtime
        except OSError:
            pass

        mine_sent = SENTINEL_DIR / f"chunk{chunk}" / "mine.done"
        mine_done_mtime: Optional[float] = None
        try:
            mine_done_mtime = mine_sent.stat().st_mtime
        except OSError:
            pass

        # Hard examples mtime for this chunk
        hard_dir = HARD_EX_DIR / f"chunk{chunk}"
        hard_mtime: Optional[float] = None
        try:
            mtimes = [p.stat().st_mtime for p in hard_dir.iterdir()]
            hard_mtime = max(mtimes) if mtimes else None
        except OSError:
            pass

        # Flag if mine.done predates train.done — mining completed before training
        # finished, so hard examples were mined from a model that hadn't converged yet.
        if train_done_mtime is not None and mine_done_mtime is not None:
            if mine_done_mtime < train_done_mtime - 60:
                age_diff = train_done_mtime - mine_done_mtime
                hard_age_h = round((time.time() - hard_mtime) / 3600, 1) if hard_mtime else None
                ckpt_age_h = round((time.time() - train_done_mtime) / 3600, 1)
                next_chunk = chunk + 1
                next_training_active = _next_chunk_training_active(chunk)

                if next_training_active:
                    next_chunk_base, _ = _chunk_base_and_end(next_chunk, steps_map)
                    # Read current training progress for contamination estimate
                    next_hb = read_heartbeat("trainer", next_chunk) or {}
                    current_step = next_hb.get("step", next_chunk_base)
                    contaminated_steps = current_step - next_chunk_base
                    contaminated_hard = max(0, round(contaminated_steps * 0.05))

                    fix_strict = (
                        f"# STRICT: stop training, re-mine with correct model, restart clean\n"
                        f"tmux kill-window -t {TMUX_SESSION}:{TMUX_TRAIN_WIN}\n"
                        f"rm -f {SENTINEL_DIR}/chunk{next_chunk}/train.done\n"
                        f"rm -f {LOG_DIR}/train_chunk{next_chunk}.log\n"
                        f"rm -f {SENTINEL_DIR}/chunk{chunk}/mine.done\n"
                        f"rm -f {SENTINEL_DIR}/chunk{chunk}/validate.done\n"
                        f"rm -rf {hard_dir}\n"
                        f"# Orchestrator will: mine chunk {chunk} → validate → train chunk {next_chunk} from step {next_chunk_base:,}"
                    )
                    fix_fast = (
                        f"# FAST: let training finish, re-mine after — stale examples affect only {contaminated_steps} steps ({contaminated_hard} hard-example samples)\n"
                        f"rm -f {SENTINEL_DIR}/chunk{chunk}/mine.done\n"
                        f"rm -f {SENTINEL_DIR}/chunk{chunk}/validate.done\n"
                        f"rm -rf {hard_dir}\n"
                        f"# Orchestrator will re-mine chunk {chunk} after chunk {next_chunk} training completes and GPU is free"
                    )
                    fix = fix_strict if _quality_mode == "strict" else fix_fast
                    sev = "CRITICAL" if _quality_mode == "strict" else "WARNING"
                    detail = (
                        f"mine.done is {age_diff/3600:.1f}h older than train.done — "
                        f"mining completed before training finished for chunk {chunk}. "
                        f"Chunk {next_chunk} is mid-training ({contaminated_steps} steps, "
                        f"~{contaminated_hard} hard-example samples affected). "
                        f"{'STRICT: stop and restart.' if _quality_mode == 'strict' else 'FAST: let training finish, re-mine after.'}"
                    )
                    ctx_extra = {"next_chunk_training_active": True, "next_chunk": next_chunk,
                                 "next_chunk_base_step": next_chunk_base,
                                 "contaminated_steps": contaminated_steps,
                                 "contaminated_hard_samples": contaminated_hard,
                                 "quality_mode": _quality_mode,
                                 "fix_strict": fix_strict, "fix_fast": fix_fast}
                else:
                    sev = "CRITICAL"
                    fix = (
                        f"rm -f {SENTINEL_DIR}/chunk{chunk}/mine.done "
                        f"{SENTINEL_DIR}/chunk{chunk}/validate.done && "
                        f"rm -rf {hard_dir}  "
                        f"# orchestrator will re-mine then re-validate chunk {chunk}"
                    )
                    detail = (
                        f"mine.done is {age_diff/3600:.1f}h older than train.done — "
                        f"mining completed before training finished for chunk {chunk}. "
                        f"Hard examples were mined from a model that hadn't converged yet."
                    )
                    ctx_extra = {"next_chunk_training_active": False, "quality_mode": _quality_mode}

                _add(sev, "ordering",
                     f"Chunk {chunk} hard examples mined before training finished (stale model)",
                     detail=detail, fix=fix, chunk=chunk,
                     ctx={"hard_ex_age_h": hard_age_h, "train_done_age_h": ckpt_age_h,
                          "diff_h": round(age_diff / 3600, 1), **ctx_extra})

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
# 11. Error sentinel analysis
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
                 chunk=chunk,
                 ctx={"step": step, "error_content": content[:200] if content else ""})


# ─────────────────────────────────────────────────────────────────────────────
# 12. Environment and disk checks
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
# 13. Pool health

def _check_cold_storage(cfg: dict, chunks: list[int]) -> None:
    """PIPELINE-26/29: Check cold storage health — precompute versions and archive freshness."""
    if not COLD_ROOT.exists():
        return  # cold volume not mounted — skip silently

    storage = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", str(COLD_ROOT)))

    # PIPELINE-26: cold precompute version matches hot
    for encoder in ("qwen3", "vae", "siglip"):
        hot_cur  = PRECOMP_DIR / encoder / "current"
        cold_cur = cold_root / "precomputed" / encoder / "current"
        if hot_cur.is_symlink() and cold_cur.is_symlink():
            hot_ver  = os.path.basename(os.readlink(str(hot_cur)))
            cold_ver = os.path.basename(os.readlink(str(cold_cur)))
            if hot_ver != cold_ver:
                _add("WARNING", "cold_storage",
                     f"Cold precompute {encoder} version mismatch: hot={hot_ver} cold={cold_ver}",
                     detail="Run data_stager.py archive --chunk N (last completed chunk) to sync.",
                     ctx={"encoder": encoder, "hot_ver": hot_ver, "cold_ver": cold_ver})

    # PIPELINE-29: stale weights archive (train.done exists but cold weights dir empty)
    weights_dir = cold_root / "weights"
    has_any_campaign = weights_dir.exists() and any(
        d.is_dir() and d.name.startswith("flywheel-")
        for d in weights_dir.iterdir()
    ) if weights_dir.exists() else False

    any_chunk_done = any(is_done(c, "train") for c in chunks)
    done_chunks = [c for c in chunks if is_done(c, "train")]
    if any_chunk_done and not has_any_campaign:
        last_done = done_chunks[-1] if done_chunks else 1
        _add("WARNING", "cold_storage",
             "Training chunks completed but no campaign weights on cold storage",
             detail="Run data_stager.py archive to copy checkpoints to cold.",
             fix=f"train/.venv/bin/python train/scripts/data_stager.py archive --chunk {last_done}")

    # PIPELINE-29: stale metadata DB archive
    hot_shard_db = DATA_ROOT / "shard_scores.db"
    cold_shard_db = cold_root / "metadata" / "shard_scores.db"
    if hot_shard_db.exists() and cold_shard_db.exists():
        try:
            hot_mtime  = hot_shard_db.stat().st_mtime
            cold_mtime = cold_shard_db.stat().st_mtime
            if hot_mtime - cold_mtime > 48 * 3600:
                _add("INFO", "cold_storage",
                     "Cold shard_scores.db is >48h behind hot copy",
                     detail="Run archive to sync.",
                     ctx={"stale_h": round((hot_mtime - cold_mtime) / 3600, 1)})
        except OSError:
            pass


def _check_pool_health(cfg: dict) -> None:
    storage = cfg.get("storage", {})
    for key, label, sentinel in [
        ("raw_pool_root",       "raw tgz pool",       ".downloaded"),
        ("converted_pool_root", "converted tar pool", ".converted"),
    ]:
        root_str = storage.get(key)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            _add("WARNING", "pool",
                 f"{label} configured but directory missing: {root}",
                 detail=f"Set storage.{key} in config. Create dir: mkdir -p {root}/{sentinel}",
                 fix=f"mkdir -p {root}/{sentinel}",
                 ctx={"key": key, "path": str(root)})
        else:
            sentinel_dir = root / sentinel
            if not sentinel_dir.exists():
                _add("WARNING", "pool",
                     f"{label} missing sentinel dir: {sentinel_dir}",
                     fix=f"mkdir -p {sentinel_dir}",
                     ctx={"key": key, "sentinel_dir": str(sentinel_dir)})


# ─────────────────────────────────────────────────────────────────────────────
# 14. Stale log detection
# ─────────────────────────────────────────────────────────────────────────────

# Log filenames for each pipeline step (must match orchestrator.py)
_STEP_LOGS: dict[str, object] = {
    "download":     lambda c: LOG_DIR / f"download_chunk{c}.log",
    "build_shards": lambda c: LOG_DIR / f"build_chunk{c}.log",
    "filter_shards":lambda c: LOG_DIR / f"filter_chunk{c}.log",
    "clip_embed":   lambda c: LOG_DIR / f"clip_embed_chunk{c}.log",
    "clip_index":   lambda c: LOG_DIR / f"clip_index_chunk{c}.log",
    "clip_dups":    lambda c: LOG_DIR / f"clip_dups_chunk{c}.log",
    "precompute":      lambda c: LOG_DIR / f"precompute_chunk{c}.log",
    "validate_shards": lambda c: LOG_DIR / f"validate_shards_chunk{c}.log",
    "mine":            lambda c: LOG_DIR / f"mine_chunk{c}.log",
    "validate":     lambda c: LOG_DIR / f"validate_chunk{c}.log",
    "train":        lambda c: LOG_DIR / f"train_chunk{c}.log",
}


def _check_data_quality(chunks: list[int]) -> None:
    """Surface per-chunk deduplication statistics from clip_dups quality reports."""
    for chunk in chunks:
        if not is_done(chunk, "clip_dups"):
            continue
        report_path = LOG_DIR / f"clip_dups_report_chunk{chunk}.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        dedup_rate = report.get("dedup_rate_pct", 0.0)
        n_total    = report.get("n_total", 0)
        n_kept     = report.get("n_kept", n_total)
        if dedup_rate > 20:
            _add("WARNING", "data_quality",
                 f"Chunk {chunk}: high deduplication rate {dedup_rate:.1f}% "
                 f"({n_total - n_kept:,} of {n_total:,} images flagged as near-duplicates)",
                 detail="A dedup rate above 20% may indicate batch overlaps or a "
                        "data source with many near-identical images. "
                        "Check the blocklist and consider reviewing source proportions.",
                 chunk=chunk,
                 ctx={"dedup_rate_pct": dedup_rate, "n_total": n_total,
                      "n_kept": n_kept, "n_dups": n_total - n_kept})
        else:
            _add("INFO", "data_quality",
                 f"Chunk {chunk}: dedup {dedup_rate:.1f}% — {n_kept:,} of {n_total:,} images retained",
                 chunk=chunk,
                 ctx={"dedup_rate_pct": dedup_rate, "n_total": n_total, "n_kept": n_kept})


def _check_stale_logs(chunks: list[int]) -> None:
    """Flag log files whose mtime predates the corresponding .done sentinel.

    A log older than its sentinel is from a prior run of that step and will
    contain misleading information (wrong step counts, wrong timestamps, etc.).
    """
    for chunk in chunks:
        for step, log_fn in _STEP_LOGS.items():
            sent = SENTINEL_DIR / f"chunk{chunk}" / f"{step}.done"
            if not sent.exists():
                continue
            log_file = log_fn(chunk)  # type: ignore[operator]
            if not log_file.exists():
                continue
            try:
                log_mtime  = log_file.stat().st_mtime
                sent_mtime = sent.stat().st_mtime
            except OSError:
                continue
            if log_mtime < sent_mtime - 300:  # 5 min: accounts for orchestrator poll delay (60s) + buffer
                age_diff_s = sent_mtime - log_mtime
                age_diff_h = age_diff_s / 3600
                age_str = (f"{age_diff_s/60:.0f}m" if age_diff_s < 3600
                           else f"{age_diff_h:.1f}h")
                _add("INFO", "stale_log",
                     f"Chunk {chunk} {step}: log predates .done sentinel by {age_str} (prior run)",
                     detail=(f"{log_file.name} written {_fmt_age(time.time() - log_mtime)}, "
                             f"sentinel written {_fmt_age(time.time() - sent_mtime)}. "
                             f"Reading this log will show output from the wrong run."),
                     fix=f"rm {log_file}",
                     chunk=chunk,
                     ctx={"step": step,
                          "log_age_s":      round(time.time() - log_mtime),
                          "sentinel_age_s": round(time.time() - sent_mtime),
                          "age_diff_s":     round(age_diff_s)})


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ablation_health() -> None:
    """Warn if an ablation campaign is running but has hit a plateau."""
    hb = read_heartbeat("ablation")
    if hb is None:
        return
    status = hb.get("status", "")
    if "plateau" in status:
        _add("WARNING", "ablation",
             f"Ablation campaign '{hb.get('run_name','')}' stopped: {status}",
             detail=(f"n_done={hb.get('n_done')}/{hb.get('n_max')}. "
                     f"Plateau info: {hb.get('plateau', '')}. "
                     "Use --force-continue to keep exploring."),
             fix="train/.venv/bin/python train/scripts/ablation_harness.py "
                 "--config <your_config.yaml> --output-dir <dir> --force-continue")
    elif tmux_window_exists(TMUX_ABLATION_WIN):
        age = heartbeat_age_secs("ablation")
        if age is not None and age > 7200:  # 2h stale while window alive
            _add("WARNING", "ablation",
                 f"Ablation window alive but heartbeat stale ({int(age//60)}m)",
                 detail="The iris-ablation window may be hung or crashed.",
                 fix=f"tmux attach -t {TMUX_SESSION}:{TMUX_ABLATION_WIN}")


def _build_summary(cfg: dict, chunks: list[int]) -> dict:
    """Build compact machine-readable pipeline health summary for --ai mode."""
    scale = cfg.get("scale", "small")
    steps_map = cfg["training"]["steps"][scale]

    # Disk
    try:
        disk_gb = round(free_gb(DATA_ROOT), 1)
    except OSError:
        disk_gb = None

    # Chunks done
    chunks_done = sum(1 for c in chunks if is_done(c, "train"))

    # Active training chunk: find the one with the freshest trainer heartbeat
    active_chunk: Optional[int] = None
    active_hb: Optional[dict] = None
    best_age = float("inf")
    for c in chunks:
        hb = read_heartbeat("trainer", c)
        age = heartbeat_age_secs("trainer", c)
        if hb is not None and age is not None and age < best_age:
            best_age = age
            active_chunk = c
            active_hb = hb

    train_info: dict = {}
    if active_chunk is not None and active_hb is not None:
        chunk_base, expected_end = _chunk_base_and_end(active_chunk, steps_map)
        train_info = {
            "chunk": active_chunk,
            "step": active_hb.get("step"),
            "total_steps": active_hb.get("total_steps") or expected_end,
            "loss": active_hb.get("loss"),
            "loss_smooth": active_hb.get("loss_smooth"),
            "lr": active_hb.get("lr"),
            "grad_norm": active_hb.get("grad_norm"),
            "eta_h": round(active_hb["eta_sec"] / 3600, 2) if active_hb.get("eta_sec") else None,
            "steps_per_sec": active_hb.get("steps_per_sec"),
            "siglip_coverage_pct": active_hb.get("siglip_coverage_pct"),
            "loss_cond": active_hb.get("loss_cond"),
            "loss_null": active_hb.get("loss_null"),
            "loss_self_ref": active_hb.get("loss_self_ref"),
            "loss_cross_ref": active_hb.get("loss_cross_ref"),
            "ip_scale_mean": active_hb.get("ip_scale_mean"),
            "ip_scale_double": active_hb.get("ip_scale_double"),
            "ip_scale_single": active_hb.get("ip_scale_single"),
            "hb_age_s": round(best_age) if best_age < float("inf") else None,
        }

    # Active GPU-bound prep step: precompute / mine / validate
    # These hold the GPU token and can't overlap with each other or with training.
    active_prep: Optional[dict] = None
    for c in chunks:
        for step, process in [("precompute", "precompute"), ("mine", "mine_hard_examples"), ("validate", "validate")]:
            if is_done(c, step) or has_error(c, step):
                continue
            hb = read_heartbeat(process, c)
            age = heartbeat_age_secs(process, c)
            if hb is not None and age is not None and age <= HEARTBEAT_STALE_SECS:
                active_prep = {
                    "chunk": c, "step": step,
                    "done": hb.get("done"), "total": hb.get("total"),
                    "pct": hb.get("pct"),
                    "eta_h": round(hb["eta_sec"] / 3600, 2) if hb.get("eta_sec") else None,
                    "hb_age_s": round(age),
                }
                break
        if active_prep:
            break

    # Throttled parallel prep steps: download / build / filter run during training
    # under taskpolicy -d throttle (PIPELINE-3) — show as informational, not GPU-blocking.
    parallel_prep: Optional[dict] = None
    if train_info:  # only meaningful when training is active
        next_chunk = train_info["chunk"] + 1
        for step, process in [
            ("download", "download_convert"),
            ("build_shards", "build_shards"),
            ("filter_shards", "filter_shards"),
        ]:
            hb = read_heartbeat(process, next_chunk)
            age = heartbeat_age_secs(process, next_chunk)
            if hb is not None and age is not None and age <= HEARTBEAT_STALE_SECS:
                parallel_prep = {
                    "chunk": next_chunk, "step": step,
                    "done": hb.get("done"), "total": hb.get("total"),
                    "pct": hb.get("pct"),
                    "hb_age_s": round(age),
                }
                break

    # Orchestrator last poll
    orch_age: Optional[int] = None
    try:
        state = read_state()
        updated = state.get("last_updated", "")
        if updated:
            ts = datetime.fromisoformat(updated)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            orch_age = round((datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:
        pass

    # Ablation harness heartbeat (process="ablation", no chunk)
    ablation_info: Optional[dict] = None
    ablation_hb = read_heartbeat("ablation")
    if ablation_hb is not None:
        abl_age = heartbeat_age_secs("ablation")
        ablation_info = {
            "run_name":  ablation_hb.get("run_name"),
            "status":    ablation_hb.get("status"),
            "n_done":    ablation_hb.get("n_done"),
            "n_max":     ablation_hb.get("n_max"),
            "plateau":   ablation_hb.get("plateau"),
            "hb_age_s":  round(abl_age) if abl_age is not None else None,
            "window":    tmux_window_exists(TMUX_ABLATION_WIN),
        }

    # Pool coverage
    storage = cfg.get("storage", {})
    pool_info: Optional[dict] = None
    raw_pool_root  = storage.get("raw_pool_root")
    conv_pool_root = storage.get("converted_pool_root")
    if raw_pool_root or conv_pool_root:
        pool_info = {}
        if raw_pool_root:
            raw_sentinel_dir = Path(raw_pool_root) / ".downloaded"
            try:
                pool_info["raw_pool_n"] = sum(1 for _ in raw_sentinel_dir.iterdir()
                                              if _.is_file()) if raw_sentinel_dir.exists() else 0
                pool_info["raw_pool_exists"] = Path(raw_pool_root).exists()
            except OSError:
                pool_info["raw_pool_n"] = None
                pool_info["raw_pool_exists"] = False
        if conv_pool_root:
            conv_sentinel_dir = Path(conv_pool_root) / ".converted"
            try:
                pool_info["converted_pool_n"] = sum(1 for _ in conv_sentinel_dir.iterdir()
                                                    if _.is_file()) if conv_sentinel_dir.exists() else 0
                pool_info["converted_pool_exists"] = Path(conv_pool_root).exists()
            except OSError:
                pool_info["converted_pool_n"] = None
                pool_info["converted_pool_exists"] = False

    summary: dict = {
        "disk_gb": disk_gb,
        "chunks_done": chunks_done,
        "total_chunks": len(chunks),
    }
    if orch_age is not None:
        summary["orch_last_poll_s"] = orch_age
    if train_info:
        summary["training"] = train_info
    if active_prep:
        summary["prep"] = active_prep
    if parallel_prep:
        summary["parallel_prep"] = parallel_prep
    if ablation_info is not None:
        summary["ablation"] = ablation_info
    if pool_info is not None:
        summary["pool"] = pool_info

    # Cold precompute version info (PIPELINE-26)
    if COLD_ROOT.exists():
        cold_precomp: dict = {}
        for encoder in ("qwen3", "vae", "siglip"):
            enc_dir = COLD_PRECOMPUTE_DIR / encoder
            cur_link = enc_dir / "current"
            if cur_link.is_symlink():
                ver = os.path.basename(os.readlink(str(cur_link)))
                manifest_path = enc_dir / ver / "manifest.json"
                entry: dict = {"version": ver}
                try:
                    m = json.loads(manifest_path.read_text())
                    entry["complete"] = m.get("complete", False)
                    entry["records"]  = m.get("record_count", m.get("records"))
                except Exception:
                    pass
                cold_precomp[encoder] = entry
        if cold_precomp:
            summary["cold_precompute"] = cold_precomp

    return summary


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


_DOC_HTML_STYLE = """
body{font-family:system-ui,sans-serif;margin:0;background:#f0f2f5;color:#222;font-size:13px}
.hdr{background:#1a1a2e;color:#eee;padding:14px 20px;display:flex;justify-content:space-between;align-items:center}
.hdr h1{margin:0;font-size:16px;font-weight:700}.ts{font-size:11px;color:#aaa}
.card{background:#fff;border-radius:6px;padding:14px 18px;margin:10px 18px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
h2{margin:0 0 10px;font-size:11px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:.8px}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:7px 10px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#f7f7f7;font-weight:600;color:#555}tr:last-child td{border-bottom:none}
pre{background:#1e1e1e;color:#ccc;padding:10px;border-radius:4px;overflow-x:auto;font-size:11px;
    margin:4px 0;white-space:pre-wrap;word-break:break-all}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}
.ok{background:#d4edda;color:#155724}.warn{background:#fff3cd;color:#856404}
.err{background:#fde8e8;color:#a00}.info{background:#d1ecf1;color:#0c5460}
.top{border-radius:6px;padding:14px 18px;margin:10px 18px;font-size:14px;font-weight:500;
     display:flex;align-items:center;gap:10px}
.top-ok{background:#d4edda;border-left:4px solid #27ae60}
.top-warn{background:#fff3cd;border-left:4px solid #e67e22}
.top-crit{background:#fde8e8;border-left:4px solid #c0392b}
.top-icon{font-size:20px}
.mrow{display:flex;gap:20px;flex-wrap:wrap}
.metric{display:flex;flex-direction:column;gap:1px}
.mlabel{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.5px}
.mval{font-size:15px;font-weight:700;font-family:monospace}
code{background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:11px;font-family:monospace}
.sev-CRITICAL{color:#c0392b;font-weight:700}.sev-WARNING{color:#856404;font-weight:700}
.sev-INFO{color:#0c5460}.category{color:#888;font-size:11px}
"""


def _dh(s) -> str:
    import html as _hm
    return _hm.escape(str(s)) if s is not None else ""


def _dmetric(label: str, val, fmt: str = "") -> str:
    if val is None:
        return ""
    v = f"{val:{fmt}}" if fmt else str(val)
    return (f'<div class="metric"><span class="mlabel">{_dh(label)}</span>'
            f'<span class="mval">{_dh(v)}</span></div>')


def render_html_doctor(summary: dict, args_chunk: Optional[int] = None) -> str:
    """Render pipeline doctor output as a self-contained HTML page."""
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]

    criticals = [i for i in issues if i.severity == "CRITICAL"]
    warnings   = [i for i in issues if i.severity == "WARNING"]

    if criticals:
        top_action = criticals[0].title
        top_cls, top_icon = "top-crit", "🔴"
    elif warnings:
        top_action = warnings[0].title
        top_cls, top_icon = "top-warn", "🟡"
    else:
        top_action = "Pipeline healthy — no issues detected"
        top_cls, top_icon = "top-ok", "🟢"

    top_html = (
        f'<div class="top {top_cls}">'
        f'<span class="top-icon">{top_icon}</span>'
        f'<span>{_dh(top_action)}</span></div>'
    )

    # Summary card (disk, training, prep)
    ti = summary.get("training", {})
    pi = summary.get("active_prep", {})
    disk = summary.get("disk_gb")
    disk_cls = ("ok" if isinstance(disk, (int, float)) and disk > 80
                else "warn" if isinstance(disk, (int, float)) and disk > 40
                else "err")
    crit_c = len(criticals)
    warn_c = len(warnings)
    info_c = len([i for i in issues if i.severity == "INFO"])

    counts_html = (
        f'<span class="badge err" style="margin-right:4px">{crit_c} critical</span>'
        f'<span class="badge warn" style="margin-right:4px">{warn_c} warning</span>'
        f'<span class="badge info">{info_c} info</span>'
    )
    summary_metrics = (
        _dmetric("Disk free", f"{disk} GB" if disk is not None else None)
        + _dmetric("Training chunk", ti.get("chunk"))
        + _dmetric("Step", f"{ti['step']:,}/{ti.get('total_steps','?'):,}" if ti.get("step") else None)
        + _dmetric("Loss", ti.get("loss_smooth") or ti.get("loss"), fmt=".4f")
        + _dmetric("ETA", f"{ti.get('eta_h','?')}h" if ti.get("eta_h") is not None else None)
        + _dmetric("Prep step", pi.get("step"))
        + _dmetric("Prep %", f"{pi.get('pct', 0):.0f}%" if pi.get("pct") is not None else None)
    )
    summary_card = (
        f'<div class="card"><h2>Summary &nbsp; {counts_html}</h2>'
        f'<div class="mrow">{summary_metrics}</div>'
        f'<div style="margin-top:8px"><span class="badge {disk_cls}">{disk} GB free</span></div>'
        f'</div>'
    )

    # Issues table
    if issues:
        rows = ""
        for i in issues:
            sev_cls = {"CRITICAL": "err", "WARNING": "warn", "INFO": "info"}.get(i.severity, "")
            chunk_s = f" (chunk {i.chunk})" if i.chunk is not None else ""
            fix_html = (
                f'<pre>{_dh(i.fix.strip())}</pre>' if i.fix else ""
            )
            detail_html = (
                f'<div style="font-size:11px;color:#555;margin:3px 0">{_dh((i.detail or "")[:300])}</div>'
                if i.detail else ""
            )
            rows += (
                f'<tr><td><span class="badge {sev_cls}">{_dh(i.severity)}</span></td>'
                f'<td><span class="category">{_dh(i.category)}{_dh(chunk_s)}</span><br>'
                f'<b>{_dh(i.title)}</b>{detail_html}{fix_html}</td></tr>'
            )
        issues_card = (
            f'<div class="card"><h2>Issues ({len(issues)})</h2>'
            f'<table><tr><th style="width:90px">Severity</th><th>Details</th></tr>'
            f'{rows}</table></div>'
        )
    else:
        issues_card = '<div class="card"><h2>Issues</h2><p style="color:#27ae60">No issues found.</p></div>'

    ts = now_iso()
    body = (
        f'<div class="hdr"><h1>iris Pipeline Doctor</h1>'
        f'<span class="ts">{_dh(ts)}</span></div>'
        f'{top_html}{summary_card}{issues_card}'
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="60">'
        f'<title>iris Pipeline Doctor</title>'
        f'<style>{_DOC_HTML_STYLE}</style></head><body>{body}</body></html>'
    )


def print_json_output(args_chunk: Optional[int]) -> None:
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]
    print(json.dumps([i.to_dict() for i in issues], indent=2))


def print_ai_output(summary: dict, args_chunk: Optional[int]) -> None:
    """Compact JSON for AI consumption: summary block + structured issues, no prose noise."""
    issues = _issues
    if args_chunk is not None:
        issues = [i for i in issues if i.chunk is None or i.chunk == args_chunk]

    criticals = [i for i in issues if i.severity == "CRITICAL"]
    warnings   = [i for i in issues if i.severity == "WARNING"]

    # top_action: highest-priority item in one sentence
    if criticals:
        top_action = criticals[0].title
    elif warnings:
        top_action = warnings[0].title
    else:
        top_action = "pipeline healthy"

    ai_issues = []
    for i in issues:
        entry: dict = {
            "severity": i.severity,
            "category": i.category,
            "title": i.title,
            "fix": i.fix,
        }
        if i.chunk is not None:
            entry["chunk"] = i.chunk
        if i.ctx:
            entry["context"] = i.ctx
        # Omit prose detail from --ai output — context dict replaces it
        ai_issues.append(entry)

    output = {
        "ts": now_iso(),
        "summary": summary,
        "top_action": top_action,
        "issue_counts": {
            "critical": len(criticals),
            "warning": len(warnings),
            "info": len([i for i in issues if i.severity == "INFO"]),
        },
        "issues": ai_issues,
    }
    print(json.dumps(output, separators=(",", ":")))


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
    parser.add_argument("--ai", action="store_true",
                        help="Compact JSON for AI consumption: structured context, no prose/ANSI")
    parser.add_argument("--fix", action="store_true",
                        help="Interactively offer to run remediation commands")
    parser.add_argument("--quality", choices=["strict", "fast"], default="strict",
                        help=(
                            "strict (default): stale hard examples mid-training → stop, re-mine, restart. "
                            "fast: let current training finish, re-mine before next chunk uses them."
                        ))
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Path to v2_pipeline.yaml (auto-detected if omitted)")
    parser.add_argument("--html", metavar="PATH",
                        help="Write rich HTML report to PATH (use - for stdout); auto-refreshes every 60s")
    parser.add_argument("--watch", action="store_true",
                        help="Re-run every --watch-interval seconds; only print when issue set changes")
    parser.add_argument("--watch-interval", type=int, default=60, metavar="SECS",
                        help="Seconds between re-runs in --watch mode (default: 60)")
    args = parser.parse_args()

    global _quality_mode
    _quality_mode = args.quality

    cfg = load_config(args.config)
    total_chunks = cfg.get("chunks", 4)
    chunks = list(range(1, total_chunks + 1))
    if args.chunk is not None:
        chunks = [args.chunk]

    if not args.json and not args.ai and not args.html and not args.watch:
        print(f"Diagnosing chunks: {chunks}  quality={_quality_mode}")

    def _run_checks() -> None:
        global _issues
        _issues = []
        _check_environment()
        _check_error_sentinels(chunks)
        _check_phantom_completions(cfg, chunks)
        _check_training_integrity(cfg, chunks)
        _check_precompute_forensics(cfg, chunks)
        _check_precompute_cache_staleness()
        _check_checkpoint_continuity(cfg, chunks)
        _check_process_liveness(chunks)
        _check_stager_health(chunks)
        _check_code_consistency(cfg, chunks)
        _check_training_anomalies(cfg, chunks)
        _check_orchestrator_log()
        _check_dispatch_queue()
        _check_ordering_sanity(cfg, chunks)
        _check_data_quality(chunks)
        _check_stale_logs(chunks)
        _check_ablation_health()
        _check_pool_health(cfg)
        _check_cold_storage(cfg, chunks)
        _issues.sort(key=lambda i: (_SEV_ORDER.get(i.severity, 9), i.chunk or 0, i.category))

    def _issue_fingerprint() -> frozenset:
        return frozenset((i.severity, i.category, i.title, i.chunk) for i in _issues)

    if args.watch:
        interval = args.watch_interval
        last_fp: Optional[frozenset] = None
        if not args.html:
            print(f"Watching pipeline — refreshing every {interval}s (Ctrl-C to stop)")
        try:
            while True:
                _run_checks()
                fp = _issue_fingerprint()
                if fp != last_fp or args.html:
                    if args.html:
                        summary = _build_summary(cfg, chunks)
                        html = render_html_doctor(summary, args.chunk)
                        if args.html == "-":
                            print(html)
                        else:
                            Path(args.html).write_text(html)
                    else:
                        os.system("clear")
                        print(f"Diagnosing chunks: {chunks}  quality={_quality_mode}")
                        print_report(args.chunk)
                    last_fp = fp
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        return

    # ── single run ────────────────────────────────────────────────────────
    _run_checks()

    # ── output ────────────────────────────────────────────────────────────
    if args.html:
        summary = _build_summary(cfg, chunks)
        html = render_html_doctor(summary, args.chunk)
        if args.html == "-":
            print(html)
        else:
            Path(args.html).write_text(html)
            print(f"HTML written to {args.html}")
    elif args.ai:
        summary = _build_summary(cfg, chunks)
        print_ai_output(summary, args.chunk)
    elif args.json:
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
