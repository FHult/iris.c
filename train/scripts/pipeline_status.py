#!/usr/bin/env python3
"""
train/scripts/pipeline_status.py — V2 pipeline status viewer.

Usage:
    python train/scripts/pipeline_status.py           # human-readable
    python train/scripts/pipeline_status.py --json    # machine-readable JSON
    python train/scripts/pipeline_status.py --watch   # refresh every 30s
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, SENTINEL_DIR, LOG_DIR, CKPT_DIR, SHARDS_DIR, PRECOMP_DIR,
    STATE_FILE, TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_ORCH_WIN,
    read_state, is_done, has_error, read_heartbeat, heartbeat_age_secs,
    tmux_session_exists, tmux_window_exists, free_gb, now_iso,
)

try:
    from orchestrator import CHUNK_STEPS, derive_chunk_state, ChunkState
    _HAS_ORCH = True
except ImportError:
    _HAS_ORCH = False
    CHUNK_STEPS = [
        "download", "convert", "build_shards", "filter_shards",
        "clip_embed", "clip_index", "clip_dups", "precompute",
        "promoted", "train", "mine", "validate",
    ]

    def derive_chunk_state(chunk):
        return "UNKNOWN"

    class ChunkState:
        DONE = "DONE"


def _age_str(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs/60:.0f}m"
    else:
        return f"{secs/3600:.1f}h"


def _tmux_status() -> dict:
    result = {"session": False, "train": False, "prep": False, "orch": False}
    result["session"] = tmux_session_exists()
    if result["session"]:
        result["train"] = tmux_window_exists(TMUX_TRAIN_WIN)
        result["prep"]  = tmux_window_exists(TMUX_PREP_WIN)
        result["orch"]  = tmux_window_exists(TMUX_ORCH_WIN)
    return result


def _chunk_status(chunk: int) -> dict:
    steps = {}
    for step in CHUNK_STEPS:
        if is_done(chunk, step):
            steps[step] = "done"
        elif has_error(chunk, step):
            steps[step] = "error"
        else:
            steps[step] = "pending"
    last_done = next((s for s in reversed(CHUNK_STEPS) if steps[s] == "done"), None)
    state = derive_chunk_state(chunk) if _HAS_ORCH else "UNKNOWN"
    return {"chunk": chunk, "state": str(state), "steps": steps, "last_done": last_done}


def _trainer_heartbeat(chunk: int) -> dict:
    hb = read_heartbeat("trainer", chunk)
    if hb is None:
        return {}
    age = heartbeat_age_secs("trainer", chunk)
    return {**hb, "age_secs": age, "stale": age is not None and age > 120}


def _count_shards(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix == ".tar")


def _count_precomputed(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*.npz"))


def build_status_dict(total_chunks: int = 4) -> dict:
    state = read_state()
    tmux = _tmux_status()
    disk_gb = free_gb()
    shards = _count_shards(SHARDS_DIR)
    precomp = _count_precomputed(PRECOMP_DIR)

    chunks = {}
    for c in range(1, total_chunks + 1):
        chunks[c] = _chunk_status(c)

    # Find active training chunk and heartbeat
    trainer_hb = {}
    active_chunk = None
    for c in range(1, total_chunks + 1):
        if str(chunks[c]["state"]) in ("TRAINING", "MINING"):
            active_chunk = c
            trainer_hb = _trainer_heartbeat(c)
            break

    return {
        "ts": now_iso(),
        "state_file": state,
        "tmux": tmux,
        "disk_free_gb": round(disk_gb, 1),
        "shards_production": shards,
        "precomputed_records": precomp,
        "chunks": chunks,
        "active_chunk": active_chunk,
        "trainer": trainer_hb,
    }


def print_human(status: dict) -> None:
    state = status["state_file"]
    tmux  = status["tmux"]
    disk  = status["disk_free_gb"]

    # Header
    run_id = state.get("run_id", "—")
    scale  = state.get("scale", "—")
    print(f"\n{'─'*60}")
    print(f"  iris pipeline  run={run_id}  scale={scale}")
    print(f"  {now_iso()}  disk={disk:.0f} GB free")
    print(f"{'─'*60}")

    # tmux windows
    sess = "✅" if tmux["session"] else "❌"
    train = "🟢 iris-train" if tmux["train"] else "⬜ iris-train"
    prep  = "🟢 iris-prep"  if tmux["prep"]  else "⬜ iris-prep"
    orch  = "🟢 iris-orch"  if tmux["orch"]  else "⬜ iris-orch"
    print(f"  tmux {sess}  {train}  {prep}  {orch}")

    # Production data
    print(f"  shards={status['shards_production']}  precomputed={status['precomputed_records']}")

    # Per-chunk status
    print()
    for c, cs in status["chunks"].items():
        state_str = cs["state"].split(".")[-1]  # strip ChunkState. prefix if present
        last = cs["last_done"] or "—"
        has_err = any(v == "error" for v in cs["steps"].values())
        err_mark = " ⚠️ ERROR" if has_err else ""
        print(f"  Chunk {c}: {state_str:<16}  last done: {last}{err_mark}")

    # Trainer heartbeat
    hb = status.get("trainer", {})
    if hb:
        step = hb.get("step", "?")
        total = hb.get("total_steps", "?")
        loss  = hb.get("loss", "?")
        eta   = hb.get("eta_sec")
        age   = hb.get("age_secs")
        eta_str = _age_str(eta) if eta else "?"
        age_str = _age_str(age) if age else "?"
        stale_mark = " ⚠️ STALE" if hb.get("stale") else ""
        print(f"\n  Training: step {step}/{total}  loss {loss}  ETA {eta_str}"
              f"  heartbeat {age_str} ago{stale_mark}")

    # Dispatch issues
    issues = state.get("issues", [])
    open_issues = [i for i in issues if not i.get("resolved")]
    if open_issues:
        print(f"\n  ⚠️  {len(open_issues)} open issue(s) in dispatch_queue.jsonl")

    print(f"{'─'*60}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",        action="store_true")
    ap.add_argument("--chunks",      type=int, default=4)
    ap.add_argument("--watch",       action="store_true", help="Refresh every 30s")
    ap.add_argument("--data-root",   default=None)
    args = ap.parse_args()

    if args.data_root:
        import pipeline_lib
        pipeline_lib.DATA_ROOT = Path(args.data_root)

    def once():
        status = build_status_dict(args.chunks)
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print_human(status)

    if args.watch:
        while True:
            os.system("clear")
            once()
            time.sleep(30)
    else:
        once()


if __name__ == "__main__":
    main()
