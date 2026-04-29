#!/usr/bin/env python3
"""
train/scripts/pipeline_status.py — V2 pipeline status viewer.

Usage:
    python train/scripts/pipeline_status.py           # human-readable
    python train/scripts/pipeline_status.py --json    # machine-readable JSON
    python train/scripts/pipeline_status.py --watch   # refresh every 30s
    python train/scripts/pipeline_status.py --verbose # extended log tails + full heartbeats
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
    DATA_ROOT, SENTINEL_DIR, LOG_DIR, CKPT_DIR, SHARDS_DIR, PRECOMP_DIR, STAGING_DIR,
    STATE_FILE, DISPATCH_QUEUE, TMUX_SESSION, TMUX_TRAIN_WIN, TMUX_PREP_WIN, TMUX_ORCH_WIN,
    read_state, is_done, has_error, read_error, read_heartbeat, heartbeat_age_secs,
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


def _log_tail(log_file: Path, n: int = 5) -> list:
    """Return last n non-empty, non-EXIT_CODE lines of a log file."""
    if not log_file or not log_file.exists():
        return []
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        tail = [l for l in lines if l.strip() and not l.startswith("EXIT_CODE=")]
        return tail[-n:]
    except OSError:
        return []


def _log_for_step(chunk: int, step: str) -> Path:
    """Map step name to its log file path."""
    mapping = {
        "download":     LOG_DIR / f"download_chunk{chunk}.log",
        "convert":      LOG_DIR / f"download_chunk{chunk}.log",
        "build_shards": LOG_DIR / f"build_chunk{chunk}.log",
        "filter_shards":LOG_DIR / f"filter_chunk{chunk}.log",
        "clip_embed":   LOG_DIR / f"clip_embed_chunk{chunk}.log",
        "clip_index":   LOG_DIR / f"clip_index_chunk{chunk}.log",
        "clip_dups":    LOG_DIR / f"clip_dups_chunk{chunk}.log",
        "precompute":   LOG_DIR / f"precompute_chunk{chunk}.log",
        "train":        LOG_DIR / f"train_chunk{chunk}.log",
        "training":     LOG_DIR / f"train_chunk{chunk}.log",
        "mine":         LOG_DIR / f"mine_chunk{chunk}.log",
        "validate":     LOG_DIR / f"validate_chunk{chunk}.log",
    }
    return mapping.get(step, Path(""))


def _chunk_status(chunk: int) -> dict:
    steps = {}
    errors = {}
    for step in CHUNK_STEPS:
        if is_done(chunk, step):
            steps[step] = "done"
        elif has_error(chunk, step):
            steps[step] = "error"
            errors[step] = read_error(chunk, step).strip()
        else:
            steps[step] = "pending"
    last_done = next((s for s in reversed(CHUNK_STEPS) if steps[s] == "done"), None)
    steps_done = sum(1 for s in steps.values() if s == "done")
    state = derive_chunk_state(chunk) if _HAS_ORCH else "UNKNOWN"
    return {"chunk": chunk, "state": str(state), "steps": steps,
            "errors": errors, "last_done": last_done,
            "steps_done": steps_done, "steps_total": len(CHUNK_STEPS)}


def _trainer_heartbeat(chunk: int) -> dict:
    hb = read_heartbeat("trainer", chunk)
    if hb is None:
        return {}
    age = heartbeat_age_secs("trainer", chunk)
    return {**hb, "age_secs": age, "stale": age is not None and age > 120}


def _worker_heartbeat(process: str, chunk: int) -> dict:
    hb = read_heartbeat(process, chunk)
    if hb is None:
        return {}
    age = heartbeat_age_secs(process, chunk)
    return {**hb, "age_secs": age, "stale": age is not None and age > 300}


def _read_dispatch_issues(resolved: bool = False) -> list:
    """Read open (or all) issues from dispatch_queue.jsonl.

    The file is append-only: dispatch_issue() appends new issues and
    cmd_dispatch_resolve() appends a resolve marker.  Last entry per ID wins,
    so we collect all entries and keep only the final state for each ID.
    """
    if not DISPATCH_QUEUE.exists():
        return []
    by_id: dict = {}
    try:
        for line in DISPATCH_QUEUE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                iid = entry.get("id")
                if iid:
                    by_id[iid] = entry  # last write wins
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return [e for e in by_id.values() if resolved or not e.get("resolved", False)]


_shard_count_cache: dict[Path, tuple] = {}  # path → (mtime, count)

def _count_shards(directory: Path) -> int:
    if not directory.exists():
        return 0
    try:
        mtime = directory.stat().st_mtime
    except OSError:
        return 0
    cached = _shard_count_cache.get(directory)
    if cached and cached[0] == mtime:
        return cached[1]
    count = sum(1 for f in directory.iterdir() if f.suffix == ".tar")
    _shard_count_cache[directory] = (mtime, count)
    return count


def _count_precomputed(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*.npz"))


def _staging_detail(chunk: int) -> dict:
    staging = STAGING_DIR / f"chunk{chunk}"
    shards = _count_shards(staging / "shards")
    precomp = _count_precomputed(staging / "precomputed")
    conv_dir = staging / "converted"
    conv_tars = sum(1 for f in conv_dir.rglob("*.tar")) if conv_dir.exists() else 0
    return {"shards": shards, "precomputed": precomp, "converted_tars": conv_tars}


def _active_step_for(chunk_status: dict, prep_running: bool, train_running: bool) -> str:
    steps = chunk_status["steps"]
    state = str(chunk_status["state"])
    chunk = chunk_status["chunk"]
    if state == "DONE" or state == "ERROR":
        return ""
    if state == "IDLE":
        # IDLE + prep running + fresh heartbeat = download actively in progress
        if prep_running:
            hb = _worker_heartbeat("download_convert", chunk)
            if hb and hb.get("age_secs", 9999) < 300:
                return "download"
        return ""
    if train_running and state in ("TRAINING", "MINING", "VALIDATING"):
        return state.lower()
    if prep_running:
        return next((s for s in CHUNK_STEPS if steps[s] == "pending"), "")
    return ""


def _active_heartbeat_for(step: str, chunk: int) -> dict:
    if step in ("download", "convert"):
        return _worker_heartbeat("download_convert", chunk)
    if step == "build_shards":
        return _worker_heartbeat("build_shards", chunk)
    if step == "filter_shards":
        return _worker_heartbeat("filter_shards", chunk)
    if step == "clip_embed":
        return _worker_heartbeat("clip_dedup", chunk)
    if step == "precompute":
        return _worker_heartbeat("precompute", chunk)
    if step in ("train", "training"):
        return _trainer_heartbeat(chunk)
    if step == "mine":
        return _worker_heartbeat("mine_hard_examples", chunk)
    return {}


def build_status_dict(total_chunks: int = 4) -> dict:
    state = read_state()
    tmux = _tmux_status()
    disk_gb = free_gb()
    shards = _count_shards(SHARDS_DIR)
    staging_shards = sum(
        _count_shards(STAGING_DIR / f"chunk{c}" / "shards")
        for c in range(1, total_chunks + 1)
    )
    precomp = _count_precomputed(PRECOMP_DIR)

    chunks = {}
    for c in range(1, total_chunks + 1):
        cs = _chunk_status(c)
        cs["active_step"] = _active_step_for(cs, tmux["prep"], tmux["train"])
        cs["staging"] = _staging_detail(c)
        chunks[c] = cs

    trainer_hb = {}
    active_chunk = None
    for c in range(1, total_chunks + 1):
        if str(chunks[c]["state"]) in ("TRAINING", "MINING"):
            active_chunk = c
            trainer_hb = _trainer_heartbeat(c)
            break

    orch_hb = read_heartbeat("orchestrator")
    orch_age = heartbeat_age_secs("orchestrator")

    return {
        "ts": now_iso(),
        "state_file": state,
        "tmux": tmux,
        "disk_free_gb": round(disk_gb, 1),
        "shards_production": shards,
        "shards_staging": staging_shards,
        "precomputed_records": precomp,
        "chunks": chunks,
        "active_chunk": active_chunk,
        "trainer": trainer_hb,
        "orchestrator_hb": orch_hb,
        "orchestrator_age_secs": orch_age,
        "dispatch_issues": _read_dispatch_issues(),
    }


def print_human(status: dict, verbose: bool = False) -> None:
    state = status["state_file"]
    tmux  = status["tmux"]
    disk  = status["disk_free_gb"]

    run_id = state.get("run_id", "—")
    scale  = state.get("scale", "—")
    print(f"\n{'─'*64}")
    print(f"  iris pipeline  run={run_id}  scale={scale}")
    print(f"  {now_iso()}  disk={disk:.0f} GB free")
    print(f"{'─'*64}")

    # tmux windows
    sess  = "✅" if tmux["session"] else "❌"
    train = "🟢 iris-train" if tmux["train"] else "⬜ iris-train"
    prep  = "🟢 iris-prep"  if tmux["prep"]  else "⬜ iris-prep"
    orch  = "🟢 iris-orch"  if tmux["orch"]  else "⬜ iris-orch"
    print(f"  tmux {sess}  {train}  {prep}  {orch}")

    # Orchestrator liveness
    orch_age = status.get("orchestrator_age_secs")
    if orch_age is not None:
        stale = " ⚠️ STALE" if orch_age > 120 else ""
        print(f"  orchestrator: last poll {_age_str(orch_age)} ago{stale}")
    elif tmux["orch"]:
        print(f"  orchestrator: running (no heartbeat yet)")

    # Production data summary
    shards = status["shards_production"]
    staging = status["shards_staging"]
    precomp = status["precomputed_records"]
    shard_str = f"shards={shards}" + (f" (+{staging} staging)" if staging else "")
    print(f"  {shard_str}  precomputed={precomp}")

    # Overall progress summary
    total_chunks = len(status["chunks"])
    active_c = next(
        (c for c, cs in status["chunks"].items()
         if cs["state"] not in ("DONE", "IDLE")),
        max(status["chunks"].keys()) if status["chunks"] else 1,
    )
    active_cs = status["chunks"].get(active_c, {})
    s_done  = active_cs.get("steps_done", 0)
    s_total = active_cs.get("steps_total", len(CHUNK_STEPS))
    done_chunks = sum(1 for cs in status["chunks"].values()
                      if cs["state"] == "DONE")
    active_step_name = active_cs.get("active_step") or active_cs.get("last_done") or "—"
    print(f"  Progress: chunk {active_c}/{total_chunks}  "
          f"step {s_done}/{s_total} ({active_step_name})  "
          f"[{done_chunks} chunk(s) complete]")

    # Per-chunk status
    print()
    for c, cs in status["chunks"].items():
        state_str = cs["state"]
        active = cs.get("active_step", "")
        if state_str == "IDLE" and active == "download":
            state_str = "DOWNLOADING"
        last = cs["last_done"] or "—"
        staging_d = cs.get("staging", {})
        errors = cs.get("errors", {})
        has_err = bool(errors)
        s_done_c  = cs.get("steps_done", 0)
        s_total_c = cs.get("steps_total", len(CHUNK_STEPS))

        stg_parts = []
        if staging_d.get("shards"):
            stg_parts.append(f"{staging_d['shards']} shards")
        if staging_d.get("precomputed"):
            stg_parts.append(f"{staging_d['precomputed']} precomp")
        if staging_d.get("converted_tars"):
            stg_parts.append(f"{staging_d['converted_tars']} conv-tars")
        stg_str = f"  [staging: {', '.join(stg_parts)}]" if stg_parts else ""

        err_mark = " ⚠️ ERROR" if has_err else ""
        active_mark = f"  → {active}" if active else ""
        step_prog = f"  step {s_done_c}/{s_total_c}"
        print(f"  Chunk {c}: {state_str:<16}{step_prog}  last: {last}{active_mark}{err_mark}{stg_str}")

        # Error details with clear instructions
        for step, emsg in errors.items():
            lines = [l for l in emsg.splitlines() if l.strip()]
            detail = lines[-1] if lines else "unknown error"
            sentinel = SENTINEL_DIR / f"chunk{c}" / f"{step}.error"
            print(f"    ✗ {step}: {detail}")
            print(f"      to retry: rm {sentinel}")

        # Active step progress from heartbeat
        if active:
            hb = _active_heartbeat_for(active, c)
            if not hb and active in ("training", "validating"):
                print(f"    {active}: loading... (no heartbeat yet)")
            if hb:
                age = hb.get("age_secs")
                age_str = _age_str(age) if age is not None else "?"
                stale_mark = " ⚠️ STALE" if hb.get("stale") else ""
                if active in ("training", "validating"):
                    step_n  = hb.get("step", "?")
                    total_s = hb.get("total_steps", "?")
                    loss    = hb.get("loss", "?")
                    grad    = hb.get("grad_norm")
                    eta     = hb.get("eta_sec")
                    grad_str = f"  grad={grad:.2f}" if grad is not None else ""
                    eta_str2 = f"  ETA {_age_str(eta)}" if eta else ""
                    print(f"    {active}: step {step_n}/{total_s}  loss={loss}{grad_str}{eta_str2}  hb {age_str} ago{stale_mark}")
                    mlx_act = hb.get("mlx_active_gb")
                    mlx_pk  = hb.get("mlx_peak_gb")
                    sys_avail = hb.get("mem_available_gb")
                    mem_parts = []
                    if mlx_act is not None:
                        mem_parts.append(f"mlx={mlx_act:.1f}GB active/{mlx_pk:.1f}GB peak" if mlx_pk is not None else f"mlx={mlx_act:.1f}GB")
                    if sys_avail is not None:
                        mem_parts.append(f"sys_free={sys_avail:.1f}GB")
                    if mem_parts:
                        print(f"           mem: {' | '.join(mem_parts)}")
                else:
                    done    = hb.get("done", 0)
                    total_n = hb.get("total", 0)
                    pct     = hb.get("pct", 0)
                    if active == "download":
                        phase = hb.get("phase", "jdb")
                        if phase == "wikiart_download":
                            label = f"wikiart: {done}/{total_n} parquets ({pct:.0f}%)"
                        elif phase == "wikiart_save":
                            label = f"wikiart: saving {done}/{total_n} records ({pct:.0f}%)"
                        else:
                            gb    = hb.get("in_flight_gb", 0)
                            speed = hb.get("dl_speed_mbps", 0)
                            tgz   = hb.get("current_tgz")
                            extra = ""
                            if tgz is not None:
                                speed_str = f"  {speed:.1f} MB/s" if speed > 0 else ""
                                extra = f"  tgz {tgz:03d}: {gb:.1f} GB{speed_str}"
                            label = f"{done}/{total_n} tgzs converted ({pct:.0f}%){extra}"
                        print(f"    {active}: {label}  hb {age_str} ago{stale_mark}")
                    elif active == "precompute":
                        shard   = hb.get("current_shard") or ""
                        cphase  = hb.get("current_phase") or ""
                        detail  = f"  [{shard} {cphase}]" if shard or cphase else ""
                        eta_sec = hb.get("eta_sec", 0)
                        eta_str2 = f"  ETA {int(eta_sec//3600)}h{int((eta_sec%3600)//60)}m" if eta_sec > 60 else ""
                        print(f"    {active}: {done}/{total_n} shards ({pct:.0f}%){eta_str2}{detail}  hb {age_str} ago{stale_mark}")
                    else:
                        print(f"    {active}: {done}/{total_n} ({pct:.0f}%)  hb {age_str} ago{stale_mark}")

        # Log tail for active or failed step
        show_step = active or (list(errors.keys())[0] if errors else None)
        if show_step:
            log = _log_for_step(c, show_step)
            n_lines = 10 if verbose else 4
            tail = _log_tail(log, n_lines)
            if tail:
                print(f"    [{show_step} log tail]")
                for line in tail:
                    print(f"      {line}")


    # Open dispatch issues (from dispatch_queue.jsonl — the authoritative escalation log)
    open_issues = status.get("dispatch_issues", [])
    if open_issues:
        print(f"\n  *** {len(open_issues)} unresolved issue(s) in dispatch queue:")
        n_show = len(open_issues) if verbose else min(5, len(open_issues))
        for i in open_issues[:n_show]:
            sev = i.get("severity", "?").upper()
            ts  = i.get("ts", "")[:19]
            msg = i.get("message", "")
            act = i.get("suggested_action", "")
            act_str = f"  → {act}" if act else ""
            print(f"    [{sev}] {ts}  {msg}{act_str}")
        if len(open_issues) > n_show:
            print(f"    ... and {len(open_issues) - n_show} more (use --verbose)")

    print(f"{'─'*64}\n")


def _auto_chunk_count(default: int = 4) -> int:
    state = read_state()
    if state.get("chunks"):
        return max(int(k) for k in state["chunks"])
    if SENTINEL_DIR.exists():
        n = sum(1 for d in SENTINEL_DIR.iterdir() if d.name.startswith("chunk"))
        if n:
            return n
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",      action="store_true")
    ap.add_argument("--chunks",    type=int, default=None,
                    help="Total chunks (default: auto-detect)")
    ap.add_argument("--watch",     action="store_true", help="Refresh every 30s")
    ap.add_argument("--verbose",   action="store_true", help="Extended log tails + full heartbeats")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    if args.data_root:
        import pipeline_lib
        pipeline_lib.DATA_ROOT = Path(args.data_root)

    total_chunks = args.chunks if args.chunks else _auto_chunk_count()

    def once():
        status = build_status_dict(total_chunks)
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print_human(status, verbose=args.verbose)

    if args.watch:
        while True:
            os.system("clear")
            once()
            time.sleep(30)
    else:
        once()


if __name__ == "__main__":
    main()
