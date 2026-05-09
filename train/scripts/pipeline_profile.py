"""
train/scripts/pipeline_profile.py — End-to-end pipeline stage profiler.

Aggregates wall-clock time per stage per chunk using:
  - Orchestrator JSONL launch events  → step start times
  - Sentinel file mtimes              → step end times
  - Trainer heartbeats                → steps/sec for training stage

Usage:
    train/.venv/bin/python train/scripts/pipeline_profile.py
    train/.venv/bin/python train/scripts/pipeline_profile.py --ai
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline_lib import (
    DATA_ROOT, LOG_DIR, SENTINEL_DIR,
)

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

# Sentinel name → human label + phase group
STAGES = [
    ("download",        "download",    "data"),
    ("convert",         "convert",     "data"),
    ("build_shards",    "build",       "data"),
    ("filter_shards",   "filter",      "data"),
    ("clip_embed",      "clip_embed",  "dedup"),
    ("clip_index",      "clip_index",  "dedup"),
    ("clip_dups",       "clip_dups",   "dedup"),
    ("precompute",      "precompute",  "precompute"),
    ("train",           "train",       "training"),
    ("mine",            "mine",        "post"),
    ("validate",        "validate",    "post"),
]

PHASES = ["data", "dedup", "precompute", "training", "post"]

# Orchestrator JSONL message patterns → (sentinel_name, chunk_group)
# Each entry: (regex, sentinel_name)  where chunk is capture group 1
LAUNCH_PATTERNS = [
    (re.compile(r"Launched: download\+convert chunk (\d+)"),  "download"),
    (re.compile(r"Launched: download\+convert chunk (\d+)"),  "convert"),
    (re.compile(r"Launched: build chunk (\d+)"),              "build_shards"),
    (re.compile(r"Launched: filter chunk (\d+)"),             "filter_shards"),
    (re.compile(r"Launched: clip_embed chunk (\d+)"),         "clip_embed"),
    (re.compile(r"Launched: clip_index chunk (\d+)"),         "clip_index"),
    (re.compile(r"Launched: clip_dups chunk (\d+)"),          "clip_dups"),
    (re.compile(r"Launched: precompute chunk (\d+)"),         "precompute"),
    (re.compile(r"Launched: mine chunk (\d+)"),               "mine"),
    (re.compile(r"Launched: validate chunk (\d+)"),           "validate"),
    (re.compile(r"Chunk (\d+): starting training"),           "train"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_launch_events() -> dict[tuple[int, str], list[datetime]]:
    """Return {(chunk, step): [launch_ts, ...]} from all orchestrator JSONL files."""
    launches: dict[tuple[int, str], list[datetime]] = {}
    jsonl_files = sorted(LOG_DIR.glob("orchestrator*.jsonl"))
    for path in jsonl_files:
        try:
            for raw in path.read_text().splitlines():
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = ev.get("message", "")
                ts_str = ev.get("ts", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                for pat, sentinel in LAUNCH_PATTERNS:
                    m = pat.search(msg)
                    if m:
                        key = (int(m.group(1)), sentinel)
                        launches.setdefault(key, []).append(ts)
                        break
        except OSError:
            continue
    return launches


def _load_sentinel_times() -> dict[tuple[int, str], datetime]:
    """Return {(chunk, step): sentinel_mtime} from pipeline/chunk*/step.done files."""
    sentinels: dict[tuple[int, str], datetime] = {}
    if not SENTINEL_DIR.exists():
        return sentinels
    for chunk_dir in sorted(SENTINEL_DIR.glob("chunk*")):
        m = re.match(r"chunk(\d+)$", chunk_dir.name)
        if not m:
            continue
        chunk = int(m.group(1))
        for sentinel in chunk_dir.glob("*.done"):
            step = sentinel.stem
            mtime = sentinel.stat().st_mtime
            sentinels[(chunk, step)] = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return sentinels


def _load_trainer_heartbeats() -> dict[int, dict]:
    """Return {chunk: heartbeat_dict} for trainer heartbeats."""
    hb_dir = DATA_ROOT / ".heartbeat"
    result: dict[int, dict] = {}
    if not hb_dir.exists():
        return result
    for f in hb_dir.glob("trainer_chunk*.json"):
        m = re.search(r"chunk(\d+)", f.name)
        if m:
            try:
                result[int(m.group(1))] = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return result


# ---------------------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------------------

def _effective_start(
    chunk: int,
    step: str,
    end_ts: datetime,
    launches: dict[tuple[int, str], list[datetime]],
) -> Optional[datetime]:
    """Return the last launch timestamp before end_ts (handles retries)."""
    candidates = launches.get((chunk, step), [])
    before = [ts for ts in candidates if ts < end_ts]
    return max(before) if before else None


def build_profile(
    launches: dict[tuple[int, str], list[datetime]],
    sentinels: dict[tuple[int, str], datetime],
    trainer_hb: dict[int, dict],
) -> dict:
    """
    Return profile dict:
      chunks: {chunk_n: {step: {start, end, duration_hours, steps_per_sec?}}}
      phase_totals: {chunk_n: {phase: duration_hours}}
      summary: {step: {min_h, max_h, mean_h, slowest_chunk}}
    """
    chunks_present = sorted({c for c, _ in sentinels})
    chunks: dict[int, dict] = {}

    for chunk in chunks_present:
        stages_out: dict[str, dict] = {}
        for sentinel_name, _label, _phase in STAGES:
            end_ts = sentinels.get((chunk, sentinel_name))
            if end_ts is None:
                continue
            start_ts = _effective_start(chunk, sentinel_name, end_ts, launches)
            if start_ts is None:
                stages_out[sentinel_name] = {"end": end_ts.isoformat(), "duration_hours": None}
                continue
            dur_h = (end_ts - start_ts).total_seconds() / 3600
            entry: dict = {
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "duration_hours": round(dur_h, 2),
            }
            if sentinel_name == "train" and chunk in trainer_hb:
                hb = trainer_hb[chunk]
                if "steps_per_sec" in hb:
                    entry["steps_per_sec"] = hb["steps_per_sec"]
                if "step" in hb:
                    entry["total_steps"] = hb.get("total_steps")
            stages_out[sentinel_name] = entry
        chunks[chunk] = stages_out

    # Phase totals per chunk
    phase_totals: dict[int, dict] = {}
    for chunk, stages_out in chunks.items():
        totals: dict[str, float] = {}
        for sentinel_name, _label, phase in STAGES:
            d = stages_out.get(sentinel_name, {}).get("duration_hours")
            if d is not None:
                totals[phase] = round(totals.get(phase, 0.0) + d, 2)
        phase_totals[chunk] = totals

    # Cross-chunk summary per stage
    summary: dict[str, dict] = {}
    for sentinel_name, _label, _phase in STAGES:
        durations = [
            (c, chunks[c][sentinel_name]["duration_hours"])
            for c in chunks
            if sentinel_name in chunks[c]
            and chunks[c][sentinel_name].get("duration_hours") is not None
        ]
        if not durations:
            continue
        vals = [d for _, d in durations]
        slowest_chunk = max(durations, key=lambda x: x[1])[0]
        summary[sentinel_name] = {
            "min_h": round(min(vals), 2),
            "max_h": round(max(vals), 2),
            "mean_h": round(sum(vals) / len(vals), 2),
            "slowest_chunk": slowest_chunk,
            "n": len(vals),
        }

    # Bottleneck: stage whose mean_h is highest
    bottleneck = max(summary, key=lambda s: summary[s]["mean_h"]) if summary else None

    return {
        "chunks": chunks,
        "phase_totals": phase_totals,
        "summary": summary,
        "bottleneck_stage": bottleneck,
    }


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

_PHASE_COLORS = {
    "data":       "\033[36m",   # cyan
    "dedup":      "\033[33m",   # yellow
    "precompute": "\033[35m",   # magenta
    "training":   "\033[32m",   # green
    "post":       "\033[34m",   # blue
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _h(hours: Optional[float]) -> str:
    if hours is None:
        return "   ?   "
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:3d}h{m:02d}m"


def _phase_color(phase: str) -> str:
    return _PHASE_COLORS.get(phase, "")


def _is_tty() -> bool:
    return sys.stdout.isatty()


def print_profile(profile: dict) -> None:
    use_color = _is_tty()

    def c(color: str, text: str) -> str:
        return f"{color}{text}{_RESET}" if use_color else text

    chunks = profile["chunks"]
    phase_totals = profile["phase_totals"]
    summary = profile["summary"]
    bottleneck = profile["bottleneck_stage"]

    if not chunks:
        print("No completed pipeline chunks found.")
        return

    chunk_list = sorted(chunks)

    # Header
    print()
    print(c(_BOLD, "Pipeline Stage Profiler"))
    print(c(_BOLD, "─" * 72))

    # Per-chunk table
    col_w = 9
    header = f"{'Stage':<18s}" + "".join(f"{'chunk'+str(n):>{col_w}}" for n in chunk_list)
    print(c(_BOLD, header))
    print("─" * (18 + col_w * len(chunk_list)))

    current_phase = None
    for sentinel_name, label, phase in STAGES:
        if sentinel_name not in summary:
            continue
        if phase != current_phase:
            current_phase = phase
            phase_label = c(_phase_color(phase) + _BOLD, f"  [{phase}]")
            print(phase_label)
        row = f"  {label:<16s}"
        for chunk in chunk_list:
            dur = chunks[chunk].get(sentinel_name, {}).get("duration_hours")
            cell = _h(dur)
            if sentinel_name == bottleneck and dur is not None:
                cell = c("\033[31m", cell)  # red for bottleneck
            row += f"{cell:>{col_w}}"
        print(row)

    # Phase totals
    print()
    print(c(_BOLD, f"{'Phase totals':<18s}" + "".join(f"{'chunk'+str(n):>{col_w}}" for n in chunk_list)))
    print("─" * (18 + col_w * len(chunk_list)))
    for phase in PHASES:
        row = f"  {phase:<16s}"
        for chunk in chunk_list:
            dur = phase_totals.get(chunk, {}).get(phase)
            row += f"{_h(dur):>{col_w}}"
        print(row)

    # Cross-chunk summary
    print()
    print(c(_BOLD, "Cross-chunk summary (mean / slowest):"))
    for sentinel_name, label, phase in STAGES:
        if sentinel_name not in summary:
            continue
        s = summary[sentinel_name]
        flag = " ← BOTTLENECK" if sentinel_name == bottleneck else ""
        color = "\033[31m" if sentinel_name == bottleneck else ""
        line = (f"  {label:<16s}  mean={_h(s['mean_h'])}  "
                f"range=[{_h(s['min_h'])}..{_h(s['max_h'])}]  "
                f"slowest=chunk{s['slowest_chunk']}{flag}")
        print(c(color, line) if use_color and flag else line)

    # Bottleneck note
    if bottleneck == "precompute":
        print()
        note = ("  Note: precompute is VAE-dominated (~80% of stage time). "
                "VAE encode at 512px has ~4096-token mid-block attention — "
                "this is the fundamental compute limit on M1 Max.")
        print(c("\033[33m", note) if use_color else note)

    print()


# ---------------------------------------------------------------------------
# --ai output
# ---------------------------------------------------------------------------

def _ai_output(profile: dict) -> dict:
    chunks = profile["chunks"]
    phase_totals = profile["phase_totals"]
    summary = profile["summary"]

    return {
        "ok": bool(chunks),
        "chunks_profiled": sorted(chunks),
        "bottleneck_stage": profile["bottleneck_stage"],
        "stage_summary": {
            s: {
                "mean_hours": v["mean_h"],
                "max_hours": v["max_h"],
                "slowest_chunk": v["slowest_chunk"],
            }
            for s, v in summary.items()
        },
        "phase_totals_by_chunk": {
            str(c): totals for c, totals in phase_totals.items()
        },
        "per_chunk": {
            str(c): {
                step: {
                    "duration_hours": data.get("duration_hours"),
                    "steps_per_sec": data.get("steps_per_sec"),
                }
                for step, data in stages.items()
            }
            for c, stages in chunks.items()
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Pipeline stage wall-clock profiler")
    p.add_argument("--ai", action="store_true",
                   help="Emit compact JSON to stdout; all prose to stderr")
    p.add_argument("--data-root", default=None,
                   help="Override PIPELINE_DATA_ROOT")
    args = p.parse_args()

    global DATA_ROOT, LOG_DIR, SENTINEL_DIR
    if args.data_root:
        from pipeline_lib import DATA_ROOT as _DR
        import pipeline_lib as _lib
        _lib.DATA_ROOT = Path(args.data_root)
        _lib.LOG_DIR   = _lib.DATA_ROOT / "logs"
        _lib.SENTINEL_DIR = _lib.DATA_ROOT / "pipeline"
        DATA_ROOT    = _lib.DATA_ROOT
        LOG_DIR      = _lib.LOG_DIR
        SENTINEL_DIR = _lib.SENTINEL_DIR

    launches = _load_launch_events()
    sentinels = _load_sentinel_times()
    trainer_hb = _load_trainer_heartbeats()

    profile = build_profile(launches, sentinels, trainer_hb)

    if args.ai:
        print(json.dumps(_ai_output(profile)))
    else:
        print_profile(profile)


if __name__ == "__main__":
    main()
