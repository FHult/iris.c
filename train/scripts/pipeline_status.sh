#!/bin/bash
# train/scripts/pipeline_status.sh — V2 pipeline status report.
#
# Reads V2 orchestrator state: pipeline_state.json, sentinel files under
# pipeline/chunk{N}/, heartbeat files under .heartbeat/, and JSONL event logs.
#
# Usage:
#   bash train/scripts/pipeline_status.sh
#   bash train/scripts/pipeline_status.sh --data-root /Volumes/2TBSSD/smoke
#   bash train/scripts/pipeline_status.sh --json
#   PIPELINE_DATA_ROOT=/Volumes/2TBSSD/smoke bash train/scripts/pipeline_status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Arg parsing ───────────────────────────────────────────────────────────────
DATA_ROOT_ARG=""
JSON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT_ARG="$2"; shift 2 ;;
        --json)      JSON_MODE=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve DATA_ROOT ─────────────────────────────────────────────────────────
# Priority: --data-root > PIPELINE_DATA_ROOT env > auto-detect
if [[ -n "$DATA_ROOT_ARG" ]]; then
    DATA_ROOT="$DATA_ROOT_ARG"
elif [[ -n "${PIPELINE_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$PIPELINE_DATA_ROOT"
else
    DATA_ROOT=""
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]] && \
           [[ -d "$candidate/pipeline" || -d "$candidate/shards" || \
              -d "$candidate/precomputed" || -f "$candidate/pipeline_state.json" ]]; then
            DATA_ROOT="$candidate"
            break
        fi
    done
    DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"
fi

PYTHON="$TRAIN_DIR/.venv/bin/python"
[[ ! -x "$PYTHON" ]] && PYTHON=python3

$PYTHON - "$DATA_ROOT" "$TRAIN_DIR" "$JSON_MODE" <<'PYEOF'
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT  = Path(sys.argv[1])
TRAIN_DIR  = Path(sys.argv[2])
JSON_MODE  = sys.argv[3] == "true"

SENTINEL_DIR = DATA_ROOT / "pipeline"
STATE_FILE   = DATA_ROOT / "pipeline_state.json"
HB_DIR       = DATA_ROOT / ".heartbeat"
LOG_DIR      = DATA_ROOT / "logs"
STAGING_DIR  = DATA_ROOT / "staging"
SHARDS_DIR   = DATA_ROOT / "shards"
PRECOMP_DIR  = DATA_ROOT / "precomputed"
CKPT_DIR     = DATA_ROOT / "checkpoints" / "stage1"
HARD_DIR     = DATA_ROOT / "hard_examples"
DEDUP_DIR    = DATA_ROOT / "dedup_ids"

# ── V2 step order per chunk ───────────────────────────────────────────────────
STEPS = [
    ("download",      "Download + Convert"),
    ("build_shards",  "Build shards"),
    ("filter_shards", "Filter shards"),
    ("clip_embed",    "CLIP embed"),
    ("clip_index",    "CLIP index"),
    ("clip_dups",     "CLIP dedup"),
    ("precompute",    "Precompute Qwen3+VAE"),
    ("train",         "Train"),
    ("mine",          "Mine hard examples"),
    ("validate",      "Validate"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def now(): return datetime.now(timezone.utc)

def age_s(ts_str):
    """Seconds since an ISO timestamp string."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return (now() - ts).total_seconds()
    except Exception:
        return -1

def count_files(d, pattern="*.tar"):
    try:
        return len(list(Path(d).glob(pattern)))
    except Exception:
        return 0

def du_h(d):
    try:
        r = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True, timeout=10)
        return r.stdout.split()[0] if r.returncode == 0 else "?"
    except Exception:
        return "?"

def is_done(chunk, step):
    return (SENTINEL_DIR / f"chunk{chunk}" / f"{step}.done").exists()

def is_error(chunk, step):
    return (SENTINEL_DIR / f"chunk{chunk}" / f"{step}.error").exists()

def read_heartbeat(process, chunk=None):
    suffix = f"_chunk{chunk}" if chunk else ""
    path = HB_DIR / f"{process}{suffix}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def last_jsonl_event(log_file, event=None):
    """Return last matching event dict from a JSONL log."""
    if not log_file or not Path(log_file).exists():
        return None
    try:
        lines = Path(log_file).read_text().splitlines()
        for line in reversed(lines):
            try:
                d = json.loads(line)
                if event is None or d.get("event") == event:
                    return d
            except Exception:
                continue
    except Exception:
        pass
    return None

def recent_orch_events(chunk=None, n=5):
    """Return last n orchestrator events (chunk-specific or global)."""
    suffix = f"_chunk{chunk}" if chunk else ""
    log = LOG_DIR / f"orchestrator{suffix}.jsonl"
    events = []
    if not log.exists():
        return events
    try:
        lines = log.read_text().splitlines()
        for line in reversed(lines):
            try:
                d = json.loads(line)
                msg = d.get("message", "")
                if msg:
                    events.append(f"[{d.get('ts','')[-8:-3]}] {msg}")
                    if len(events) >= n:
                        break
            except Exception:
                continue
    except Exception:
        pass
    return list(reversed(events))

def tmux_windows():
    try:
        r = subprocess.run(
            ["tmux", "list-windows", "-t", "iris", "-F", "#{window_name}"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.splitlines() if r.returncode == 0 else []
    except Exception:
        return []

def tmux_sessions():
    try:
        r = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}: #{session_windows} windows (#{?session_attached,attached,detached})"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.splitlines() if r.returncode == 0 else []
    except Exception:
        return []

# ── Read pipeline state ───────────────────────────────────────────────────────
state = {}
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        pass

scale       = state.get("scale", "?")
recipe      = state.get("recipe", "?")
chunks_cfg  = state.get("chunks", {})
last_upd    = state.get("last_updated", "")
run_id      = state.get("run_id", "")
issues      = state.get("issues", [])

# Infer total_chunks from state or sentinel dirs
sentinel_chunks = [int(p.name.replace("chunk",""))
                   for p in SENTINEL_DIR.glob("chunk*") if p.is_dir()] if SENTINEL_DIR.exists() else []
total_chunks = max([int(k) for k in chunks_cfg] + sentinel_chunks + [1]) if (chunks_cfg or sentinel_chunks) else 1

# ── Counts ────────────────────────────────────────────────────────────────────
shard_count   = count_files(SHARDS_DIR)
qwen3_count   = count_files(PRECOMP_DIR / "qwen3", "*.npz")
vae_count     = count_files(PRECOMP_DIR / "vae",   "*.npz")
siglip_count  = count_files(PRECOMP_DIR / "siglip","*.npz")
hard_count    = count_files(HARD_DIR)
ckpt_count    = len(list(CKPT_DIR.glob("step_*.safetensors"))) if CKPT_DIR.exists() else 0
latest_ckpt   = ""
if CKPT_DIR.exists():
    ckpts = sorted(CKPT_DIR.glob("step_*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        latest_ckpt = ckpts[0].name.replace(".safetensors", "")

windows = tmux_windows()
prep_active  = "iris-prep"  in windows
train_active = "iris-train" in windows

# ── Training heartbeat ────────────────────────────────────────────────────────
train_hb = read_heartbeat("train_ip_adapter")
train_hb_str = ""
train_hb_stale = False
if train_hb:
    step      = train_hb.get("step", 0)
    total_s   = train_hb.get("total_steps", 0)
    loss      = train_hb.get("loss", 0)
    loss_sm   = train_hb.get("loss_smooth", 0)
    sps       = train_hb.get("steps_per_sec", 0)
    eta_s     = train_hb.get("eta_seconds", 0)
    ts        = train_hb.get("timestamp", "")
    pct       = step / total_s * 100 if total_s else 0
    eta_h, r  = divmod(int(eta_s), 3600)
    eta_m     = r // 60
    train_hb_str = (f"step {step:,}/{total_s:,} ({pct:.1f}%)  "
                    f"loss {loss:.4f} (avg {loss_sm:.4f})  "
                    f"{sps:.3f} steps/s  ETA {eta_h}h{eta_m:02d}m")
    age = age_s(ts)
    if age > 300:
        train_hb_stale = True
        train_hb_str += f"  ⚠️  stale ({int(age)}s ago)"

# ── Human-readable output ─────────────────────────────────────────────────────
if not JSON_MODE:
    W = 65
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * W)
    print(f"  Pipeline status — {ts_now}")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    if run_id:
        print(f"  Run: {run_id}  recipe={recipe}  scale={scale}")
    print("=" * W)

    def step_icon(done, error, active):
        if error:   return "❌"
        if done:    return "✅"
        if active:  return "⏳"
        return "⬜"

    for chunk in range(1, total_chunks + 1):
        chunk_info = chunks_cfg.get(str(chunk), {})
        chunk_state = chunk_info.get("state", "IDLE")
        print(f"\n── Chunk {chunk}  [{chunk_state}] {'─'*(W-18-len(chunk_state))}")

        for key, label in STEPS:
            done  = is_done(chunk, key)
            error = is_error(chunk, key)

            # Determine if this step is currently active
            if key == "train":
                active = train_active and not done
            elif key in ("download", "convert", "build_shards", "filter_shards",
                         "clip_embed", "clip_index", "clip_dups",
                         "precompute", "mine", "validate"):
                active = prep_active and not done and not error
            else:
                active = False

            icon = step_icon(done, error, active)

            # Progress detail
            detail = ""
            if key == "download" and (done or active):
                hb = read_heartbeat("download_convert", chunk)
                if hb:
                    detail = f"{hb.get('done',0)}/{hb.get('total',0)} tgzs ({hb.get('pct',0):.0f}%)"
                elif done:
                    # Read from event log
                    ev = last_jsonl_event(LOG_DIR / f"download_convert_chunk{chunk}.jsonl")
                    if ev:
                        detail = f"done ({ev.get('ts','')[:16]})"
            elif key == "build_shards":
                n = count_files(STAGING_DIR / f"chunk{chunk}" / "shards")
                if n or done:
                    detail = f"{n} shard(s)"
                hb = read_heartbeat("build_shards", chunk)
                if hb and active:
                    detail = f"{hb.get('done',0)}/{hb.get('total',0)} ({hb.get('pct',0):.0f}%)"
            elif key == "clip_embed":
                hb = read_heartbeat("clip_dedup", chunk)
                if hb and active:
                    detail = f"{hb.get('done',0)}/{hb.get('total',0)} shards ({hb.get('pct',0):.0f}%)"
                elif done:
                    n = count_files(STAGING_DIR / f"chunk{chunk}" / "embeddings", "*.npz")
                    detail = f"{n} embeddings"
            elif key == "clip_dups":
                dup_file = DEDUP_DIR / "duplicate_ids.txt"
                if dup_file.exists():
                    try:
                        n = len(dup_file.read_text().splitlines())
                        detail = f"{n:,} IDs blocked (cumulative)"
                    except Exception:
                        pass
            elif key == "precompute":
                if done or active:
                    detail = f"qwen3={qwen3_count}  vae={vae_count}  siglip={siglip_count}"
                hb = read_heartbeat("precompute", chunk)
                if hb and active:
                    detail = f"{hb.get('done',0)}/{hb.get('total',0)} shards ({hb.get('pct',0):.0f}%)"
            elif key == "train":
                if train_hb_str and active:
                    detail = train_hb_str
                elif done:
                    detail = f"{ckpt_count} checkpoint(s){', latest: ' + latest_ckpt if latest_ckpt else ''}"
                elif train_active:
                    detail = train_hb_str or "running..."
            elif key == "mine":
                if done:
                    detail = f"{hard_count} hard-example shard(s)"
            elif key == "validate":
                if done:
                    val_dir = LOG_DIR / f"val_chunk{chunk}"
                    rpt = val_dir / "validation_report.json"
                    if rpt.exists():
                        try:
                            r = json.loads(rpt.read_text())
                            verdict = r.get("verdict", "?")
                            reason  = r.get("reason", "")
                            detail  = f"{verdict} — {reason}"
                        except Exception:
                            detail = "report available"

            if detail:
                print(f"  {icon} {label:<26} {detail}")
            else:
                print(f"  {icon} {label}")

        # Recent orchestrator events for this chunk
        events = recent_orch_events(chunk, n=3)
        if events:
            print(f"     Recent:")
            for e in events:
                print(f"       {e}")

    # ── Global shards + precompute ────────────────────────────────────────────
    print(f"\n── Global data {'─'*(W-17)}")
    print(f"  Unified shards:   {shard_count} tars  ({du_h(SHARDS_DIR)})")
    print(f"  Precomputed:      qwen3={qwen3_count}  vae={vae_count}  siglip={siglip_count}")
    print(f"  Hard examples:    {hard_count} tars  ({du_h(HARD_DIR)})")
    print(f"  Checkpoints:      {ckpt_count}{', latest: ' + latest_ckpt if latest_ckpt else ''}")

    # ── Issues ────────────────────────────────────────────────────────────────
    if issues:
        print(f"\n── Issues {'─'*(W-11)}")
        for iss in issues[-5:]:
            print(f"  ⚠️  [{iss.get('severity','?')}] {iss.get('message','')}")

    # ── tmux ─────────────────────────────────────────────────────────────────
    print(f"\n── tmux {'─'*(W-9)}")
    sessions = tmux_sessions()
    if sessions:
        for s in sessions:
            print(f"  {s}")
        if windows:
            print(f"  iris windows: {', '.join(windows)}")
    else:
        print("  (none)")

    # ── Active log tail ───────────────────────────────────────────────────────
    print(f"\n── Active log {'─'*(W-15)}")
    active_log = None
    # Most recently modified .log file in LOG_DIR
    try:
        logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            active_log = logs[0]
    except Exception:
        pass
    if active_log and active_log.exists():
        print(f"  {active_log}")
        try:
            tail = active_log.read_text().splitlines()[-12:]
            for line in tail:
                print(f"  {line}")
        except Exception:
            pass
    else:
        print("  (no log found)")

    # ── Disk ─────────────────────────────────────────────────────────────────
    print(f"\n── Disk {'─'*(W-9)}")
    try:
        r = subprocess.run(["df", "-h", str(DATA_ROOT)], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.splitlines()[1].split()
            print(f"  {parts[0]:<30} used={parts[2]:<8} avail={parts[3]:<8} {parts[4]}")
    except Exception:
        pass
    for label, path in [
        ("staging/",          STAGING_DIR),
        ("shards/",           SHARDS_DIR),
        ("precomputed/",      PRECOMP_DIR),
        ("hard_examples/",    HARD_DIR),
        ("checkpoints/",      CKPT_DIR),
        ("dedup_ids/",        DEDUP_DIR),
    ]:
        if Path(path).exists():
            print(f"  {label:<26} {du_h(path)}")

    print("\n" + "=" * W)

# ── JSON output ───────────────────────────────────────────────────────────────
if JSON_MODE:
    chunk_states = {}
    for chunk in range(1, total_chunks + 1):
        steps_done  = {key: is_done(chunk, key)  for key, _ in STEPS}
        steps_error = {key: is_error(chunk, key) for key, _ in STEPS}
        chunk_states[chunk] = {
            "state":      chunks_cfg.get(str(chunk), {}).get("state", "IDLE"),
            "steps_done":  steps_done,
            "steps_error": steps_error,
        }

    disk_info = {}
    try:
        r = subprocess.run(["df", "-k", str(DATA_ROOT)], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.splitlines()[1].split()
            disk_info = {
                "device": parts[0],
                "used_gib":  round(int(parts[2]) / 1048576, 1),
                "avail_gib": round(int(parts[3]) / 1048576, 1),
            }
    except Exception:
        pass

    out = {
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root":   str(DATA_ROOT),
        "run_id":      run_id,
        "recipe":      recipe,
        "scale":       scale,
        "total_chunks": total_chunks,
        "chunks":      chunk_states,
        "issues":      issues,
        "tmux": {
            "iris_session": len(tmux_sessions()) > 0,
            "prep_window":  prep_active,
            "train_window": train_active,
        },
        "data": {
            "shard_count":   shard_count,
            "qwen3_count":   qwen3_count,
            "vae_count":     vae_count,
            "siglip_count":  siglip_count,
            "hard_count":    hard_count,
            "ckpt_count":    ckpt_count,
            "latest_ckpt":   latest_ckpt,
        },
        "training": ({"running": train_active, **train_hb} if train_hb
                     else {"running": train_active}),
        "disk": disk_info,
    }
    print(json.dumps(out, indent=2))

PYEOF
