#!/bin/bash
# train/scripts/pipeline_status.sh — V2 pipeline status report.
#
# Usage:
#   bash train/scripts/pipeline_status.sh
#   bash train/scripts/pipeline_status.sh --data-root /Volumes/2TBSSD/smoke
#   bash train/scripts/pipeline_status.sh --json
#   PIPELINE_DATA_ROOT=/Volumes/2TBSSD/smoke bash train/scripts/pipeline_status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

DATA_ROOT_ARG=""
JSON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT_ARG="$2"; shift 2 ;;
        --json)      JSON_MODE=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

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

# Step key → display label (grouped: clip steps shown as one row)
STEPS = [
    ("download",      "Download + Convert"),
    ("build_shards",  "Build shards"),
    ("filter_shards", "Filter shards"),
    ("clip_embed",    "CLIP dedup"),        # representative of the 3-step clip pipeline
    ("precompute",    "Precompute Qwen3+VAE"),
    ("train",         "Train"),
    ("mine",          "Mine hard examples"),
    ("validate",      "Validate"),
]
# clip_index and clip_dups are sub-steps of clip_embed row; all must be done for ✅
CLIP_SUBSTEPS = ["clip_embed", "clip_index", "clip_dups"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def now():
    return datetime.now(timezone.utc)

def age_s(ts_str):
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return (now() - ts).total_seconds()
    except Exception:
        return -1

def rel_time(ts_str):
    """Human-readable relative time: '2m ago', '1h 4m ago'."""
    a = age_s(ts_str)
    if a < 0:    return ""
    if a < 60:   return f"{int(a)}s ago"
    if a < 3600: return f"{int(a/60)}m ago"
    h, m = divmod(int(a/60), 60)
    return f"{h}h{m:02d}m ago"

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
    if step == "clip_embed":
        return all((SENTINEL_DIR / f"chunk{chunk}" / f"{s}.done").exists()
                   for s in CLIP_SUBSTEPS)
    return (SENTINEL_DIR / f"chunk{chunk}" / f"{step}.done").exists()

def is_error(chunk, step):
    if step == "clip_embed":
        return any((SENTINEL_DIR / f"chunk{chunk}" / f"{s}.error").exists()
                   for s in CLIP_SUBSTEPS)
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

def last_log_lines(log_file, n=10):
    """Return last n lines of a log, with timestamps cleaned to HH:MM:SS."""
    if not log_file or not Path(log_file).exists():
        return []
    try:
        lines = Path(log_file).read_text().splitlines()[-n:]
        out = []
        for line in lines:
            # Convert "[2026-04-19T17:38:43+00:00] [INFO] msg" → "17:38:43  msg"
            if line.startswith("[20") and "]" in line:
                ts_part = line[1:line.index("]")]
                rest = line[line.index("]")+1:].strip()
                # Strip level tag [INFO]/[WARNING]/[ERROR]
                if rest.startswith("["):
                    rest = rest[rest.index("]")+1:].strip()
                try:
                    t = datetime.fromisoformat(ts_part.replace("Z","+00:00"))
                    out.append(f"  {t.strftime('%H:%M:%S')}  {rest}")
                except Exception:
                    out.append(f"  {line}")
            elif line.strip():
                out.append(f"  {line}")
        return out
    except Exception:
        return []

def tmux_windows():
    try:
        r = subprocess.run(
            ["tmux", "list-windows", "-t", "iris", "-F", "#{window_name}"],
            capture_output=True, text=True, timeout=5)
        return r.stdout.splitlines() if r.returncode == 0 else []
    except Exception:
        return []

def tmux_sessions():
    try:
        r = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                           capture_output=True, text=True, timeout=5)
        return set(r.stdout.splitlines()) if r.returncode == 0 else set()
    except Exception:
        return set()

# ── Read state ────────────────────────────────────────────────────────────────
state = {}
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        pass

scale       = state.get("scale", "?")
recipe      = state.get("recipe", "?")
chunks_cfg  = state.get("chunks", {})
issues      = state.get("issues", [])
last_upd    = state.get("last_updated", "")

sentinel_chunks = ([int(p.name.replace("chunk",""))
                    for p in SENTINEL_DIR.glob("chunk*") if p.is_dir()]
                   if SENTINEL_DIR.exists() else [])
total_chunks = max([int(k) for k in chunks_cfg] + sentinel_chunks + [1]) if (chunks_cfg or sentinel_chunks) else 1

# ── Counts ────────────────────────────────────────────────────────────────────
shard_count  = count_files(SHARDS_DIR)
qwen3_count  = count_files(PRECOMP_DIR / "qwen3", "*.npz")
vae_count    = count_files(PRECOMP_DIR / "vae",   "*.npz")
siglip_count = count_files(PRECOMP_DIR / "siglip","*.npz")
hard_count   = count_files(HARD_DIR)
ckpt_count   = len(list(CKPT_DIR.glob("step_*.safetensors"))) if CKPT_DIR.exists() else 0
latest_ckpt  = ""
if CKPT_DIR.exists():
    ckpts = sorted(CKPT_DIR.glob("step_*.safetensors"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        latest_ckpt = ckpts[0].name.replace(".safetensors", "")

windows      = tmux_windows()
sessions     = tmux_sessions()
prep_active  = "iris-prep"  in windows
train_active = "iris-train" in windows
orch_running = "iris" in sessions

# ── Training heartbeat ────────────────────────────────────────────────────────
train_hb     = read_heartbeat("train_ip_adapter")
train_hb_str = ""
train_stale  = False
if train_hb:
    step    = train_hb.get("step", 0)
    total_s = train_hb.get("total_steps", 0)
    loss    = train_hb.get("loss", 0)
    loss_sm = train_hb.get("loss_smooth", 0)
    sps     = train_hb.get("steps_per_sec", 0)
    eta_s   = train_hb.get("eta_seconds", 0)
    ts      = train_hb.get("timestamp", "")
    pct     = step / total_s * 100 if total_s else 0
    eta_h, r = divmod(int(eta_s), 3600)
    eta_m    = r // 60
    train_hb_str = (f"step {step:,}/{total_s:,} ({pct:.0f}%)  "
                    f"loss {loss:.4f}  ETA {eta_h}h{eta_m:02d}m")
    if age_s(ts) > 300:
        train_stale  = True
        train_hb_str += "  ⚠️  heartbeat stale"

# ── Determine which step is currently active per chunk ────────────────────────
def active_step_for(chunk):
    """The first undone step when the prep/train window is live, else None."""
    for key, _ in STEPS:
        if not is_done(chunk, key):
            if key == "train" and train_active:
                return key
            if key != "train" and prep_active:
                return key
            return None  # window not live; pending but not running
    return None

# ── Step detail string ────────────────────────────────────────────────────────
def step_detail(chunk, key):
    if key == "download":
        hb = read_heartbeat("download_convert", chunk)
        if hb:
            done_n, total_n, pct = hb.get("done",0), hb.get("total",0), hb.get("pct",0)
            return f"tgz {done_n}/{total_n}  ({pct:.0f}%)"
    elif key == "build_shards":
        n = count_files(STAGING_DIR / f"chunk{chunk}" / "shards")
        hb = read_heartbeat("build_shards", chunk)
        if hb:
            return f"{hb.get('done',0)}/{hb.get('total',0)} shards ({hb.get('pct',0):.0f}%)"
        if n:
            return f"{n} shard(s) written"
    elif key == "clip_embed":
        hb = read_heartbeat("clip_dedup", chunk)
        if hb:
            phase = hb.get("phase", "embed")
            return f"{phase}  {hb.get('done',0)}/{hb.get('total',0)} ({hb.get('pct',0):.0f}%)"
        if is_done(chunk, "clip_embed"):
            dup_file = DEDUP_DIR / "duplicate_ids.txt"
            if dup_file.exists():
                try:
                    n = len(dup_file.read_text().splitlines())
                    return f"{n:,} duplicates blocked"
                except Exception:
                    pass
    elif key == "precompute":
        hb = read_heartbeat("precompute", chunk)
        if hb:
            return f"{hb.get('done',0)}/{hb.get('total',0)} shards ({hb.get('pct',0):.0f}%)"
        if is_done(chunk, "precompute"):
            return f"qwen3={qwen3_count}  vae={vae_count}"
    elif key == "train":
        if train_hb_str:
            return train_hb_str
        if is_done(chunk, "train"):
            return f"{ckpt_count} checkpoint(s)" + (f"  latest {latest_ckpt}" if latest_ckpt else "")
    elif key == "mine":
        if is_done(chunk, "mine"):
            return f"{hard_count} hard-example shard(s)"
    elif key == "validate":
        if is_done(chunk, "validate"):
            rpt = LOG_DIR / f"val_chunk{chunk}" / "validation_report.json"
            if rpt.exists():
                try:
                    r = json.loads(rpt.read_text())
                    return f"{r.get('verdict','?')} — {r.get('reason','')}"
                except Exception:
                    pass
    return ""

# ── Most relevant active log ──────────────────────────────────────────────────
def active_log():
    try:
        logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[0] if logs else None
    except Exception:
        return None

# ── Human-readable output ─────────────────────────────────────────────────────
if not JSON_MODE:
    W = 62
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Overall status line
    if not STATE_FILE.exists():
        overall = "NOT STARTED"
    elif not orch_running and not prep_active and not train_active:
        all_done = all(is_done(c, "validate") for c in range(1, total_chunks+1))
        overall = "COMPLETE" if all_done else "STOPPED"
    elif train_active:
        overall = "TRAINING"
    elif prep_active:
        overall = "RUNNING"
    else:
        overall = "RUNNING"

    print("=" * W)
    print(f"  iris pipeline  {ts_now}  [{overall}]")
    print(f"  scale={scale}  chunks={total_chunks}  {DATA_ROOT}")
    print("=" * W)

    # "Now:" banner — what is actually happening right now
    now_lines = []
    for chunk in range(1, total_chunks + 1):
        act = active_step_for(chunk)
        if act:
            label = dict(STEPS)[act]
            detail = step_detail(chunk, act)
            now_lines.append(f"  chunk {chunk} · {label}" + (f"  {detail}" if detail else ""))
    if now_lines:
        print(f"\n  Now:")
        for l in now_lines:
            print(l)
    elif overall in ("STOPPED", "COMPLETE", "NOT STARTED"):
        pass
    else:
        print(f"\n  Now:  waiting for next step...")

    # Per-chunk step list
    for chunk in range(1, total_chunks + 1):
        act = active_step_for(chunk)
        all_chunk_done = all(is_done(chunk, k) for k, _ in STEPS)
        chunk_label = f"chunk {chunk}/{total_chunks}"
        if all_chunk_done:
            chunk_label += "  ✅ done"
        print(f"\n── {chunk_label} {'─'*(W-4-len(chunk_label))}")

        for key, label in STEPS:
            done  = is_done(chunk, key)
            error = is_error(chunk, key)
            active = (act == key)
            detail = step_detail(chunk, key)

            if error:
                icon = "❌"
            elif done:
                icon = "✅"
            elif active:
                icon = "⏳"
            else:
                icon = "  "   # blank — not done, not running

            if done or active or error:
                if detail:
                    print(f"  {icon} {label:<26} {detail}")
                else:
                    print(f"  {icon} {label}")
            else:
                print(f"  {icon} {label}")

    # Issues
    if issues:
        print(f"\n── Issues {'─'*(W-11)}")
        for iss in issues[-5:]:
            sev = iss.get("severity","?").upper()
            print(f"  ⚠️  [{sev}] {iss.get('message','')}")

    # Active log
    log = active_log()
    if log:
        lines = last_log_lines(log, n=10)
        if lines:
            print(f"\n── Log: {log.name} {'─'*(max(1, W-10-len(log.name)))}")
            for l in lines:
                print(l)

    # Disk — compact, only populated dirs
    print(f"\n── Disk {'─'*(W-9)}")
    try:
        r = subprocess.run(["df", "-h", str(DATA_ROOT)],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.splitlines()[1].split()
            print(f"  {p[0]}  used {p[2]}  free {p[3]}  ({p[4]})")
    except Exception:
        pass
    for label, path in [("staging",     STAGING_DIR),
                        ("shards",       SHARDS_DIR),
                        ("precomputed",  PRECOMP_DIR),
                        ("checkpoints",  CKPT_DIR),
                        ("hard_examples",HARD_DIR)]:
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            print(f"  {label:<16} {du_h(p)}")

    print("=" * W)

# ── JSON output ───────────────────────────────────────────────────────────────
if JSON_MODE:
    chunk_states = {}
    for chunk in range(1, total_chunks + 1):
        chunk_states[chunk] = {
            "state":       chunks_cfg.get(str(chunk), {}).get("state", "IDLE"),
            "active_step": active_step_for(chunk),
            "steps_done":  {key: is_done(chunk, key)  for key, _ in STEPS},
            "steps_error": {key: is_error(chunk, key) for key, _ in STEPS},
        }
    disk_info = {}
    try:
        r = subprocess.run(["df", "-k", str(DATA_ROOT)],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.splitlines()[1].split()
            disk_info = {"device": parts[0],
                         "used_gib":  round(int(parts[2])/1048576, 1),
                         "avail_gib": round(int(parts[3])/1048576, 1)}
    except Exception:
        pass

    print(json.dumps({
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root":    str(DATA_ROOT),
        "scale":        scale,
        "total_chunks": total_chunks,
        "running":      orch_running or prep_active or train_active,
        "chunks":       chunk_states,
        "issues":       issues,
        "data": {
            "shard_count":  shard_count,
            "qwen3_count":  qwen3_count,
            "vae_count":    vae_count,
            "hard_count":   hard_count,
            "ckpt_count":   ckpt_count,
            "latest_ckpt":  latest_ckpt,
        },
        "training": ({"running": train_active, **train_hb}
                     if train_hb else {"running": train_active}),
        "disk": disk_info,
    }, indent=2))

PYEOF
