"""
train/scripts/pipeline_lib.py — Shared primitives for the V2 pipeline.

All pipeline scripts import from here for consistent state file I/O,
sentinel management, structured event logging, and heartbeat handling.
"""

import json
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Default paths — override via config or DATA_ROOT env var
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
TRAIN_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = TRAIN_DIR / "scripts"
VENV_PYTHON = TRAIN_DIR / ".venv" / "bin" / "python"

STATE_FILE     = DATA_ROOT / "pipeline_state.json"
CONTROL_FILE   = DATA_ROOT / "pipeline_control.json"
DISPATCH_QUEUE = DATA_ROOT / "dispatch_queue.jsonl"
DISPATCH_RESP  = DATA_ROOT / "dispatch_responses.jsonl"
SENTINEL_DIR   = DATA_ROOT / "pipeline"
LOG_DIR        = DATA_ROOT / "logs"
STAGING_DIR    = DATA_ROOT / "staging"
SHARDS_DIR        = DATA_ROOT / "shards"
PRECOMP_DIR       = DATA_ROOT / "precomputed"
HARD_EX_DIR       = DATA_ROOT / "hard_examples"
ANCHOR_SHARDS_DIR = DATA_ROOT / "anchor_shards"
DEDUP_DIR         = DATA_ROOT / "dedup_ids"
CKPT_DIR          = DATA_ROOT / "checkpoints" / "stage1"
CKPT_ARCHIVE_DIR  = DATA_ROOT / "checkpoints" / "stage1" / "archive"
RAW_POOL_DIR       = DATA_ROOT / "raw" / "journeydb"
CONVERTED_POOL_DIR = DATA_ROOT / "converted" / "journeydb"
# HuggingFace cache lives on cold storage; ~/.cache/huggingface symlinks here.
HF_CACHE_DIR = Path("/Volumes/16TBCold/hf_cache")
GPU_LOCK_FILE     = DATA_ROOT / ".gpu_lock"
RUN_METADATA_FILE = DATA_ROOT / "run_metadata.json"

TMUX_SESSION   = "iris"
TMUX_TRAIN_WIN = "iris-train"
TMUX_PREP_WIN  = "iris-prep"
TMUX_ORCH_WIN  = "iris-orch"
TMUX_WATCH_WIN = "iris-watchdog"
TMUX_STAGE_WIN    = "iris-stage"     # data_stager.py — cold/hot staging, runs alongside training
TMUX_ABLATION_WIN = "iris-ablation"  # ablation_harness.py — long-term autonomous ablation
TMUX_FLYWHEEL_WIN = "iris-flywheel"  # orchestrator.py --flywheel-config — sref optimization loop

ABLATION_CONTROL_FILE = DATA_ROOT / "ablation_control.json"
ABLATION_DB_PATH      = DATA_ROOT / "ablation_history.db"

FLYWHEEL_CONTROL_FILE = DATA_ROOT / "flywheel_control.json"
FLYWHEEL_DB_PATH      = DATA_ROOT / "flywheel_history.db"
SHARD_SCORES_DB_PATH  = DATA_ROOT / "shard_scores.db"
FLYWHEEL_REPORTS_DIR  = DATA_ROOT / "reports"

# Trainer heartbeat is written every min(log_every, 100) steps.
# At 0.19 steps/s: 100 steps ≈ 526s.  900s = ~1.7× that interval.
# NOTE: heartbeat is decoupled from log_every in train_ip_adapter.py so that
# large log intervals (e.g. 500 steps ≈ 44 min) do not exceed this threshold.
HEARTBEAT_STALE_SECS = 900
DISK_WARN_GB  = 80
DISK_ABORT_GB = 40

# Shard ID space reserved per chunk.  Chunk N owns IDs [(N-1)*SHARD_BLOCK, N*SHARD_BLOCK).
# 200 000 shards × 5 000 images/shard = 1 billion images per chunk — ample headroom.
# Single authoritative definition; imported by orchestrator.py and data_stager.py.
SHARD_BLOCK = 200_000


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> dict:
    """Load pipeline YAML config. Falls back to built-in defaults."""
    import yaml
    if path is None:
        path = TRAIN_DIR / "configs" / "v2_pipeline.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# State file — atomic R/W
# ---------------------------------------------------------------------------

def read_state() -> dict:
    """Read pipeline_state.json. Returns empty dict if missing or corrupt."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(state: dict) -> None:
    """Atomically write pipeline_state.json."""
    state["last_updated"] = now_iso()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


def update_state(**kwargs) -> dict:
    """Read-modify-write the state file."""
    state = read_state()
    _deep_update(state, kwargs)
    write_state(state)
    return state


def _deep_update(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Sentinel files
# ---------------------------------------------------------------------------

def sentinel_path(chunk: int, step: str) -> Path:
    return SENTINEL_DIR / f"chunk{chunk}" / f"{step}.done"


def error_path(chunk: int, step: str) -> Path:
    return SENTINEL_DIR / f"chunk{chunk}" / f"{step}.error"


def is_done(chunk: int, step: str) -> bool:
    return sentinel_path(chunk, step).exists()


def mark_done(chunk: int, step: str) -> None:
    p = sentinel_path(chunk, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def mark_error(chunk: int, step: str, message: str = "") -> None:
    p = error_path(chunk, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{now_iso()}\n{message}\n")


def has_error(chunk: int, step: str) -> bool:
    return error_path(chunk, step).exists()


def read_error(chunk: int, step: str) -> str:
    p = error_path(chunk, step)
    return p.read_text() if p.exists() else ""


def clear_error(chunk: int, step: str) -> None:
    p = error_path(chunk, step)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Structured event logging (JSONL)
# ---------------------------------------------------------------------------

def log_event(process: str, event: str, chunk: Optional[int] = None, **fields) -> None:
    """Append a structured JSON event to {process}_chunk{N}.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{chunk}" if chunk is not None else ""
    log_path = LOG_DIR / f"{process}{suffix}.jsonl"
    entry = {"ts": now_iso(), "process": process, "event": event}
    if chunk is not None:
        entry["chunk"] = chunk
    entry.update(fields)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_orch(message: str, level: str = "info", **fields) -> None:
    """Log an orchestrator message to console and JSONL."""
    ts = now_iso()
    print(f"[{ts}] [{level.upper()}] {message}", flush=True)
    log_event("orchestrator", level, message=message, **fields)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def heartbeat_path(process: str, chunk: Optional[int] = None) -> Path:
    suffix = f"_chunk{chunk}" if chunk is not None else ""
    return DATA_ROOT / ".heartbeat" / f"{process}{suffix}.json"


def write_heartbeat(process: str, chunk: Optional[int] = None, **fields) -> None:
    p = heartbeat_path(process, chunk)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"ts": now_iso(), "process": process}
    if chunk is not None:
        data["chunk"] = chunk
    data.update(fields)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(p)


def read_heartbeat(process: str, chunk: Optional[int] = None) -> Optional[dict]:
    p = heartbeat_path(process, chunk)
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def heartbeat_age_secs(process: str, chunk: Optional[int] = None) -> Optional[float]:
    hb = read_heartbeat(process, chunk)
    if hb is None:
        return None
    # Accept both "ts" (UTC ISO, written by write_heartbeat) and "timestamp"
    # (legacy local-time string written by older trainer versions).
    raw = hb.get("ts") or hb.get("timestamp")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except ValueError:
        return None


def is_heartbeat_stale(process: str, chunk: Optional[int] = None) -> bool:
    age = heartbeat_age_secs(process, chunk)
    return age is None or age > HEARTBEAT_STALE_SECS


# ---------------------------------------------------------------------------
# Dispatch (Claude / CLI / web interface)
# ---------------------------------------------------------------------------

def dispatch_issue(issue_id: str, severity: str, message: str,
                   chunk: Optional[int] = None, process: str = "",
                   context: Optional[dict] = None,
                   suggested_action: str = "") -> None:
    """Append an issue to dispatch_queue.jsonl for operator consumption."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": issue_id,
        "severity": severity,
        "ts": now_iso(),
        "process": process,
        "message": message,
        "context": context or {},
        "suggested_action": suggested_action,
        "resolved": False,
    }
    if chunk is not None:
        entry["chunk"] = chunk
    with open(DISPATCH_QUEUE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# tmux helpers
# ---------------------------------------------------------------------------

def tmux_window_exists(window_name: str, session: str = TMUX_SESSION) -> bool:
    import subprocess
    result = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        capture_output=True, text=True
    )
    return window_name in result.stdout.splitlines()


def tmux_session_exists(session: str = TMUX_SESSION) -> bool:
    import subprocess
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True
    )
    return result.returncode == 0


def gpu_is_free() -> bool:
    """GPU is free when no iris-train window exists in the tmux session."""
    if not tmux_session_exists():
        return True
    return not tmux_window_exists(TMUX_TRAIN_WIN)


def tmux_new_window(window_name: str, cmd: str, log_file: Path,
                    session: str = TMUX_SESSION) -> None:
    """Launch cmd in a new tmux window, logging stdout+stderr to log_file."""
    import subprocess
    full_cmd = f"({cmd}) >> '{log_file}' 2>&1; echo EXIT_CODE=$? >> '{log_file}'"
    try:
        subprocess.run([
            "tmux", "new-window", "-t", f"{session}:", "-n", window_name,
            "bash", "-c", full_cmd
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"tmux new-window failed (session={session!r}, window={window_name!r}): {e}"
        ) from e


def tmux_send_keys(window_name: str, cmd: str, session: str = TMUX_SESSION) -> None:
    import subprocess
    subprocess.run([
        "tmux", "send-keys", "-t", f"{session}:{window_name}", cmd, "Enter"
    ], check=True)


def last_exit_code(log_file: Path) -> Optional[int]:
    """Read EXIT_CODE=N from the last line of a tmux log."""
    if not log_file.exists():
        return None
    try:
        with open(log_file, "rb") as _f:
            _f.seek(0, 2)
            _tail = min(512, _f.tell())
            _f.seek(-_tail, 2)
            _chunk = _f.read(_tail).decode("utf-8", errors="replace")
        for line in reversed(_chunk.splitlines()):
            if line.startswith("EXIT_CODE="):
                return int(line.split("=", 1)[1])
    except (ValueError, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# GPU file lock — cross-process mutex for all GPU-bound steps
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it


def gpu_lock_holder() -> Optional[dict]:
    """Return the lock info dict if a live process holds the GPU lock, else None."""
    if not GPU_LOCK_FILE.exists():
        return None
    try:
        info = json.loads(GPU_LOCK_FILE.read_text())
        if _pid_alive(info.get("pid", 0)):
            return info
    except (ValueError, OSError):
        pass
    return None


def acquire_gpu_lock(label: str) -> bool:
    """
    Acquire the cross-process GPU exclusive lock file.
    Returns True if acquired, False if a live process already holds it.
    Stale locks from dead processes are silently stolen.
    Uses O_EXCL (open mode 'x') for atomic acquisition — no TOCTOU window.
    """
    record = json.dumps({
        "pid": os.getpid(),
        "label": label,
        "started": now_iso(),
    })
    for _ in range(2):
        try:
            with open(GPU_LOCK_FILE, "x") as _f:
                _f.write(record)
            return True
        except (FileExistsError, PermissionError) as exc:
            # Some Apple filesystems return EPERM instead of EEXIST for O_EXCL
            # on an existing file.  Re-raise only if the file genuinely doesn't
            # exist (real permission problem, not a collision).
            if isinstance(exc, PermissionError) and not GPU_LOCK_FILE.exists():
                raise
            if gpu_lock_holder() is not None:
                return False
            try:
                GPU_LOCK_FILE.unlink()
            except OSError:
                pass
    return False


def release_gpu_lock() -> None:
    """Release the GPU lock if held by the current process. Safe to call on exit."""
    try:
        info = json.loads(GPU_LOCK_FILE.read_text())
        if info.get("pid") == os.getpid():
            GPU_LOCK_FILE.unlink()
    except (ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

def free_gb(path: Path = DATA_ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


# ---------------------------------------------------------------------------
# macOS notification
# ---------------------------------------------------------------------------

def notify(title: str, message: str) -> None:
    import subprocess
    try:
        safe_msg   = message.replace('\\', '\\\\').replace('"', '\\"')
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        subprocess.run([
            "osascript", "-e",
            f'display notification "{safe_msg}" with title "{safe_title}"'
        ], capture_output=True)
    except Exception:
        pass
