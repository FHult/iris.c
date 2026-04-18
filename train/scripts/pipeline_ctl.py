#!/usr/bin/env python3
"""
train/scripts/pipeline_ctl.py — V2 pipeline control interface.

Usage:
    python train/scripts/pipeline_ctl.py pause
    python train/scripts/pipeline_ctl.py resume
    python train/scripts/pipeline_ctl.py abort
    python train/scripts/pipeline_ctl.py restart-orchestrator
    python train/scripts/pipeline_ctl.py force-next-chunk 2
    python train/scripts/pipeline_ctl.py clear-error 2 build_shards
    python train/scripts/pipeline_ctl.py dispatch-read       # show open issues
    python train/scripts/pipeline_ctl.py dispatch-resolve I-001
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, CONTROL_FILE, DISPATCH_QUEUE, LOG_DIR, TMUX_SESSION,
    TMUX_ORCH_WIN, TMUX_TRAIN_WIN, TMUX_PREP_WIN, SCRIPTS_DIR, TRAIN_DIR,
    clear_error, mark_done, tmux_session_exists, tmux_window_exists,
    tmux_new_window, now_iso,
)


def _write_control(action: str, **kwargs) -> None:
    payload = {"action": action, "ts": now_iso(), **kwargs}
    CONTROL_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Control signal written: {action}")


def cmd_pause(_args) -> None:
    _write_control("pause")


def cmd_resume(_args) -> None:
    CONTROL_FILE.unlink(missing_ok=True)
    print("Resumed (cleared pause signal)")


def cmd_abort(_args) -> None:
    confirm = input("Abort pipeline? This stops orchestrator and prep (y/N): ").strip()
    if confirm.lower() != "y":
        print("Aborted.")
        return
    _write_control("abort")
    # Kill prep window immediately
    if tmux_window_exists(TMUX_PREP_WIN):
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_PREP_WIN}"])
        print("Killed iris-prep window")


def cmd_restart_orchestrator(_args) -> None:
    if not tmux_session_exists():
        print(f"tmux session '{TMUX_SESSION}' not found — create it first", file=sys.stderr)
        sys.exit(1)
    # Kill existing orch window if any
    if tmux_window_exists(TMUX_ORCH_WIN):
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_ORCH_WIN}"])
        print("Killed existing iris-orch window")
    config = TRAIN_DIR / "configs" / "v2_pipeline.yaml"
    log_file = LOG_DIR / "orchestrator.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (f"source '{TRAIN_DIR}/.venv/bin/activate' && "
           f"python -u '{SCRIPTS_DIR}/orchestrator.py' --resume --config '{config}'")
    tmux_new_window(TMUX_ORCH_WIN, cmd, log_file)
    print(f"Orchestrator restarted → {log_file}")


def cmd_force_next_chunk(args) -> None:
    chunk = args.chunk
    _write_control("force-next-chunk", chunk=chunk)
    print(f"Force-advancing chunk {chunk} past validation")


def cmd_clear_error(args) -> None:
    chunk = args.chunk
    step  = args.step
    clear_error(chunk, step)
    print(f"Cleared error sentinel for chunk {chunk} step {step}")


def cmd_mark_done(args) -> None:
    chunk = args.chunk
    step  = args.step
    mark_done(chunk, step)
    print(f"Marked chunk {chunk} step {step} as done")


def cmd_dispatch_read(_args) -> None:
    if not DISPATCH_QUEUE.exists():
        print("No dispatch queue found")
        return
    lines = DISPATCH_QUEUE.read_text().strip().splitlines()
    issues = [json.loads(l) for l in lines if l.strip()]
    open_issues = [i for i in issues if not i.get("resolved")]
    if not open_issues:
        print("No open issues")
        return
    for issue in open_issues:
        sev  = issue.get("severity", "?").upper()
        iid  = issue.get("id", "?")
        ts   = issue.get("ts", "?")
        msg  = issue.get("message", "?")
        act  = issue.get("suggested_action", "")
        chunk = issue.get("chunk")
        chunk_str = f" chunk={chunk}" if chunk else ""
        print(f"[{sev}] {iid}{chunk_str}  {ts}")
        print(f"       {msg}")
        if act:
            print(f"       suggested: {act}")
        print()


def cmd_dispatch_resolve(args) -> None:
    issue_id = args.issue_id
    if not DISPATCH_QUEUE.exists():
        print("No dispatch queue")
        return
    lines = DISPATCH_QUEUE.read_text().strip().splitlines()
    issues = [json.loads(l) for l in lines if l.strip()]
    found = False
    for issue in issues:
        if issue.get("id") == issue_id:
            issue["resolved"] = True
            issue["resolved_at"] = now_iso()
            found = True
    if not found:
        print(f"Issue {issue_id} not found")
        return
    DISPATCH_QUEUE.write_text("\n".join(json.dumps(i) for i in issues) + "\n")
    print(f"Marked {issue_id} as resolved")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline V2 control")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("pause",               help="Pause orchestrator")
    sub.add_parser("resume",              help="Clear pause signal")
    sub.add_parser("abort",               help="Abort orchestrator and prep")
    sub.add_parser("restart-orchestrator",help="Restart iris-orch tmux window")

    p = sub.add_parser("force-next-chunk",help="Force-pass validation for a chunk")
    p.add_argument("chunk", type=int)

    p = sub.add_parser("clear-error",    help="Clear error sentinel for a step")
    p.add_argument("chunk", type=int)
    p.add_argument("step")

    p = sub.add_parser("mark-done",      help="Manually mark a step done")
    p.add_argument("chunk", type=int)
    p.add_argument("step")

    sub.add_parser("dispatch-read",      help="Show open dispatch issues")

    p = sub.add_parser("dispatch-resolve",help="Mark a dispatch issue resolved")
    p.add_argument("issue_id")

    args = ap.parse_args()
    handlers = {
        "pause":                cmd_pause,
        "resume":               cmd_resume,
        "abort":                cmd_abort,
        "restart-orchestrator": cmd_restart_orchestrator,
        "force-next-chunk":     cmd_force_next_chunk,
        "clear-error":          cmd_clear_error,
        "mark-done":            cmd_mark_done,
        "dispatch-read":        cmd_dispatch_read,
        "dispatch-resolve":     cmd_dispatch_resolve,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
