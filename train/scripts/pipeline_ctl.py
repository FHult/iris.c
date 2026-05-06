#!/usr/bin/env python3
"""
train/scripts/pipeline_ctl.py — V2 pipeline control interface.

Usage:
    python train/scripts/pipeline_ctl.py status            # status + doctor summary
    python train/scripts/pipeline_ctl.py status --brief    # one-line summary
    python train/scripts/pipeline_ctl.py pause
    python train/scripts/pipeline_ctl.py resume
    python train/scripts/pipeline_ctl.py abort
    python train/scripts/pipeline_ctl.py restart-orchestrator
    python train/scripts/pipeline_ctl.py force-next-chunk 2
    python train/scripts/pipeline_ctl.py clear-error 2 build_shards
    python train/scripts/pipeline_ctl.py dispatch-read       # show open issues
    python train/scripts/pipeline_ctl.py dispatch-resolve I-001
    python train/scripts/pipeline_ctl.py restart-from-chunk 2  # restart from chunk 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, CONTROL_FILE, DISPATCH_QUEUE, LOG_DIR, SENTINEL_DIR,
    TMUX_SESSION, TMUX_ORCH_WIN, TMUX_TRAIN_WIN, TMUX_PREP_WIN, SCRIPTS_DIR, TRAIN_DIR,
    CKPT_ARCHIVE_DIR, CKPT_DIR, HARD_EX_DIR, SHARDS_DIR, ANCHOR_SHARDS_DIR,
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
    # Create the session if it doesn't exist yet.
    if not tmux_session_exists():
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-n", TMUX_ORCH_WIN],
                       check=True)
    config = TRAIN_DIR / "configs" / "v2_pipeline.yaml"
    log_file = LOG_DIR / "orchestrator.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (f"source '{TRAIN_DIR}/.venv/bin/activate' && "
           f"caffeinate -dim python -u '{SCRIPTS_DIR}/orchestrator.py' --resume --config '{config}'")
    # If an old orch window exists, rename it so the session stays alive while we
    # create the replacement, then kill it after.
    if tmux_window_exists(TMUX_ORCH_WIN):
        subprocess.run(["tmux", "rename-window", "-t", f"{TMUX_SESSION}:{TMUX_ORCH_WIN}", "_old-orch"],
                       check=True)
    tmux_new_window(TMUX_ORCH_WIN, cmd, log_file)
    if tmux_window_exists("_old-orch"):
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:_old-orch"])
    print(f"Orchestrator restarted → {log_file}")


def cmd_force_next_chunk(args) -> None:
    chunk = args.chunk
    _write_control("force-next-chunk", chunk=chunk)
    print(f"Force-advancing chunk {chunk} past validation")


def cmd_retry(args) -> None:
    """Clear error sentinel and tell the running orchestrator to reset its retry counter."""
    chunk = args.chunk
    step  = args.step
    clear_error(chunk, step)
    _write_control("retry", chunk=chunk, step=step)
    print(f"Retry signal sent for chunk {chunk} step {step} — orchestrator will pick up on next poll")


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
    # Verify the issue exists before appending a resolve marker.
    lines = DISPATCH_QUEUE.read_text().strip().splitlines()
    ids = {json.loads(l).get("id") for l in lines if l.strip()}
    if issue_id not in ids:
        print(f"Issue {issue_id} not found")
        return
    # Append a resolve marker — the queue is append-only; last entry per ID wins.
    marker = json.dumps({"id": issue_id, "resolved": True, "resolved_at": now_iso()})
    with open(DISPATCH_QUEUE, "a") as f:
        f.write(marker + "\n")
    print(f"Marked {issue_id} as resolved")


def cmd_restart_from_chunk(args) -> None:
    """
    Safely restart the pipeline from chunk N:
      1. Kill iris-train window if running (with confirmation)
      2. Clear all sentinels for chunks N..total_chunks
      3. Delete hard_examples/chunk{M}/ for M >= N so mining re-runs against
         the new checkpoint (stale manifests would otherwise cause a zero-op re-mine)
      4. Restore chunk N-1 final checkpoint from archive if available
      5. Restart orchestrator
    """
    import shutil
    chunk = args.chunk

    # Detect total_chunks from sentinel dirs
    if SENTINEL_DIR.exists():
        total = sum(1 for d in SENTINEL_DIR.iterdir() if d.name.startswith("chunk"))
    else:
        total = 4  # fallback

    # Identify hard example dirs that would be stale after the restart
    hard_ex_to_delete = [
        HARD_EX_DIR / f"chunk{c}"
        for c in range(chunk, total + 1)
        if (HARD_EX_DIR / f"chunk{c}").exists()
    ]

    print(f"Restarting pipeline from chunk {chunk} (total={total})")
    print(f"This will:")
    print(f"  - Kill iris-train if running")
    print(f"  - Delete sentinels for chunks {chunk}..{total}")
    if hard_ex_to_delete:
        for d in hard_ex_to_delete:
            tars = list(d.glob("*.tar"))
            print(f"  - Delete hard_examples/{d.name}/ ({len(tars)} tar(s)) — stale without re-training")
    if chunk > 1:
        arch = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.safetensors"
        if arch.exists():
            print(f"  - Restore chunk {chunk - 1} final checkpoint from archive")
        else:
            print(f"  - WARNING: no archived checkpoint for chunk {chunk - 1} — training will resume from whatever is in {CKPT_DIR}")
    confirm = input("Continue? (y/N): ").strip()
    if confirm.lower() != "y":
        print("Aborted.")
        return

    # Kill training window
    if tmux_window_exists(TMUX_TRAIN_WIN):
        subprocess.run(["tmux", "kill-window", "-t", f"{TMUX_SESSION}:{TMUX_TRAIN_WIN}"])
        print("Killed iris-train window")

    # Clear sentinels for chunks N..total
    for c in range(chunk, total + 1):
        chunk_dir = SENTINEL_DIR / f"chunk{c}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared sentinels for chunk {c}")

    # Delete stale hard example directories so mining re-runs against the new
    # checkpoint.  Without this, mine_hard_examples.py reads the .existing_ids.txt
    # manifest, finds all records already extracted, and exits immediately — producing
    # a zero-op re-mine that leaves examples scored by the old checkpoint in place.
    for d in hard_ex_to_delete:
        shutil.rmtree(d)
        print(f"  Deleted hard_examples/{d.name}/ (will be re-mined after re-training)")

    # Restore archived checkpoint if available
    if chunk > 1:
        arch_st  = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.safetensors"
        arch_js  = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.json"
        arch_ema = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.ema.safetensors"
        if arch_st.exists():
            # Find or infer the step number from the archive json
            step_name = "restored"
            if arch_js.exists():
                try:
                    meta = json.loads(arch_js.read_text())
                    step_name = f"step_{meta.get('step', 'restored'):07d}"
                except Exception:
                    pass
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            dst_st  = CKPT_DIR / f"{step_name}.safetensors"
            dst_js  = CKPT_DIR / f"{step_name}.json"
            dst_ema = CKPT_DIR / f"{step_name}.ema.safetensors"
            shutil.copy2(arch_st, dst_st)
            if arch_js.exists():
                shutil.copy2(arch_js, dst_js)
            if arch_ema.exists():
                shutil.copy2(arch_ema, dst_ema)
            print(f"  Restored checkpoint → {dst_st.name}")

    # Restart orchestrator
    cmd_restart_orchestrator(args)


def cmd_populate_anchor_shards(args) -> None:
    """Copy every Nth shard from SHARDS_DIR to ANCHOR_SHARDS_DIR.

    Used to manually populate anchor shards when the orchestrator's
    auto-populate (PIPELINE-17) did not run or ran against the wrong shards.
    Anchor shards are mixed into chunk 2+ training to prevent forgetting.
    """
    import shutil
    rate = args.rate
    force = args.force

    if not SHARDS_DIR.exists() or not any(SHARDS_DIR.glob("*.tar")):
        print(f"No shards found in {SHARDS_DIR}")
        return

    shards = sorted(SHARDS_DIR.glob("*.tar"))
    selected = shards[::rate]

    if ANCHOR_SHARDS_DIR.exists() and any(ANCHOR_SHARDS_DIR.glob("*.tar")):
        existing = list(ANCHOR_SHARDS_DIR.glob("*.tar"))
        if not force:
            print(f"Anchor shards already populated ({len(existing)} tars). "
                  f"Use --force to overwrite.")
            return
        print(f"Clearing {len(existing)} existing anchor shards (--force)")
        for f in existing:
            f.unlink()

    ANCHOR_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    for shard in selected:
        shutil.copy2(shard, ANCHOR_SHARDS_DIR / shard.name)
    print(f"Populated {len(selected)} anchor shards (1/{rate} sample of "
          f"{len(shards)} shards) → {ANCHOR_SHARDS_DIR}")


def cmd_status(args) -> None:
    """Run pipeline_status.py (brief summary) then pipeline_doctor.py --ai."""
    import subprocess
    venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")
    status_script = str(SCRIPTS_DIR / "pipeline_status.py")
    doctor_script = str(SCRIPTS_DIR / "pipeline_doctor.py")

    if args.brief:
        subprocess.run([venv_python, status_script, "--brief"], check=False)
    else:
        subprocess.run([venv_python, status_script], check=False)
        print()
        subprocess.run([venv_python, doctor_script, "--ai"], check=False)


def cmd_dispatch_resolve_all(args) -> None:
    """Resolve all open dispatch issues, optionally filtered by chunk or severity."""
    if not DISPATCH_QUEUE.exists():
        print("No dispatch queue")
        return
    lines = DISPATCH_QUEUE.read_text().strip().splitlines()
    by_id: dict = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            iid = entry.get("id")
            if iid:
                by_id[iid] = entry
        except json.JSONDecodeError:
            pass
    open_issues = [e for e in by_id.values() if not e.get("resolved", False)]
    if args.chunk:
        open_issues = [e for e in open_issues if e.get("chunk") == args.chunk]
    if not open_issues:
        print("No matching open issues")
        return
    ts = now_iso()
    with open(DISPATCH_QUEUE, "a") as f:
        for issue in open_issues:
            marker = json.dumps({"id": issue["id"], "resolved": True, "resolved_at": ts})
            f.write(marker + "\n")
    print(f"Resolved {len(open_issues)} issue(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline V2 control")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status",           help="Show pipeline status + doctor summary")
    p.add_argument("--brief", action="store_true", help="One-line summary only")

    p = sub.add_parser("restart-from-chunk",
                       help="Safely restart pipeline from chunk N (clears sentinels, restores checkpoint)")
    p.add_argument("chunk", type=int)

    sub.add_parser("pause",               help="Pause orchestrator")
    sub.add_parser("resume",              help="Clear pause signal")
    sub.add_parser("abort",               help="Abort orchestrator and prep")
    sub.add_parser("restart-orchestrator",help="Restart iris-orch tmux window")

    p = sub.add_parser("force-next-chunk",help="Force-pass validation for a chunk")
    p.add_argument("chunk", type=int)

    p = sub.add_parser("retry",          help="Reset retry counter and restart a failed step")
    p.add_argument("chunk", type=int)
    p.add_argument("step")

    p = sub.add_parser("clear-error",    help="Clear error sentinel for a step")
    p.add_argument("chunk", type=int)
    p.add_argument("step")

    p = sub.add_parser("mark-done",      help="Manually mark a step done")
    p.add_argument("chunk", type=int)
    p.add_argument("step")

    sub.add_parser("dispatch-read",      help="Show open dispatch issues")

    p = sub.add_parser("dispatch-resolve",    help="Mark a single dispatch issue resolved")
    p.add_argument("issue_id")

    p = sub.add_parser("dispatch-resolve-all", help="Resolve all open dispatch issues")
    p.add_argument("--chunk", type=int, default=None, help="Limit to a specific chunk")

    p = sub.add_parser("populate-anchor-shards",
                       help="Sample production shards into anchor_shards/ for chunk 2+ training")
    p.add_argument("--rate", type=int, default=10,
                   help="Copy every Nth shard (default: 10 = 10%% of shards)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing anchor shards")

    args = ap.parse_args()
    handlers = {
        "status":                  cmd_status,
        "restart-from-chunk":      cmd_restart_from_chunk,
        "pause":                   cmd_pause,
        "resume":                  cmd_resume,
        "abort":                   cmd_abort,
        "restart-orchestrator":    cmd_restart_orchestrator,
        "force-next-chunk":        cmd_force_next_chunk,
        "retry":                   cmd_retry,
        "clear-error":             cmd_clear_error,
        "mark-done":               cmd_mark_done,
        "dispatch-read":           cmd_dispatch_read,
        "dispatch-resolve":        cmd_dispatch_resolve,
        "dispatch-resolve-all":    cmd_dispatch_resolve_all,
        "populate-anchor-shards":  cmd_populate_anchor_shards,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
