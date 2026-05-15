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
    python train/scripts/pipeline_ctl.py clear-phantoms         # fix phantom sentinels
    python train/scripts/pipeline_ctl.py create-val-set         # one-time: build held-out val set

Flywheel:
    python train/scripts/pipeline_ctl.py start-flywheel train/configs/flywheel_sref_v1.yaml
    python train/scripts/pipeline_ctl.py flywheel-status
    python train/scripts/pipeline_ctl.py pause-flywheel
    python train/scripts/pipeline_ctl.py resume-flywheel
    python train/scripts/pipeline_ctl.py stop-flywheel
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, CONTROL_FILE, DISPATCH_QUEUE, LOG_DIR, SENTINEL_DIR,
    TMUX_SESSION, TMUX_ORCH_WIN, TMUX_TRAIN_WIN, TMUX_PREP_WIN,
    TMUX_ABLATION_WIN, TMUX_FLYWHEEL_WIN,
    SCRIPTS_DIR, TRAIN_DIR,
    CKPT_ARCHIVE_DIR, CKPT_DIR, HARD_EX_DIR, SHARDS_DIR, ANCHOR_SHARDS_DIR,
    ABLATION_CONTROL_FILE, ABLATION_DB_PATH,
    FLYWHEEL_CONTROL_FILE, FLYWHEEL_DB_PATH, SHARD_SCORES_DB_PATH,
    COLD_ROOT, COLD_WEIGHTS_DIR, COLD_VAL_SHARDS_DIR, COLD_VAL_PRECOMP_DIR,
    VAL_SHARDS_DIR, VAL_PRECOMP_DIR,
    clear_error, mark_done, tmux_session_exists, tmux_window_exists,
    tmux_new_window, now_iso, read_heartbeat, heartbeat_age_secs,
    load_config,
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
    config = Path(getattr(_args, "config", None) or (TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    log_file = LOG_DIR / "orchestrator.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (f"source '{TRAIN_DIR}/.venv/bin/activate' && "
           f"caffeinate -dims python -u '{SCRIPTS_DIR}/orchestrator.py' --resume --config '{config}'")
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


def _find_cold_checkpoint(chunk: int):
    """Return (safetensors, json, ema) paths from cold weights, or (None, None, None)."""
    if not COLD_ROOT.exists():
        return None, None, None
    # Scan flywheel-YYYYMMDD campaign dirs, newest first
    campaigns = sorted(COLD_WEIGHTS_DIR.glob("flywheel-*"), reverse=True)
    for campaign in campaigns:
        st  = campaign / "final.safetensors"
        js  = campaign / "final.json"
        ema = campaign / "final.ema.safetensors"
        if st.exists():
            return st, (js if js.exists() else None), (ema if ema.exists() else None)
    return None, None, None


def cmd_restart_from_chunk(args) -> None:
    """
    Safely restart the pipeline from chunk N:
      1. Pre-flight: warn if cold storage not mounted
      2. Kill iris-train window if running (with confirmation)
      3. Clear all sentinels for chunks N..total_chunks
      4. Delete hard_examples/chunk{M}/ for M >= N so mining re-runs against
         the new checkpoint (stale manifests would otherwise cause a zero-op re-mine)
      5. Restore chunk N-1 final checkpoint — from hot archive, falling back to cold
      6. Restart orchestrator
    """
    import shutil
    chunk = args.chunk

    # Cold mount pre-flight
    if not COLD_ROOT.exists():
        print(f"WARNING: cold storage not mounted at {COLD_ROOT}")
        print("         Checkpoint restore will use hot archive only.")
        print("         Mount cold storage before restarting if hot archive is missing.")
        print()

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

    # Resolve checkpoint source
    ckpt_source = None
    ckpt_source_label = None
    if chunk > 1:
        arch_st  = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.safetensors"
        arch_js  = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.json"
        arch_ema = CKPT_ARCHIVE_DIR / f"chunk{chunk - 1}_final.ema.safetensors"
        if arch_st.exists():
            ckpt_source = (arch_st, arch_js if arch_js.exists() else None,
                           arch_ema if arch_ema.exists() else None)
            ckpt_source_label = f"hot archive ({arch_st})"
        else:
            cold_st, cold_js, cold_ema = _find_cold_checkpoint(chunk)
            if cold_st:
                ckpt_source = (cold_st, cold_js, cold_ema)
                ckpt_source_label = f"cold storage ({cold_st})"

    print(f"Restarting pipeline from chunk {chunk} (total={total})")
    print(f"This will:")
    print(f"  - Kill iris-train if running")
    print(f"  - Delete sentinels for chunks {chunk}..{total}")
    if hard_ex_to_delete:
        for d in hard_ex_to_delete:
            tars = list(d.glob("*.tar"))
            print(f"  - Delete hard_examples/{d.name}/ ({len(tars)} tar(s)) — stale without re-training")
    if chunk > 1:
        if ckpt_source:
            print(f"  - Restore chunk {chunk - 1} checkpoint from {ckpt_source_label}")
        else:
            print(f"  - WARNING: no checkpoint found for chunk {chunk - 1} in hot archive or cold")
            print(f"             Training will resume from whatever is currently in {CKPT_DIR}")
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

    # Restore checkpoint from hot archive or cold fallback
    if ckpt_source:
        src_st, src_js, src_ema = ckpt_source
        step_name = "restored"
        if src_js is not None:
            try:
                meta = json.loads(src_js.read_text())
                step_name = f"step_{meta.get('step', 'restored'):07d}"
            except Exception:
                pass
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        dst_st  = CKPT_DIR / f"{step_name}.safetensors"
        dst_js  = CKPT_DIR / f"{step_name}.json"
        dst_ema = CKPT_DIR / f"{step_name}.ema.safetensors"
        shutil.copy2(src_st, dst_st)
        if src_js is not None:
            shutil.copy2(src_js, dst_js)
        if src_ema is not None:
            shutil.copy2(src_ema, dst_ema)
        print(f"  Restored checkpoint → {dst_st.name}  (from {ckpt_source_label})")

    # Restart orchestrator
    cmd_restart_orchestrator(args)


def cmd_clear_phantoms(_args) -> None:
    """
    Run pipeline_doctor --ai, find phantom sentinel issues, and execute their
    fix commands with a single confirmation prompt.

    A phantom sentinel is one where the sentinel file claims a step completed
    but the underlying hot data no longer exists (e.g. after hot storage was
    cleaned and cold wasn't yet staged back).
    """
    import shlex
    venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")
    doctor_script = str(SCRIPTS_DIR / "pipeline_doctor.py")
    result = subprocess.run(
        [venv_python, doctor_script, "--ai"],
        capture_output=True, text=True,
    )
    # Doctor exits 1 when there are CRITICAL issues — that's the normal case here.
    # Only treat it as a hard failure if stdout is not valid JSON.
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("ERROR: pipeline_doctor --ai failed or returned non-JSON output")
        print(result.stderr[-2000:] if result.stderr else result.stdout[:500])
        return

    issues = report.get("issues", [])
    phantoms = [i for i in issues if i.get("category") == "phantom"]
    if not phantoms:
        print("No phantom sentinel issues found.")
        return

    print(f"Found {len(phantoms)} phantom sentinel(s):")
    fix_commands = []
    for issue in phantoms:
        title = issue.get("title", "?")
        chunk = issue.get("chunk", "?")
        ctx   = issue.get("context", {})
        step  = ctx.get("step", "")
        fix   = issue.get("fix", "")
        step_str = f" step={step}" if step else ""
        print(f"  chunk={chunk}{step_str}: {title}")
        if fix:
            print(f"    fix: {fix}")
            # A fix may be multi-line; treat each non-empty line as a command
            for line in fix.splitlines():
                line = line.strip()
                if line:
                    fix_commands.append(line)

    if not fix_commands:
        print("No fix commands available — clear sentinels manually.")
        return

    if getattr(_args, "yes", False):
        confirm = "y"
    else:
        confirm = input(f"\nRun {len(fix_commands)} fix command(s)? (y/N): ").strip()
    if confirm.lower() != "y":
        print("Aborted.")
        return

    for cmd in fix_commands:
        print(f"  Running: {cmd}")
        try:
            subprocess.run(shlex.split(cmd), check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: command exited {e.returncode}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("Done. Run 'pipeline_ctl.py status' to verify.")


def cmd_create_val_set(args) -> None:
    """
    One-time operation: build and precompute the permanent held-out validation set.

    Reads tgz 0 (000.tar) from the cold converted pool, builds 2 shards (~1000 records),
    and runs precompute (qwen3 + vae + siglip) — all stored in cold/validation/.
    Idempotent: exits immediately if the sentinel already exists.
    To rebuild: delete cold/validation/held_out/.val_set_created and re-run.

    Run this ONCE before the first training run.  pipeline_setup.py stages the results
    to hot before each training session automatically.
    """
    import os
    import tempfile

    config_path = getattr(args, "config", None) or (TRAIN_DIR / "configs" / "v2_pipeline.yaml")
    try:
        cfg = load_config(str(config_path))
    except Exception as e:
        print(f"ERROR: could not load config {config_path}: {e}")
        return

    storage    = cfg.get("storage", {})
    cold_root  = Path(storage.get("cold_root", str(COLD_ROOT)))
    val_shards = cold_root / "validation" / "held_out"
    val_precomp= cold_root / "validation" / "precomputed"
    sentinel   = val_shards / ".val_set_created"

    if sentinel.exists():
        print(f"Val set already created ({sentinel})")
        print("  To rebuild: delete the sentinel and re-run.")
        return

    if not cold_root.exists():
        print(f"ERROR: cold storage not mounted at {cold_root}")
        return

    conv_pool_str = storage.get("converted_pool_root")
    if not conv_pool_str:
        print("ERROR: converted_pool_root not set in config storage block.")
        print("       Enable it and run download_convert.py --cold-only first.")
        return
    conv_pool = Path(conv_pool_str)

    val_tgzs    = cfg.get("jdb", {}).get("validation_tgzs", [0, 0])
    val_tgz_idx = val_tgzs[0]
    val_tar     = conv_pool / f"{val_tgz_idx:03d}.tar"
    pool_sent   = conv_pool / ".converted" / f"{val_tgz_idx:03d}.done"

    if not pool_sent.exists() or not val_tar.exists():
        print(f"Val tgz {val_tgz_idx:03d} not found in converted pool ({conv_pool}).")
        if not getattr(args, "yes", False):
            ans = input("  Download and convert now? (y/N): ").strip()
            if ans.lower() != "y":
                print("Aborted. Run: download_convert.py --cold-only to populate the pool first.")
                return
        venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")
        print(f"  Downloading/converting tgz {val_tgz_idx:03d}...")
        r = subprocess.run([
            venv_python, str(SCRIPTS_DIR / "download_convert.py"),
            "--config", str(config_path),
            "--chunk", "1",
            "--tgz-start", str(val_tgz_idx),
            "--tgz-end",   str(val_tgz_idx),
            "--cold-only",
        ])
        if r.returncode != 0:
            print(f"ERROR: download_convert failed (exit {r.returncode})")
            return
        if not val_tar.exists():
            print(f"ERROR: converted tar still not found at {val_tar}")
            return

    val_shards.mkdir(parents=True, exist_ok=True)
    venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")

    # Step 1: build val shards from just the val tgz via a temp source dir.
    print(f"\nStep 1/2: building val shards from {val_tar.name} → {val_shards}")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.symlink(val_tar.resolve(), Path(tmpdir) / val_tar.name)
        r = subprocess.run([
            venv_python, str(SCRIPTS_DIR / "build_shards.py"),
            "--sources",    tmpdir,
            "--output",     str(val_shards),
            "--shard_size", "500",
            "--max-shards", "2",
            "--workers",    "1",
            "--seed",       "999",
        ])
    if r.returncode != 0:
        print(f"ERROR: build_shards failed (exit {r.returncode})")
        return

    built = sorted(val_shards.glob("*.tar"))
    if not built:
        print("ERROR: no shards produced — build_shards may have found no records in the source.")
        return
    print(f"  Built {len(built)} val shard(s): {', '.join(s.name for s in built)}")

    # Step 2: precompute qwen3 + vae + (optionally) siglip for val shards.
    print(f"\nStep 2/2: precomputing embeddings for val shards → {val_precomp}")
    model_cfg  = cfg.get("model", {})
    flux_model = model_cfg.get("flux_model", "flux-klein-model")
    do_siglip  = cfg.get("training", {}).get("siglip", True)

    precomp_cmd = [
        venv_python, str(SCRIPTS_DIR / "precompute_all.py"),
        "--shards",       str(val_shards),
        "--qwen3-output", str(val_precomp / "qwen3"),
        "--vae-output",   str(val_precomp / "vae"),
        "--siglip-output",str(val_precomp / "siglip"),
        "--flux-model",   flux_model,
    ]
    if do_siglip:
        precomp_cmd.append("--siglip")

    r = subprocess.run(precomp_cmd)
    if r.returncode != 0:
        print(f"ERROR: precompute_all failed (exit {r.returncode})")
        return

    sentinel.write_text(now_iso() + "\n")
    print(f"\nVal set created:")
    print(f"  shards:      {val_shards}  ({len(built)} shard(s))")
    print(f"  precomputed: {val_precomp}")
    print(f"  sentinel:    {sentinel}")
    print(f"\nRun 'pipeline_setup.py' before training to stage the val set to hot.")


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


def cmd_start_ablation(args) -> None:
    """Launch ablation_harness.py in a dedicated iris-ablation tmux window."""
    from pipeline_lib import (
        SHARDS_DIR, PRECOMP_DIR,
    )
    config = Path(args.ablation_config)
    if not config.exists():
        print(f"ERROR: config not found: {config}")
        return
    output_dir = Path(args.output_dir) if args.output_dir else DATA_ROOT / "ablation_long"
    db_path = ABLATION_DB_PATH

    if tmux_window_exists(TMUX_ABLATION_WIN):
        print(f"ERROR: {TMUX_ABLATION_WIN} window already running — stop it first with ablation-stop")
        return

    if not tmux_session_exists():
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-n", TMUX_ABLATION_WIN],
                       check=True)

    log_file = LOG_DIR / "ablation.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any stale stop signal
    ABLATION_CONTROL_FILE.unlink(missing_ok=True)

    shards      = args.shards      or str(SHARDS_DIR)
    qwen3_cache = args.qwen3_cache or str(PRECOMP_DIR / "qwen3")
    vae_cache   = args.vae_cache   or str(PRECOMP_DIR / "vae")
    siglip_cache= args.siglip_cache or str(PRECOMP_DIR / "siglip")

    cmd = (
        f"source '{TRAIN_DIR}/.venv/bin/activate' && "
        f"caffeinate -dims python -u '{SCRIPTS_DIR}/ablation_harness.py' "
        f"--config '{config}' --output-dir '{output_dir}' --db '{db_path}' "
        f"--shards '{shards}' "
        f"--qwen3-cache '{qwen3_cache}' "
        f"--vae-cache '{vae_cache}' "
        f"--siglip-cache '{siglip_cache}'"
    )
    tmux_new_window(TMUX_ABLATION_WIN, cmd, log_file)
    print(f"Ablation started → {TMUX_ABLATION_WIN} window")
    print(f"  config:  {config}")
    print(f"  output:  {output_dir}")
    print(f"  db:      {db_path}")
    print(f"  shards:  {shards}")
    print(f"  log:     {log_file}")


def cmd_stop_ablation(_args) -> None:
    payload = {"action": "stop", "ts": now_iso()}
    ABLATION_CONTROL_FILE.write_text(json.dumps(payload))
    print("Stop signal written — harness will exit after the current run completes")
    print(f"  control file: {ABLATION_CONTROL_FILE}")


def cmd_pause_ablation(_args) -> None:
    payload = {"action": "pause", "ts": now_iso()}
    ABLATION_CONTROL_FILE.write_text(json.dumps(payload))
    print("Pause signal written — harness will pause after the current run")


def cmd_resume_ablation(_args) -> None:
    ABLATION_CONTROL_FILE.unlink(missing_ok=True)
    print("Resumed (cleared ablation control signal)")


def cmd_ablation_status(_args) -> None:
    """Show ablation heartbeat, window status, and DB summary."""
    running = tmux_window_exists(TMUX_ABLATION_WIN)
    print(f"  iris-ablation window: {'🟢 running' if running else '⬜ not running'}")

    hb = read_heartbeat("ablation")
    if hb:
        age = heartbeat_age_secs("ablation")
        age_str = f"{int(age)}s ago" if age is not None else "unknown"
        status = hb.get("status", "")
        plateau_str = f"  plateau={hb['plateau']}" if hb.get("plateau") else ""
        print(f"  heartbeat: {age_str}  run={hb.get('run_name','')}  "
              f"status={status}  "
              f"n_done={hb.get('n_done','?')}/{hb.get('n_max','?')}"
              f"{plateau_str}")
        if "plateau" in status:
            print(f"  ⚠  Campaign plateau detected — use --force-continue to keep exploring")
        if hb.get("current_combo"):
            print(f"  current:   {hb['current_combo']}  "
                  f"step={hb.get('current_step','?')}  "
                  f"loss={hb.get('current_loss','?')}  "
                  f"ref_gap={hb.get('current_ref_gap','?')}")
    else:
        print("  heartbeat: not found")

    ctrl_str = "none"
    try:
        ctrl = json.loads(ABLATION_CONTROL_FILE.read_text())
        ctrl_str = ctrl.get("action", "none")
    except (OSError, json.JSONDecodeError):
        pass
    print(f"  control signal: {ctrl_str}")

    if ABLATION_DB_PATH.exists():
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(ABLATION_DB_PATH))
            rows = conn.execute(
                "SELECT run_name, COUNT(*) as n, "
                "SUM(CASE WHEN score IS NOT NULL THEN 1 ELSE 0 END) as scored, "
                "MAX(score) as best_score "
                "FROM experiments GROUP BY run_name ORDER BY run_name"
            ).fetchall()
            conn.close()
            if rows:
                print(f"\n  DB: {ABLATION_DB_PATH}")
                for run, n, scored, best in rows:
                    best_str = f"{best:.2f}" if best is not None else "—"
                    print(f"    {run}: {n} experiments ({scored} scored)  best_score={best_str}")
        except Exception as e:
            print(f"  DB: error reading {ABLATION_DB_PATH}: {e}")
    else:
        print(f"  DB: not found ({ABLATION_DB_PATH})")


def cmd_start_flywheel(args) -> None:
    """Launch the flywheel loop via orchestrator.py in a dedicated tmux window."""
    config = Path(args.flywheel_config)
    if not config.exists():
        print(f"ERROR: config not found: {config}")
        return

    if tmux_window_exists(TMUX_FLYWHEEL_WIN):
        print(f"ERROR: {TMUX_FLYWHEEL_WIN} window already running — stop it first")
        return

    if not tmux_session_exists():
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION,
                        "-n", TMUX_FLYWHEEL_WIN], check=True)

    FLYWHEEL_CONTROL_FILE.unlink(missing_ok=True)

    log_file = LOG_DIR / "flywheel.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")
    cmd = (
        f"source '{TRAIN_DIR}/.venv/bin/activate' && "
        f"caffeinate -dims {venv_python} -u '{SCRIPTS_DIR}/orchestrator.py' "
        f"--flywheel-config '{config}'"
    )
    tmux_new_window(TMUX_FLYWHEEL_WIN, cmd, log_file)
    print(f"Flywheel started → {TMUX_FLYWHEEL_WIN} window")
    print(f"  config: {config}")
    print(f"  log:    {log_file}")


def cmd_stop_flywheel(_args) -> None:
    FLYWHEEL_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLYWHEEL_CONTROL_FILE.write_text(json.dumps({"action": "stop", "ts": now_iso()}))
    print("Stop signal written — flywheel will exit after the current iteration")
    print(f"  control file: {FLYWHEEL_CONTROL_FILE}")


def cmd_pause_flywheel(_args) -> None:
    FLYWHEEL_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLYWHEEL_CONTROL_FILE.write_text(json.dumps({"action": "pause", "ts": now_iso()}))
    print("Pause signal written — flywheel will pause after the current training window exits")


def cmd_resume_flywheel(_args) -> None:
    FLYWHEEL_CONTROL_FILE.write_text(json.dumps({"action": "resume", "ts": now_iso()}))
    print("Resume signal written")


def cmd_force_continue_flywheel(_args) -> None:
    """Clear any auto-pause (plateau or manual) and resume the flywheel immediately."""
    FLYWHEEL_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLYWHEEL_CONTROL_FILE.write_text(json.dumps({"action": "resume", "ts": now_iso(),
                                                  "force": True}))
    print("Force-continue signal written — flywheel will resume on next poll")


def cmd_flywheel_status(_args) -> None:
    """Show flywheel heartbeat, window status, and iteration DB summary."""
    running = tmux_window_exists(TMUX_FLYWHEEL_WIN)
    print(f"  {TMUX_FLYWHEEL_WIN} window: {'running' if running else 'not running'}")

    hb = read_heartbeat("flywheel")
    if hb:
        age = heartbeat_age_secs("flywheel")
        age_str = f"{int(age)}s ago" if age is not None else "unknown"
        print(f"  heartbeat:  {age_str}  name={hb.get('flywheel_name','')}  "
              f"iter={hb.get('iteration','')}  status={hb.get('status','')}")
    else:
        print("  heartbeat:  not found")

    ctrl_str = "none"
    plateau_reason_str = ""
    try:
        ctrl = json.loads(FLYWHEEL_CONTROL_FILE.read_text())
        ctrl_str = ctrl.get("action", "none")
        if ctrl.get("auto") and ctrl.get("reason"):
            plateau_reason_str = f"  [auto-pause: {ctrl['reason']}]"
    except (OSError, json.JSONDecodeError):
        pass
    print(f"  control:    {ctrl_str}{plateau_reason_str}")

    if FLYWHEEL_DB_PATH.exists():
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(FLYWHEEL_DB_PATH))
            rows = conn.execute(
                "SELECT flywheel_name, COUNT(*) as n, "
                "SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) as done, "
                "MAX(ref_gap) as best_ref_gap "
                "FROM iterations GROUP BY flywheel_name ORDER BY flywheel_name"
            ).fetchall()
            conn.close()
            if rows:
                print(f"\n  DB: {FLYWHEEL_DB_PATH}")
                for fw_name, n, done, best_rg in rows:
                    best_str = f"{best_rg:.4f}" if best_rg is not None else "—"
                    print(f"    {fw_name}: {n} iters ({done} done)  best_ref_gap={best_str}")
        except Exception as e:
            print(f"  DB: error reading {FLYWHEEL_DB_PATH}: {e}")
    else:
        print(f"  DB: not found ({FLYWHEEL_DB_PATH})")

    if SHARD_SCORES_DB_PATH.exists():
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(SHARD_SCORES_DB_PATH))
            total  = conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            scored = conn.execute("SELECT COUNT(*) FROM shards WHERE n_scored>0").fetchone()[0]
            conn.close()
            print(f"  shard DB:   {total} shards, {scored} scored")
        except Exception:
            pass


def cmd_data_explorer(_args) -> None:
    """Passthrough to data_explorer.py — data intelligence layer."""
    import subprocess
    venv_python = str(TRAIN_DIR / ".venv" / "bin" / "python")
    explorer = str(SCRIPTS_DIR / "data_explorer.py")
    # Forward all remaining CLI args directly (strip 'data-explorer' from sys.argv)
    pos = sys.argv.index("data-explorer")
    forwarded = sys.argv[pos + 1:]
    result = subprocess.run([venv_python, explorer] + forwarded)
    sys.exit(result.returncode)


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
    # data-explorer is a transparent passthrough; intercept before argparse sees the flags.
    if len(sys.argv) >= 2 and sys.argv[1] == "data-explorer":
        cmd_data_explorer(None)
        return

    ap = argparse.ArgumentParser(description="Pipeline V2 control")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status",           help="Show pipeline status + doctor summary")
    p.add_argument("--brief", action="store_true", help="One-line summary only")

    p = sub.add_parser("restart-from-chunk",
                       help="Safely restart pipeline from chunk N (clears sentinels, restores checkpoint)")
    p.add_argument("chunk", type=int)

    p = sub.add_parser("clear-phantoms",
                   help="Find and remove phantom sentinels (step claims done but hot data is missing)")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    p = sub.add_parser("create-val-set",
                       help="One-time: build and precompute the permanent held-out validation set from tgz 0")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Pipeline config file (default: train/configs/v2_pipeline.yaml)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Auto-confirm downloading the val tgz if not yet in pool")

    sub.add_parser("pause",               help="Pause orchestrator")
    sub.add_parser("resume",              help="Clear pause signal")
    sub.add_parser("abort",               help="Abort orchestrator and prep")
    p = sub.add_parser("restart-orchestrator", help="Restart iris-orch tmux window")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Pipeline config file (default: train/configs/v2_pipeline.yaml)")

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

    p = sub.add_parser("start-ablation",
                       help="Launch long-term ablation harness in iris-ablation tmux window")
    p.add_argument("ablation_config", metavar="CONFIG",
                   help="Harness config YAML (ablation: name/strategy/variables/...)")
    p.add_argument("--output-dir",    default=None, metavar="PATH",
                   help=f"Output dir (default: {DATA_ROOT}/ablation_long)")
    p.add_argument("--shards",        default=None, metavar="PATH",
                   help="Shard directory (default: DATA_ROOT/shards)")
    p.add_argument("--qwen3-cache",   default=None, metavar="PATH",
                   help="Qwen3 cache dir (default: DATA_ROOT/precomputed/qwen3)")
    p.add_argument("--vae-cache",     default=None, metavar="PATH",
                   help="VAE cache dir (default: DATA_ROOT/precomputed/vae)")
    p.add_argument("--siglip-cache",  default=None, metavar="PATH",
                   help="SigLIP cache dir (default: DATA_ROOT/precomputed/siglip)")

    sub.add_parser("stop-ablation",    help="Send stop signal to running ablation harness")
    sub.add_parser("pause-ablation",   help="Pause harness after current run")
    sub.add_parser("resume-ablation",  help="Clear pause/stop signal")
    sub.add_parser("ablation-status",  help="Show ablation heartbeat and DB summary")

    p = sub.add_parser("start-flywheel",
                       help="Launch self-improving sref flywheel in iris-flywheel tmux window")
    p.add_argument("flywheel_config", metavar="CONFIG",
                   help="Flywheel config YAML (flywheel: name/max_iterations/...)")

    sub.add_parser("stop-flywheel",          help="Stop flywheel after current iteration")
    sub.add_parser("pause-flywheel",         help="Pause flywheel (resumes on resume-flywheel)")
    sub.add_parser("resume-flywheel",        help="Resume a paused flywheel")
    sub.add_parser("force-continue-flywheel",
                   help="Clear auto-pause (plateau) and resume flywheel immediately")
    sub.add_parser("flywheel-status",        help="Show flywheel heartbeat and iteration DB summary")

    sub.add_parser("data-explorer",
                   help="Data intelligence layer: shards, checkpoints, coverage, warm-start")

    args = ap.parse_args()
    handlers = {
        "status":                  cmd_status,
        "restart-from-chunk":      cmd_restart_from_chunk,
        "clear-phantoms":          cmd_clear_phantoms,
        "create-val-set":          cmd_create_val_set,
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
        "start-ablation":          cmd_start_ablation,
        "stop-ablation":           cmd_stop_ablation,
        "pause-ablation":          cmd_pause_ablation,
        "resume-ablation":         cmd_resume_ablation,
        "ablation-status":         cmd_ablation_status,
        "start-flywheel":          cmd_start_flywheel,
        "stop-flywheel":           cmd_stop_flywheel,
        "pause-flywheel":          cmd_pause_flywheel,
        "resume-flywheel":         cmd_resume_flywheel,
        "force-continue-flywheel": cmd_force_continue_flywheel,
        "flywheel-status":         cmd_flywheel_status,
        "data-explorer":           cmd_data_explorer,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
