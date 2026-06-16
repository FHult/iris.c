#!/bin/bash
# flywheel_chain_iter24.sh — two-hop chain: archive source-probe → launch iter24
# replay → archive iter24 replay. One-shot, unattended.
#
# Built on the same primitives as flywheel_chain.sh (watch the iris-flywheel tmux
# window for closure as the "campaign ended" signal; 2-poll debounce; MAX_WAIT cap;
# settle for GPU-lock release; sentinel idempotency). Adds archive-to-cold steps
# the single-hop watcher doesn't have.
#
# SEQUENCE
#   [source-probe's iris-flywheel window closes]
#     -> settle, then archive source-probe champion  -> /Volumes/16TBCold/weights/source-probe-YYYYMMDD/
#     -> launch iter24-replay (pipeline_ctl start-flywheel)
#   [iter24-replay's iris-flywheel window closes]
#     -> settle, then archive iter24-replay champion -> /Volumes/16TBCold/weights/iter24-replay-YYYYMMDD/
#     -> write sentinel (.flywheel_chain_iter24.done)
#
# CHAMPION SELECTION (important): the archived checkpoint is the campaign champion
# by held-out cond_gap, read from campaign_summary.best_checkpoint in
# flywheel_history.db — NOT best.json (which is the last train-run's loss-best) and
# NOT iterNNNN_best.safetensors by guesswork. The orchestrator preserves the
# champion's step weights AND its EMA (iterNNNN_best.safetensors) through pruning
# (orchestrator.py FLYWHEEL-CKPT-1), so both still exist when this fires. We copy
# the step weights as best.safetensors and the EMA as best_ema.safetensors, plus a
# manifest.json recording cond_gap / source paths / git sha.
#
# RUN IT IN TMUX, not launchd (launchd can't write /Volumes without Full Disk
# Access). Arm:
#   tmux new-window -t iris -n iris-chain \
#     "caffeinate -dims env PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
#      bash $PWD/train/scripts/flywheel_chain_iter24.sh"
# Disarm:
#   tmux kill-window -t iris:iris-chain ; rm -f /Volumes/2TBSSD/.flywheel_chain_iter24.done
#
# Env (all have defaults):
#   PIPELINE_DATA_ROOT   data root          (default /Volumes/2TBSSD)
#   CHAIN_REPLAY_CONFIG  replay config      (default train/configs/flywheel_iter24_replay.yaml)
#   CHAIN_COLD_DIR       cold weights dir   (default /Volumes/16TBCold/weights)
#   CHAIN_POLL_SECS      poll interval      (default 300)
#   CHAIN_MAX_WAIT_H     per-hop give-up cap (default 72)
#   CHAIN_SETTLE_SECS    GPU-lock settle    (default 90)
#   CHAIN_SENTINEL       one-shot marker    (default $DATA_ROOT/.flywheel_chain_iter24.done)
#   CHAIN_DELETE_SCRATCH 1 = delete shard_scores_scratch.db before launch (default 1;
#                        replay uses ephemeral_scores so a stale scratch DB would
#                        widen the pool past the intended 40 shards — see replay
#                        config header "LAUNCH PREREQ")
#   CHAIN_DRY_RUN        1 = log decisions, never archive/launch/sentinel
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

DATA_ROOT="${PIPELINE_DATA_ROOT:-/Volumes/2TBSSD}"
REPLAY_CONFIG="${CHAIN_REPLAY_CONFIG:-$REPO/train/configs/flywheel_iter24_replay.yaml}"
COLD_DIR="${CHAIN_COLD_DIR:-/Volumes/16TBCold/weights}"
WIN="iris-flywheel"
POLL="${CHAIN_POLL_SECS:-300}"
MAX_WAIT_H="${CHAIN_MAX_WAIT_H:-72}"
SETTLE="${CHAIN_SETTLE_SECS:-90}"
SENTINEL="${CHAIN_SENTINEL:-$DATA_ROOT/.flywheel_chain_iter24.done}"
DELETE_SCRATCH="${CHAIN_DELETE_SCRATCH:-1}"
DRY_RUN="${CHAIN_DRY_RUN:-0}"
# Hop 1's source-probe orchestrator runs in detached shell s001 (PID 9929), NOT in
# the iris-flywheel tmux window, so PID death — not window closure — is its end signal.
ORCH_PID="${CHAIN_ORCH_PID:-9929}"

DB="$DATA_ROOT/flywheel_history.db"
SCRATCH_DB="$DATA_ROOT/shard_scores_scratch.db"
LOG="$DATA_ROOT/logs/flywheel_chain_iter24.log"
CTL="$REPO/train/.venv/bin/python $REPO/train/scripts/pipeline_ctl.py"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

# --- hard preconditions ----------------------------------------------------
# tmux IS the completion signal. Without it, win_gone would falsely report "gone"
# and we'd archive/launch on top of a running campaign. Refuse rather than misfire.
TMUX_BIN="$(command -v tmux || true)"
if [ -z "$TMUX_BIN" ]; then
  log "FATAL: tmux not found on PATH — refusing to run (cannot detect campaign end safely)"; exit 3
fi
if ! command -v sqlite3 >/dev/null 2>&1; then
  log "FATAL: sqlite3 not found on PATH — cannot read champion from $DB"; exit 3
fi
if [ -f "$SENTINEL" ]; then
  log "sentinel $SENTINEL present — chain already fired, exiting"; exit 0
fi
if [ ! -f "$REPLAY_CONFIG" ]; then
  log "FATAL: replay config not found: $REPLAY_CONFIG"; exit 1
fi
if [ ! -d "$COLD_DIR" ]; then
  log "FATAL: cold weights dir not found: $COLD_DIR (is /Volumes/16TBCold mounted?)"; exit 1
fi

win_gone() { ! "$TMUX_BIN" list-windows -a -F '#{window_name}' 2>/dev/null | grep -qx "$WIN"; }

# wait_close — block until the iris-flywheel window is gone for 2 consecutive
# polls, or the per-hop cap is exceeded. Returns 0 on close, 2 on cap.
wait_close() {
  local what="$1" start streak
  start=$(date +%s); streak=0
  log "[$what] watching '$WIN'; poll ${POLL}s, cap ${MAX_WAIT_H}h"
  while true; do
    if win_gone; then
      streak=$((streak + 1))
      log "[$what] window '$WIN' absent (streak $streak/2)"
      [ "$streak" -ge 2 ] && return 0
    else
      streak=0
    fi
    if [ $(( $(date +%s) - start )) -gt $((MAX_WAIT_H * 3600)) ]; then
      log "[$what] MAX_WAIT_H (${MAX_WAIT_H}h) exceeded, window still present — giving up WITHOUT proceeding (possible hang)."
      return 2
    fi
    sleep "$POLL"
  done
}

# wait_pid_close — block until the orchestrator PID is gone for 2 consecutive
# polls, or the per-hop cap is exceeded. Returns 0 on close, 2 on cap.
# Hop 1 uses this instead of wait_close: the source-probe orchestrator runs in
# detached shell s001 (PID $ORCH_PID), not the iris-flywheel tmux window, so the
# process exiting — not a window closing — is the "campaign ended" signal.
wait_pid_close() {
  local what="$1" start streak
  start=$(date +%s); streak=0
  log "[$what] watching orchestrator PID $ORCH_PID; poll ${POLL}s, cap ${MAX_WAIT_H}h"
  while true; do
    if ! kill -0 "$ORCH_PID" 2>/dev/null; then
      streak=$((streak + 1))
      log "[$what] orchestrator PID $ORCH_PID gone (streak $streak/2)"
      [ "$streak" -ge 2 ] && return 0
    else
      streak=0
    fi
    if [ $(( $(date +%s) - start )) -gt $((MAX_WAIT_H * 3600)) ]; then
      log "[$what] MAX_WAIT_H (${MAX_WAIT_H}h) exceeded, PID $ORCH_PID still alive — giving up WITHOUT proceeding (possible hang)."
      return 2
    fi
    sleep "$POLL"
  done
}

# wait_open — after launching, confirm the window actually appears (start-flywheel
# creates it synchronously, but guard against tmux lag / launch failure).
wait_open() {
  local what="$1" i
  for i in $(seq 1 30); do
    win_gone || { log "[$what] window '$WIN' is up"; return 0; }
    sleep 2
  done
  log "[$what] window '$WIN' never appeared after launch — aborting chain"
  return 1
}

# archive_champion <flywheel_name> <archive_label>
#   Reads campaign_summary.best_checkpoint (the max-cond_gap champion) and copies it
#   + its EMA sibling + a manifest into $COLD_DIR/<archive_label>-YYYYMMDD/.
#   Returns 0 on success, non-zero on any problem (caller decides whether to abort).
archive_champion() {
  local name="$1" label="$2"
  local ckpt cond_gap date dest base dir iter ema git_sha
  ckpt="$(sqlite3 "$DB" "SELECT best_checkpoint FROM campaign_summary WHERE flywheel_name='$name';" 2>/dev/null)"
  cond_gap="$(sqlite3 "$DB" "SELECT best_cond_gap FROM campaign_summary WHERE flywheel_name='$name';" 2>/dev/null)"
  if [ -z "$ckpt" ]; then
    log "[archive:$name] FAIL: no best_checkpoint in $DB (campaign produced no scored iteration?)"; return 1
  fi
  if [ ! -f "$ckpt" ]; then
    log "[archive:$name] FAIL: champion file missing on disk: $ckpt"; return 1
  fi
  date="$(date +%Y%m%d)"
  dest="$COLD_DIR/${label}-${date}"
  base="$(basename "$ckpt")"; dir="$(dirname "$ckpt")"
  iter="${base%%_step_*}"                        # e.g. iter0024
  ema="$dir/${iter}_best.safetensors"            # EMA sibling, if present
  git_sha="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

  if [ "$DRY_RUN" = "1" ]; then
    log "[archive:$name] DRY_RUN: would copy '$ckpt' (cond_gap=$cond_gap) -> $dest/best.safetensors (+EMA, +manifest)"
    return 0
  fi

  mkdir -p "$dest" || { log "[archive:$name] FAIL: cannot mkdir $dest"; return 1; }
  # copy-to-tmp + rename so an interrupted copy never leaves a partial best.safetensors
  if ! cp "$ckpt" "$dest/best.safetensors.tmp"; then
    log "[archive:$name] FAIL: copy of $ckpt failed"; rm -f "$dest/best.safetensors.tmp"; return 1
  fi
  mv "$dest/best.safetensors.tmp" "$dest/best.safetensors"
  log "[archive:$name] archived champion -> $dest/best.safetensors (from $base, cond_gap=$cond_gap)"

  if [ -f "$ema" ]; then
    if cp "$ema" "$dest/best_ema.safetensors.tmp"; then
      mv "$dest/best_ema.safetensors.tmp" "$dest/best_ema.safetensors"
      log "[archive:$name] archived EMA -> $dest/best_ema.safetensors (from $(basename "$ema"))"
    else
      log "[archive:$name] WARN: EMA copy failed ($ema); step weights archived OK"
      rm -f "$dest/best_ema.safetensors.tmp"
    fi
  else
    log "[archive:$name] note: no EMA sibling at $ema (step weights archived)"
  fi

  local ema_field="null"
  [ -f "$ema" ] && ema_field="\"$ema\""
  cat > "$dest/manifest.json" <<EOF
{
  "flywheel_name": "$name",
  "archive_label": "$label",
  "champion_checkpoint": "$ckpt",
  "champion_ema": $ema_field,
  "best_cond_gap": $cond_gap,
  "git_sha": "$git_sha",
  "archived_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  log "[archive:$name] manifest written -> $dest/manifest.json"
  return 0
}

# === HOP 1: source-probe ends -> archive -> launch replay ===================
log "chain armed. Hop 1: watching orchestrator PID $ORCH_PID for source-probe to end."
wait_pid_close "source-probe" || { log "aborting (hop-1 cap or error); nothing launched."; exit 2; }
log "source-probe ended. Settling ${SETTLE}s for GPU-lock release…"
sleep "$SETTLE"

archive_champion "source-probe" "source-probe" \
  || log "WARN: source-probe archive failed — continuing to launch replay so the GPU is not wasted. Investigate $LOG."

# Replay prereq: a stale ephemeral scratch DB would widen the pinned 40-shard pool.
if [ -f "$SCRATCH_DB" ]; then
  if [ "$DELETE_SCRATCH" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    rm -f "$SCRATCH_DB" && log "deleted stale $SCRATCH_DB (replay uses ephemeral_scores; pool must be exactly 40)"
  else
    log "WARN: $SCRATCH_DB present and DELETE_SCRATCH!=1 — replay pool may exceed 40 shards. Remove it before launch."
  fi
fi

# Guard against stacking: the window must be gone before we launch the replay.
if ! win_gone; then
  log "FATAL: '$WIN' reappeared before replay launch — refusing to stack a campaign."; exit 4
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: would launch replay -> $REPLAY_CONFIG"
else
  log "launching iter24-replay: $REPLAY_CONFIG"
  OUT=$($CTL start-flywheel "$REPLAY_CONFIG" 2>&1); RC=$?
  log "start-flywheel rc=$RC: $(echo "$OUT" | tr '\n' ' ')"
  if [ "$RC" -ne 0 ]; then
    log "FATAL: replay launch failed (rc=$RC) — not waiting/archiving. Fix and re-run."; exit "$RC"
  fi
  wait_open "iter24-replay" || exit 5
fi

# === HOP 2: replay ends -> archive -> sentinel ==============================
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: would wait for iter24-replay to finish, archive it, write sentinel. Done."
  exit 0
fi

log "Hop 2: waiting for iter24-replay (window '$WIN') to close."
wait_close "iter24-replay" || { log "aborting (hop-2 cap or error); replay champion NOT archived — archive manually from $DB."; exit 2; }
log "iter24-replay ended. Settling ${SETTLE}s…"
sleep "$SETTLE"

if archive_champion "iter24-replay" "iter24-replay"; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SENTINEL"
  log "CHAIN COMPLETE — both champions archived; sentinel written ($SENTINEL). Re-arm by removing it."
  exit 0
else
  log "iter24-replay archive FAILED — sentinel NOT written so the failure is visible. Archive manually from $DB, then touch $SENTINEL."
  exit 1
fi
