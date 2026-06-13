#!/bin/bash
# flywheel_chain.sh — launch the NEXT flywheel campaign when the current one ends.
#
# Use case: unattended GPU utilisation. The orchestrator has no campaign queue,
# so this watcher waits for the active campaign's tmux window to close (which
# happens on completion, crash, OR manual stop — any way the GPU frees), then
# starts the configured next campaign exactly once.
#
# RUN IT IN TMUX, not launchd: launchd agents cannot write the external /Volumes
# mount without Full Disk Access (fails EX_CONFIG/Operation-not-permitted), and
# start-flywheel needs that access. tmux runs in the user session (full access)
# and persists across logout-free absences exactly like the flywheel it chains.
#   ARM:
#     tmux new-window -t iris -n iris-chain \
#       "caffeinate -dims env \
#        CHAIN_NEXT_CONFIG=$PWD/train/configs/flywheel_source_probe.yaml \
#        PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
#        bash $PWD/train/scripts/flywheel_chain.sh"
#   DISARM: tmux kill-window -t iris:iris-chain ; rm -f /Volumes/2TBSSD/.flywheel_chain.done
#
# Robust + idempotent:
#   - completion signal = the iris-flywheel tmux window is gone (debounced 2×),
#     which a PAUSE does NOT trigger (the orchestrator stays alive while paused).
#   - a sentinel prevents a second launch if this script is ever re-run.
#   - a safety cap: if the wait exceeds MAX_WAIT_H, exit WITHOUT launching
#     (a hung campaign must not get a colliding second campaign on top).
#   - runs under caffeinate (via the launchd plist) so the Mac stays awake in the
#     gap between the two campaigns' own caffeinate wrappers.
#
# Env (all have defaults):
#   CHAIN_NEXT_CONFIG   flywheel config to launch next (REQUIRED)
#   CHAIN_WAIT_WINDOW   tmux window to watch (default iris-flywheel)
#   CHAIN_POLL_SECS     poll interval (default 300)
#   CHAIN_MAX_WAIT_H    give-up cap in hours (default 72)
#   CHAIN_SENTINEL      one-shot marker (default $DATA_ROOT/.flywheel_chain.done)
#   CHAIN_DRY_RUN       1 = report detection, never launch
set -u
# launchd hands us a minimal PATH; ensure the tools we shell out to are found.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# CRITICAL: tmux is the completion signal. If it can't be found, win_gone would
# falsely report "gone" and launch the next campaign ON TOP of a running one.
# Refuse to run rather than misfire.
TMUX_BIN="$(command -v tmux || true)"
if [ -z "$TMUX_BIN" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FATAL: tmux not found on PATH — refusing to run (cannot detect campaign end safely)" \
    >> "${PIPELINE_DATA_ROOT:-/Volumes/2TBSSD}/logs/flywheel_chain.log"
  exit 3
fi
DATA_ROOT="${PIPELINE_DATA_ROOT:-/Volumes/2TBSSD}"
WIN="${CHAIN_WAIT_WINDOW:-iris-flywheel}"
POLL="${CHAIN_POLL_SECS:-300}"
MAX_WAIT_H="${CHAIN_MAX_WAIT_H:-72}"
SENTINEL="${CHAIN_SENTINEL:-$DATA_ROOT/.flywheel_chain.done}"
LOG="$DATA_ROOT/logs/flywheel_chain.log"
CTL="$REPO/train/.venv/bin/python $REPO/train/scripts/pipeline_ctl.py"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

if [ -z "${CHAIN_NEXT_CONFIG:-}" ]; then
  log "ERROR: CHAIN_NEXT_CONFIG unset — nothing to chain"; exit 1
fi
if [ -f "$SENTINEL" ]; then
  log "sentinel $SENTINEL present — chain already fired, exiting"; exit 0
fi

win_gone() { ! "$TMUX_BIN" list-windows -a -F '#{window_name}' 2>/dev/null | grep -qx "$WIN"; }

log "watching window '$WIN'; will launch '$CHAIN_NEXT_CONFIG' when it closes (poll ${POLL}s, cap ${MAX_WAIT_H}h)"
START=$(date +%s)
gone_streak=0
while true; do
  if win_gone; then
    gone_streak=$((gone_streak + 1))
    log "window '$WIN' absent (streak $gone_streak/2)"
    if [ "$gone_streak" -ge 2 ]; then break; fi
  else
    gone_streak=0
  fi
  if [ $(( $(date +%s) - START )) -gt $((MAX_WAIT_H * 3600)) ]; then
    log "MAX_WAIT_H (${MAX_WAIT_H}h) exceeded and window still present — giving up WITHOUT launching (possible hang; not stacking a campaign)."
    exit 2
  fi
  sleep "$POLL"
done

# Window gone for two consecutive polls → the prior campaign's orchestrator exited.
SETTLE="${CHAIN_SETTLE_SECS:-90}"
log "prior campaign ended. Settling ${SETTLE}s for GPU lock release…"
sleep "$SETTLE"

if [ "${CHAIN_DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN: would launch → $CHAIN_NEXT_CONFIG"; exit 0
fi

log "launching next campaign: $CHAIN_NEXT_CONFIG"
OUT=$($CTL start-flywheel "$CHAIN_NEXT_CONFIG" 2>&1)
RC=$?
log "start-flywheel rc=$RC: $(echo "$OUT" | tr '\n' ' ')"
if [ "$RC" -eq 0 ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SENTINEL"
  log "chain complete — sentinel written ($SENTINEL). Re-arm by removing it."
fi
exit "$RC"
