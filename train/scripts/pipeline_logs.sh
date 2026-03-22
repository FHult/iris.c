#!/bin/bash
# train/scripts/pipeline_logs.sh — View the active pipeline log.
#
# Auto-selects the most relevant log for what is currently running:
#   build_shards running  → /tmp/build_shards.log
#   precompute running    → DATA_ROOT/logs/precompute*.log (most recent)
#   training running      → DATA_ROOT/logs/pipeline_chunk*.log (most recent)
#   nothing running       → most recent log file found
#
# Usage:
#   bash train/scripts/pipeline_logs.sh                 # last 60 lines
#   bash train/scripts/pipeline_logs.sh --lines 200     # more context
#   bash train/scripts/pipeline_logs.sh --follow        # stream live (Ctrl-C to exit)
#   bash train/scripts/pipeline_logs.sh --all           # list all log files
#
# Safe to call from Claude CoWork Dispatch (--follow not suitable for dispatch).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Args ──────────────────────────────────────────────────────────────────────
LINES=60
FOLLOW=false
LIST_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lines)   LINES="$2"; shift 2 ;;
        --follow)  FOLLOW=true; shift ;;
        --all)     LIST_ALL=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
DATA_ROOT=""
for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
    if [[ -d "$candidate" ]] && \
       [[ -d "$candidate/shards" || -d "$candidate/raw" || -d "$candidate/logs" ]]; then
        DATA_ROOT="$candidate"
        break
    fi
done
DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"

# ── List all logs mode ────────────────────────────────────────────────────────
if $LIST_ALL; then
    echo "All log files (newest first):"
    ls -lt "$DATA_ROOT/logs/"*.log "$TRAIN_DIR/data/logs/"*.log /tmp/build_shards.log 2>/dev/null \
        | awk '{print "  " $6, $7, $8, $9}' | head -20 || echo "  (none found)"
    exit 0
fi

# ── Find active log ───────────────────────────────────────────────────────────
ACTIVE_LOG=""

if pgrep -f build_shards &>/dev/null && [[ -f /tmp/build_shards.log ]]; then
    ACTIVE_LOG=/tmp/build_shards.log
elif pgrep -f "precompute_qwen3\|precompute_vae\|precompute_siglip" &>/dev/null; then
    ACTIVE_LOG=$(ls -t "$DATA_ROOT/logs"/precompute*.log 2>/dev/null | head -1 || true)
    [[ -z "$ACTIVE_LOG" ]] && ACTIVE_LOG=$(ls -t "$DATA_ROOT/logs"/*.log 2>/dev/null | head -1 || true)
elif pgrep -f "train_ip_adapter\|run_training_pipeline" &>/dev/null; then
    ACTIVE_LOG=$(ls -t "$DATA_ROOT/logs"/pipeline_chunk*.log "$TRAIN_DIR/data/logs"/pipeline_chunk*.log 2>/dev/null | head -1 || true)
fi

# Fall back to most recently modified log file
if [[ -z "$ACTIVE_LOG" ]]; then
    ACTIVE_LOG=$(ls -t \
        /tmp/build_shards.log \
        "$DATA_ROOT/logs"/pipeline_chunk*.log \
        "$DATA_ROOT/logs"/*.log \
        "$TRAIN_DIR/data/logs"/pipeline_chunk*.log \
        2>/dev/null | head -1 || true)
fi

if [[ -z "$ACTIVE_LOG" ]]; then
    echo "No log file found. Pipeline may not have run yet."
    echo "  DATA_ROOT: $DATA_ROOT"
    exit 0
fi

echo "Log: $ACTIVE_LOG"
echo "─────────────────────────────────────────────────────────────────"

if $FOLLOW; then
    tail -n "$LINES" -f "$ACTIVE_LOG"
else
    tail -n "$LINES" "$ACTIVE_LOG"
fi
