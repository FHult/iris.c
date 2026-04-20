#!/bin/bash
# train/scripts/pipeline_logs.sh — View the active pipeline log.
#
# Auto-selects the most relevant log for what is currently running:
#   build_shards running  → /tmp/build_shards.log
#   precompute running    → DATA_ROOT/logs/precompute*.log (most recent)
#   training running      → DATA_ROOT/logs/pipeline_chunk*.log (most recent)
#   nothing running       → most recent log file found
#
# Heartbeat line formats (emitted by each step):
#   build_shards:     [worker N] src X/Y | written N records | shards A/B full
#   clip_dedup:       [X/N] N duplicates found
#   filter_shards:    [X/Y] kept=N  dropped=N  X.X shards/s  ETA Xm
#   precompute_qwen3: [X/Y] N,NNN embeddings  X.XX shards/s  ETA Xm
#   precompute_vae:   [X/Y] N,NNN latents  X.XX shards/s  ETA Xm
#   precompute_siglip:[X/Y] N,NNN features  X.XX shards/s  ETA Xm
#   train_ip_adapter: step X,XXX/105,000  loss X.XXXX (avg X.XXXX)  lr X  X steps/s  ETA Xh XXm
#
# Usage:
#   bash train/scripts/pipeline_logs.sh                 # last 60 lines
#   bash train/scripts/pipeline_logs.sh --lines 200     # more context
#   bash train/scripts/pipeline_logs.sh --follow        # stream live (Ctrl-C to exit)
#   bash train/scripts/pipeline_logs.sh --progress      # only heartbeat/progress lines
#   bash train/scripts/pipeline_logs.sh --all           # list all log files
#
# Safe to call from Claude CoWork Dispatch (--follow not suitable for dispatch).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Args ──────────────────────────────────────────────────────────────────────
LINES=60
FOLLOW=false
LIST_ALL=false
PROGRESS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lines)    LINES="$2"; shift 2 ;;
        --follow)   FOLLOW=true; shift ;;
        --all)      LIST_ALL=true; shift ;;
        --progress) PROGRESS_ONLY=true; shift ;;
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
elif pgrep -f "precompute_qwen3\|precompute_vae\|precompute_siglip\|filter_shards\|clip_dedup" &>/dev/null; then
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

# ── Progress-only mode ────────────────────────────────────────────────────────
# Extracts only heartbeat/progress lines — useful for a quick status overview
# without noise from startup messages, stack traces, or verbose model output.
if $PROGRESS_ONLY; then
    # Pattern covers all heartbeat formats from the pipeline scripts:
    #   [X/Y] ...               — shard-level progress (filter, precompute, dedup)
    #   [worker N] src X/Y ...  — build_shards worker heartbeat
    #   step X,XXX/...          — training step progress
    HEARTBEAT_PAT='(\[worker [0-9]+\] src [0-9]+/[0-9]+|\[[0-9,]+/[0-9,]+\] (kept=|[0-9,]+ (embeddings|latents|features|duplicates))|step +[0-9,]+/[0-9,]+.*steps/s)'
    MATCHES=$(grep -E "$HEARTBEAT_PAT" "$ACTIVE_LOG" 2>/dev/null | tail -"$LINES" | sed 's/^[[:space:]]*//')
    if [[ -n "$MATCHES" ]]; then
        echo "$MATCHES"
    else
        echo "(no heartbeat lines found yet — pipeline may be starting up)"
        echo ""
        echo "Last 10 lines:"
        tail -10 "$ACTIVE_LOG"
    fi
    exit 0
fi

# ── Normal mode ───────────────────────────────────────────────────────────────
if $FOLLOW; then
    tail -n "$LINES" -f "$ACTIVE_LOG"
else
    tail -n "$LINES" "$ACTIVE_LOG"
fi
