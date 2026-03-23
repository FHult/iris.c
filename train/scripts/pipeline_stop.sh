#!/bin/bash
# train/scripts/pipeline_stop.sh — Gracefully stop all pipeline processes.
#
# Sends SIGTERM to running pipeline Python processes (training will complete the
# current step and save a periodic checkpoint before exiting), then kills the
# tmux sessions. Reports the latest checkpoint so you know where to resume from.
#
# Usage:
#   bash train/scripts/pipeline_stop.sh
#
# Safe to call from Claude CoWork Dispatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -d "/Volumes/2TBSSD" ]]; then
    DATA_ROOT="/Volumes/2TBSSD"
else
    DATA_ROOT="$TRAIN_DIR/data"
fi

echo "================================================================="
echo "  Stopping training pipeline"
echo "================================================================="
echo ""

# ── Collect PIDs to stop ──────────────────────────────────────────────────────
# 1. Main pipeline bash process recorded in the lock file.
LOCK_FILE="$DATA_ROOT/logs/pipeline.lock"
LOCK_PID=""
if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(grep '^pid=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2 || true)
    kill -0 "$LOCK_PID" 2>/dev/null || LOCK_PID=""  # discard if not alive
fi

# 2. Python worker processes by name pattern.
PIPELINE_PATTERNS="train_ip_adapter|build_shards|filter_shards|precompute_qwen3|precompute_vae|precompute_siglip|clip_dedup|recaption"
WORKER_PIDS=$(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true)

# Combine, deduplicate.
ALL_PIDS=$(printf '%s\n' $LOCK_PID $WORKER_PIDS | sort -u | grep -v '^$' || true)

if [[ -n "$ALL_PIDS" ]]; then
    echo "Sending SIGTERM to pipeline processes..."
    while IFS= read -r pid; do
        NAME=$(ps -p "$pid" -o comm= 2>/dev/null | xargs basename 2>/dev/null || echo "?")
        echo "  SIGTERM PID $pid ($NAME)"
        kill -TERM "$pid" 2>/dev/null || true
    done <<< "$ALL_PIDS"

    # Wait up to 30s for graceful shutdown
    echo "Waiting up to 30s for processes to exit..."
    for i in $(seq 1 30); do
        REMAINING=$(printf '%s\n' $LOCK_PID $(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true) \
                    | sort -u | grep -v '^$' \
                    | while IFS= read -r pid; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done || true)
        [[ -z "$REMAINING" ]] && break
        sleep 1
    done

    # Force-kill anything still running
    REMAINING=$(printf '%s\n' $LOCK_PID $(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true) \
                | sort -u | grep -v '^$' \
                | while IFS= read -r pid; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done || true)
    if [[ -n "$REMAINING" ]]; then
        echo "Force-killing remaining processes..."
        while IFS= read -r pid; do
            echo "  SIGKILL PID $pid"
            kill -KILL "$pid" 2>/dev/null || true
        done <<< "$REMAINING"
    fi
    echo "  Processes stopped."
else
    echo "  No pipeline processes found."
fi

echo ""

# ── Kill tmux sessions ────────────────────────────────────────────────────────
echo "Stopping tmux sessions..."
for session in pipeline build_shards precompute; do
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
        echo "  Killed tmux session: $session"
    fi
done

# ── Report latest checkpoint ──────────────────────────────────────────────────
echo ""
CKPT_DIR="$TRAIN_DIR/checkpoints"
LATEST=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | grep -v ema | head -1)
if [[ -n "$LATEST" ]]; then
    STEP=$(basename "$LATEST" .safetensors)
    echo "Latest checkpoint: $LATEST"
    echo "  Resume with: bash train/scripts/pipeline_resume.sh"
    echo "  Or manually: bash train/scripts/pipeline_start.sh --resume $LATEST"
else
    echo "No checkpoint found in $CKPT_DIR"
fi

echo ""
echo "Pipeline stopped."
