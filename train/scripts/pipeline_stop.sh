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

echo "================================================================="
echo "  Stopping training pipeline"
echo "================================================================="
echo ""

# ── SIGTERM pipeline Python processes (graceful) ──────────────────────────────
PIPELINE_PATTERNS="train_ip_adapter|build_shards|filter_shards|precompute_qwen3|precompute_vae|precompute_siglip|clip_dedup|recaption"
PIDS=$(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true)

if [[ -n "$PIDS" ]]; then
    echo "Sending SIGTERM to pipeline processes..."
    while IFS= read -r pid; do
        NAME=$(ps -p "$pid" -o comm= 2>/dev/null | xargs basename 2>/dev/null || echo "?")
        echo "  SIGTERM PID $pid ($NAME)"
        kill -TERM "$pid" 2>/dev/null || true
    done <<< "$PIDS"

    # Wait up to 30s for graceful shutdown
    echo "Waiting up to 30s for processes to exit..."
    for i in $(seq 1 30); do
        REMAINING=$(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true)
        [[ -z "$REMAINING" ]] && break
        sleep 1
    done

    # Force-kill anything still running
    REMAINING=$(pgrep -f "$PIPELINE_PATTERNS" 2>/dev/null || true)
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
