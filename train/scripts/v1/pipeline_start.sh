#!/bin/bash
# train/scripts/pipeline_start.sh — Start the training pipeline in a tmux session.
#
# Guards against double-starting: exits if a 'pipeline' tmux session already exists.
# All output is logged to DATA_ROOT/logs/pipeline_chunk<N>_<timestamp>.log.
#
# Usage:
#   bash train/scripts/pipeline_start.sh                         # chunk 1, auto DATA_ROOT
#   bash train/scripts/pipeline_start.sh --chunk 2 --resume /path/to/step_105000.safetensors
#   bash train/scripts/pipeline_start.sh --skip-dedup --skip-train
#
# Options:
#   --chunk N          Training chunk 1–4 (default: 1)
#   --data-root PATH   Override auto-detected data root
#   --resume PATH      Checkpoint to resume from (required for chunks 2–4)
#   --config PATH      Training config YAML (default: train/configs/stage1_512px.yaml)
#   --scale PRESET     Training scale: small|medium|large|god-like or a step count N
#                        small:     50K / 15K steps,  21 /  7 shards  (fast iteration)
#                        medium:   105K / 40K steps,  43 / 17 shards  (default)
#                        large:    200K / 60K steps,  81 / 25 shards  (recommended)
#                        god-like: 400K /120K steps, 162 / 50 shards  (max quality)
#                        all-in:   540K /200K steps,  ALL shards      (~18 days total)
#                      Controls both step count and how many shards are precomputed.
#   --steps N          Override step count only (shard count still follows --scale)
#   --lr RATE          Override learning rate for this chunk
#   --siglip           Include SigLIP precompute step
#   --skip-dedup       Skip CLIP deduplication
#   --skip-train       Run pipeline steps but not the final training step
#
# Safe to call from Claude CoWork Dispatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Arg passthrough ───────────────────────────────────────────────────────────
# Collect all args and forward them verbatim to run_training_pipeline.sh.
PIPELINE_ARGS=()
CHUNK=1
DATA_ROOT_EXPLICIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk)      CHUNK="$2";             PIPELINE_ARGS+=(--chunk "$2"); shift 2 ;;
        --data-root)  DATA_ROOT_EXPLICIT="$2"; PIPELINE_ARGS+=(--data-root "$2"); shift 2 ;;
        --resume)     PIPELINE_ARGS+=(--resume "$2"); shift 2 ;;
        --config)     PIPELINE_ARGS+=(--config "$2"); shift 2 ;;
        --scale)      PIPELINE_ARGS+=(--scale "$2"); shift 2 ;;
        --steps)      PIPELINE_ARGS+=(--steps "$2"); shift 2 ;;
        --lr)         PIPELINE_ARGS+=(--lr "$2"); shift 2 ;;
        --siglip)     PIPELINE_ARGS+=(--siglip); shift ;;
        --skip-dedup) PIPELINE_ARGS+=(--skip-dedup); shift ;;
        --skip-train) PIPELINE_ARGS+=(--skip-train); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SESSION="pipeline"

# ── Guard: already running? ───────────────────────────────────────────────────
if /opt/homebrew/bin/tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists — pipeline may already be running." >&2
    echo "  Run 'bash train/scripts/pipeline_status.sh' to check what is active." >&2
    echo "  Run 'bash train/scripts/pipeline_stop.sh' to stop it first." >&2
    exit 1
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
VENV="$TRAIN_DIR/.venv/bin/activate"
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV — run 'bash train/setup.sh' first." >&2
    exit 1
fi

PIPELINE_SCRIPT="$SCRIPT_DIR/run_training_pipeline.sh"
if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
    echo "ERROR: pipeline script not found: $PIPELINE_SCRIPT" >&2
    exit 1
fi

# ── Auto-detect DATA_ROOT for log message ─────────────────────────────────────
if [[ -z "$DATA_ROOT_EXPLICIT" ]]; then
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]] && \
           [[ -d "$candidate/shards" || -d "$candidate/raw" || -d "$candidate/dedup_ids" ]]; then
            DATA_ROOT_EXPLICIT="$candidate"
            break
        fi
    done
    DATA_ROOT_EXPLICIT="${DATA_ROOT_EXPLICIT:-$TRAIN_DIR/data}"
fi

mkdir -p "$DATA_ROOT_EXPLICIT/logs"

# ── Pass auto-detected data root unless the user already provided --data-root ──
# Without this, run_training_pipeline.sh defaults to train/data (local SSD).
# Use ${#...} guard to avoid bash 3.2 nounset failure on empty arrays.
if [[ ${#PIPELINE_ARGS[@]} -eq 0 || " ${PIPELINE_ARGS[*]} " != *"--data-root"* ]]; then
    PIPELINE_ARGS+=(--data-root "$DATA_ROOT_EXPLICIT")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
CMD="caffeinate -i -d bash $PIPELINE_SCRIPT ${PIPELINE_ARGS[*]:-}"
echo "Starting pipeline in tmux session '$SESSION' (chunk $CHUNK)..."
echo "  Command: $CMD"
echo "  DATA_ROOT: $DATA_ROOT_EXPLICIT"
echo ""
echo "Follow progress:"
echo "  tmux attach -t $SESSION"
echo "  bash train/scripts/pipeline_logs.sh --follow"
echo "  bash train/scripts/pipeline_status.sh"

/opt/homebrew/bin/tmux new-session -d -s "$SESSION" "$CMD"

echo ""
echo "Pipeline started. Session: $SESSION"
