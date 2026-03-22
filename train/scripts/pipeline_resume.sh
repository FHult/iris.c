#!/bin/bash
# train/scripts/pipeline_resume.sh — Resume training from the latest checkpoint.
#
# Auto-detects the latest checkpoint and infers the current training chunk from
# its step count vs the per-chunk step budgets. Pass --chunk to override if the
# inference is wrong (e.g. you want to start the next chunk, not re-run this one).
#
# Chunk step budgets:
#   Chunk 1: 0 – 105,000 steps
#   Chunk 2: 105,001 – 145,000 steps
#   Chunk 3: 145,001 – 185,000 steps
#   Chunk 4: 185,001 – 225,000 steps
#
# Usage:
#   bash train/scripts/pipeline_resume.sh             # auto-detect chunk
#   bash train/scripts/pipeline_resume.sh --chunk 2   # force chunk
#   bash train/scripts/pipeline_resume.sh --chunk 2 --data-root /Volumes/2TBSSD
#
# Safe to call from Claude CoWork Dispatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="$TRAIN_DIR/checkpoints"

# ── Arg parsing ───────────────────────────────────────────────────────────────
CHUNK_OVERRIDE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk)      CHUNK_OVERRIDE="$2"; shift 2 ;;
        --data-root)  EXTRA_ARGS+=(--data-root "$2"); shift 2 ;;
        --config)     EXTRA_ARGS+=(--config "$2"); shift 2 ;;
        --steps)      EXTRA_ARGS+=(--steps "$2"); shift 2 ;;
        --lr)         EXTRA_ARGS+=(--lr "$2"); shift 2 ;;
        --siglip)     EXTRA_ARGS+=(--siglip); shift ;;
        --skip-dedup) EXTRA_ARGS+=(--skip-dedup); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Find latest checkpoint ────────────────────────────────────────────────────
LATEST=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | grep -v ema | head -1 || true)
if [[ -z "$LATEST" ]]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR" >&2
    echo "  If this is a fresh run, use: bash train/scripts/pipeline_start.sh" >&2
    exit 1
fi

# ── Infer step count from filename (e.g. step_105000.safetensors) ─────────────
STEP=$(basename "$LATEST" .safetensors | grep -oE '[0-9]+' || echo "0")

# ── Infer chunk from step count ───────────────────────────────────────────────
if [[ -n "$CHUNK_OVERRIDE" ]]; then
    CHUNK="$CHUNK_OVERRIDE"
else
    if   [[ "$STEP" -le 105000 ]]; then CHUNK=1
    elif [[ "$STEP" -le 145000 ]]; then CHUNK=2
    elif [[ "$STEP" -le 185000 ]]; then CHUNK=3
    else                                CHUNK=4
    fi
fi

echo "================================================================="
echo "  Resuming training pipeline"
echo "  Latest checkpoint: $LATEST"
echo "  Step:    $STEP"
echo "  Chunk:   $CHUNK"
echo "================================================================="
echo ""

# ── Launch via pipeline_start.sh ─────────────────────────────────────────────
exec bash "$SCRIPT_DIR/pipeline_start.sh" \
    --chunk "$CHUNK" \
    --resume "$LATEST" \
    "${EXTRA_ARGS[@]}"
