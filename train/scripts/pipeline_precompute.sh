#!/bin/bash
# train/scripts/pipeline_precompute.sh — Run precompute steps only.
#
# Runs steps 8a (Qwen3 embeddings, ~8h) and 8b (VAE latents, ~6h) in sequence.
# Use this after build_shards and filter_shards are complete if you want to run
# precompute independently from the main pipeline.
#
# Guards against starting when a 'precompute' tmux session already exists.
# Each step is idempotent: already-computed .npz files are skipped per-shard.
#
# Usage:
#   bash train/scripts/pipeline_precompute.sh
#   bash train/scripts/pipeline_precompute.sh --siglip              # include SigLIP (420 GB)
#   bash train/scripts/pipeline_precompute.sh --data-root /Volumes/2TBSSD
#
# Safe to call from Claude CoWork Dispatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Args ──────────────────────────────────────────────────────────────────────
SIGLIP=false
DATA_ROOT_EXPLICIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --siglip)     SIGLIP=true; shift ;;
        --data-root)  DATA_ROOT_EXPLICIT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SESSION="precompute"

# ── Guard: already running? ───────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists — precompute may already be running." >&2
    echo "  Run 'bash train/scripts/pipeline_status.sh' to check." >&2
    echo "  Run 'bash train/scripts/pipeline_stop.sh' to stop it first." >&2
    exit 1
fi

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
if [[ -z "$DATA_ROOT_EXPLICIT" ]]; then
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]] && \
           [[ -d "$candidate/shards" || -d "$candidate/precomputed" ]]; then
            DATA_ROOT_EXPLICIT="$candidate"
            break
        fi
    done
    DATA_ROOT_EXPLICIT="${DATA_ROOT_EXPLICIT:-$TRAIN_DIR/data}"
fi

VENV="$TRAIN_DIR/.venv/bin/activate"
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV — run 'bash train/setup.sh' first." >&2
    exit 1
fi

# ── Check shards exist ────────────────────────────────────────────────────────
SHARD_COUNT=$(find "$DATA_ROOT_EXPLICIT/shards" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$SHARD_COUNT" -eq 0 ]]; then
    echo "ERROR: No shards found in $DATA_ROOT_EXPLICIT/shards/" >&2
    echo "  Run steps 4 (build_shards) and 5 (filter_shards) first." >&2
    exit 1
fi

echo "Starting precompute in tmux session '$SESSION'..."
echo "  DATA_ROOT: $DATA_ROOT_EXPLICIT"
echo "  Shards:    $SHARD_COUNT"
echo "  SigLIP:    $SIGLIP"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
SIGLIP_FLAG=""
$SIGLIP && SIGLIP_FLAG="--siglip"

CMD="caffeinate -i -d bash $SCRIPT_DIR/run_shard_and_precompute.sh --data-root $DATA_ROOT_EXPLICIT $SIGLIP_FLAG"

tmux new-session -d -s "$SESSION" "$CMD"

echo "Precompute started."
echo "  Follow: tmux attach -t $SESSION"
echo "  Log:    bash train/scripts/pipeline_logs.sh --follow"
echo "  Status: bash train/scripts/pipeline_status.sh"
