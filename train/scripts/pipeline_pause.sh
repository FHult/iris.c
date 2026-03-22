#!/bin/bash
# train/scripts/pipeline_pause.sh — Pause the pipeline and save state.
#
# Gracefully stops all pipeline processes. The training loop completes its
# current step and writes a periodic checkpoint before exiting. Use
# pipeline_resume.sh to continue from where you left off.
#
# This is identical to pipeline_stop.sh — 'pause' and 'stop' are the same
# operation because the training state is fully captured in checkpoints.
# There is no in-memory state to preserve; MLX writes a checkpoint every
# 1000 steps by default.
#
# Usage:
#   bash train/scripts/pipeline_pause.sh
#
# Safe to call from Claude CoWork Dispatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Pausing pipeline (saving checkpoint state)..."
echo "Use 'bash train/scripts/pipeline_resume.sh' to continue."
echo ""

bash "$SCRIPT_DIR/pipeline_stop.sh"

echo ""
echo "Pipeline paused. Run 'bash train/scripts/pipeline_resume.sh' to resume."
