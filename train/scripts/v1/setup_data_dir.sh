#!/bin/bash
# train/scripts/setup_data_dir.sh — Set up the training data directory.
#
# train/data/ is ALWAYS a real local directory.
# train/data/raw/ is the only subdirectory that may be symlinked to an external
# SSD — the raw source datasets (LAION ~150 GB, JourneyDB ~80 GB, COYO ~25 GB)
# are only needed during Phase 1–2 data preparation.
#
# Everything else (shards, precomputed, weights, checkpoints) stays local to
# eliminate TB4 disconnection risk during multi-day unattended training.
#
# Space requirements:
#   raw/ on external SSD:        ~257 GB
#   Local (recommended minimum):
#     shards/                    ~260 GB  (read every training step — must be local)
#     precomputed/qwen3/         ~143 GB  (saves 200ms/step)
#     precomputed/vae/           ~198 GB  (saves 180ms/step)
#     weights/ + checkpoints/    ~16 GB
#   Minimum local total:         ~617 GB
#
# Usage:
#   bash train/scripts/setup_data_dir.sh
#
# Force raw/ to external path now (e.g. SSD already mounted):
#   bash train/scripts/setup_data_dir.sh --external /Volumes/IrisData
#
# Force everything local:
#   bash train/scripts/setup_data_dir.sh --local

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$TRAIN_DIR/data"
RAW_DIR="$DATA_DIR/raw"

LOCAL_IDEAL_GB=617   # shards + qwen3 + vae + weights + checkpoints

EXTERNAL_PATH=""
FORCE_LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --external) EXTERNAL_PATH="$2"; shift 2 ;;
        --local)    FORCE_LOCAL=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helper: create always-local subdirs ───────────────────────────────────────
create_local_layout() {
    echo "Creating local directory structure under $DATA_DIR ..."
    mkdir -p \
        "$DATA_DIR/shards" \
        "$DATA_DIR/precomputed/qwen3" \
        "$DATA_DIR/precomputed/vae" \
        "$DATA_DIR/precomputed/siglip" \
        "$DATA_DIR/embeddings" \
        "$DATA_DIR/dedup_ids" \
        "$DATA_DIR/weights" \
        "$DATA_DIR/checkpoints" \
        "$DATA_DIR/logs"
    echo "  Done (shards/, precomputed/, weights/, checkpoints/ — all local)."
}

# ── Helper: create raw/ as a real local directory ─────────────────────────────
create_raw_local() {
    mkdir -p \
        "$RAW_DIR/laion" \
        "$RAW_DIR/journeydb" \
        "$RAW_DIR/coyo" \
        "$RAW_DIR/wikiart"
    echo "  raw/ created locally."
}

# ── Helper: symlink raw/ to external SSD ─────────────────────────────────────
symlink_raw() {
    local ext="$1"
    local ext_raw="$ext/raw"

    # Create subdirs on the external volume
    mkdir -p \
        "$ext_raw/laion" \
        "$ext_raw/journeydb" \
        "$ext_raw/coyo" \
        "$ext_raw/wikiart"

    # Remove existing empty raw/ stub
    if [[ -L "$RAW_DIR" ]]; then
        rm "$RAW_DIR"
    elif [[ -d "$RAW_DIR" ]]; then
        if [[ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]]; then
            rmdir "$RAW_DIR"
        else
            echo "Warning: $RAW_DIR is non-empty. Cannot replace with symlink."
            echo "Move its contents to $ext_raw/ first, then re-run."
            exit 1
        fi
    fi

    ln -s "$ext_raw" "$RAW_DIR"
    echo "  Symlink: train/data/raw/ -> $ext_raw"
    echo "  (shards/, precomputed/, weights/, checkpoints/ remain local)"
}

# ── Helper: prompt for external volume path ───────────────────────────────────
prompt_external() {
    # Try to auto-detect mounted external volumes
    local detected=""
    for vol in /Volumes/IrisData /Volumes/iris_data /Volumes/IrisDataSSD /Volumes/Samsung /Volumes/SanDisk; do
        if [[ -d "$vol" ]]; then
            detected="$vol"
            break
        fi
    done

    if [[ -n "$detected" ]]; then
        echo "Detected external volume: $detected"
        # Skip interactive prompt when no TTY (remote/phone dispatch)
        if [[ -t 0 ]]; then
            read -r -p "Symlink train/data/raw/ -> $detected/raw/? [Y/n] " REPLY
            REPLY="${REPLY:-Y}"
            [[ "$REPLY" =~ ^[Yy]$ ]] && EXTERNAL_PATH="$detected"
        else
            echo "  (no TTY — pass --external $detected to symlink automatically)"
        fi
    fi

    if [[ -z "$EXTERNAL_PATH" ]]; then
        echo ""
        echo "External SSD not yet mounted (or not confirmed)."
        echo "Creating train/data/raw/ as a local stub for now."
        echo ""
        echo "When the SSD arrives, run:"
        echo "  bash train/scripts/setup_data_dir.sh --external /Volumes/YourDriveName"
        create_raw_local
        return
    fi

    if [[ ! -d "$EXTERNAL_PATH" ]]; then
        echo "Error: $EXTERNAL_PATH does not exist. Mount the drive and re-run."
        exit 1
    fi

    symlink_raw "$EXTERNAL_PATH"
}

# ── Print next steps ──────────────────────────────────────────────────────────
print_next_steps() {
    echo ""
    echo "==================================================================="
    echo "  train/data/ is ready. What you can do right now:"
    echo ""
    echo "  # Pre-filter LAION metadata (~1 hour, no images needed)"
    echo "  python train/scripts/prepare_laion.py \\"
    echo "    --output train/data/raw/laion_filtered.parquet"
    echo ""
    echo "  # Download WikiArt (~2 GB, tiny — goes to train/data/raw/wikiart/)"
    echo "  hf download Artificio/WikiArt \\"
    echo "    --repo-type dataset --local-dir train/data/raw/wikiart"
    echo ""
    echo "  # Download warmstart weights (~5.3 GB → train/data/weights/)"
    echo "  hf download InstantX/FLUX.1-dev-IP-Adapter \\"
    echo "    --local-dir train/data/weights/flux_dev_ipadapter"
    echo ""
    echo "  When external SSD arrives:"
    echo "    bash train/scripts/download_datasets.sh"
    echo "==================================================================="
}

# =============================================================================
# Main
# =============================================================================

echo "==================================================================="
echo "  iris.c IP-Adapter training data directory setup"
echo "==================================================================="
echo ""

# Always create local layout first
create_local_layout
echo ""

# Get available local space
AVAILABLE_GB=$(df -g "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -z "$AVAILABLE_GB" ]]; then
    AVAILABLE_GB=$(df -BG "$DATA_DIR" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
fi
AVAILABLE_GB="${AVAILABLE_GB:-0}"

echo "Internal disk available: ${AVAILABLE_GB} GB"
echo "  Ideal local minimum (shards + Qwen3 + VAE): ${LOCAL_IDEAL_GB} GB"
echo ""

# Decide what to do with raw/
if $FORCE_LOCAL; then
    echo "--local: creating raw/ locally."
    create_raw_local

elif [[ -n "$EXTERNAL_PATH" ]]; then
    # Explicit external path provided on command line
    if [[ ! -d "$EXTERNAL_PATH" ]]; then
        echo "Error: external path not found: $EXTERNAL_PATH"
        echo "Mount the drive first, then re-run."
        exit 1
    fi
    symlink_raw "$EXTERNAL_PATH"

elif [[ "$AVAILABLE_GB" -ge 800 ]]; then
    # Plenty of space — keep everything local
    echo "Plenty of local space. Creating raw/ locally."
    echo "Raw source datasets (~257 GB) can download directly to train/data/raw/."
    echo "After sharding, raw/ can be deleted to reclaim space."
    create_raw_local

else
    # Not enough for ideal local setup — recommend symlinking raw/ to external
    echo "Local space may be tight for raw datasets (~257 GB) alongside training data."
    echo ""
    echo "Recommended split:"
    echo "  train/data/raw/          → symlink to external SSD  (~257 GB)"
    echo "  train/data/shards/       → local  (~260 GB, read every training step)"
    echo "  train/data/precomputed/  → local  (~341 GB, saves 14h training time)"
    echo "  train/data/weights/ etc  → local  (~16 GB)"
    echo ""
    echo "If no external SSD yet, raw/ will be a local stub — re-run with"
    echo "--external when the SSD arrives to convert it to a symlink."
    echo ""

    prompt_external
fi

print_next_steps
