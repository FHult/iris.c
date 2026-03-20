#!/bin/bash
# train/scripts/setup_data_dir.sh — Set up the training data directory.
#
# Creates the canonical train/data/ layout as either:
#   A) A real directory tree on the internal SSD (if ≥ 450 GB free), OR
#   B) A symlink to an external volume (e.g. /Volumes/IrisData)
#
# After this script runs, all training scripts and configs can use
# train/data/ as the data root regardless of where the actual data lives.
#
# Usage:
#   bash train/scripts/setup_data_dir.sh
#   bash train/scripts/setup_data_dir.sh --external /Volumes/IrisData
#   bash train/scripts/setup_data_dir.sh --local

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$TRAIN_DIR/data"
REQUIRED_GB=450
EXTERNAL_PATH=""
FORCE_LOCAL=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --external) EXTERNAL_PATH="$2"; shift 2 ;;
        --local)    FORCE_LOCAL=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==================================================================="
echo "  iris.c IP-Adapter training data directory setup"
echo "==================================================================="
echo ""

# ── Sub-directory layout to create ───────────────────────────────────────────
SUBDIRS=(
    "raw/laion"
    "raw/journeydb"
    "raw/coyo"
    "raw/wikiart"
    "shards"
    "precomputed/qwen3"
    "precomputed/vae"
    "precomputed/siglip"
    "embeddings"
    "dedup_ids"
    "checkpoints"
    "weights"
    "logs"
)

create_layout() {
    local root="$1"
    echo "Creating directory structure under $root ..."
    for subdir in "${SUBDIRS[@]}"; do
        mkdir -p "$root/$subdir"
    done
    echo "  Done."
}

# ── Handle --external flag ────────────────────────────────────────────────────
if [[ -n "$EXTERNAL_PATH" ]]; then
    if [[ ! -d "$EXTERNAL_PATH" ]]; then
        echo "Error: external path not found: $EXTERNAL_PATH"
        echo "Mount the drive first, then re-run."
        exit 1
    fi
    create_layout "$EXTERNAL_PATH"
    # Remove stub dir if it exists (but not if it's already a symlink to somewhere else)
    if [[ -L "$DATA_DIR" ]]; then
        rm "$DATA_DIR"
    elif [[ -d "$DATA_DIR" ]]; then
        # Keep README but remove if it's an empty stub
        non_readme=$(find "$DATA_DIR" -mindepth 1 -not -name "README.md" | head -1)
        if [[ -z "$non_readme" ]]; then
            rm -rf "$DATA_DIR"
        else
            echo "Warning: $DATA_DIR is a non-empty directory — not replacing with symlink."
            echo "Move or delete it first, then re-run with --external."
            exit 1
        fi
    fi
    ln -s "$EXTERNAL_PATH" "$DATA_DIR"
    echo ""
    echo "Symlink created: $DATA_DIR -> $EXTERNAL_PATH"
    echo ""
    _print_next_steps "$DATA_DIR"
    exit 0
fi

# ── Check internal disk space ─────────────────────────────────────────────────
AVAILABLE_GB=$(df -g "$TRAIN_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -z "$AVAILABLE_GB" ]]; then
    # df -g not available (Linux fallback)
    AVAILABLE_GB=$(df -BG "$TRAIN_DIR" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
fi
AVAILABLE_GB="${AVAILABLE_GB:-0}"

echo "Internal disk available: ${AVAILABLE_GB} GB (required: ${REQUIRED_GB} GB)"
echo ""

# ── Decision: local or symlink ────────────────────────────────────────────────
USE_LOCAL=false
if $FORCE_LOCAL; then
    USE_LOCAL=true
elif [[ "$AVAILABLE_GB" -ge "$REQUIRED_GB" ]]; then
    echo "Sufficient space for local storage."
    echo ""
    echo "Recommendation: store training shards + precomputed caches locally."
    echo "  Benefits: eliminates TB4 disconnection risk during multi-day training."
    echo "  Raw source datasets (~257 GB) can stay on external SSD after sharding."
    echo ""
    read -r -p "Use internal SSD for train/data/? [Y/n] " REPLY
    REPLY="${REPLY:-Y}"
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        USE_LOCAL=true
    fi
fi

if $USE_LOCAL; then
    # ── Option A: local directory ─────────────────────────────────────────────
    if [[ -L "$DATA_DIR" ]]; then
        echo "Removing existing symlink $DATA_DIR ..."
        rm "$DATA_DIR"
    fi
    create_layout "$DATA_DIR"
    echo ""
    echo "train/data/ is ready as a local directory."

else
    # ── Option B: symlink to external volume ──────────────────────────────────
    echo ""
    echo "Not enough local space (or external preferred)."
    echo ""

    # Try to auto-detect mounted external volumes
    DETECTED=""
    for vol in /Volumes/IrisData /Volumes/iris_data /Volumes/IrisDataSSD; do
        if [[ -d "$vol" ]]; then
            DETECTED="$vol"
            break
        fi
    done

    if [[ -n "$DETECTED" ]]; then
        echo "Detected external volume: $DETECTED"
        read -r -p "Use $DETECTED as data root? [Y/n] " REPLY
        REPLY="${REPLY:-Y}"
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            EXTERNAL_PATH="$DETECTED"
        fi
    fi

    if [[ -z "$EXTERNAL_PATH" ]]; then
        read -r -p "Enter external volume path (e.g. /Volumes/IrisData): " EXTERNAL_PATH
    fi

    if [[ ! -d "$EXTERNAL_PATH" ]]; then
        echo "Error: $EXTERNAL_PATH does not exist. Mount the drive and re-run."
        exit 1
    fi

    create_layout "$EXTERNAL_PATH"

    if [[ -L "$DATA_DIR" ]]; then
        rm "$DATA_DIR"
    elif [[ -d "$DATA_DIR" ]]; then
        non_readme=$(find "$DATA_DIR" -mindepth 1 -not -name "README.md" | head -1)
        if [[ -z "$non_readme" ]]; then
            rm -rf "$DATA_DIR"
        else
            echo "Warning: $DATA_DIR is a non-empty directory. Move it first."
            exit 1
        fi
    fi

    ln -s "$EXTERNAL_PATH" "$DATA_DIR"
    echo ""
    echo "Symlink created: $DATA_DIR -> $EXTERNAL_PATH"
fi

# ── Next steps ────────────────────────────────────────────────────────────────
echo ""
echo "==================================================================="
echo "  Next steps"
echo "==================================================================="
echo ""
echo "1. Pre-filter LAION metadata (no images needed, ~1 hour):"
echo "   python train/scripts/prepare_laion.py \\"
echo "     --output train/data/raw/laion_filtered.parquet"
echo ""
echo "2. Download datasets — see train/README.md Step 2:"
echo "   bash train/scripts/download_datasets.sh"
echo "   (uses train/data/ automatically)"
echo ""
echo "3. After downloads, run the preprocessing pipeline:"
echo "   caffeinate -i -d bash train/scripts/run_preprocessing.sh"
echo "   (builds shards, deduplicates, recaptions, precomputes)"
echo ""
echo "4. Train:"
echo "   caffeinate -i -d python train/train_ip_adapter.py \\"
echo "     --config train/configs/stage1_512px.yaml"
echo ""
