#!/bin/bash
# train/scripts/run_preprocessing.sh — Full preprocessing pipeline.
#
# Run this ONCE after all dataset downloads complete (after download_datasets.sh).
# Wraps the entire pipeline in a single caffeinate invocation.
#
# Usage:
#   caffeinate -i -d bash train/scripts/run_preprocessing.sh
#
# Override data root if train/data is not already set up:
#   DATA_ROOT=/Volumes/IrisData caffeinate -i -d bash train/scripts/run_preprocessing.sh
#
# Steps:
#   1. CLIP deduplication (~1.5h embed + 20min FAISS)
#   2. Merge + shuffle all sources into unified shards (COMPUTE_WORKERS=6)
#   3. Filter pass — drop corrupt/small/bad-caption records (PERF_CORES=8)
#   4. Precompute Qwen3 embeddings (~8h, ~143 GB)
#   5. Precompute VAE latents (~6h, ~198 GB)
#
# Recaption (step 2b) is optional and runs separately in two terminals.
# SigLIP precompute is excluded by default (420 GB); add --siglip to enable.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"
VENV="$TRAIN_DIR/.venv/bin/activate"
ENABLE_SIGLIP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --siglip)    ENABLE_SIGLIP=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate venv if not already active
if [[ -z "$VIRTUAL_ENV" && -f "$VENV" ]]; then
    source "$VENV"
fi

echo "==================================================================="
echo "  IP-Adapter preprocessing pipeline"
echo "  DATA_ROOT: $DATA_ROOT"
echo "==================================================================="
echo ""

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: data root not found: $DATA_ROOT"
    echo "Run: bash train/scripts/setup_data_dir.sh"
    exit 1
fi

# ── Step 1: CLIP deduplication ────────────────────────────────────────────────
echo "[1/5] CLIP deduplication (~2h)..."
python "$SCRIPT_DIR/clip_dedup.py" all \
    --shards "$DATA_ROOT/raw/laion" \
    --embeddings "$DATA_ROOT/embeddings" \
    --output "$DATA_ROOT/dedup_ids"

# ── Step 2: Merge and shuffle into unified shards ─────────────────────────────
echo "[2/5] Merging sources into unified shards..."
python "$SCRIPT_DIR/build_shards.py" \
    --sources "$DATA_ROOT/raw/laion" \
              "$DATA_ROOT/raw/journeydb" \
              "$DATA_ROOT/raw/coyo" \
              "$DATA_ROOT/raw/wikiart" \
    --output "$DATA_ROOT/shards" \
    --blocklist "$DATA_ROOT/dedup_ids/duplicate_ids.txt"

# ── Step 3: Filter pass ───────────────────────────────────────────────────────
echo "[3/5] Filter pass (validate all shards)..."
python "$SCRIPT_DIR/filter_shards.py" \
    --shards "$DATA_ROOT/shards"

# ── Step 4: Precompute Qwen3 (~8h, ~143 GB) ──────────────────────────────────
echo "[4/5] Precomputing Qwen3 text embeddings (~8h)..."
python "$SCRIPT_DIR/precompute_qwen3.py" \
    --shards "$DATA_ROOT/shards" \
    --output "$DATA_ROOT/precomputed/qwen3"

# ── Step 5: Precompute VAE latents (~6h, ~198 GB) ────────────────────────────
echo "[5/5] Precomputing VAE latents (~6h)..."
python "$SCRIPT_DIR/precompute_vae.py" \
    --shards "$DATA_ROOT/shards" \
    --output "$DATA_ROOT/precomputed/vae"

# ── Optional: SigLIP (~420 GB) ────────────────────────────────────────────────
if $ENABLE_SIGLIP; then
    echo "[+] Precomputing SigLIP features (~420 GB)..."
    python "$SCRIPT_DIR/precompute_siglip.py" \
        --shards "$DATA_ROOT/shards" \
        --output "$DATA_ROOT/precomputed/siglip"
fi

echo ""
echo "==================================================================="
echo "  Preprocessing complete."
echo "  Shards:      $DATA_ROOT/shards/"
echo "  Qwen3 cache: $DATA_ROOT/precomputed/qwen3/"
echo "  VAE cache:   $DATA_ROOT/precomputed/vae/"
echo ""
echo "  NOTE: Recaptioning runs separately — see train/README.md Step 4."
echo ""
echo "  Next: caffeinate -i -d python train/train_ip_adapter.py \\"
echo "            --config train/configs/stage1_512px.yaml"
echo "==================================================================="
