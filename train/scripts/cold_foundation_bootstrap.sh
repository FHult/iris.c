#!/usr/bin/env bash
# train/scripts/cold_foundation_bootstrap.sh
#
# One-time bootstrap + command generator for the "cold-only full shard build +
# at-scale precompute + progressive medium foundation runs" campaign.
#
# Run this the moment your large cold volume (HDD) is mounted, and preferably also
# when the hot external SSD is attached.
#
# It creates the directory skeleton on cold, validates your four converted pools,
# and prints the exact copy-paste commands for the full cold-only shard build
# followed by proper three-tier foundation runs.
#
# Usage:
#   bash train/scripts/cold_foundation_bootstrap.sh
#   bash train/scripts/cold_foundation_bootstrap.sh /Volumes/16TBCold
#   bash train/scripts/cold_foundation_bootstrap.sh /Volumes/16TBCold /Volumes/2TBSSD /Users/fredrikhult/ultrahot

set -euo pipefail

COLD="${1:-/Volumes/16TBCold}"      # 16 TB HDD - source of truth + archive
HOT="${2:-/Volumes/2TBSSD}"       # 2 TB external SSD - primary compute
ULTRA="${3:-/Users/fredrikhult/ultrahot}"  # internal NVMe - small/fast runs (WIP logistics)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLD_PREFIX="/Volumes/16TBCold"   # must match the constant in build_shards.py

echo "=== iris.c Cold Foundation Bootstrap (Three-Tier) ==="
echo "Cold   (HDD, source of truth) : $COLD"
echo "Hot    (external SSD, compute): $HOT"
echo "Ultrahot (internal NVMe)      : $ULTRA"
echo "Repo                          : $REPO_ROOT"
echo

if [[ ! -d "$COLD" ]]; then
    echo "ERROR: Cold volume (HDD) not found at $COLD"
    echo "Mount the 16 TB HDD first for the full cold-only shard build."
    exit 1
fi

echo "Note: For the very first full shard build you will run build_shards.py --cold-only"
echo "      directly against the HDD while it is the only (or deliberately chosen) volume."
echo "      After shards exist, the stager will copy small active subsets to Hot."
echo

# 1. Directory skeleton (idempotent)
echo "Creating persistent directory skeleton on cold..."
mkdir -p \
    "$COLD/shards" \
    "$COLD/precomputed"/{qwen3,vae,siglip} \
    "$COLD/hard_examples" \
    "$COLD/anchor_shards" \
    "$COLD/dedup_ids" \
    "$COLD/checkpoints/stage1/archive" \
    "$COLD/logs" \
    "$COLD/pipeline" \
    "$COLD/.heartbeat" \
    "$COLD/metadata"/{checkouts,validation}

# 2. Validate the four converted pools (you said they are already present)
echo
echo "Checking for the four converted source pools on cold..."
MISSING=()
for ds in laion journeydb coyo wikiart; do
    if [[ ! -d "$COLD/converted/$ds" ]]; then
        MISSING+=("$ds")
        echo "  MISSING: $COLD/converted/$ds"
    else
        count=$(find "$COLD/converted/$ds" -name '*.tar' 2>/dev/null | wc -l | tr -d ' ')
        echo "  OK     : $COLD/converted/$ds  ($count .tar files)"
    fi
done

if (( ${#MISSING[@]} > 0 )); then
    echo
    echo "WARNING: One or more converted pools are missing."
    echo "The cold-only shard build will fail for those sources."
    echo "Fix the paths or download/convert the missing data before proceeding."
fi

# 3. Check that build_shards.py --cold-only will accept the paths
echo
echo "Validating --cold-only path rules (must live under $COLD_PREFIX)..."
if [[ "$COLD" != "$COLD_PREFIX"* ]]; then
    echo "  WARNING: Your cold mount ($COLD) does not start with $COLD_PREFIX"
    echo "           You will need to either:"
    echo "             a) create a symlink: sudo ln -s '$COLD' '$COLD_PREFIX', or"
    echo "             b) edit COLD_PREFIX in train/scripts/build_shards.py (last resort)"
fi

# 4. Print the exact next commands (the user can copy-paste when ready)
echo
echo "======================================================================"
echo "  NEXT COMMANDS — COPY AND RUN WHEN READY (after cold is mounted)"
echo "======================================================================"
echo
echo "# --- Phase 1: Full cold-only shard build (resume-safe) ---"
cat <<CMD
python train/scripts/build_shards.py \\
  --sources \\
    "$COLD/converted/laion" \\
    "$COLD/converted/journeydb" \\
    "$COLD/converted/coyo" \\
    "$COLD/converted/wikiart" \\
  --output "$COLD/shards" \\
  --shard_size 5000 \\
  --workers 6 \\
  --compression zstd \\
  --compression_level 1 \\
  --blocklist "$COLD/metadata/duplicate_ids.txt" \\
  --cold-only \\
  2>&1 | tee "$COLD/logs/build_shards_cold_full_\$(date +%Y%m%d-%H%M).log"
CMD

echo
echo "# --- Phase 2: Foundation precompute (start with 300 shards for a fast first pass) ---"
cat <<CMD
python train/scripts/precompute_all.py \\
  --shards "$COLD/shards" \\
  --qwen3-output "$COLD/precomputed/qwen3" \\
  --vae-output   "$COLD/precomputed/vae" \\
  --siglip-output "$COLD/precomputed/siglip" \\
  --siglip \\
  --flux-model flux-klein-model \\
  --qwen3-model flux-klein-model \\
  --vae-batch 4 \\
  --max-shards 300 \\
  2>&1 | tee "$COLD/logs/precompute_foundation_\$(date +%Y%m%d).log"
CMD

echo
echo "# --- Phase 3: First progressive medium foundation run (proper three-tier) ---"
echo "# (Run this after the cold-only shard build is complete and hot external SSD is attached)"
cat <<CMD
./train/start_pipeline.sh \\
  --data-root "$HOT" \\
  --config train/configs/cold_foundation_v1.yaml
CMD

echo
echo "# For quick/small ablation or smoke runs on the internal NVMe (ultrahot):"
cat <<CMD
./train/start_pipeline.sh \\
  --data-root "$ULTRA" \\
  --config train/configs/cold_foundation_v1.yaml   # (you may want a separate small-run config)
CMD

echo
echo "# --- Parallel: Long-term ablation harness (QUALITY feature discovery on your real data) ---"
cat <<CMD
python train/scripts/ablation_harness.py \\
  --config train/configs/ablation_sref_v1.yaml \\
  --output-dir "$COLD/ablation_foundation_sref_v1" \\
  2>&1 | tee "$COLD/logs/ablation_foundation_\$(date +%Y%m%d).log"
CMD

echo
echo "# After the first medium run finishes, mine hard examples + score shards (orchestrator usually does this)"
echo "# Later medium runs warm-start from the best checkpoint in $COLD/checkpoints/stage1"
echo
echo "Hot ↔ Ultrahot movement policy is still WIP per your note."
echo "For now:"
echo "  - Use Hot (external 2 TB SSD) for all real medium foundation runs."
echo "  - Use Ultrahot (internal NVMe) only for small ablation bursts, smoke tests, or quick experiments."
echo "  - Manually stage small subsets to ultrahot when needed, or wait for a future small helper."
echo
echo "Full detailed runbook: plans/cold-full-shard-build-foundation-runs.md"
echo "======================================================================"

# 5. Quick next-step reminder
echo
echo "Recommended immediate next manual step (while cold is attached):"
echo "  1. Run the shard build command above (it will take days — use tmux + caffeinate)."
echo "  2. While it runs, start the ablation harness in a second window."
echo "  3. When shards are ready, start the foundation precompute + first medium training run."
echo
echo "Bootstrap complete. All directories created. Commands printed above."
