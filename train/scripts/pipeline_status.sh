#!/bin/bash
# train/scripts/pipeline_status.sh — Training pipeline status report.
#
# Prints a full status snapshot: all 9 pipeline steps, running processes,
# recent log lines, disk usage, and SSD health.
#
# Usage:
#   bash train/scripts/pipeline_status.sh
#   bash train/scripts/pipeline_status.sh --data-root /Volumes/2TBSSD
#
# Works from any working directory. Safe to run from Claude CoWork Dispatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAIN_DIR")"
DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
count_files() { find "${1}" -maxdepth 1 -name "${2:-*}" 2>/dev/null | wc -l | tr -d ' '; }
count_tars()  { count_files "${1}" "*.tar"; }
du_h()        { du -sh "$1" 2>/dev/null | cut -f1; }
ts()          { date '+%Y-%m-%d %H:%M:%S'; }

step_status() {
    local label="$1" done_cond="$2" run_cond="$3"
    if eval "$done_cond" &>/dev/null; then
        printf "  ✅ %-40s %s\n" "$label" "$4"
    elif [[ -n "$run_cond" ]] && eval "$run_cond" &>/dev/null; then
        printf "  ⏳ %-40s %s\n" "$label" "$5"
    else
        printf "  ⬜ %-40s %s\n" "$label" "${6:-Pending}"
    fi
}

echo "================================================================="
echo "  Pipeline status — $(ts)"
echo "  DATA_ROOT: $DATA_ROOT"
echo "================================================================="
echo ""

# ── Step counts ───────────────────────────────────────────────────────────────
LAION_TARS=$(count_tars "$DATA_ROOT/raw/laion")
COYO_TARS=$(count_tars "$DATA_ROOT/raw/coyo")
JDB_WDS_TARS=$(count_tars "$DATA_ROOT/raw/journeydb_wds")
WIKIART_WDS_TARS=$(count_tars "$DATA_ROOT/raw/wikiart_wds")
DEDUP_IDS="$DATA_ROOT/dedup_ids/duplicate_ids.txt"
SHARDS_DIR="$DATA_ROOT/shards"
SHARDS_DONE="$SHARDS_DIR/.filtered_chunk1"
DEDUP_INDEX="$DATA_ROOT/dedup_ids/dedup_index.faiss"
ANCHOR_DIR="$DATA_ROOT/anchor_shards"
PRECOMP_DIR="$DATA_ROOT/precomputed"
CKPT_DIR="$TRAIN_DIR/checkpoints"

SHARD_COUNT=$(count_tars "$SHARDS_DIR")
QWEN3_COUNT=$(count_files "$PRECOMP_DIR/qwen3" "*.npz")
VAE_COUNT=$(count_files "$PRECOMP_DIR/vae" "*.npz")
ANCHOR_COUNT=$(count_tars "$ANCHOR_DIR")
CKPT_COUNT=$(find "$CKPT_DIR" -name "step_*.safetensors" 2>/dev/null | wc -l | tr -d ' ')

# Estimate total expected shards (rough — 2.47M records / 5000)
TOTAL_SHARDS_EST=933

echo "── Steps ────────────────────────────────────────────────────────"
step_status "[1/9] Verify downloads" \
    "[[ $LAION_TARS -gt 0 ]]" "" \
    "(LAION:$LAION_TARS COYO:$COYO_TARS JDB_WDS:$JDB_WDS_TARS)"

step_status "[2a/9] WikiArt → WDS" \
    "[[ $WIKIART_WDS_TARS -gt 0 ]]" "" \
    "($WIKIART_WDS_TARS shards)"

step_status "[2b/9] JourneyDB → WDS" \
    "[[ $JDB_WDS_TARS -gt 0 ]]" "" \
    "($JDB_WDS_TARS shards)"

DEDUP_LINES=0
[[ -f "$DEDUP_IDS" ]] && DEDUP_LINES=$(wc -l < "$DEDUP_IDS" | tr -d ' ')
step_status "[3/9] CLIP deduplication" \
    "[[ -f $DEDUP_IDS ]]" \
    "pgrep -f clip_dedup" \
    "($DEDUP_LINES IDs blocked)" \
    "running..."

BUILD_RUNNING=false
pgrep -f build_shards &>/dev/null && BUILD_RUNNING=true
step_status "[4/9] Build unified shards" \
    "[[ $SHARD_COUNT -ge $TOTAL_SHARDS_EST ]]" \
    "$BUILD_RUNNING" \
    "($SHARD_COUNT/$TOTAL_SHARDS_EST shards)" \
    "($SHARD_COUNT/$TOTAL_SHARDS_EST shards, writing...)"

step_status "[5/9] Filter shards" \
    "[[ -f $SHARDS_DONE ]]" \
    "pgrep -f filter_shards" \
    "done" "running..."

step_status "[6/9] Cross-chunk dedup index" \
    "[[ -f $DEDUP_INDEX ]]" \
    "pgrep -f 'clip_dedup.*build-index'" \
    "($(du_h "$DEDUP_INDEX"))" "building..."

step_status "[7/9] Anchor set" \
    "[[ $ANCHOR_COUNT -gt 0 ]]" "" \
    "($ANCHOR_COUNT shards)"

QWEN3_RUNNING=false; pgrep -f precompute_qwen3 &>/dev/null && QWEN3_RUNNING=true
VAE_RUNNING=false;   pgrep -f precompute_vae   &>/dev/null && VAE_RUNNING=true

# Expected precomputed counts = shard_count (one file per shard)
step_status "[8a/9] Precompute Qwen3 embeddings" \
    "[[ $QWEN3_COUNT -ge $SHARD_COUNT && $SHARD_COUNT -gt 0 ]]" \
    "$QWEN3_RUNNING" \
    "($QWEN3_COUNT/$SHARD_COUNT shards)" \
    "($QWEN3_COUNT/$SHARD_COUNT shards, running...)"

step_status "[8b/9] Precompute VAE latents" \
    "[[ $VAE_COUNT -ge $SHARD_COUNT && $SHARD_COUNT -gt 0 ]]" \
    "$VAE_RUNNING" \
    "($VAE_COUNT/$SHARD_COUNT shards)" \
    "($VAE_COUNT/$SHARD_COUNT shards, running...)"

TRAIN_RUNNING=false; pgrep -f train_ip_adapter &>/dev/null && TRAIN_RUNNING=true
LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | grep -v ema | head -1)
LATEST_STEP=""
[[ -n "$LATEST_CKPT" ]] && LATEST_STEP="(latest: $(basename "$LATEST_CKPT" .safetensors))"
step_status "[9/9] Train" \
    "[[ $CKPT_COUNT -gt 0 ]]" \
    "$TRAIN_RUNNING" \
    "$LATEST_STEP ($CKPT_COUNT checkpoints)" \
    "running... $LATEST_STEP"

echo ""
echo "── Running processes ────────────────────────────────────────────"
PROCS=$(pgrep -a -l -f "build_shards|clip_dedup|filter_shards|precompute_qwen3|precompute_vae|precompute_siglip|train_ip_adapter|caffeinate|run_training_pipeline" 2>/dev/null \
    | grep -v "grep\|pipeline_status")
if [[ -n "$PROCS" ]]; then
    echo "$PROCS" | while read -r line; do echo "  $line"; done
else
    echo "  (none)"
fi

echo ""
echo "── tmux sessions ────────────────────────────────────────────────"
TMUX_SESSIONS=$(tmux list-sessions 2>/dev/null)
if [[ -n "$TMUX_SESSIONS" ]]; then
    echo "$TMUX_SESSIONS" | while read -r line; do echo "  $line"; done
else
    echo "  (none)"
fi

echo ""
echo "── Recent log ───────────────────────────────────────────────────"
# Logs may be in DATA_ROOT/logs/ or train/data/logs/ depending on invocation
LATEST_LOG=$(ls -t "$DATA_ROOT/logs"/pipeline_chunk*.log "$TRAIN_DIR/data/logs"/pipeline_chunk*.log 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    echo "  $LATEST_LOG"
    tail -6 "$LATEST_LOG" | while read -r line; do echo "  $line"; done
else
    echo "  (no pipeline log found)"
fi

echo ""
echo "── Disk ─────────────────────────────────────────────────────────"
df -h "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {
    printf "  %-30s used=%-8s avail=%-8s capacity=%s\n", $1, $3, $4, $5
}'
echo ""
printf "  %-28s %s\n" "raw/"          "$(du_h "$DATA_ROOT/raw")"
printf "  %-28s %s\n" "shards/"       "$(du_h "$SHARDS_DIR") ($SHARD_COUNT tars)"
printf "  %-28s %s\n" "precomputed/qwen3/"  "$(du_h "$PRECOMP_DIR/qwen3") ($QWEN3_COUNT files)"
printf "  %-28s %s\n" "precomputed/vae/"    "$(du_h "$PRECOMP_DIR/vae") ($VAE_COUNT files)"
printf "  %-28s %s\n" "embeddings/"   "$(du_h "$DATA_ROOT/embeddings")"
printf "  %-28s %s\n" "checkpoints/"  "$(du_h "$CKPT_DIR") ($CKPT_COUNT checkpoints)"
echo ""
echo "================================================================="
