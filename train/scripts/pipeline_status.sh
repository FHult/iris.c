#!/bin/bash
# train/scripts/pipeline_status.sh — Training pipeline status report.
#
# Prints a full status snapshot: all 9 pipeline steps with live heartbeat
# progress from each step's log, active log tail, disk usage.
# Auto-detects DATA_ROOT from common SSD mount points so a single call
# gives a complete picture with no follow-up queries needed.
#
# Usage:
#   bash train/scripts/pipeline_status.sh
#   bash train/scripts/pipeline_status.sh --data-root /Volumes/2TBSSD
#
# Works from any working directory. Safe to run from Claude CoWork Dispatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Arg parsing ───────────────────────────────────────────────────────────────
DATA_ROOT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
if [[ -z "$DATA_ROOT" ]]; then
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]] && \
           [[ -d "$candidate/shards" || -d "$candidate/raw" || -d "$candidate/dedup_ids" || -d "$candidate/precomputed" ]]; then
            DATA_ROOT="$candidate"
            break
        fi
    done
    DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
count_files() { find -L "${1}" -maxdepth 1 -name "${2:-*}" 2>/dev/null | wc -l | tr -d ' '; }
count_tars()  { count_files "${1}" "*.tar"; }
du_h()        { du -sh "$1" 2>/dev/null | cut -f1; }
ts()          { date '+%Y-%m-%d %H:%M:%S'; }

# last_match <file> <egrep-pattern>
# Returns the last line in <file> matching <pattern>, stripped of leading whitespace.
last_match() {
    [[ -f "${1:-}" ]] && grep -E "$2" "$1" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' || true
}

step_status() {
    local label="$1" done_cond="$2" run_cond="$3"
    if eval "$done_cond" &>/dev/null; then
        printf "  ✅ %-42s %s\n" "$label" "$4"
    elif [[ -n "$run_cond" ]] && eval "$run_cond" &>/dev/null; then
        printf "  ⏳ %-42s %s\n" "$label" "$5"
    else
        printf "  ⬜ %-42s %s\n" "$label" "${6:-Pending}"
    fi
}

# ── Counts (gathered once, reused throughout) ─────────────────────────────────
LAION_TARS=$(count_tars "$DATA_ROOT/raw/laion")
COYO_TARS=$(count_tars "$DATA_ROOT/raw/coyo")
WIKIART_WDS_TARS=$(count_tars "$DATA_ROOT/raw/wikiart_wds")
# JourneyDB is downloaded and converted in 4 phases across training chunks.
# Chunk 1 writes to journeydb_wds/; chunks 2-4 each write to journeydb_wds_chunkN/.
JDB_WDS_TARS=$(count_tars "$DATA_ROOT/raw/journeydb_wds")
JDB_WDS2_TARS=$(count_tars "$DATA_ROOT/raw/journeydb_wds_chunk2")
JDB_WDS3_TARS=$(count_tars "$DATA_ROOT/raw/journeydb_wds_chunk3")
JDB_WDS4_TARS=$(count_tars "$DATA_ROOT/raw/journeydb_wds_chunk4")
DEDUP_IDS="$DATA_ROOT/dedup_ids/duplicate_ids.txt"
SHARDS_DIR="$DATA_ROOT/shards"
SHARD_COUNT=$(count_tars "$SHARDS_DIR")
DEDUP_INDEX="$DATA_ROOT/dedup_ids/dedup_index.faiss"
ANCHOR_DIR="$DATA_ROOT/anchor_shards"
PRECOMP_DIR="$DATA_ROOT/precomputed"
CKPT_DIR="$TRAIN_DIR/checkpoints"

# JourneyDB per-phase summary for display (phase N: shard count or ⬜ if not yet downloaded)
_jdb_phase() {
    local n="$1" tars="$2" label="$3"
    if [[ $tars -gt 0 ]]; then printf "ph.%s:%s(%s)" "$n" "$tars" "$label"
    else printf "ph.%s:⬜" "$n"; fi
}
JDB_PHASE_INFO="$(_jdb_phase 1 $JDB_WDS_TARS 000-049)  $(_jdb_phase 2 $JDB_WDS2_TARS 050-099)  $(_jdb_phase 3 $JDB_WDS3_TARS 100-149)  $(_jdb_phase 4 $JDB_WDS4_TARS 150-201)"

# ── Per-source raw state for step 1 detail lines ──────────────────────────────
WIKIART_RAW_EXISTS=false; [[ -d "$DATA_ROOT/raw/wikiart" ]] && WIKIART_RAW_EXISTS=true
JDB_IMGS_DIR="$DATA_ROOT/raw/journeydb/data/train/imgs"
JDB_RAW1_TGZS=$(count_files "$JDB_IMGS_DIR" "0[0-4][0-9].tgz")
JDB_RAW2_TGZS=$(count_files "$JDB_IMGS_DIR" "0[5-9][0-9].tgz")
JDB_RAW3_TGZS=$(count_files "$JDB_IMGS_DIR" "1[0-4][0-9].tgz")
JDB_RAW4_TGZS=$(( $(count_files "$JDB_IMGS_DIR" "1[5-9][0-9].tgz") + $(count_files "$JDB_IMGS_DIR" "20[01].tgz") ))
JDB_PREFETCH2_RUNNING=false; JDB_PREFETCH2_DONE=false
_pf_pid_file="$DATA_ROOT/raw/journeydb/.prefetch_chunk2_pid"
[[ -f "$DATA_ROOT/raw/journeydb/.prefetch_chunk2_done" ]] && JDB_PREFETCH2_DONE=true
if [[ -f "$_pf_pid_file" ]]; then
    _pf_pid=$(cat "$_pf_pid_file" 2>/dev/null)
    kill -0 "$_pf_pid" 2>/dev/null && JDB_PREFETCH2_RUNNING=true
fi

# Helper: per-source state string for step 1 detail lines
_jdb_src_state() {
    local raw="$1" expected="$2" wds="$3" extra="${4:-pending}"
    if [[ $raw -gt 0 ]]; then echo "${raw}/${expected} tgz · WDS ${wds} shards"
    elif [[ $wds -gt 0 ]]; then echo "raw deleted · WDS ready (${wds} shards)"
    else echo "$extra"; fi
}

if [[ $LAION_TARS -gt 0 ]]; then         LAION_STATE="${LAION_TARS} tars"
elif [[ $SHARD_COUNT -gt 0 ]]; then       LAION_STATE="raw deleted · used in step 4"
else                                       LAION_STATE="not downloaded"; fi

if [[ $COYO_TARS -gt 0 ]]; then           COYO_STATE="${COYO_TARS} tars"
elif [[ $SHARD_COUNT -gt 0 ]]; then       COYO_STATE="raw deleted · used in step 4"
else                                       COYO_STATE="not downloaded"; fi

if $WIKIART_RAW_EXISTS; then
    WIKIART_STATE="raw present"
    [[ $WIKIART_WDS_TARS -gt 0 ]] && WIKIART_STATE+=" · WDS ${WIKIART_WDS_TARS} shards"
elif [[ $WIKIART_WDS_TARS -gt 0 ]]; then  WIKIART_STATE="raw deleted · WDS ready (${WIKIART_WDS_TARS} shards)"
else                                       WIKIART_STATE="not downloaded"; fi

JDB1_STATE=$(_jdb_src_state $JDB_RAW1_TGZS 50 $JDB_WDS_TARS)
if $JDB_PREFETCH2_RUNNING; then   JDB2_EXTRA="prefetching in background..."
elif $JDB_PREFETCH2_DONE; then    JDB2_EXTRA="prefetch done · not converted yet"
else                               JDB2_EXTRA="pending"; fi
JDB2_STATE=$(_jdb_src_state $JDB_RAW2_TGZS 50 $JDB_WDS2_TARS "$JDB2_EXTRA")
JDB3_STATE=$(_jdb_src_state $JDB_RAW3_TGZS 50 $JDB_WDS3_TARS)
JDB4_STATE=$(_jdb_src_state $JDB_RAW4_TGZS 52 $JDB_WDS4_TARS)

QWEN3_COUNT=$(count_files "$PRECOMP_DIR/qwen3" "*.npz")
VAE_COUNT=$(count_files "$PRECOMP_DIR/vae" "*.npz")
ANCHOR_COUNT=$(count_tars "$ANCHOR_DIR")
CKPT_COUNT=$(find "$CKPT_DIR" -name "step_*.safetensors" 2>/dev/null | wc -l | tr -d ' ')

# ── Log file discovery ────────────────────────────────────────────────────────
BUILD_LOG=/tmp/build_shards.log
TRAIN_LOG=$(ls -t "$DATA_ROOT/logs"/pipeline_chunk*.log "$TRAIN_DIR/data/logs"/pipeline_chunk*.log 2>/dev/null | head -1 || true)
# Dedicated precompute log (legacy); fall back to the pipeline chunk log which
# captures all step output via tee when no separate precompute log exists.
PRECOMPUTE_LOG=$(ls -t "$DATA_ROOT/logs"/precompute*.log 2>/dev/null | head -1 || true)
[[ -z "$PRECOMPUTE_LOG" ]] && PRECOMPUTE_LOG="${TRAIN_LOG:-}"

# ── Expected shard total (for in-progress display only) ───────────────────────
# Parse from build_shards log: "writing 000000–XXXXXX of N total"
# Only used when build is actively running; not used for done detection.
# Do NOT use the "across N shards" completion line — that reflects the pre-filter
# planned count (e.g. 495), while filter_shards may later remove empty shards.
TOTAL_SHARDS_EST=""
if [[ -f "$BUILD_LOG" ]]; then
    parsed=$(grep -oE 'of ([0-9]+) total' "$BUILD_LOG" | tail -1 | grep -oE '[0-9]+')
    [[ -n "$parsed" ]] && TOTAL_SHARDS_EST="$parsed"
fi

# ── Process state ─────────────────────────────────────────────────────────────
BUILD_RUNNING=false;  pgrep -f build_shards     &>/dev/null && BUILD_RUNNING=true

# ── Build_shards done condition ───────────────────────────────────────────────
# Match run_training_pipeline.sh logic: done if any shards exist and not running.
BUILD_DONE=false
if [[ $SHARD_COUNT -gt 0 && $BUILD_RUNNING == false ]]; then BUILD_DONE=true; fi
FILTER_RUNNING=false; pgrep -f filter_shards    &>/dev/null && FILTER_RUNNING=true
QWEN3_RUNNING=false;  pgrep -f precompute_qwen3 &>/dev/null && QWEN3_RUNNING=true
VAE_RUNNING=false;    pgrep -f precompute_vae   &>/dev/null && VAE_RUNNING=true
SIGLIP_RUNNING=false; pgrep -f precompute_siglip &>/dev/null && SIGLIP_RUNNING=true
TRAIN_RUNNING=false;  pgrep -f train_ip_adapter &>/dev/null && TRAIN_RUNNING=true

# ── Heartbeat progress lines (from logs) ──────────────────────────────────────
# Each running step emits heartbeat lines we can parse for live progress.

# Step 3 — clip_dedup: "[X/N] X duplicates found"
DEDUP_LINES=0
[[ -f "$DEDUP_IDS" ]] && DEDUP_LINES=$(wc -l < "$DEDUP_IDS" | tr -d ' ')
DEDUP_RUN_INFO="running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9,]+/[0-9,]+\] [0-9,]+ duplicates found')
[[ -n "$hb" ]] && DEDUP_RUN_INFO="$hb"

# Step 4 — build_shards: "[worker N] src X/Y | written N records | shards A/B full"
if [[ -n "$TOTAL_SHARDS_EST" ]]; then
    BUILD_RUN_INFO="$SHARD_COUNT/$TOTAL_SHARDS_EST shards, writing..."
else
    BUILD_RUN_INFO="$SHARD_COUNT shards written so far..."
fi
hb=$(last_match "$BUILD_LOG" '\[worker [0-9]+\] src [0-9]+/[0-9]+')
[[ -n "$hb" ]] && BUILD_RUN_INFO="$hb"

# Step 5 — filter_shards: "[X/Y] kept=N  dropped=N  X.X shards/s  ETA Xm"
FILTER_RUN_INFO="running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] kept=')
[[ -n "$hb" ]] && FILTER_RUN_INFO="$hb"

# Step 8a — precompute_qwen3: "[X/Y] N,NNN embeddings  X.XX shards/s  ETA Xm"
QWEN3_RUN_INFO="$QWEN3_COUNT/$SHARD_COUNT shards, running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9,]+ embeddings')
[[ -n "$hb" ]] && QWEN3_RUN_INFO="$hb"

# Step 8b — precompute_vae: "[X/Y] N,NNN latents  X.XX shards/s  ETA Xm"
VAE_RUN_INFO="$VAE_COUNT/$SHARD_COUNT shards, running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9,]+ latents')
[[ -n "$hb" ]] && VAE_RUN_INFO="$hb"

# Step 9 — training: "step X,XXX/105,000  loss X.XXXX (avg X.XXXX)  lr ...  ETA Xh XXm"
LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | grep -v ema | head -1)
LATEST_STEP=""
[[ -n "$LATEST_CKPT" ]] && LATEST_STEP="($(basename "$LATEST_CKPT" .safetensors)) "
TRAIN_RUN_INFO="${LATEST_STEP}running..."
hb=$(last_match "$TRAIN_LOG" 'steps/s.*ETA [0-9]+h')
[[ -n "$hb" ]] && TRAIN_RUN_INFO="$hb"

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================="
echo "  Pipeline status — $(ts)"
echo "  DATA_ROOT: $DATA_ROOT"
echo "================================================================="
echo ""

# ── Steps ─────────────────────────────────────────────────────────────────────
echo "── Steps ────────────────────────────────────────────────────────"
step_status "[1/9] Verify downloads" \
    "[[ $LAION_TARS -gt 0 || $SHARD_COUNT -gt 0 || $WIKIART_WDS_TARS -gt 0 || $JDB_WDS_TARS -gt 0 ]]" "" \
    ""
printf "       LAION:   %-40s COYO: %s\n" "$LAION_STATE" "$COYO_STATE"
printf "       WikiArt: %s\n" "$WIKIART_STATE"
printf "       JDB ph.1(000-049): %-36s ph.2(050-099): %s\n" "$JDB1_STATE" "$JDB2_STATE"
printf "       JDB ph.3(100-149): %-36s ph.4(150-201): %s\n" "$JDB3_STATE" "$JDB4_STATE"

step_status "[2a/9] WikiArt → WDS" \
    "[[ $WIKIART_WDS_TARS -gt 0 ]]" "" \
    "($WIKIART_WDS_TARS shards)"

step_status "[2b/9] JourneyDB → WDS" \
    "[[ $JDB_WDS_TARS -gt 0 ]]" "" \
    "($JDB_PHASE_INFO)"

step_status "[3/9] CLIP deduplication" \
    "[[ -f $DEDUP_IDS ]]" \
    "pgrep -f clip_dedup" \
    "($DEDUP_LINES IDs blocked)" \
    "$DEDUP_RUN_INFO"

step_status "[4/9] Build unified shards" \
    "$BUILD_DONE" \
    "$BUILD_RUNNING" \
    "($SHARD_COUNT shards)" \
    "$BUILD_RUN_INFO"

step_status "[5/9] Filter shards" \
    "[[ -f $SHARDS_DIR/.filtered_chunk1 ]]" \
    "$FILTER_RUNNING" \
    "done" "$FILTER_RUN_INFO"

step_status "[6/9] Cross-chunk dedup index" \
    "[[ -f $DEDUP_INDEX ]]" \
    "pgrep -f 'clip_dedup.*build-index'" \
    "($(du_h "$DEDUP_INDEX"))" "building..."

step_status "[7/9] Anchor set" \
    "[[ $ANCHOR_COUNT -gt 0 ]]" "" \
    "($ANCHOR_COUNT shards)"

step_status "[8a/9] Precompute Qwen3 embeddings" \
    "[[ -f $PRECOMP_DIR/qwen3/.done || ($QWEN3_COUNT -ge $SHARD_COUNT && $SHARD_COUNT -gt 0) ]]" \
    "$QWEN3_RUNNING" \
    "($QWEN3_COUNT/$SHARD_COUNT shards)" \
    "$QWEN3_RUN_INFO"

step_status "[8b/9] Precompute VAE latents" \
    "[[ -f $PRECOMP_DIR/vae/.done || ($VAE_COUNT -ge $SHARD_COUNT && $SHARD_COUNT -gt 0) ]]" \
    "$VAE_RUNNING" \
    "($VAE_COUNT/$SHARD_COUNT shards)" \
    "$VAE_RUN_INFO"

step_status "[9/9] Train" \
    "[[ $CKPT_COUNT -gt 0 ]]" \
    "$TRAIN_RUNNING" \
    "${LATEST_STEP}($CKPT_COUNT checkpoints)" \
    "$TRAIN_RUN_INFO"

# ── Active processes (names only) ─────────────────────────────────────────────
echo ""
echo "── Active processes ─────────────────────────────────────────────"
PROC_LINES=$(pgrep -a -l -f "build_shards|clip_dedup|filter_shards|precompute_qwen3|precompute_vae|precompute_siglip|train_ip_adapter|caffeinate|run_training_pipeline|run_shard_and_precompute" 2>/dev/null \
    | grep -v "grep\|pipeline_status")
if [[ -n "$PROC_LINES" ]]; then
    echo "$PROC_LINES" | while read -r pid rest; do
        script=$(echo "$rest" | grep -oE '[^ /]+\.(py|sh)' | head -1)
        [[ -z "$script" ]] && script=$(echo "$rest" | awk '{print $1}' | xargs basename 2>/dev/null)
        printf "  PID %-8s %s\n" "$pid" "$script"
    done
else
    echo "  (none)"
fi

# ── tmux sessions ─────────────────────────────────────────────────────────────
echo ""
echo "── tmux sessions ────────────────────────────────────────────────"
TMUX_SESSIONS=$(tmux list-sessions 2>/dev/null)
if [[ -n "$TMUX_SESSIONS" ]]; then
    echo "$TMUX_SESSIONS" | while read -r line; do echo "  $line"; done
else
    echo "  (none)"
fi

# ── Active log tail ───────────────────────────────────────────────────────────
# Show the most relevant log for the currently running step.
# Priority: build_shards → precompute/filter → training → most recent
echo ""
echo "── Active log ───────────────────────────────────────────────────"
ACTIVE_LOG=""
if $BUILD_RUNNING; then
    # Prefer live pipeline log; fall back to /tmp/build_shards.log for heartbeat
    ACTIVE_LOG="${TRAIN_LOG:-}"
    [[ -z "$ACTIVE_LOG" && -f "$BUILD_LOG" ]] && ACTIVE_LOG="$BUILD_LOG"
elif $FILTER_RUNNING || $QWEN3_RUNNING || $VAE_RUNNING || $SIGLIP_RUNNING; then
    ACTIVE_LOG="${PRECOMPUTE_LOG:-}"
    [[ -z "$ACTIVE_LOG" ]] && ACTIVE_LOG=$(ls -t "$DATA_ROOT/logs"/*.log 2>/dev/null | head -1 || true)
elif $TRAIN_RUNNING; then
    ACTIVE_LOG="${TRAIN_LOG:-}"
fi
# Fall back to most recent log if nothing running
if [[ -z "$ACTIVE_LOG" ]]; then
    ACTIVE_LOG=$(ls -t \
        "$BUILD_LOG" \
        "$DATA_ROOT/logs"/pipeline_chunk*.log \
        "$DATA_ROOT/logs"/*.log \
        "$TRAIN_DIR/data/logs"/pipeline_chunk*.log \
        2>/dev/null | head -1 || true)
fi

if [[ -n "$ACTIVE_LOG" ]]; then
    echo "  $ACTIVE_LOG"
    tail -15 "$ACTIVE_LOG" | while read -r line; do echo "  $line"; done
else
    echo "  (no log found)"
fi

# ── Disk ──────────────────────────────────────────────────────────────────────
echo ""
echo "── Disk ─────────────────────────────────────────────────────────"
df -h "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {
    printf "  %-30s used=%-8s avail=%-8s capacity=%s\n", $1, $3, $4, $5
}'
echo ""
printf "  %-28s %s\n" "raw/"                "$(du_h "$DATA_ROOT/raw")"
printf "  %-28s %s\n" "shards/"             "$(du_h "$SHARDS_DIR") ($SHARD_COUNT tars)"
printf "  %-28s %s\n" "precomputed/qwen3/"  "$(du_h "$PRECOMP_DIR/qwen3") ($QWEN3_COUNT files)"
printf "  %-28s %s\n" "precomputed/vae/"    "$(du_h "$PRECOMP_DIR/vae") ($VAE_COUNT files)"
printf "  %-28s %s\n" "embeddings/"         "$(du_h "$DATA_ROOT/embeddings")"
printf "  %-28s %s\n" "checkpoints/"        "$(du_h "$CKPT_DIR") ($CKPT_COUNT checkpoints)"
echo ""
echo "================================================================="
