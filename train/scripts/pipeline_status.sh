#!/bin/bash
# train/scripts/pipeline_status.sh — Training pipeline status report.
#
# Prints a full status snapshot: all pipeline steps (10 including hard-example
# mining) with live heartbeat progress from each step's log, active log tail,
# disk usage.
# Auto-detects DATA_ROOT from common SSD mount points so a single call
# gives a complete picture with no follow-up queries needed.
#
# Usage:
#   bash train/scripts/pipeline_status.sh
#   bash train/scripts/pipeline_status.sh --data-root /Volumes/2TBSSD
#   bash train/scripts/pipeline_status.sh --json        # machine-readable JSON
#
# Works from any working directory. Safe to run from Claude CoWork Dispatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAIN_DIR")"

# ── Arg parsing ───────────────────────────────────────────────────────────────
DATA_ROOT=""
JSON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --json)      JSON_MODE=true; shift ;;
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
# du_h <dir> — returns human-readable size, cached for 60s to avoid traversing
# 350+ GB trees on every pipeline_status invocation.
_DU_CACHE="${DATA_ROOT}/logs/.du_cache"
du_h() {
    local dir="$1" now size elapsed line ts cached
    [[ -d "$dir" ]] || { echo "—"; return; }
    now=$(date +%s)
    if [[ -f "$_DU_CACHE" ]]; then
        # Cache format: timestamp<TAB>path<TAB>size
        line=$(grep -F "	${dir}	" "$_DU_CACHE" 2>/dev/null | tail -1)
        if [[ -n "$line" ]]; then
            ts=$(printf '%s' "$line" | cut -f1)
            cached=$(printf '%s' "$line" | cut -f3)
            elapsed=$(( now - ts ))
            if [[ $elapsed -lt 60 ]]; then
                echo "$cached"; return
            fi
        fi
    fi
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    # Atomically update cache: remove stale entry, append fresh one
    local tmp
    tmp=$(mktemp "${DATA_ROOT}/logs/.du_cache.XXXXXX" 2>/dev/null) || { echo "$size"; return; }
    { [[ -f "$_DU_CACHE" ]] && grep -vF "	${dir}	" "$_DU_CACHE" 2>/dev/null; true; } > "$tmp"
    printf '%s\t%s\t%s\n' "$now" "$dir" "$size" >> "$tmp"
    mv "$tmp" "$_DU_CACHE" 2>/dev/null || rm -f "$tmp"
    echo "$size"
}
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
LOCK_FILE="$DATA_ROOT/logs/pipeline.lock"

# JourneyDB per-phase WDS shard count for [2b/10] summary line
_jdb_phase() {
    local n="$1" tars="$2"
    if [[ $tars -gt 0 ]]; then printf "ph.%s:%-5s" "$n" "$tars"
    else printf "ph.%s:⬜    " "$n"; fi
}
JDB_PHASE_INFO="$(_jdb_phase 1 $JDB_WDS_TARS)  $(_jdb_phase 2 $JDB_WDS2_TARS)  $(_jdb_phase 3 $JDB_WDS3_TARS)  $(_jdb_phase 4 $JDB_WDS4_TARS)"

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
    if [[ $raw -gt 0 && $wds -gt 0 ]]; then echo "${raw}/${expected} tgz [retained] · WDS ${wds} shards [retained]"
    elif [[ $raw -gt 0 ]]; then              echo "${raw}/${expected} tgz [retained] · WDS not yet converted"
    elif [[ $wds -gt 0 ]]; then              echo "tgz deleted · WDS ${wds} shards [retained]"
    else echo "$extra"; fi
}

if [[ $LAION_TARS -gt 0 ]]; then         LAION_STATE="${LAION_TARS} tars [retained]"
elif [[ $SHARD_COUNT -gt 0 ]]; then       LAION_STATE="deleted · used in step 4"
else                                       LAION_STATE="not downloaded"; fi

if [[ $COYO_TARS -gt 0 ]]; then           COYO_STATE="${COYO_TARS} tars [retained]"
elif [[ $SHARD_COUNT -gt 0 ]]; then       COYO_STATE="deleted · used in step 4"
else                                       COYO_STATE="not downloaded"; fi

if $WIKIART_RAW_EXISTS; then
    WIKIART_STATE="raw retained"
    [[ $WIKIART_WDS_TARS -gt 0 ]] && WIKIART_STATE+=" · WDS ${WIKIART_WDS_TARS} shards"
elif [[ $WIKIART_WDS_TARS -gt 0 ]]; then  WIKIART_STATE="raw deleted · WDS ${WIKIART_WDS_TARS} shards [retained]"
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
SIGLIP_COUNT=$(count_files "$PRECOMP_DIR/siglip" "*.npz")
ANCHOR_COUNT=$(count_tars "$ANCHOR_DIR")
HARD_DIR="$DATA_ROOT/hard_examples"
HARD_COUNT=$(count_tars "$HARD_DIR")

# ── Resolve checkpoint dir from config (may be on SSD) ───────────────────────
# Read checkpoint_dir from the active training config so we always find
# checkpoints regardless of whether they are on the local disk or SSD.
# Uses grep+sed to avoid depending on PyYAML (not always in system python3).
CONFIG="${TRAIN_DIR}/configs/stage1_512px.yaml"
CKPT_DIR_FROM_CONFIG=$(grep -E '^\s*checkpoint_dir:' "$CONFIG" 2>/dev/null \
    | head -1 | sed 's/.*checkpoint_dir:[[:space:]]*//' | tr -d '"'"'" | tr -d "'")
# Resolve relative paths against repo root
if [[ -n "$CKPT_DIR_FROM_CONFIG" && "$CKPT_DIR_FROM_CONFIG" != /* ]]; then
    CKPT_DIR_FROM_CONFIG="$REPO_DIR/$CKPT_DIR_FROM_CONFIG"
fi
# Fallback cascade: config → DATA_ROOT/checkpoints/stage1 → REPO_DIR/checkpoints/stage1
if [[ -n "$CKPT_DIR_FROM_CONFIG" ]]; then
    CKPT_DIR="$CKPT_DIR_FROM_CONFIG"
elif [[ -d "$DATA_ROOT/checkpoints/stage1" ]]; then
    CKPT_DIR="$DATA_ROOT/checkpoints/stage1"
elif [[ -d "$DATA_ROOT/checkpoints" ]]; then
    CKPT_DIR="$DATA_ROOT/checkpoints"
else
    CKPT_DIR="$REPO_DIR/checkpoints/stage1"
fi

CKPT_COUNT=$(find "$CKPT_DIR" -name "step_*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
# Heartbeat file written by train_ip_adapter.py every log_every steps
HEARTBEAT_FILE=$(ls -t "$CKPT_DIR"/heartbeat.json "$CKPT_DIR"/stage*/heartbeat.json \
    "$DATA_ROOT"/checkpoints/stage*/heartbeat.json \
    "$REPO_DIR"/checkpoints/stage*/heartbeat.json \
    2>/dev/null | head -1 || true)

# ── Log file discovery ────────────────────────────────────────────────────────
BUILD_LOG="$DATA_ROOT/logs/build_shards.log"
TRAIN_LOG=$(ls -t "$DATA_ROOT/logs"/pipeline_chunk*.log "$TRAIN_DIR/data/logs"/pipeline_chunk*.log 2>/dev/null | head -1 || true)
# Dedicated precompute log (legacy); fall back to the pipeline chunk log which
# captures all step output via tee when no separate precompute log exists.
PRECOMPUTE_LOG=$(ls -t "$DATA_ROOT/logs"/precompute*.log 2>/dev/null | head -1 || true)
[[ -z "$PRECOMPUTE_LOG" ]] && PRECOMPUTE_LOG="${TRAIN_LOG:-}"

# ── Expected shard total (for in-progress display only) ───────────────────────
# Parse from build_shards log or pipeline chunk log: "of N total"
# Only used when build is actively running; not used for done detection.
# Do NOT use the "across N shards" completion line — that reflects the pre-filter
# planned count, while filter_shards may later remove empty shards.
TOTAL_SHARDS_EST=""
for _log in "$BUILD_LOG" "${TRAIN_LOG:-}"; do
    [[ -z "$_log" || ! -f "$_log" ]] && continue
    parsed=$(grep -oE 'of ([0-9]+) total' "$_log" | tail -1 | grep -oE '[0-9]+')
    if [[ -n "$parsed" ]]; then TOTAL_SHARDS_EST="$parsed"; break; fi
done

# ── Process state ─────────────────────────────────────────────────────────────
BUILD_RUNNING=false;  pgrep -f "build_shards\.py"    &>/dev/null && BUILD_RUNNING=true

# ── Build_shards done condition ───────────────────────────────────────────────
# Match run_training_pipeline.sh logic: done if any shards exist and not running.
BUILD_DONE=false
if [[ $SHARD_COUNT -gt 0 && $BUILD_RUNNING == false ]]; then BUILD_DONE=true; fi
FILTER_RUNNING=false;     pgrep -f "filter_shards\.py"    &>/dev/null && FILTER_RUNNING=true
PRECOMPUTE_RUNNING=false; pgrep -f "precompute_all\.py"   &>/dev/null && PRECOMPUTE_RUNNING=true
TRAIN_RUNNING=false;      pgrep -f "train_ip_adapter\.py" &>/dev/null && TRAIN_RUNNING=true

# ── Heartbeat progress lines (from logs) ──────────────────────────────────────
# Each running step emits heartbeat lines we can parse for live progress.

# Step 3 — clip_dedup: "[X/N] X duplicates found"
DEDUP_LINES=0
[[ -f "$DEDUP_IDS" ]] && DEDUP_LINES=$(wc -l < "$DEDUP_IDS" | tr -d ' ')
DEDUP_RUN_INFO="running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9,]+/[0-9,]+\] [0-9,]+ duplicates found')
[[ -n "$hb" ]] && DEDUP_RUN_INFO="$hb"

# Step 4 — build_shards: "[worker N] src X/Y | written N records | shards A/B done"
if [[ -n "$TOTAL_SHARDS_EST" ]]; then
    BUILD_RUN_INFO="$SHARD_COUNT/$TOTAL_SHARDS_EST shards, writing..."
else
    BUILD_RUN_INFO="$SHARD_COUNT shards written so far..."
fi
hb=$(last_match "${TRAIN_LOG:-}" '\[worker [0-9]+\] src [0-9]+/[0-9]+')
[[ -z "$hb" ]] && hb=$(last_match "$BUILD_LOG" '\[worker [0-9]+\] src [0-9]+/[0-9]+')
[[ -n "$hb" ]] && BUILD_RUN_INFO="$hb"

# Step 5 — filter_shards: "[X/Y] kept=N  dropped=N  X.X s/shard  ETA Xm"
FILTER_RUN_INFO="running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] kept=')
[[ -n "$hb" ]] && FILTER_RUN_INFO="$hb"

# Step 6 — clip_dedup build-index: "[X/Y] N,NNN images embedded  X.X s/shard  ETA Xm"
DEDUP_INDEX_RUN_INFO="building..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9,]+ images embedded')
[[ -n "$hb" ]] && DEDUP_INDEX_RUN_INFO="$hb"

# Step 8 — precompute_all: "[X/Y] PCT%  X.X s/shard  ETA Xm"
PRECOMPUTE_RUN_INFO="running... ($SHARD_COUNT shards)"
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9]+%.*ETA [0-9]+h')
[[ -z "$hb" ]] && hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\]  qwen3=[0-9,]+  vae=')
[[ -n "$hb" ]] && PRECOMPUTE_RUN_INFO="$hb"
# Legacy individual scripts (for old runs that used precompute_qwen3/vae separately)
QWEN3_RUN_INFO="$QWEN3_COUNT embeddings, running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9,]+ embeddings')
[[ -n "$hb" ]] && QWEN3_RUN_INFO="$hb"
VAE_RUN_INFO="$VAE_COUNT latents, running..."
hb=$(last_match "$PRECOMPUTE_LOG" '\[[0-9]+/[0-9]+\] [0-9,]+ latents')
[[ -n "$hb" ]] && VAE_RUN_INFO="$hb"

# Step 9a — training: read heartbeat.json for live progress
LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | head -1 || true)
LATEST_STEP=""
[[ -n "$LATEST_CKPT" ]] && LATEST_STEP="($(basename "$LATEST_CKPT" .safetensors)) "
TRAIN_RUN_INFO="${LATEST_STEP}running..."
HB_STALE=""
if [[ -n "$HEARTBEAT_FILE" && -f "$HEARTBEAT_FILE" ]]; then
    # Parse heartbeat.json with python3 (always available in venv context)
    _hb_out=$(python3 - "$HEARTBEAT_FILE" 2>/dev/null <<'PYEOF'
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
    step      = d.get("step", 0)
    total     = d.get("total_steps", 0)
    loss      = d.get("loss", 0)
    loss_s    = d.get("loss_smooth", 0)
    lr        = d.get("lr", 0)
    sps       = d.get("steps_per_sec", 0)
    eta_s     = d.get("eta_seconds", 0)
    elapsed_s = d.get("elapsed_seconds", 0)
    ts        = d.get("timestamp", "")
    pct       = step / total * 100 if total else 0
    eta_h, rem = divmod(int(eta_s), 3600)
    eta_m     = rem // 60
    age_s     = int(time.time()) - int(time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))) if ts else -1
    stale     = age_s > 300  # >5 min since last log = likely stuck/dead
    print(f"step {step:,}/{total:,} ({pct:.1f}%)  loss {loss:.4f} (avg {loss_s:.4f})  {sps:.3f} steps/s  ETA {eta_h}h{eta_m:02d}m  [{ts}]")
    if stale:
        print(f"STALE:{age_s}")
    # Emit structured data for --json mode
    jd = {"step": step, "total_steps": total, "pct": round(pct, 2),
          "loss": round(loss, 6), "loss_smooth": round(loss_s, 6),
          "lr": lr, "steps_per_sec": round(sps, 4),
          "eta_seconds": int(eta_s), "eta_hours": round(eta_s/3600, 2),
          "elapsed_seconds": int(elapsed_s),
          "heartbeat_age_s": age_s, "heartbeat_stale": stale,
          "timestamp": ts}
    print("JSON:" + json.dumps(jd))
except Exception as e:
    print(f"ERR:{e}")
PYEOF
    )
    _stale_line=$(echo "$_hb_out" | grep "^STALE:")
    _hb_main=$(echo "$_hb_out" | grep -v "^STALE:\|^ERR:\|^JSON:")
    _hb_json=$(echo "$_hb_out" | grep "^JSON:" | sed 's/^JSON://')
    [[ -n "$_hb_main" ]] && TRAIN_RUN_INFO="$_hb_main"
    HB_JSON="${_hb_json:-}"
    HB_AGE_S=""
    if [[ -n "$_stale_line" ]]; then
        _age=$(echo "$_stale_line" | cut -d: -f2)
        HB_AGE_S="$_age"
        HB_STALE="  ⚠️  heartbeat stale (${_age}s ago — process may be stuck or dead)"
    fi
fi

# Step 9b — hard example mining: runs after each chunk's training
MINE_RUNNING=false; pgrep -f "mine_hard_examples\.py" &>/dev/null && MINE_RUNNING=true
MINE_RUN_INFO="running..."
hb=$(last_match "$TRAIN_LOG" 'Mining hard\|hard examples.*Done\|top-[0-9]+ records')
[[ -n "$hb" ]] && MINE_RUN_INFO="$hb"
BEST_CKPT_EXISTS=false
[[ -f "$CKPT_DIR/best.safetensors" || -f "$CKPT_DIR/stage1/best.safetensors" || \
   -f "$DATA_ROOT/checkpoints/stage1/best.safetensors" ]] && BEST_CKPT_EXISTS=true

# ── Suppress human-readable output in JSON mode ───────────────────────────────
if $JSON_MODE; then exec 3>&1 1>/dev/null; fi

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================="
echo "  Pipeline status — $(ts)"
echo "  DATA_ROOT: $DATA_ROOT"
echo "================================================================="
echo ""

# ── Steps ─────────────────────────────────────────────────────────────────────
echo "── Steps ────────────────────────────────────────────────────────"
step_status "[1/10] Verify downloads" \
    "[[ $LAION_TARS -gt 0 || $SHARD_COUNT -gt 0 || $WIKIART_WDS_TARS -gt 0 || $JDB_WDS_TARS -gt 0 ]]" "" \
    ""
printf "       LAION:   %s\n" "$LAION_STATE"
printf "       COYO:    %s\n" "$COYO_STATE"
printf "       WikiArt: %s\n" "$WIKIART_STATE"
printf "       JDB 1 (000-049):  %s\n" "$JDB1_STATE"
printf "       JDB 2 (050-099):  %s\n" "$JDB2_STATE"
printf "       JDB 3 (100-149):  %s\n" "$JDB3_STATE"
printf "       JDB 4 (150-201):  %s\n" "$JDB4_STATE"

step_status "[2a/10] WikiArt → WDS" \
    "[[ $WIKIART_WDS_TARS -gt 0 ]]" "" \
    "($WIKIART_WDS_TARS shards)"

step_status "[2b/10] JourneyDB → WDS" \
    "[[ $JDB_WDS_TARS -gt 0 ]]" "" \
    "($JDB_PHASE_INFO)"

step_status "[3/10] CLIP deduplication" \
    "[[ -f $DEDUP_IDS ]]" \
    "pgrep -f 'clip_dedup\.py all'" \
    "($DEDUP_LINES IDs blocked)" \
    "$DEDUP_RUN_INFO"

step_status "[4/10] Build unified shards" \
    "$BUILD_DONE" \
    "$BUILD_RUNNING" \
    "($SHARD_COUNT shards)" \
    "$BUILD_RUN_INFO"

step_status "[5/10] Filter shards" \
    "[[ -f $SHARDS_DIR/.filtered_chunk1 && $SHARD_COUNT -gt 0 ]]" \
    "$FILTER_RUNNING" \
    "($SHARD_COUNT shards)" "$FILTER_RUN_INFO"

step_status "[7/10] Anchor set" \
    "[[ $ANCHOR_COUNT -gt 0 ]]" "" \
    "($ANCHOR_COUNT shards · built from raw sources, not from unified shards)"

step_status "[8/10] Precompute Qwen3 + VAE [+ SigLIP]" \
    "[[ -f $PRECOMP_DIR/.done ]]" \
    "$PRECOMPUTE_RUNNING" \
    "(qwen3=$QWEN3_COUNT  vae=$VAE_COUNT  siglip=$SIGLIP_COUNT  shards=$SHARD_COUNT)" \
    "$PRECOMPUTE_RUN_INFO"

step_status "[9a/10] Train" \
    "[[ $CKPT_COUNT -gt 0 && $TRAIN_RUNNING == false ]]" \
    "$TRAIN_RUNNING" \
    "${LATEST_STEP}($CKPT_COUNT checkpoints)" \
    "$TRAIN_RUN_INFO"
[[ -n "$HB_STALE" ]] && echo "$HB_STALE"

step_status "[9b/10] Mine hard examples" \
    "$BEST_CKPT_EXISTS && [[ $HARD_COUNT -gt 0 && $MINE_RUNNING == false ]]" \
    "$MINE_RUNNING" \
    "($HARD_COUNT hard-example shards)" \
    "$MINE_RUN_INFO" \
    "$(if $BEST_CKPT_EXISTS; then echo 'Pending (runs after training)'; else echo 'Pending (needs best.safetensors)'; fi)"

step_status "[10/10] Cross-chunk dedup index  (runs after training)" \
    "[[ -f $DEDUP_INDEX ]]" \
    "pgrep -f 'clip_dedup\.py.*build-index'" \
    "($(du_h "$DEDUP_INDEX"))" "$DEDUP_INDEX_RUN_INFO"

# ── Active processes ──────────────────────────────────────────────────────────
# Show the pipeline as one entry (top-level PID only) plus individual step processes.
echo ""
echo "── Active processes ─────────────────────────────────────────────"
# Read the root pipeline PID from the lock file written by run_training_pipeline.sh.
# This is more reliable than the ppid heuristic which breaks in containers/debuggers.
PIPELINE_ROOT=""
if [[ -f "$LOCK_FILE" ]]; then
    _lock_pid=$(grep '^pid=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2)
    if [[ -n "$_lock_pid" ]] && kill -0 "$_lock_pid" 2>/dev/null; then
        PIPELINE_ROOT="$_lock_pid"
    fi
fi

if [[ -n "$PIPELINE_ROOT" ]]; then
    lock_chunk=$(grep '^chunk=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2)
    lock_scale=$(grep '^cmdline=' "$LOCK_FILE" 2>/dev/null | grep -oE '\-\-scale [^ ]+' | awk '{print $2}')
    printf "  PID %-8s run_training_pipeline.sh%s%s\n" "$PIPELINE_ROOT" \
        "${lock_chunk:+  (chunk $lock_chunk)}" \
        "${lock_scale:+  scale=$lock_scale}"
fi

# Step-level worker processes (exclude pipeline shell subprocesses)
STEP_PROCS=$(pgrep -a -l -f "build_shards\.py|clip_dedup\.py|filter_shards\.py|precompute_all\.py|precompute_qwen3\.py|precompute_vae\.py|precompute_siglip\.py|train_ip_adapter\.py" 2>/dev/null \
    | grep -v "grep\|pipeline_status" || true)
if [[ -n "$STEP_PROCS" ]]; then
    echo "$STEP_PROCS" | while read -r pid rest; do
        script=$(echo "$rest" | grep -oE '[^ /]+\.(py|sh)' | head -1)
        printf "  PID %-8s %s\n" "$pid" "$script"
    done
fi

if [[ -z "$PIPELINE_ROOT" && -z "$STEP_PROCS" ]]; then
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
elif $FILTER_RUNNING || $PRECOMPUTE_RUNNING; then
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
printf "  %-28s %s\n" "anchor_shards/ [keep]" "$(du_h "$ANCHOR_DIR") ($ANCHOR_COUNT tars)"
printf "  %-28s %s\n" "hard_examples/ [keep]" "$(du_h "$HARD_DIR") ($HARD_COUNT tars)"
printf "  %-28s %s\n" "precomputed/qwen3/"  "$(du_h "$PRECOMP_DIR/qwen3") ($QWEN3_COUNT files)"
printf "  %-28s %s\n" "precomputed/vae/"    "$(du_h "$PRECOMP_DIR/vae") ($VAE_COUNT files)"
printf "  %-28s %s\n" "precomputed/siglip/" "$(du_h "$PRECOMP_DIR/siglip") ($SIGLIP_COUNT files)"
printf "  %-28s %s\n" "embeddings/"         "$(du_h "$DATA_ROOT/embeddings")"
printf "  %-28s %s\n" "checkpoints/"        "$(du_h "$CKPT_DIR") ($CKPT_COUNT checkpoints)"
[[ -n "$HEARTBEAT_FILE" ]] && printf "  %-28s %s\n" "heartbeat:"  "$HEARTBEAT_FILE"
echo ""
echo "================================================================="

# ── JSON output (--json flag) ─────────────────────────────────────────────────
# Emits a single JSON object summarising pipeline state for AI agent consumption.
# All fields always present; unknown values use null. Human-readable output is
# suppressed; only JSON is printed when --json is set.
if $JSON_MODE; then
    # Restore stdout before emitting JSON
    exec 1>&3 3>&-

    # Determine active step
    _active_step="idle"
    $BUILD_RUNNING      && _active_step="build_shards"
    $FILTER_RUNNING     && _active_step="filter_shards"
    $PRECOMPUTE_RUNNING && _active_step="precompute"
    $TRAIN_RUNNING      && _active_step="train"
    $MINE_RUNNING       && _active_step="mine_hard_examples"

    # Pipeline metadata from lock file
    _lock_chunk=$(grep '^chunk='   "$LOCK_FILE" 2>/dev/null | cut -d= -f2 || true)
    _lock_scale=$(grep '^cmdline=' "$LOCK_FILE" 2>/dev/null | grep -oE '\-\-scale [^ ]+' | awk '{print $2}' || true)
    _latest_ckpt_name=$(basename "${LATEST_CKPT:-}" .safetensors)

    # Disk stats (raw values for JSON)
    _disk_info=$(df -k "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {print $1, $3, $4}')
    _disk_dev=$(echo "$_disk_info"  | awk '{print $1}')
    _disk_used=$(echo "$_disk_info" | awk '{printf "%.1f", $2/1048576}')   # GiB
    _disk_avail=$(echo "$_disk_info"| awk '{printf "%.1f", $3/1048576}')   # GiB

    # Step done/running/pending state as strings
    _s_downloads="pending"; { [[ $LAION_TARS -gt 0 || $SHARD_COUNT -gt 0 ]]; } 2>/dev/null && _s_downloads="done"
    _s_dedup="pending";     [[ -f "$DEDUP_IDS" ]] && _s_dedup="done";     $([[ "$DEDUP_RUN_INFO" != "running..." ]]) 2>/dev/null && [[ "$_s_dedup" != "done" ]] && _s_dedup="running"
    _s_build="pending";     [[ "$BUILD_DONE" == true ]] && _s_build="done"; [[ "$BUILD_RUNNING" == true ]] && _s_build="running"
    _s_filter="pending";    [[ -f "$SHARDS_DIR/.filtered_chunk1" ]] && _s_filter="done"; [[ "$FILTER_RUNNING" == true ]] && _s_filter="running"
    _s_precompute="pending"; [[ -f "$PRECOMP_DIR/.done" ]] && _s_precompute="done"; [[ "$PRECOMPUTE_RUNNING" == true ]] && _s_precompute="running"
    _s_train="pending";     [[ $CKPT_COUNT -gt 0 && "$TRAIN_RUNNING" == false ]] && _s_train="done"; [[ "$TRAIN_RUNNING" == true ]] && _s_train="running"
    _s_mine="pending";      [[ $HARD_COUNT -gt 0 && "$MINE_RUNNING" == false ]] && _s_mine="done"; [[ "$MINE_RUNNING" == true ]] && _s_mine="running"

    python3 - \
        "$DATA_ROOT" "$CKPT_DIR" "${HEARTBEAT_FILE:-}" \
        "${PIPELINE_ROOT:-}" "${_lock_chunk:-}" "${_lock_scale:-}" \
        "$_active_step" \
        "$SHARD_COUNT" "$QWEN3_COUNT" "$VAE_COUNT" "$SIGLIP_COUNT" \
        "$ANCHOR_COUNT" "$HARD_COUNT" "$CKPT_COUNT" "${_latest_ckpt_name:-}" \
        "${HB_JSON:-}" \
        "$_disk_dev" "$_disk_used" "$_disk_avail" \
        "$_s_downloads" "$_s_dedup" "$_s_build" "$_s_filter" \
        "$_s_precompute" "$_s_train" "$_s_mine" \
        "$TRAIN_RUNNING" \
        <<'PYEOF'
import json, sys, datetime
a = sys.argv[1:]
def s(v): return v if v else None

out = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "data_root":      a[0],
    "checkpoint_dir": a[1],
    "heartbeat_file": s(a[2]),
    "pipeline": {
        "running": bool(a[3]),
        "pid":     int(a[3]) if a[3] else None,
        "chunk":   a[4] or None,
        "scale":   a[5] or None,
    },
    "active_step": a[6],
    "steps": {
        "downloads":  a[19],
        "dedup":      a[20],
        "build_shards": a[21],
        "filter":     a[22],
        "precompute": a[23],
        "train":      a[24],
        "mine_hard":  a[25],
    },
    "training": None,
    "data": {
        "shard_count":          int(a[7]),
        "qwen3_count":          int(a[8]),
        "vae_count":            int(a[9]),
        "siglip_count":         int(a[10]),
        "anchor_shards":        int(a[11]),
        "hard_example_shards":  int(a[12]),
        "checkpoints":          int(a[13]),
        "latest_checkpoint":    s(a[14]),
    },
    "disk": {
        "device":    a[16],
        "used_gib":  float(a[17]),
        "avail_gib": float(a[18]),
    },
}

# Parse heartbeat JSON if available
hb_raw = a[15]
if hb_raw:
    try:
        hb = json.loads(hb_raw)
        hb["running"] = (a[26] == "true")
        hb["checkpoints"] = int(a[13])
        hb["latest_checkpoint"] = s(a[14])
        out["training"] = hb
    except Exception:
        out["training"] = {"running": (a[26] == "true"), "checkpoints": int(a[13])}
else:
    out["training"] = {"running": (a[26] == "true"), "checkpoints": int(a[13])}

print(json.dumps(out, indent=2))
PYEOF
fi
