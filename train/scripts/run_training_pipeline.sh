#!/bin/bash
# train/scripts/run_training_pipeline.sh — End-to-end IP-Adapter training pipeline.
#
# Handles the full lifecycle from raw downloads through training, including
# incremental chunked training on additional JourneyDB data.
#
# ── Recommended: run inside tmux so the pipeline survives shell disconnects ───
#
#   tmux new-session -d -s pipeline \
#     "caffeinate -i -d bash train/scripts/run_training_pipeline.sh \
#         --data-root /Volumes/2TBSSD 2>&1 | tee train/data/logs/pipeline_tmux.log"
#   tmux attach -t pipeline          # attach to watch progress
#
# ── Stage 1 (initial): convert → dedup → shard → precompute → train ──────────
#
#   caffeinate -i -d bash train/scripts/run_training_pipeline.sh \
#       --data-root /Volumes/2TBSSD
#
# ── Incremental chunks (after Stage 1 completes): ────────────────────────────
#
#   caffeinate -i -d bash train/scripts/run_training_pipeline.sh \
#       --chunk 2 \
#       --resume checkpoints/stage1/best.safetensors \
#       --data-root /Volumes/2TBSSD
#
# ── Options: ─────────────────────────────────────────────────────────────────
#   --chunk N          Which pipeline phase (default: 1)
#                        1 = initial: convert + dedup + shard + precompute + train
#                        2 = JourneyDB files 050–099 (resume stage 1)
#                        3 = JourneyDB files 100–149 (resume chunk 2)
#                        4 = JourneyDB files 150–201 (resume chunk 3)
#   --resume PATH      Checkpoint .safetensors to resume from (auto-detected if omitted)
#   --data-root PATH   Root for all data (default: the train/data symlink)
#   --config PATH      Training config yaml (default: train/configs/stage1_512px.yaml)
#   --steps N          Override num_steps for this chunk's training run
#   --lr LR            Override learning rate for this chunk
#   --siglip           Also precompute SigLIP features (~420 GB extra)
#   --recaption        Re-caption short captions before precompute (enforces correct ordering)
#   --skip-train       Run data prep only, stop before training
#   --skip-dedup       Skip CLIP deduplication (safe if LAION not changed)
#
# ── JourneyDB chunk file ranges: ─────────────────────────────────────────────
#   Chunk 1: 000–049  (already downloaded by download_datasets.sh)
#   Chunk 2: 050–099  (~800 GB download, ~160 GB shards)
#   Chunk 3: 100–149  (~800 GB download, ~160 GB shards)
#   Chunk 4: 150–201  (~832 GB download, ~166 GB shards)
#
# ── Per-chunk LR schedule (plans/ip-adapter-training.md §Phase 5b): ──────────
#   Chunk 1: lr=1e-4, steps=105000
#   Chunk 2: lr=3e-5, steps=40000
#   Chunk 3: lr=1e-5, steps=40000
#   Chunk 4: lr=1e-5, steps=40000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAIN_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
CHUNK=1
RESUME=""
DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"
CONFIG="$TRAIN_DIR/configs/stage1_512px.yaml"
OVERRIDE_STEPS=""
OVERRIDE_LR=""
ENABLE_SIGLIP=false
ENABLE_RECAPTION=false
SKIP_TRAIN=false
SKIP_DEDUP=false
VENV="$TRAIN_DIR/.venv/bin/activate"

# ── Per-chunk defaults ────────────────────────────────────────────────────────
declare -a CHUNK_LR=(""  "1e-4"  "3e-5"  "1e-5"  "1e-5")
declare -a CHUNK_STEPS=("" "105000" "40000" "40000" "40000")

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk)       CHUNK="$2";       shift 2 ;;
        --resume)      RESUME="$2";      shift 2 ;;
        --data-root)   DATA_ROOT="$2";   shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        --steps)       OVERRIDE_STEPS="$2"; shift 2 ;;
        --lr)          OVERRIDE_LR="$2"; shift 2 ;;
        --siglip)      ENABLE_SIGLIP=true;     shift ;;
        --recaption)   ENABLE_RECAPTION=true;  shift ;;
        --skip-train)  SKIP_TRAIN=true;        shift ;;
        --skip-dedup)  SKIP_DEDUP=true;        shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Apply per-chunk defaults unless overridden
if [[ "$CHUNK" -lt 1 || "$CHUNK" -gt 4 ]]; then
    echo "ERROR: --chunk must be 1–4, got: $CHUNK" >&2; exit 1
fi
TRAIN_LR="${OVERRIDE_LR:-${CHUNK_LR[$CHUNK]}}"
TRAIN_STEPS="${OVERRIDE_STEPS:-${CHUNK_STEPS[$CHUNK]}}"

# ── Early checks (before logging so we can fail cleanly) ─────────────────────
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: data root not found: $DATA_ROOT — run setup_data_dir.sh first" >&2; exit 1; }
[[ -f "$VENV" ]]      || { echo "ERROR: venv not found: $VENV — run train/setup.sh first" >&2; exit 1; }
if [[ -z "${TMUX:-}" ]]; then
    echo "WARNING: not running inside tmux — pipeline will be killed if this shell exits." >&2
    echo "         Recommended: tmux new-session -d -s pipeline \"caffeinate -i -d bash $0 $*\"" >&2
fi

# ── Pipeline lock — prevent concurrent runs against the same DATA_ROOT ────────
mkdir -p "$DATA_ROOT/logs"
LOCK_FILE="$DATA_ROOT/logs/pipeline.lock"

_acquire_lock() {
    local lock_content
    lock_content="$(printf 'pid=%s\nhost=%s\nchunk=%s\ndata_root=%s\nstarted=%s\ncmdline=%s\n' \
        "$$" "$(hostname)" "$CHUNK" "$DATA_ROOT" "$(date '+%Y-%m-%d %H:%M:%S')" "$0 $*")"
    if ( set -C; echo "$lock_content" > "$LOCK_FILE" ) 2>/dev/null; then
        return 0  # acquired
    fi
    # Lock exists — check whether owner is still alive
    local existing_pid
    existing_pid=$(grep '^pid=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2)
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "ERROR: pipeline already running. Lock: $LOCK_FILE" >&2
        echo "" >&2
        cat "$LOCK_FILE" >&2
        echo "" >&2
        echo "To force-clear a stale lock: rm $LOCK_FILE" >&2
        return 1
    fi
    # Stale lock — remove and retry once
    echo "WARNING: removing stale lock (PID ${existing_pid:-unknown} not running)" >&2
    rm -f "$LOCK_FILE"
    if ( set -C; echo "$lock_content" > "$LOCK_FILE" ) 2>/dev/null; then
        return 0
    fi
    echo "ERROR: failed to acquire lock after removing stale lock" >&2
    return 1
}

_release_lock() { [[ "${LOCK_FILE:-}" == "$DATA_ROOT/logs/pipeline.lock" ]] && rm -f "$LOCK_FILE"; }

_acquire_lock "$@" || exit 1
trap _release_lock EXIT INT TERM

# ── Logging ───────────────────────────────────────────────────────────────────
LOG="$DATA_ROOT/logs/pipeline_chunk${CHUNK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

# retry MAX_ATTEMPTS DELAY_SECS CMD [ARGS...]
# Runs CMD up to MAX_ATTEMPTS times, sleeping DELAY_SECS between attempts.
# Returns 0 on first success; returns 1 (triggering set -e exit) on exhaustion.
retry() {
    local max="$1" delay="$2"; shift 2
    local attempt=1
    while true; do
        if "$@"; then return 0; fi
        [[ "$attempt" -ge "$max" ]] && { log "ERROR: command failed after $max attempts: $*"; return 1; }
        log "  Attempt $attempt/$max failed — retrying in ${delay}s..."
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

log "================================================================="
log "  IP-Adapter training pipeline  chunk=$CHUNK"
log "  DATA_ROOT: $DATA_ROOT"
log "  CONFIG:    $CONFIG"
log "  LR:        $TRAIN_LR    STEPS: $TRAIN_STEPS"
log "  Log:       $LOG"
log "  Lock:      $LOCK_FILE  (PID $$)"
log "================================================================="

source "$VENV"

# ── Helper: count .tar files in a directory ───────────────────────────────────
# Use find (exits 0 even on no matches); ls|wc fails under set -o pipefail.
count_tars() { find -L "${1}" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l; }

# ── Helper: auto-detect latest checkpoint ────────────────────────────────────
latest_checkpoint() {
    local dir="$1"
    ls -t "$dir"/step_*.safetensors 2>/dev/null | grep -v ema | head -1
}

# ═════════════════════════════════════════════════════════════════════════════
# CHUNK 1 — initial pipeline
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$CHUNK" -eq 1 ]]; then

    # ── Note: LAION and COYO must be downloaded separately before running ─────
    # LAION aesthetic subset (~150 shards, already in WebDataset .tar format):
    #   pip install img2dataset
    #   img2dataset --url_list laion_aesthetic_urls.parquet \
    #       --input_format parquet --url_col url --caption_col caption \
    #       --output_format webdataset --output_folder "$DATA_ROOT/raw/laion" \
    #       --resize_mode keep_ratio --min_image_size 256 --processes_count 8
    # COYO-700M subset (optional, also WebDataset):
    #   img2dataset --url_list coyo_urls.parquet ... --output_folder "$DATA_ROOT/raw/coyo"

    # ── 1a–3. Parallel: (download if needed) → convert → dedup ──────────────
    # Each source runs as an independent background job:
    #   WikiArt:   download huggan/wikiart if absent → convert_wikiart.py
    #   JourneyDB: download JourneyDB/JourneyDB if absent → convert_journeydb.py
    #   CLIP dedup: runs immediately on LAION (already on disk)
    # Jobs that find their WDS already present exit immediately.
    # LAION must exist already (WebDataset output of img2dataset, no HF source).

    WIKIART_WDS="$DATA_ROOT/raw/wikiart_wds"
    JDB_WDS="$DATA_ROOT/raw/journeydb_wds"
    DEDUP_IDS="$DATA_ROOT/dedup_ids/duplicate_ids.txt"
    WIKIART_LOG="$DATA_ROOT/logs/wikiart.log"
    JDB_LOG="$DATA_ROOT/logs/journeydb.log"
    DEDUP_LOG="$DATA_ROOT/logs/clip_dedup.log"

    [[ -d "$DATA_ROOT/raw/laion" ]] || \
        die "raw/laion not found or not accessible: $DATA_ROOT/raw/laion — run img2dataset first"
    LAION_TARS=$(count_tars "$DATA_ROOT/raw/laion")
    [[ "$LAION_TARS" -gt 0 ]] || \
        die "LAION shards not found in $DATA_ROOT/raw/laion/ — run img2dataset first"
    COYO_TARS=$(count_tars "$DATA_ROOT/raw/coyo" || echo 0)
    log "[1/9] LAION: $LAION_TARS shards  COYO: $COYO_TARS shards"
    log "      Launching WikiArt, JourneyDB, and CLIP dedup in parallel..."

    # Pre-create output directories for all parallel background jobs so they
    # never fail due to a missing parent directory.
    mkdir -p "$WIKIART_WDS" "$JDB_WDS" \
             "$DATA_ROOT/embeddings" "$DATA_ROOT/dedup_ids"

    # ── WikiArt background job ────────────────────────────────────────────────
    WIKIART_PID=""
    if [[ $(count_tars "$WIKIART_WDS") -gt 0 ]]; then
        log "  WikiArt:   WDS already present ($(count_tars "$WIKIART_WDS") shards) — skipping"
    else
        log "  WikiArt:   launching → $WIKIART_LOG"
        (
            set -euo pipefail
            _wa_sentinel="${DATA_ROOT}/raw/wikiart/.download_complete"
            if [[ ! -f "$_wa_sentinel" ]]; then
                echo "[$(date '+%T')] Downloading huggan/wikiart (~27 GB)..."
                python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download('huggan/wikiart', repo_type='dataset',
    local_dir='${DATA_ROOT}/raw/wikiart')
print('Download complete.')
PYEOF
                touch "$_wa_sentinel"
            fi
            echo "[$(date '+%T')] Converting WikiArt to WebDataset..."
            python "$SCRIPT_DIR/convert_wikiart.py" \
                --input  "$DATA_ROOT/raw/wikiart/data" \
                --output "$WIKIART_WDS" \
                --shard-size 1000
            echo "[$(date '+%T')] Done: $(count_tars "$WIKIART_WDS") shards."
            [[ -d "$DATA_ROOT/raw/wikiart" ]] && \
                { echo "Removing raw WikiArt (~1.6 GB)..."; rm -rf "$DATA_ROOT/raw/wikiart"; }
        ) > "$WIKIART_LOG" 2>&1 &
        WIKIART_PID=$!
    fi

    # ── JourneyDB background job ──────────────────────────────────────────────
    JDB_PID=""
    if [[ $(count_tars "$JDB_WDS") -gt 0 ]]; then
        log "  JourneyDB: WDS already present ($(count_tars "$JDB_WDS") shards) — skipping"
    else
        log "  JourneyDB: launching → $JDB_LOG"
        (
            set -euo pipefail
            shopt -s nullglob
            _tgz=("$DATA_ROOT/raw/journeydb/data/train/imgs"/0[0-4][0-9].tgz)
            shopt -u nullglob
            _jdb_sentinel="${DATA_ROOT}/raw/journeydb/.download_complete_chunk1"
            if [[ ${#_tgz[@]} -lt 50 && ! -f "$_jdb_sentinel" ]]; then
                echo "[$(date '+%T')] ${#_tgz[@]}/50 tgz files — downloading JourneyDB chunk 1 (~800 GB)..."
                python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    'JourneyDB/JourneyDB',
    repo_type='dataset',
    local_dir='${DATA_ROOT}/raw/journeydb',
    allow_patterns=[
        'data/train/imgs/0[0-4][0-9].tgz',
        'data/train/train_anno_realease_repath.jsonl.tgz',
    ],
)
print('Download complete.')
PYEOF
                touch "$_jdb_sentinel"
            fi
            echo "[$(date '+%T')] Converting JourneyDB chunk 1 to WebDataset..."
            python "$SCRIPT_DIR/convert_journeydb.py" \
                --input      "$DATA_ROOT/raw/journeydb" \
                --output     "$JDB_WDS" \
                --shard-size 5000 \
                --start-tgz  0 --end-tgz 49
            echo "[$(date '+%T')] Done: $(count_tars "$JDB_WDS") shards."
            [[ -d "$DATA_ROOT/raw/journeydb" ]] && \
                { echo "Removing original JourneyDB tgz (~730 GB)..."; rm -rf "$DATA_ROOT/raw/journeydb"; }
        ) > "$JDB_LOG" 2>&1 &
        JDB_PID=$!
    fi

    # ── CLIP dedup background job (LAION already on disk — starts immediately) ─
    DEDUP_PID=""
    if [[ -f "$DEDUP_IDS" ]] || $SKIP_DEDUP; then
        log "  CLIP dedup: already done (or --skip-dedup) — skipping"
    else
        log "  CLIP dedup: launching → $DEDUP_LOG"
        python "$SCRIPT_DIR/clip_dedup.py" all \
            --shards     "$DATA_ROOT/raw/laion" \
            --embeddings "$DATA_ROOT/embeddings" \
            --output     "$DATA_ROOT/dedup_ids" > "$DEDUP_LOG" 2>&1 &
        DEDUP_PID=$!
    fi

    # ── Wait for all three ────────────────────────────────────────────────────
    PARALLEL_FAIL=0
    if [[ -n "$WIKIART_PID" ]]; then
        if wait "$WIKIART_PID"; then
            log "[2a/9] WikiArt done: $(count_tars "$WIKIART_WDS") shards"
        else
            log "ERROR: WikiArt job failed — see $WIKIART_LOG" >&2; PARALLEL_FAIL=1
        fi
    fi
    if [[ -n "$JDB_PID" ]]; then
        if wait "$JDB_PID"; then
            log "[2b/9] JourneyDB done: $(count_tars "$JDB_WDS") shards"
        else
            log "ERROR: JourneyDB job failed — see $JDB_LOG" >&2; PARALLEL_FAIL=1
        fi
    fi
    if [[ -n "$DEDUP_PID" ]]; then
        if wait "$DEDUP_PID"; then
            log "[3/9] CLIP dedup done: $DEDUP_IDS"
            [[ -d "$DATA_ROOT/embeddings" ]] && \
                { log "  Removing CLIP embeddings (~3 GB)..."; rm -rf "$DATA_ROOT/embeddings"; }
        else
            log "ERROR: CLIP dedup failed — see $DEDUP_LOG" >&2; PARALLEL_FAIL=1
        fi
    fi
    [[ "$PARALLEL_FAIL" -eq 0 ]] || die "One or more parallel steps failed"

    # ── 1e. Build unified shards (+ concurrent filter) ────────────────────────
    SHARDS_DIR="$DATA_ROOT/shards"
    FILTER_DONE="$SHARDS_DIR/.filtered_chunk1"
    if [[ $(count_tars "$SHARDS_DIR") -gt 0 ]]; then
        log "[4/9] Unified shards already built ($(count_tars "$SHARDS_DIR") shards) — skipping"
    else
        log "[4/9] Building unified shards from all sources..."
        SOURCES=("$DATA_ROOT/raw/laion" "$JDB_WDS" "$WIKIART_WDS")
        [[ "${COYO_TARS:-0}" -gt 0 ]] && SOURCES+=("$DATA_ROOT/raw/coyo")

        BUILD_ARGS=("--sources" "${SOURCES[@]}" "--output" "$SHARDS_DIR")
        [[ -f "$DEDUP_IDS" ]] && BUILD_ARGS+=("--blocklist" "$DEDUP_IDS")

        # Filter shards as they arrive: build_shards publishes atomically (tmp→rename),
        # so filter_shards never sees a partial shard.  Per-shard .filtered sentinels
        # make repeated runs idempotent.  Output goes to a separate log to avoid
        # interleaving with build_shards progress lines.
        FILTER_BG_LOG="$DATA_ROOT/logs/filter_background.log"
        (while true; do
            python "$SCRIPT_DIR/filter_shards.py" --shards "$SHARDS_DIR" \
                >> "$FILTER_BG_LOG" 2>&1 || true
            sleep 60
        done) &
        FILTER_LOOP_PID=$!

        python "$SCRIPT_DIR/build_shards.py" "${BUILD_ARGS[@]}" 2>&1 | tee /tmp/build_shards.log
        log "  Done: $(count_tars "$SHARDS_DIR") unified shards"

        # Stop loop; final pass catches any shards written in the last 60s window
        kill "$FILTER_LOOP_PID" 2>/dev/null || true
        wait "$FILTER_LOOP_PID" 2>/dev/null || true
        log "[5/9] Final filter pass..."
        python "$SCRIPT_DIR/filter_shards.py" --shards "$SHARDS_DIR"
        touch "$FILTER_DONE"
        log "  Done"
    fi

    # ── 1f. Filter shards ─────────────────────────────────────────────────────
    if [[ -f "$FILTER_DONE" ]]; then
        log "[5/9] Shards already filtered — skipping"
    else
        log "[5/9] Filtering shards (drop corrupt/small/bad-caption)..."
        python "$SCRIPT_DIR/filter_shards.py" --shards "$SHARDS_DIR"
        touch "$FILTER_DONE"
        log "  Done"
    fi

    # ── 1gr. Recaption (optional, must run BEFORE precompute AND anchor set) ──────
    # Enforces correct ordering: recaption updates shard captions; both precompute
    # and anchor set must see the final captions or they will silently diverge.
    RECAPTION_DONE="$SHARDS_DIR/.recaption_done"
    if $ENABLE_RECAPTION; then
        if [[ -f "$RECAPTION_DONE" ]]; then
            log "[8r/9] Recaption already done — skipping"
        else
            log "[8r/9] Re-captioning short captions (this takes ~2 days, resumable)..."
            python "$SCRIPT_DIR/recaption.py" --shards "$SHARDS_DIR"
            touch "$RECAPTION_DONE"
            log "  Done: recaptioning complete"
        fi
    fi

    # ── 1g. Launch anchor set in background (LAION + WikiArt only, no JourneyDB) ─
    # JourneyDB uses synthetic Midjourney-style prompts; anchors should be natural
    # language only so they don't skew the style distribution during chunk training.
    # LAION is never deleted; WIKIART_WDS was just converted above and still exists.
    ANCHOR_DIR="$DATA_ROOT/anchor_shards"
    ANCHOR_SOURCES=("$DATA_ROOT/raw/laion" "$WIKIART_WDS")
    [[ "${COYO_TARS:-0}" -gt 0 ]] && ANCHOR_SOURCES+=("$DATA_ROOT/raw/coyo")
    mkdir -p "$ANCHOR_DIR"
    ANCHOR_PID=""
    if [[ $(count_tars "$ANCHOR_DIR") -gt 0 ]]; then
        log "[7/9] Anchor set already exists ($(count_tars "$ANCHOR_DIR") shards) — skipping"
    else
        log "[7/9] Creating anchor set in background (LAION + WikiArt, no JourneyDB)..."
        python "$SCRIPT_DIR/create_anchor_set.py" \
            --shards  "${ANCHOR_SOURCES[@]}" \
            --output  "$ANCHOR_DIR" \
            --n       10000 &
        ANCHOR_PID=$!
    fi

    # Wait for anchor set (should be done well before precompute finishes, but check)
    if [[ -n "$ANCHOR_PID" ]]; then
        log "[7/9] Waiting for anchor set creation..."
        wait "$ANCHOR_PID" || die "create_anchor_set.py failed"
        log "  Done: anchor set in $ANCHOR_DIR"
    fi

    # ── 1h. Unified precompute: Qwen3 + VAE [+ SigLIP] ───────────────────────
    # Single pass over all shards — reads each tar once instead of 2-3 times.
    QWEN3_DIR="$DATA_ROOT/precomputed/qwen3"
    VAE_DIR="$DATA_ROOT/precomputed/vae"
    SIGLIP_DIR="$DATA_ROOT/precomputed/siglip"
    PRECOMPUTE_DONE="$DATA_ROOT/precomputed/.done"

    mkdir -p "$QWEN3_DIR" "$VAE_DIR"
    $ENABLE_SIGLIP && mkdir -p "$SIGLIP_DIR"
    if [[ -f "$PRECOMPUTE_DONE" ]]; then
        log "[8/9] Precompute already done — skipping"
    else
        log "[8/9] Precomputing Qwen3 + VAE (~14h, ~341 GB)..."
        PRECOMPUTE_ARGS=(
            --shards        "$SHARDS_DIR"
            --qwen3-output  "$QWEN3_DIR"
            --vae-output    "$VAE_DIR"
        )
        $ENABLE_SIGLIP && PRECOMPUTE_ARGS+=(--siglip --siglip-output "$SIGLIP_DIR")
        retry 3 60 python "$SCRIPT_DIR/precompute_all.py" "${PRECOMPUTE_ARGS[@]}"
        touch "$PRECOMPUTE_DONE"
        log "  Done: $DATA_ROOT/precomputed/"
    fi

    # ── 1i. Background-prefetch JDB chunk 2 (front-run next phase, GPU-bound window) ──
    JDB_RAW="$DATA_ROOT/raw/journeydb"
    JDB_WDS_CHUNK2="$DATA_ROOT/raw/journeydb_wds_chunk2"
    PREFETCH2_DONE="$JDB_RAW/.prefetch_chunk2_done"
    PREFETCH2_PID_FILE="$JDB_RAW/.prefetch_chunk2_pid"
    PREFETCH2_LOG="$DATA_ROOT/logs/prefetch_chunk2.log"
    PREFETCH2_PID=""
    if [[ $(count_tars "$JDB_WDS_CHUNK2") -gt 0 || -f "$PREFETCH2_DONE" ]]; then
        log "[1i/9] JDB chunk 2 already downloaded or converted — no prefetch needed"
    else
        log "[1i/9] Launching JDB chunk 2 background prefetch (~800 GB, resumable)..."
        mkdir -p "$JDB_RAW" "$DATA_ROOT/logs"
        (python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    'JourneyDB/JourneyDB',
    repo_type='dataset',
    local_dir='${JDB_RAW}',
    allow_patterns=[
        'data/train/imgs/0[5-9][0-9].tgz',
        'data/train/train_anno_realease_repath.jsonl.tgz',
    ],
)
PYEOF
        touch "${PREFETCH2_DONE}"
        ) >> "$PREFETCH2_LOG" 2>&1 &
        PREFETCH2_PID=$!
        echo "$PREFETCH2_PID" > "$PREFETCH2_PID_FILE"
        log "  Prefetch PID $PREFETCH2_PID → $PREFETCH2_LOG"
    fi

    # ── 1j. Train ─────────────────────────────────────────────────────────────
    if $SKIP_TRAIN; then
        log "[9/9] --skip-train set — data prep complete."
        log "  To train:  caffeinate -i -d python train/train_ip_adapter.py \\"
        log "                 --config $CONFIG"
    else
        log "[9/9] Starting Stage 1 training ($TRAIN_STEPS steps, lr=$TRAIN_LR)..."
        TRAIN_ARGS=("--config" "$CONFIG")
        [[ -n "$TRAIN_LR" ]]    && TRAIN_ARGS+=("--lr" "$TRAIN_LR")
        [[ -n "$TRAIN_STEPS" ]] && TRAIN_ARGS+=("--max-steps" "$TRAIN_STEPS")
        [[ -n "$ANCHOR_DIR" && $(count_tars "$ANCHOR_DIR") -gt 0 ]] && \
            TRAIN_ARGS+=("--anchor-shards" "$ANCHOR_DIR")

        retry 2 30 python "$TRAIN_DIR/train_ip_adapter.py" "${TRAIN_ARGS[@]}"
        log "Stage 1 training complete."

        # Wait for prefetch if it's still running
        if [[ -n "$PREFETCH2_PID" ]] && kill -0 "$PREFETCH2_PID" 2>/dev/null; then
            log "Training done. Waiting for JDB chunk 2 prefetch (PID $PREFETCH2_PID)..."
            wait "$PREFETCH2_PID" || true
            log "Prefetch complete."
        fi
    fi

    # ── 1k. Build cross-chunk dedup index (after training — not on critical path) ──
    # DEPENDENCY: chunks 2–4 step 2b (cross-chunk dedup) requires this index.
    # Building here (after training) is safe for chunk 1 because chunk 2 cannot
    # start until chunk 1 training completes and this index exists.
    # Do NOT skip this step if you plan to run chunk 2+.
    DEDUP_INDEX="$DATA_ROOT/dedup_ids/dedup_index.faiss"
    ALL_EMBEDDINGS="$DATA_ROOT/embeddings/all"
    mkdir -p "$ALL_EMBEDDINGS"
    if [[ -f "$DEDUP_INDEX" ]]; then
        log "[6/9] Cross-chunk dedup index already built — skipping"
    else
        log "[6/9] Building cross-chunk dedup index (~1.5h)..."
        log "  (Required before starting --chunk 2; chunks 2-4 step 2b depends on this)"
        python "$SCRIPT_DIR/clip_dedup.py" build-index \
            --shards     "$SHARDS_DIR" \
            --embeddings "$ALL_EMBEDDINGS" \
            --index      "$DEDUP_INDEX"
        log "  Done: $DEDUP_INDEX"
    fi

# ═════════════════════════════════════════════════════════════════════════════
# CHUNKS 2–4 — incremental JourneyDB
# ═════════════════════════════════════════════════════════════════════════════
elif [[ "$CHUNK" -ge 2 && "$CHUNK" -le 4 ]]; then

    # ── File ranges per chunk ─────────────────────────────────────────────────
    case $CHUNK in
        2) JDB_PATTERNS=("data/train/imgs/0[5-9][0-9].tgz")
           JDB_TGZ_START=50;  JDB_TGZ_END=99  ;;
        3) JDB_PATTERNS=("data/train/imgs/1[0-4][0-9].tgz")
           JDB_TGZ_START=100; JDB_TGZ_END=149 ;;
        4) JDB_PATTERNS=("data/train/imgs/1[5-9][0-9].tgz"
                         "data/train/imgs/200.tgz"
                         "data/train/imgs/201.tgz")
           JDB_TGZ_START=150; JDB_TGZ_END=201 ;;
    esac

    JDB_RAW="$DATA_ROOT/raw/journeydb"
    JDB_WDS="$DATA_ROOT/raw/journeydb_wds_chunk${CHUNK}"
    SHARDS_DIR="$DATA_ROOT/shards"
    ANCHOR_DIR="$DATA_ROOT/anchor_shards"
    QWEN3_DIR="$DATA_ROOT/precomputed/qwen3"
    VAE_DIR="$DATA_ROOT/precomputed/vae"
    CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints/stage1}"
    DEDUP_INDEX="$DATA_ROOT/dedup_ids/dedup_index.faiss"
    DEDUP_IDS="$DATA_ROOT/dedup_ids/duplicate_ids.txt"
    CHUNK_EMBEDDINGS="$DATA_ROOT/embeddings/chunk${CHUNK}"

    # ── 1. Download JourneyDB chunk N ─────────────────────────────────────────
    # Count present files using nullglob-safe array expansion (avoids ls|wc pipefail)
    EXPECTED_TGZ_COUNT=$((JDB_TGZ_END - JDB_TGZ_START + 1))

    shopt -s nullglob
    IMGS_DIR="$JDB_RAW/data/train/imgs"
    case $CHUNK in
        2) PRESENT_FILES=("$IMGS_DIR"/0[5-9][0-9].tgz) ;;
        3) PRESENT_FILES=("$IMGS_DIR"/1[0-4][0-9].tgz) ;;
        4) PRESENT_FILES=("$IMGS_DIR"/1[5-9][0-9].tgz)
           [[ -f "$IMGS_DIR/200.tgz" ]] && PRESENT_FILES+=("$IMGS_DIR/200.tgz")
           [[ -f "$IMGS_DIR/201.tgz" ]] && PRESENT_FILES+=("$IMGS_DIR/201.tgz") ;;
    esac
    shopt -u nullglob
    PRESENT=${#PRESENT_FILES[@]}

    if [[ "$PRESENT" -ge "$EXPECTED_TGZ_COUNT" ]]; then
        log "[1/6] JourneyDB chunk $CHUNK already downloaded ($PRESENT files) — skipping"
    else
        log "[1/6] Downloading JourneyDB chunk $CHUNK (${EXPECTED_TGZ_COUNT} files, ~$((EXPECTED_TGZ_COUNT * 16)) GB)..."

        # Build Python allow_patterns list
        PATTERNS_PY="["
        for p in "${JDB_PATTERNS[@]}"; do
            PATTERNS_PY+="'$p', "
        done
        # Always include annotation files (idempotent if already present)
        PATTERNS_PY+="'data/train/train_anno_realease_repath.jsonl.tgz']"

        python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    'JourneyDB/JourneyDB',
    repo_type='dataset',
    local_dir='$JDB_RAW',
    allow_patterns=$PATTERNS_PY,
)
print('Download complete.')
PYEOF
        [[ $? -eq 0 ]] || die "HuggingFace snapshot_download failed for chunk $CHUNK"
        # Recount after download
        shopt -s nullglob
        case $CHUNK in
            2) PRESENT_FILES=("$IMGS_DIR"/0[5-9][0-9].tgz) ;;
            3) PRESENT_FILES=("$IMGS_DIR"/1[0-4][0-9].tgz) ;;
            4) PRESENT_FILES=("$IMGS_DIR"/1[5-9][0-9].tgz)
               [[ -f "$IMGS_DIR/200.tgz" ]] && PRESENT_FILES+=("$IMGS_DIR/200.tgz")
               [[ -f "$IMGS_DIR/201.tgz" ]] && PRESENT_FILES+=("$IMGS_DIR/201.tgz") ;;
        esac
        shopt -u nullglob
        log "  Done: ${#PRESENT_FILES[@]} files in $IMGS_DIR"
    fi

    # ── 2. Convert chunk N to WebDataset ─────────────────────────────────────
    if [[ $(count_tars "$JDB_WDS") -gt 0 ]]; then
        log "[2/6] Chunk $CHUNK WebDataset already exists ($(count_tars "$JDB_WDS") shards) — skipping"
    else
        log "[2/6] Converting JourneyDB chunk $CHUNK to WebDataset (tgz $JDB_TGZ_START–$JDB_TGZ_END)..."
        python "$SCRIPT_DIR/convert_journeydb.py" \
            --input      "$JDB_RAW" \
            --output     "$JDB_WDS" \
            --shard-size 5000 \
            --start-tgz  "$JDB_TGZ_START" --end-tgz "$JDB_TGZ_END"
        log "  Done: $(count_tars "$JDB_WDS") shards in $JDB_WDS"
    fi

    # ── 2b. Cross-chunk dedup: flag images already seen in previous chunks ────
    DEDUP_DONE="$SHARDS_DIR/.deduped_chunk${CHUNK}"
    if [[ -f "$DEDUP_DONE" ]]; then
        log "[2b/6] Cross-chunk dedup already done for chunk $CHUNK — skipping"
    elif [[ ! -f "$DEDUP_INDEX" ]]; then
        log "[2b/6] WARNING: no dedup index found at $DEDUP_INDEX"
        log "         Skipping cross-chunk dedup (run build-index on chunk 1 shards first)"
    else
        log "[2b/6] Running cross-chunk dedup (query $JDB_WDS against existing index)..."
        mkdir -p "$CHUNK_EMBEDDINGS"
        python "$SCRIPT_DIR/clip_dedup.py" incremental \
            --shards     "$JDB_WDS" \
            --embeddings "$CHUNK_EMBEDDINGS" \
            --index      "$DEDUP_INDEX" \
            --blocklist  "$DEDUP_IDS"
        touch "$DEDUP_DONE"
        log "  Done: updated blocklist at $DEDUP_IDS"
    fi

    # ── 3. Build + filter new shards (appended to main shards/) ──────────────
    FILTER_DONE="$SHARDS_DIR/.filtered_chunk${CHUNK}"
    if [[ -f "$FILTER_DONE" ]]; then
        log "[3/6] Chunk $CHUNK shards already built and filtered — skipping"
    else
        # Capture shard count before appending — needed for filter --start-idx
        EXISTING=$(count_tars "$SHARDS_DIR")
        log "[3/6] Building new shards for chunk $CHUNK (appending after shard $EXISTING)..."
        CHUNK_BUILD_ARGS=("--sources" "$JDB_WDS" "--output" "$SHARDS_DIR" "--start-idx" "$EXISTING")
        [[ -f "$DEDUP_IDS" ]] && CHUNK_BUILD_ARGS+=("--blocklist" "$DEDUP_IDS")
        python "$SCRIPT_DIR/build_shards.py" "${CHUNK_BUILD_ARGS[@]}"
        log "  Done: $(count_tars "$SHARDS_DIR") total shards (was $EXISTING)"

        log "       Filtering new shards only (--start-idx $EXISTING)..."
        python "$SCRIPT_DIR/filter_shards.py" \
            --shards    "$SHARDS_DIR" \
            --start-idx "$EXISTING"
        touch "$FILTER_DONE"
        log "  Done"
    fi

    # ── 4. Precompute embeddings for new shards (single pass, resume-safe) ──────
    # precompute_all.py reads each shard once and writes Qwen3 + VAE [+ SigLIP].
    # Per-sample .npz skip logic means only new shards do real work on resume.
    SIGLIP_DIR="$DATA_ROOT/precomputed/siglip"
    log "[4/6] Precomputing Qwen3 + VAE embeddings for chunk $CHUNK shards..."
    CHUNK_PRECOMPUTE_ARGS=(
        --shards       "$SHARDS_DIR"
        --qwen3-output "$QWEN3_DIR"
        --vae-output   "$VAE_DIR"
    )
    $ENABLE_SIGLIP && CHUNK_PRECOMPUTE_ARGS+=(--siglip --siglip-output "$SIGLIP_DIR")
    retry 3 60 python "$SCRIPT_DIR/precompute_all.py" "${CHUNK_PRECOMPUTE_ARGS[@]}"
    log "  Done: embeddings in $QWEN3_DIR and $VAE_DIR"

    # ── 5. Resume training ────────────────────────────────────────────────────
    if $SKIP_TRAIN; then
        log "[5/6] --skip-train set — data prep for chunk $CHUNK complete."
    else
        # Auto-detect checkpoint if not specified
        if [[ -z "$RESUME" ]]; then
            RESUME=$(latest_checkpoint "$CKPT_DIR")
            [[ -n "$RESUME" ]] || die "No checkpoint found in $CKPT_DIR — pass --resume PATH"
            log "  Auto-detected checkpoint: $RESUME"
        fi

        log "[5/6] Resuming training for chunk $CHUNK ($TRAIN_STEPS steps, lr=$TRAIN_LR)..."
        log "       Checkpoint: $RESUME"
        log "       Anchor shards: $ANCHOR_DIR"

        retry 2 30 python "$TRAIN_DIR/train_ip_adapter.py" \
            --config    "$CONFIG" \
            --resume    "$RESUME" \
            --lr        "$TRAIN_LR" \
            --max-steps "$TRAIN_STEPS" \
            --anchor-shards "$ANCHOR_DIR"

        log "Chunk $CHUNK training complete."
    fi

    # ── 6. Clean up raw chunk files to free SSD space ─────────────────────────
    log "[6/6] Chunk $CHUNK complete."
    log ""
    log "  Raw chunk files: $JDB_RAW/data/train/imgs/ ($((EXPECTED_TGZ_COUNT * 16)) GB approx)"
    log "  Chunk WDS shards: $JDB_WDS"
    log "  These can be deleted to free SSD space once training is verified:"
    log ""
    log "  To free raw chunk data (~$((EXPECTED_TGZ_COUNT * 16)) GB):"
    case $CHUNK in
        2) log "    rm $JDB_RAW/data/train/imgs/0[5-9][0-9].tgz" ;;
        3) log "    rm $JDB_RAW/data/train/imgs/1[0-4][0-9].tgz" ;;
        4) log "    rm $JDB_RAW/data/train/imgs/1[5-9][0-9].tgz $JDB_RAW/data/train/imgs/200.tgz $JDB_RAW/data/train/imgs/201.tgz" ;;
    esac
    log "  To free chunk WDS shards (after precompute verified):"
    log "    rm -rf $JDB_WDS"
    log ""
    log "  !! Delete only after confirming embeddings are in $QWEN3_DIR and $VAE_DIR !!"

else
    die "Invalid --chunk $CHUNK (must be 1–4)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "================================================================="
log "  Pipeline chunk $CHUNK complete."
log "  Full log: $LOG"
log "================================================================="
