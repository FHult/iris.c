#!/bin/bash
# train/scripts/pipeline_reset.sh — Clean all pipeline-generated files, preserving
# downloaded datasets.
#
# Usage:
#   bash train/scripts/pipeline_reset.sh [OPTIONS]
#
# Options:
#   --data-root PATH   Data root (default: auto-detect /Volumes/2TBSSD or train/data)
#   --full             Also delete downloaded raw datasets (wikiart, journeydb).
#                      LAION and COYO are never deleted — they are pre-existing and
#                      not managed by this pipeline.
#   --dry-run          Print what would be deleted; make no changes.
#   --keep-journeydb   Keep raw/journeydb_wds* (converted JourneyDB WDS shards).
#   --yes, -y          Skip the confirmation prompt.
#
# What is cleaned (standard reset):
#   Converted WebDataset dirs   raw/wikiart_wds, raw/journeydb_wds{,_chunk2..4}
#   Unified shards              shards/
#   Anchor set                  anchor_shards/
#   CLIP dedup output           dedup_ids/
#   Precomputed caches          precomputed/
#   Cross-chunk embeddings      embeddings/
#   Logs and lock file          logs/
#   Prefetch sentinels          raw/journeydb/.prefetch_chunk2_{done,pid}
#
# What is kept (standard reset):
#   Downloaded WikiArt          raw/wikiart/         (~27 GB)
#   Downloaded JourneyDB        raw/journeydb/       (~800 GB+ per chunk)
#   LAION / COYO shards         raw/laion/, raw/coyo/ (pre-existing, never touched)
#   Training checkpoints        train/checkpoints/
#
# What additionally is cleaned (--full):
#   Downloaded WikiArt          raw/wikiart/
#   Downloaded JourneyDB        raw/journeydb/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Arg parsing ───────────────────────────────────────────────────────────────
DATA_ROOT=""
FULL_RESET=false
KEEP_JOURNEYDB=false
DRY_RUN=false
YES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)       DATA_ROOT="$2"; shift 2 ;;
        --full)            FULL_RESET=true;      shift ;;
        --keep-journeydb)  KEEP_JOURNEYDB=true;  shift ;;
        --dry-run)         DRY_RUN=true;         shift ;;
        --yes|-y)          YES=true;             shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
if [[ -z "$DATA_ROOT" ]]; then
    if [[ -d "/Volumes/2TBSSD" ]]; then
        DATA_ROOT="/Volumes/2TBSSD"
    elif [[ -d "$TRAIN_DIR/data" ]]; then
        DATA_ROOT="$TRAIN_DIR/data"
    else
        echo "ERROR: could not auto-detect data root." >&2
        echo "       Pass --data-root PATH explicitly." >&2
        exit 1
    fi
fi

[[ -d "$DATA_ROOT" ]] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 1; }

# ── Guard: refuse if pipeline is running ─────────────────────────────────────
LOCK_FILE="$DATA_ROOT/logs/pipeline.lock"
if [[ -f "$LOCK_FILE" ]]; then
    existing_pid=$(grep '^pid=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2 || true)
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "ERROR: pipeline is currently running (PID $existing_pid)." >&2
        echo "       Run pipeline_stop.sh first, then retry." >&2
        exit 1
    fi
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
_du() {
    # Human-readable size of a path, or empty string if absent.
    local path="$1"
    if [[ -e "$path" ]]; then
        du -sh "$path" 2>/dev/null | cut -f1
    fi
}

_remove() {
    local path="$1"
    [[ -e "$path" ]] || return 0
    local size
    size=$(_du "$path")
    if $DRY_RUN; then
        printf "  [dry-run]  %-55s %s\n" "$path" "${size:-}"
    else
        printf "  Deleting   %-55s %s\n" "$path" "${size:-}"
        rm -rf "$path"
    fi
}

# ── Build target lists ────────────────────────────────────────────────────────

# Converted WebDataset dirs (intermediate, always cleaned unless --keep-journeydb)
WDS_DIRS=("$DATA_ROOT/raw/wikiart_wds")
if ! $KEEP_JOURNEYDB; then
    WDS_DIRS+=(
        "$DATA_ROOT/raw/journeydb_wds"
        "$DATA_ROOT/raw/journeydb_wds_chunk2"
        "$DATA_ROOT/raw/journeydb_wds_chunk3"
        "$DATA_ROOT/raw/journeydb_wds_chunk4"
    )
fi

# Pipeline output directories (always cleaned)
OUTPUT_DIRS=(
    "$DATA_ROOT/shards"
    "$DATA_ROOT/anchor_shards"
    "$DATA_ROOT/dedup_ids"
    "$DATA_ROOT/precomputed"
    "$DATA_ROOT/embeddings"
    "$DATA_ROOT/logs"
)

# Sentinel / pid files that live inside dirs we keep in a standard reset.
# (Sentinels inside OUTPUT_DIRS are cleaned automatically with their parent dir.)
LOOSE_SENTINELS=(
    "$DATA_ROOT/raw/journeydb/.prefetch_chunk2_done"
    "$DATA_ROOT/raw/journeydb/.prefetch_chunk2_pid"
)

# Downloaded raw datasets (--full only)
RAW_DATASETS=(
    "$DATA_ROOT/raw/wikiart"
    "$DATA_ROOT/raw/journeydb"
)

# ── Summary header ────────────────────────────────────────────────────────────
echo "================================================================="
if $FULL_RESET; then
    echo "  Pipeline FULL reset  (includes downloaded datasets)"
else
    echo "  Pipeline reset  (preserves downloaded datasets)"
fi
echo "  Data root : $DATA_ROOT"
$DRY_RUN && echo "  Mode      : dry-run (no files will be deleted)"
echo "================================================================="
echo ""

# ── Show what will be deleted ─────────────────────────────────────────────────
_present() { [[ -e "$1" ]]; }
_any_present() { for p in "$@"; do _present "$p" && return 0; done; return 1; }

echo "Will delete:"

_any_present "${WDS_DIRS[@]}" && {
    echo "  Converted WebDataset dirs:"
    for d in "${WDS_DIRS[@]}"; do
        _present "$d" && printf "    %-58s %s\n" "$d" "$(_du "$d")"
    done
}

_any_present "${OUTPUT_DIRS[@]}" && {
    echo "  Pipeline output dirs:"
    for d in "${OUTPUT_DIRS[@]}"; do
        _present "$d" && printf "    %-58s %s\n" "$d" "$(_du "$d")"
    done
}

_any_present "${LOOSE_SENTINELS[@]}" && {
    echo "  Sentinel / pid files:"
    for s in "${LOOSE_SENTINELS[@]}"; do
        _present "$s" && echo "    $s"
    done
}

if $FULL_RESET; then
    _any_present "${RAW_DATASETS[@]}" && {
        echo "  Downloaded raw datasets (--full):"
        for d in "${RAW_DATASETS[@]}"; do
            _present "$d" && printf "    %-58s %s\n" "$d" "$(_du "$d")"
        done
    }
fi

echo ""
echo "Will keep:"
echo "  $DATA_ROOT/raw/laion           (pre-existing, not pipeline-managed)"
echo "  $DATA_ROOT/raw/coyo            (pre-existing, not pipeline-managed)"
$KEEP_JOURNEYDB && echo "  $DATA_ROOT/raw/journeydb_wds*  (--keep-journeydb)"
if ! $FULL_RESET; then
    echo "  $DATA_ROOT/raw/wikiart         (downloaded dataset)"
    echo "  $DATA_ROOT/raw/journeydb       (downloaded dataset)"
fi
echo "  $TRAIN_DIR/checkpoints/        (training checkpoints)"
echo ""

# ── Confirmation ──────────────────────────────────────────────────────────────
if ! $YES && ! $DRY_RUN; then
    if $FULL_RESET; then
        echo "WARNING: --full will permanently delete all downloaded datasets."
    fi
    read -r -p "Proceed? [y/N] " ans
    case "$ans" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# ── Delete ────────────────────────────────────────────────────────────────────
for d in "${WDS_DIRS[@]}";        do _remove "$d"; done
for d in "${OUTPUT_DIRS[@]}";     do _remove "$d"; done
for s in "${LOOSE_SENTINELS[@]}"; do _remove "$s"; done
if $FULL_RESET; then
    for d in "${RAW_DATASETS[@]}"; do _remove "$d"; done
fi

echo ""
if $DRY_RUN; then
    echo "Dry run complete — no files deleted."
else
    echo "Reset complete."
    echo "Run pipeline_start.sh to begin a fresh run."
fi
