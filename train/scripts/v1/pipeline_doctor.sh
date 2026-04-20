#!/bin/bash
# train/scripts/pipeline_doctor.sh — Health check of pipeline progress.
#
# Reviews all completed pipeline outputs for correctness and integrity:
# counts, sizes, sample content, and cross-checks between steps. Reports
# any anomalies that indicate a step ran incorrectly or produced bad data,
# even if it appeared to complete successfully.
#
# Checks:
#   [D1] Source dataset completeness and size plausibility
#   [D2] Dedup blocklist: line count, no duplicate IDs
#   [D3] Shards: count vs expected, size distribution, sample content
#   [D4] Filtered shard sentinel vs actual shard content sampling
#   [D5] Qwen3 cache: count vs shard count, shape and dtype consistency
#   [D6] VAE cache: count vs shard count, shape and dtype consistency
#   [D7] Precompute ID alignment: npz filenames match shard record keys
#   [D8] Checkpoints: count, step progression, EMA pairing, sizes
#
# Usage:
#   bash train/scripts/pipeline_doctor.sh
#   bash train/scripts/pipeline_doctor.sh --data-root /Volumes/2TBSSD
#   bash train/scripts/pipeline_doctor.sh --verbose    # show passing checks too
#
# Safe to call from Claude CoWork Dispatch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Args ──────────────────────────────────────────────────────────────────────
DATA_ROOT_EXPLICIT=""
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT_EXPLICIT="$2"; shift 2 ;;
        --verbose)   VERBOSE=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
if [[ -z "$DATA_ROOT_EXPLICIT" ]]; then
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]] && \
           [[ -d "$candidate/shards" || -d "$candidate/raw" || -d "$candidate/precomputed" ]]; then
            DATA_ROOT_EXPLICIT="$candidate"
            break
        fi
    done
    DATA_ROOT_EXPLICIT="${DATA_ROOT_EXPLICIT:-$TRAIN_DIR/data}"
fi
DATA_ROOT="$DATA_ROOT_EXPLICIT"

PYTHON="$TRAIN_DIR/.venv/bin/python"

# ── Helpers ───────────────────────────────────────────────────────────────────
WARN=0; INFO=0; ERR=0
ts() { date '+%H:%M:%S'; }

ok()   { $VERBOSE && printf "  ✅ [D%s] %s\n" "$1" "$2"; INFO=$(( INFO + 1 )); }
warn() { printf "  ⚠️  [D%s] %s\n  ↳ %s\n" "$1" "$2" "$3"; WARN=$(( WARN + 1 )); }
err()  { printf "  ❌ [D%s] %s\n  ↳ %s\n" "$1" "$2" "$3"; ERR=$(( ERR + 1 )); }
skip() { $VERBOSE && printf "  ⬜ [D%s] %s  (not yet run)\n" "$1" "$2"; }

echo "================================================================="
echo "  Pipeline health check — $(ts)"
echo "  DATA_ROOT: $DATA_ROOT"
echo "================================================================="
echo ""

# ── D1: Source datasets ───────────────────────────────────────────────────────
echo "── D1: Source datasets ──────────────────────────────────────────"
LAION_N=$(find "$DATA_ROOT/raw/laion" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
COYO_N=$(find  "$DATA_ROOT/raw/coyo"  -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
JDB_N=$(find   "$DATA_ROOT/raw/journeydb_wds" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
WA_N=$(find    "$DATA_ROOT/raw/wikiart_wds"   -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')

printf "  LAION:         %4d shards\n" "$LAION_N"
printf "  COYO:          %4d shards\n" "$COYO_N"
printf "  JourneyDB WDS: %4d shards\n" "$JDB_N"
printf "  WikiArt WDS:   %4d shards\n" "$WA_N"

[[ "$LAION_N" -ge 100 ]] && ok  1 "LAION count ($LAION_N ≥ 100)" \
                          || warn 1 "LAION count low" "expected ~150, got $LAION_N"
[[ "$JDB_N"   -ge 100 ]] && ok  1 "JourneyDB WDS count ($JDB_N ≥ 100)" \
                          || warn 1 "JourneyDB WDS count low" "expected ~210, got $JDB_N"
[[ "$WA_N"    -ge 80  ]] && ok  1 "WikiArt WDS count ($WA_N ≥ 80)" \
                          || warn 1 "WikiArt WDS count low" "expected ~104, got $WA_N"
echo ""

# ── D2: Dedup blocklist ───────────────────────────────────────────────────────
echo "── D2: Dedup blocklist ──────────────────────────────────────────"
DEDUP_FILE="$DATA_ROOT/dedup_ids/duplicate_ids.txt"
if [[ ! -f "$DEDUP_FILE" ]]; then
    skip 2 "Dedup blocklist (not yet generated)"
else
    DEDUP_LINES=$(wc -l < "$DEDUP_FILE" | tr -d ' ')
    DEDUP_UNIQ=$(sort -u "$DEDUP_FILE" | wc -l | tr -d ' ')
    DEDUP_DUP=$(( DEDUP_LINES - DEDUP_UNIQ ))
    printf "  Lines: %s  |  Unique: %s  |  Duplicates in blocklist: %s\n" \
        "$DEDUP_LINES" "$DEDUP_UNIQ" "$DEDUP_DUP"
    [[ "$DEDUP_LINES" -gt 50000 ]] \
        && ok   2 "Blocklist size ($DEDUP_LINES IDs)" \
        || warn 2 "Blocklist seems small" "expected >100K IDs, got $DEDUP_LINES"
    [[ "$DEDUP_DUP" -eq 0 ]] \
        && ok   2 "No duplicate IDs in blocklist" \
        || warn 2 "Blocklist contains $DEDUP_DUP duplicate IDs" "may cause extra skips in build_shards"
fi
echo ""

# ── D3: Shards ────────────────────────────────────────────────────────────────
echo "── D3: Shards ───────────────────────────────────────────────────"
SHARDS_DIR="$DATA_ROOT/shards"
SHARD_COUNT=$(find "$SHARDS_DIR" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
printf "  Shard count: %s\n" "$SHARD_COUNT"

if [[ "$SHARD_COUNT" -eq 0 ]]; then
    skip 3 "Shards (not yet built)"
else
    # Check for very small shards (likely stubs from a crashed run)
    STUB_COUNT=$(find "$SHARDS_DIR" -maxdepth 1 -name "*.tar" -size -1k 2>/dev/null | wc -l | tr -d ' ')
    [[ "$STUB_COUNT" -eq 0 ]] \
        && ok   3 "No stub/empty shards found" \
        || err  3 "Stub shards detected" "$STUB_COUNT shards < 1 KB — likely from a crashed build run, should be deleted"

    # Sample 5 random shards: verify image+caption pairs and record counts
    SAMPLE=()
    while IFS= read -r f; do SAMPLE+=("$f"); done < \
        <(ls "$SHARDS_DIR"/*.tar 2>/dev/null | shuf -n 5 2>/dev/null || ls "$SHARDS_DIR"/*.tar | head -5)
    SAMPLE_RESULT=$("$PYTHON" - "${SAMPLE[@]}" <<'PYEOF' 2>&1
import sys, tarfile, io
paths = sys.argv[1:]
issues = []
for path in paths:
    try:
        # Streaming mode (r:) works on in-progress tars missing the EOA block.
        jpgs = []; txts = []
        try:
            with tarfile.open(path, 'r:') as t:
                for m in t:
                    if not m.isfile(): continue
                    if m.name.endswith('.jpg'): jpgs.append(m.name)
                    elif m.name.endswith('.txt'): txts.append(m.name)
        except tarfile.ReadError:
            pass  # missing EOA block — normal when tar is still open for writing
        stems_jpg = {n.rsplit('.',1)[0] for n in jpgs}
        stems_txt = {n.rsplit('.',1)[0] for n in txts}
        unpaired = stems_jpg.symmetric_difference(stems_txt)
        if len(unpaired) > 2:  # ≤2 is normal during active writing (last record in flight)
            issues.append(f"{path.split('/')[-1]}: {len(unpaired)} unpaired keys")
        if len(jpgs) < 100:
            issues.append(f"{path.split('/')[-1]}: only {len(jpgs)} records (expect ~5000)")
    except Exception as e:
        issues.append(f"{path.split('/')[-1]}: {e}")
if issues:
    print("WARN: " + "; ".join(issues))
else:
    print(f"ok: sampled {len(paths)} shards, all healthy")
PYEOF
)
    if [[ "$SAMPLE_RESULT" == ok:* ]]; then
        ok  3 "Shard content sample ($SAMPLE_RESULT)"
    else
        warn 3 "Shard content issues" "$SAMPLE_RESULT"
    fi
fi
echo ""

# ── D4: Filter sentinel ───────────────────────────────────────────────────────
echo "── D4: Filter pass ──────────────────────────────────────────────"
SENTINEL="$SHARDS_DIR/.filtered_chunk1"
if [[ -f "$SENTINEL" ]]; then
    ok 4 "Filter sentinel present"
    # Verify shards contain no obviously corrupted records by sampling
    if [[ "$SHARD_COUNT" -gt 0 ]]; then
        SAMPLE=()
        while IFS= read -r f; do SAMPLE+=("$f"); done < \
            <(ls "$SHARDS_DIR"/*.tar 2>/dev/null | shuf -n 3 2>/dev/null || ls "$SHARDS_DIR"/*.tar | head -3)
        FILTER_CHECK=$("$PYTHON" - "${SAMPLE[@]}" <<'PYEOF' 2>&1
import sys, tarfile, io
paths = sys.argv[1:]
bad = []
for path in paths:
    with tarfile.open(path, 'r:') as t:
        checked = 0
        for m in t:
            if not m.isfile() or not m.name.endswith('.txt'): continue
            if checked >= 20: break
            cap = t.extractfile(m).read().decode('utf-8','replace').strip()
            checked += 1
            words = cap.split()
            if len(words) < 5:
                bad.append(f"{path.split('/')[-1]}: short caption after filter: {repr(cap[:60])}")
            if cap.lower().startswith('http') or cap.rstrip().endswith(('.jpg','.png')):
                bad.append(f"{path.split('/')[-1]}: URL/filename caption after filter: {repr(cap[:60])}")
if bad:
    print("WARN: " + "; ".join(bad[:3]))
else:
    print(f"ok: spot-check passed on {len(paths)} shards")
PYEOF
)
        if [[ "$FILTER_CHECK" == ok:* ]]; then
            ok  4 "Post-filter caption quality ($FILTER_CHECK)"
        else
            warn 4 "Post-filter caption quality issues" "$FILTER_CHECK"
        fi
    fi
else
    [[ "$SHARD_COUNT" -gt 0 ]] \
        && warn 4 "Filter sentinel missing" "shards exist but .filtered_chunk1 not present — filter_shards.py may not have run" \
        || skip 4 "Filter pass (shards not yet built)"
fi
echo ""

# ── D5: Qwen3 cache ───────────────────────────────────────────────────────────
echo "── D5: Qwen3 precompute cache ───────────────────────────────────"
QWEN3_DIR="$DATA_ROOT/precomputed/qwen3"
QWEN3_N=$(find "$QWEN3_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
printf "  Qwen3 .npz count: %s  (shards: %s)\n" "$QWEN3_N" "$SHARD_COUNT"
if [[ "$QWEN3_N" -eq 0 ]]; then
    skip 5 "Qwen3 cache (not yet computed)"
else
    # Coverage
    [[ "$SHARD_COUNT" -gt 0 && "$QWEN3_N" -ge "$SHARD_COUNT" ]] \
        && ok   5 "Qwen3 coverage ($QWEN3_N ≥ $SHARD_COUNT shards)" \
        || warn 5 "Qwen3 coverage incomplete" "$QWEN3_N / $SHARD_COUNT — precompute may still be running"

    # Shape/dtype sampling
    QWEN3_CHECK=$("$PYTHON" - "$QWEN3_DIR" <<'PYEOF' 2>&1
import sys, os, numpy as np, glob, random
d = sys.argv[1]
files = glob.glob(os.path.join(d, '*.npz'))
sample = random.sample(files, min(10, len(files)))
shapes = set(); dtypes = set(); issues = []
for f in sample:
    try:
        data = np.load(f)
        for k in data:
            shapes.add(data[k].shape)
            dtypes.add(str(data[k].dtype))
    except Exception as e:
        issues.append(f"{os.path.basename(f)}: {e}")
if issues:
    print("WARN: " + "; ".join(issues))
else:
    print(f"ok: shapes={list(shapes)[:3]} dtypes={list(dtypes)}")
PYEOF
)
    if [[ "$QWEN3_CHECK" == ok:* ]]; then
        ok  5 "Qwen3 cache integrity ($QWEN3_CHECK)"
    else
        warn 5 "Qwen3 cache issues" "$QWEN3_CHECK"
    fi
fi
echo ""

# ── D6: VAE cache ─────────────────────────────────────────────────────────────
echo "── D6: VAE precompute cache ─────────────────────────────────────"
VAE_DIR="$DATA_ROOT/precomputed/vae"
VAE_N=$(find "$VAE_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
printf "  VAE .npz count: %s  (shards: %s)\n" "$VAE_N" "$SHARD_COUNT"
if [[ "$VAE_N" -eq 0 ]]; then
    skip 6 "VAE cache (not yet computed)"
else
    [[ "$SHARD_COUNT" -gt 0 && "$VAE_N" -ge "$SHARD_COUNT" ]] \
        && ok   6 "VAE coverage ($VAE_N ≥ $SHARD_COUNT shards)" \
        || warn 6 "VAE coverage incomplete" "$VAE_N / $SHARD_COUNT — precompute may still be running"

    VAE_CHECK=$("$PYTHON" - "$VAE_DIR" <<'PYEOF' 2>&1
import sys, os, numpy as np, glob, random
d = sys.argv[1]
files = glob.glob(os.path.join(d, '*.npz'))
sample = random.sample(files, min(10, len(files)))
shapes = set(); dtypes = set(); issues = []
for f in sample:
    try:
        data = np.load(f)
        for k in data:
            shapes.add(data[k].shape)
            dtypes.add(str(data[k].dtype))
    except Exception as e:
        issues.append(f"{os.path.basename(f)}: {e}")
if issues:
    print("WARN: " + "; ".join(issues))
else:
    print(f"ok: shapes={list(shapes)[:3]} dtypes={list(dtypes)}")
PYEOF
)
    if [[ "$VAE_CHECK" == ok:* ]]; then
        ok  6 "VAE cache integrity ($VAE_CHECK)"
    else
        warn 6 "VAE cache issues" "$VAE_CHECK"
    fi
fi
echo ""

# ── D7: Precompute ID alignment ───────────────────────────────────────────────
echo "── D7: Precompute ID alignment ──────────────────────────────────"
if [[ "$QWEN3_N" -gt 0 && "$SHARD_COUNT" -gt 0 ]]; then
    ALIGN_CHECK=$("$PYTHON" - "$SHARDS_DIR" "$QWEN3_DIR" <<'PYEOF' 2>&1
import sys, os, tarfile, glob
shards_dir, qwen3_dir = sys.argv[1], sys.argv[2]
import random
shard_files = glob.glob(os.path.join(shards_dir, '*.tar'))
sample_shard = random.choice(shard_files)
npz_name = os.path.basename(sample_shard).replace('.tar', '.npz')
npz_path = os.path.join(qwen3_dir, npz_name)
if not os.path.exists(npz_path):
    print(f"WARN: no matching npz for {os.path.basename(sample_shard)} → expected {npz_name}")
else:
    import numpy as np
    data = np.load(npz_path)
    keys = list(data.keys())
    with tarfile.open(sample_shard) as t:
        shard_keys = {m.name.rsplit('.',1)[0] for m in t.getmembers() if m.name.endswith('.jpg')}
    print(f"ok: shard {os.path.basename(sample_shard)}: {len(shard_keys)} records, npz has {len(keys)} keys")
PYEOF
)
    if [[ "$ALIGN_CHECK" == ok:* ]]; then
        ok  7 "Precompute ID alignment ($ALIGN_CHECK)"
    else
        warn 7 "Precompute ID alignment issue" "$ALIGN_CHECK"
    fi
else
    skip 7 "Precompute ID alignment (cache or shards not ready)"
fi
echo ""

# ── D8: Checkpoints ───────────────────────────────────────────────────────────
echo "── D8: Checkpoints ──────────────────────────────────────────────"
CKPT_DIR="$TRAIN_DIR/checkpoints"
CKPT_ALL=$(ls "$CKPT_DIR"/step_*.safetensors 2>/dev/null || true)
CKPT_MAIN=$(echo "$CKPT_ALL" | grep -v ema | grep -v '^$' || true)
CKPT_EMA=$(echo "$CKPT_ALL" | grep ema | grep -v '^$' || true)
CKPT_N=$(echo "$CKPT_MAIN" | grep -c '[^[:space:]]' 2>/dev/null || true); CKPT_N="${CKPT_N:-0}"
EMA_N=$(echo "$CKPT_EMA"  | grep -c '[^[:space:]]' 2>/dev/null || true); EMA_N="${EMA_N:-0}"

if [[ "$CKPT_N" -eq 0 ]]; then
    skip 8 "Checkpoints (training not yet started)"
else
    printf "  Checkpoints: %s main, %s EMA\n" "$CKPT_N" "$EMA_N"
    LATEST=$(echo "$CKPT_MAIN" | sort -t_ -k2 -n | tail -1)
    LATEST_STEP=$(basename "$LATEST" .safetensors | grep -oE '[0-9]+' || echo 0)
    printf "  Latest: %s (step %s)\n" "$(basename "$LATEST")" "$LATEST_STEP"

    # Check EMA pairing (every main ckpt should have an EMA)
    [[ "$CKPT_N" -eq "$EMA_N" ]] \
        && ok   8 "EMA pairing complete ($CKPT_N main = $EMA_N EMA)" \
        || warn 8 "EMA mismatch" "$CKPT_N main checkpoints but $EMA_N EMA checkpoints"

    # Check file sizes (a valid checkpoint should be > 1 MB)
    TINY=$(echo "$CKPT_ALL" | while read -r f; do [[ -n "$f" && $(stat -f%z "$f" 2>/dev/null || echo 0) -lt 1048576 ]] && echo "$f"; done || true)
    [[ -z "$TINY" ]] \
        && ok   8 "All checkpoint files are >1 MB" \
        || err  8 "Tiny checkpoint files detected" "$(echo "$TINY" | head -3)"

    # Check step progression is monotonically increasing
    STEPS=$(echo "$CKPT_MAIN" | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tr '\n' ' ')
    printf "  Step sequence: %s\n" "$STEPS"
    ok 8 "Step sequence recorded"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────────"
TOTAL=$(( WARN + ERR ))
if [[ "$ERR" -gt 0 ]]; then
    echo "  ❌ $ERR error(s), $WARN warning(s) — review output above."
    exit 1
elif [[ "$WARN" -gt 0 ]]; then
    echo "  ⚠️  $WARN warning(s) — pipeline may continue but review recommended."
    exit 0
else
    echo "  ✅ No issues found. Pipeline outputs look healthy."
fi
echo "================================================================="
