#!/bin/bash
# nightly_health.sh — scheduled GPU-free health run (early-warning harness).
#
# Why: make test only runs when someone remembers — the QWEN-1 regression
# shipped because the mps build was broken for three weeks unnoticed. This run
# executes everything that needs NO GPU and NO model weights, every night,
# and writes a machine-readable verdict that pipeline_doctor surfaces:
#   1. pytest train/tests  (CPU-only)
#   2. make mps            (full compile of the production target — build check)
#   3. make test-unit      (C unit suites: LoRA/kernels/VAE/safetensors/IP/CSDModulation/JPEG/PNG)
#   4. pytest web/tests    (Flask API suite, incl. SREF routing — CPU-only, no GPU/model)
#
# Output: $DATA_ROOT/health/nightly_health.json  (doctor: _check_nightly_health)
# Log:    $DATA_ROOT/logs/nightly_health.log
#
# Installed via launchd (train/launchd/com.iris.nightly-health.plist, 05:30).
# Safe to run manually any time:  ./train/scripts/nightly_health.sh

set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${PIPELINE_DATA_ROOT:-/Volumes/2TBSSD}"
# Results go on the BOOT VOLUME, not /Volumes: launchd agents cannot write an
# external mount without Full Disk Access (Operation-not-permitted), so a
# /Volumes result path silently fails every 05:30 run. The doctor reads here
# first (HEALTH_DIR), /Volumes second.
OUT_DIR="${IRIS_HEALTH_DIR:-$HOME/Library/iris-health}"
LOG="$OUT_DIR/nightly_health.log"
mkdir -p "$OUT_DIR"

cd "$REPO" || exit 1
T0=$(date +%s)
{
  echo "===== nightly_health $(date '+%Y-%m-%d %H:%M:%S') ====="
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >> "$LOG"

# 1. Python test suite (CPU-only)
PYTEST_OUT=$(train/.venv/bin/python -m pytest train/tests/ -q --tb=line 2>&1)
PYTEST_EXIT=$?
echo "$PYTEST_OUT" | tail -30 >> "$LOG"
PY_PASSED=$(echo "$PYTEST_OUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
PY_FAILED=$(echo "$PYTEST_OUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
PY_PASSED=${PY_PASSED:-0}; PY_FAILED=${PY_FAILED:-0}

# 2. Production-target build check (compile only; no GPU execution)
make mps >> "$LOG" 2>&1
BUILD_EXIT=$?

# 3. C unit suites (CPU; no model weights)
make test-unit >> "$LOG" 2>&1
UNIT_EXIT=$?

# 4. Web API suite (Flask test client; CPU-only, no GPU/model) — catches web-surface rot
#    that steps 1-3 miss (e.g. SREF routing / style-code contracts).
WEB_OUT=$(web/venv/bin/python3 -m pytest web/tests/ -q --tb=line 2>&1)
WEB_EXIT=$?
echo "$WEB_OUT" | tail -15 >> "$LOG"
WEB_PASSED=$(echo "$WEB_OUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
WEB_FAILED=$(echo "$WEB_OUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
WEB_PASSED=${WEB_PASSED:-0}; WEB_FAILED=${WEB_FAILED:-0}

DUR=$(( $(date +%s) - T0 ))
STATUS=ok
[ "$PY_FAILED" -gt 0 ] || [ "$PYTEST_EXIT" -ne 0 ] && STATUS=pytest_failed
[ "$UNIT_EXIT" -ne 0 ] && STATUS=unit_failed
[ "$WEB_FAILED" -gt 0 ] || [ "$WEB_EXIT" -ne 0 ] && STATUS=web_failed
[ "$BUILD_EXIT" -ne 0 ] && STATUS=build_failed

TMP="$OUT_DIR/nightly_health.json.tmp"
cat > "$TMP" <<EOF
{
  "ts": "$(date '+%Y-%m-%dT%H:%M:%S')",
  "git_sha": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "status": "$STATUS",
  "pytest_passed": $PY_PASSED,
  "pytest_failed": $PY_FAILED,
  "pytest_exit": $PYTEST_EXIT,
  "build_exit": $BUILD_EXIT,
  "unit_exit": $UNIT_EXIT,
  "web_passed": $WEB_PASSED,
  "web_failed": $WEB_FAILED,
  "web_exit": $WEB_EXIT,
  "duration_s": $DUR,
  "log": "$LOG"
}
EOF
mv "$TMP" "$OUT_DIR/nightly_health.json"
echo "status=$STATUS pytest=$PY_PASSED/$((PY_PASSED+PY_FAILED)) build=$BUILD_EXIT unit=$UNIT_EXIT web=$WEB_PASSED/$((WEB_PASSED+WEB_FAILED)) (${DUR}s)" >> "$LOG"
[ "$STATUS" = ok ]
