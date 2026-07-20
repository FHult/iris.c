#!/bin/bash
#
# debug/metal_regression_suite.sh
#
# Basic Metal regression / parity / perf smoke test for the inference engine.
# Intended to be run before and after Metal kernel changes.
#
# Usage:
#   bash debug/metal_regression_suite.sh
#   IRIS_METAL_DEBUG=1 bash debug/metal_regression_suite.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== iris.c Metal Regression Suite ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# 1. Build the Metal backend (MPS)
echo "[1/5] Building MPS backend..."
make clean -s >/dev/null 2>&1 || true
make mps -j4 > /tmp/metal_build.log 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: MPS build failed. See /tmp/metal_build.log"
    exit 1
fi
echo "    Build successful."

# 2. Run the existing kernel correctness tests (they exercise Metal paths when available)
echo "[2/5] Running existing kernel tests (debug/test_kernels)..."
if [ -x ./debug/test_kernels ]; then
    ./debug/test_kernels 2>&1 | tail -20
else
    echo "    (debug/test_kernels not built — skipping)"
fi

# 3. Basic generation smoke on Metal (exercises transformer + attention)
echo "[3/5] Running generation smoke test (exercises Metal attention paths)..."
if [ -x ./iris ]; then
    ./iris -d flux-klein-4b \
           -p "a small test prompt for metal regression" \
           -W 256 -H 256 --steps 4 --seed 42 -o /tmp/metal_reg_test.png \
           2>&1 | grep -E "(Metal|attention|time|Done)" || true
else
    echo "    (iris binary not found — skipping generation smoke)"
fi

# 4. (Placeholder) Attention parity micro-benchmark
# When B-METAL-05 style test is implemented, it would be called here.
echo "[4/5] Attention parity micro-benchmark (placeholder)"
echo "    (Future: debug/metal_attention_parity_test would run here)"
echo "    Current recommendation: implement a small C test that compares"
echo "    iris_metal_attention_fused vs CPU reference on multiple (seq, heads) combos."

# 5. Memory / residency sanity (very rough)
echo "[5/5] Quick Metal memory sanity..."
if [ -x ./iris ]; then
    ./iris -d flux-klein-4b -p "memory smoke" --steps 2 -o /tmp/metal_mem_smoke.png 2>&1 | grep -i "metal\|memory" || true
fi

echo
echo "=== Metal Regression Suite Complete ==="
echo "Review output above for anomalies before committing Metal changes."
echo "For stricter testing, extend debug/metal_attention_parity_test.c (see plans/metal_optimization_backlog.md)"