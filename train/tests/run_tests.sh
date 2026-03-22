#!/usr/bin/env bash
# train/tests/run_tests.sh — Run the training pipeline test suite.
#
# Usage:
#   bash train/tests/run_tests.sh            # all tests
#   bash train/tests/run_tests.sh smoke      # smoke tests only (fast)
#   bash train/tests/run_tests.sh loss       # loss function tests
#   bash train/tests/run_tests.sh dataset    # dataset utility tests
#   bash train/tests/run_tests.sh ema        # EMA tests
#   bash train/tests/run_tests.sh model      # model shape tests (slower)
#   bash train/tests/run_tests.sh scripts    # data script tests
#
# Requires: source train/.venv/bin/activate

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_DIR/train/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: train venv not found at $VENV"
    echo "Run: bash train/setup.sh"
    exit 1
fi

PYTHON="$VENV/bin/python"
PYTEST="$VENV/bin/pytest"

if [[ ! -x "$PYTEST" ]]; then
    echo "Installing pytest into venv..."
    "$PYTHON" -m pip install pytest -q
fi

cd "$REPO_DIR/train"

# Resolve test file from optional argument
case "${1:-all}" in
    smoke)   TARGET="tests/test_smoke.py" ;;
    loss)    TARGET="tests/test_loss.py" ;;
    dataset) TARGET="tests/test_dataset.py" ;;
    ema)     TARGET="tests/test_ema.py" ;;
    model)   TARGET="tests/test_model.py" ;;
    scripts) TARGET="tests/test_scripts.py" ;;
    all)     TARGET="tests/" ;;
    *)
        echo "Unknown target: $1"
        echo "Valid: smoke loss dataset ema model scripts all"
        exit 1
        ;;
esac

echo "========================================"
echo "  Training pipeline tests: ${1:-all}"
echo "========================================"

"$PYTEST" "$TARGET" -v --tb=short 2>&1
