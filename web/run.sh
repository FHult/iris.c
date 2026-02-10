#!/bin/bash
# Start the FLUX.2 Web UI server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check for virtual environment
if [ ! -d "web/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv web/venv
    source web/venv/bin/activate
    pip install -r web/requirements.txt
else
    source web/venv/bin/activate
fi

# Check for flux binary
if [ ! -f "./flux" ]; then
    echo "Error: flux binary not found. Build it first:"
    echo "  make mps    # Apple Silicon (fastest)"
    echo "  make blas   # CPU with BLAS acceleration"
    echo "  make generic  # Pure C fallback"
    exit 1
fi

# Auto-detect model directory
MODEL_DIR=""
for dir in flux-klein-4b flux-klein-model flux-klein-9b flux-klein-4b-base flux-klein-9b-base; do
    if [ -d "./$dir" ]; then
        MODEL_DIR="./$dir"
        break
    fi
done

if [ -z "$MODEL_DIR" ]; then
    echo "Error: No model found. Download one first:"
    echo "  ./download_model.sh 4b"
    exit 1
fi

echo ""
python web/server.py --model-dir "$MODEL_DIR" "$@"
