#!/bin/bash
# Start the FLUX.2 Web UI server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check for virtual environment
if [ ! -d "web/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv web/venv
    source web/venv/bin/activate
    pip install flask
else
    source web/venv/bin/activate
fi

# Check for flux binary
if [ ! -f "./flux" ]; then
    echo "Error: flux binary not found. Build it first with: make mps (or make blas)"
    exit 1
fi

# Check for model
if [ ! -d "./flux-klein-model" ]; then
    echo "Error: Model not found. Download it first with: ./download_model.sh"
    exit 1
fi

echo ""
python web/server.py "$@"
