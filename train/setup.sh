#!/bin/bash
# train/setup.sh — Create Python venv and install all training dependencies.
# Run once from the iris.c repo root before any other train/ scripts.
#
# Usage:
#   cd /path/to/iris.c
#   bash train/setup.sh

set -e

VENV_DIR="train/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
    echo "To recreate: rm -rf $VENV_DIR && bash train/setup.sh"
else
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing packages ..."
pip install --upgrade pip

# MLX + training utilities
pip install mlx mlx-lm

# Flux Klein MLX inference (mflux provides the frozen Flux forward pass)
pip install mflux

# Data / I/O (huggingface_hub[cli] installs the 'hf' CLI binary)
pip install safetensors webdataset "huggingface_hub[cli]" datasets pyarrow pandas

# Image processing (turbojpeg is 2-4x faster than Pillow for JPEG decode)
# PyTurboJPEG requires libjpeg-turbo system library: brew install libjpeg-turbo
pip install Pillow PyTurboJPEG

# Dataset tools
pip install img2dataset clip-retrieval faiss-cpu

# Training utilities
pip install tqdm wandb

echo ""
echo "Setup complete. Activate with:"
echo "  source train/.venv/bin/activate"
