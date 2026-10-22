#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== [1/4] Creating project directories ==="
mkdir -p weights outputs

echo "=== [2/4] Setting up Python virtual environment (.venv) ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== [3/4] Installing PyTorch with CUDA support ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision

echo "=== [4/4] Installing requirements ==="
pip install -r requirements.txt

echo "=== Setup Complete! Activate environment with: source $SCRIPT_DIR/.venv/bin/activate ==="
