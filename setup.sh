#!/bin/bash

set -e

echo "🚀 Setting up VFGNN environment..."

# -------------------------
# Check Python
# -------------------------
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 not found"
    exit
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "📌 Found Python $PYTHON_VERSION"

# -------------------------
# Create venv
# -------------------------
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "⚠️  venv already exists"
fi

# -------------------------
# Activate venv
# -------------------------
source venv/bin/activate

echo " Virtual environment activated"

# -------------------------
# Upgrade pip
# -------------------------
pip install --upgrade pip setuptools wheel

# -------------------------
# Install PyTorch (CPU by default)
# -------------------------
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# -------------------------
# Install PyG (Correct Wheels)
# -------------------------
# echo "Installing PyTorch Geometric..."
# pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# -------------------------
# Install project requirements
# -------------------------
echo "Installing project dependencies..."
pip install --no-cache-dir -r requirements.txt

# -------------------------
# Create report directory
# -------------------------
mkdir -p report

echo " Setup complete!"
echo "  To activate later: source venv/bin/activate"
echo " To run project: python src/main.py"
