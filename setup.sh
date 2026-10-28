#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " SubClean: Deep Generative Video Subtitle Removal - Setup Script"
echo "====================================================================="
echo ""

# Check python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install PyTorch with CUDA or CPU
echo "[INFO] Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision

# Install dependencies
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# Download model weights
echo "[INFO] Downloading pretrained weights..."
python scripts/download_weights.py

echo ""
echo "====================================================================="
echo " Setup Completed Successfully!"
echo " To start the Web UI:"
echo "     source venv/bin/activate"
echo "     python app.py"
echo ""
echo " To process via CLI:"
echo "     python run.py --input input.mp4 --output cleaned.mp4"
echo "====================================================================="
