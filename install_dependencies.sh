#!/bin/bash
set -e

echo "Installing Protein Benchmarker Dependencies"
echo "==========================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ Warning: No virtual environment detected. Consider activating one."
fi

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "✓ CUDA detected: $CUDA_VERSION"
    
    # Map CUDA version to PyTorch index
    case $CUDA_VERSION in
        "11.8")
            TORCH_INDEX="cu118"
            ;;
        "12.1")
            TORCH_INDEX="cu121"
            ;;
        "11.7")
            TORCH_INDEX="cu117"
            ;;
        *)
            echo "⚠ Warning: CUDA version $CUDA_VERSION may not be fully supported"
            TORCH_INDEX="cu118"  # Default fallback
            ;;
    esac
else
    echo "⚠ No CUDA detected, installing CPU version"
    TORCH_INDEX="cpu"
fi

echo "Using PyTorch index: $TORCH_INDEX"

# Step 1: Install PyTorch
echo ""
echo "Step 1: Installing PyTorch..."
if [[ "$TORCH_INDEX" == "cpu" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX
fi

# Step 1.5: Get PyTorch version for PyG dependencies
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d+ -f1)
echo "✓ PyTorch version: $TORCH_VERSION"

# Step 3: Install PyTorch Geometric and its dependencies
echo ""
echo "Step 2: Installing PyTorch Geometric and dependencies..."
pip install torch-geometric

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_INDEX}.html"
echo "Installing PyG extensions from: $PYG_URL"
pip install torch-scatter torch-sparse torch-cluster --find-links $PYG_URL

# Step 4: Install the package and all other dependencies
echo ""
echo "Step 3: Installing protein-benchmarker package and remaining dependencies..."
pip install -e .

# Verification
echo ""
echo "Step 4: Verifying installation..."
python -c "
import torch
import torch_geometric
import torch_cluster
print('✓ PyTorch:', torch.__version__)
print('✓ PyTorch Geometric:', torch_geometric.__version__)
print('✓ torch-cluster: Available')
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA devices:', torch.cuda.device_count())
"

echo ""
echo "✓ Installation completed successfully!"
echo ""