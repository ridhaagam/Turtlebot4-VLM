#!/bin/bash
# =============================================================================
# Complete Setup Script for Edge Detection System
# Works on: NVIDIA Jetson Orin, Raspberry Pi 4, and generic Linux
# Downloads all dependencies and optionally pre-downloads the model
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Edge Detection System - Full Setup${NC}"
echo -e "${GREEN}========================================${NC}"

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Detect platform
detect_platform() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
        if echo "$MODEL" | grep -qi "nvidia\|jetson\|orin"; then
            echo "jetson"
            return
        elif echo "$MODEL" | grep -qi "raspberry pi"; then
            echo "rpi"
            return
        fi
    fi
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "jetson"
        return
    fi
    echo "generic"
}

PLATFORM=$(detect_platform)
echo -e "${CYAN}Detected platform: ${PLATFORM}${NC}"

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${CYAN}Python version: ${PYTHON_VERSION}${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Install with: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq

    # Common packages
    PACKAGES="python3-pip python3-dev python3-venv libopencv-dev v4l-utils"

    # Platform-specific packages
    case $PLATFORM in
        rpi)
            PACKAGES="$PACKAGES libatlas-base-dev libhdf5-dev libharfbuzz-dev libwebp-dev"
            PACKAGES="$PACKAGES libtiff5-dev libjasper-dev libilmbase-dev libopenexr-dev"
            PACKAGES="$PACKAGES libgstreamer1.0-dev libavcodec-dev libavformat-dev libswscale-dev"
            ;;
        jetson)
            PACKAGES="$PACKAGES libhdf5-dev"
            ;;
    esac

    sudo apt-get install -y $PACKAGES 2>/dev/null || {
        echo -e "${YELLOW}Some packages may not be available, continuing...${NC}"
    }
fi

# Remove old venv if exists and is broken
VENV_DIR="$PROJECT_ROOT/edge/venv"
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${YELLOW}Removing broken virtual environment...${NC}"
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR" || {
        echo -e "${RED}Failed to create venv. Installing python3-venv...${NC}"
        sudo apt-get install -y python3-venv python3.${PYTHON_VERSION#*.}-venv 2>/dev/null || true
        python3 -m venv "$VENV_DIR"
    }
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools --quiet

# Install PyTorch based on platform
echo -e "${YELLOW}Installing PyTorch...${NC}"
case $PLATFORM in
    jetson)
        echo -e "${CYAN}Installing PyTorch for Jetson Orin...${NC}"

        # Check if PyTorch is already installed with CUDA
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo -e "${GREEN}PyTorch with CUDA already installed${NC}"
        else
            # Try NVIDIA's JetPack 6.x wheel first
            echo "Trying NVIDIA JetPack 6.x PyTorch..."
            pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60 2>/dev/null || {
                # Try JetPack 5.x
                echo "Trying NVIDIA JetPack 5.x PyTorch..."
                pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v51 2>/dev/null || {
                    # Fallback to standard PyTorch
                    echo -e "${YELLOW}NVIDIA wheels not available, using standard PyTorch...${NC}"
                    pip install torch torchvision
                }
            }
        fi
        ;;
    rpi)
        echo -e "${CYAN}Installing PyTorch for Raspberry Pi 4 (CPU only)...${NC}"
        # Use PyTorch CPU wheel optimized for ARM
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || {
            # Fallback: install from source-compatible wheel
            echo -e "${YELLOW}Trying alternative PyTorch installation...${NC}"
            pip install torch torchvision
        }
        ;;
    *)
        echo -e "${CYAN}Installing standard PyTorch...${NC}"
        pip install torch torchvision
        ;;
esac

# Install base requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r "$PROJECT_ROOT/edge/requirements/base.txt" --quiet

# Verify PyTorch installation
echo -e "${YELLOW}Verifying PyTorch...${NC}"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# Pre-download the model (optional but recommended)
echo ""
echo -e "${YELLOW}Do you want to pre-download the Florence-2 model? (~500MB)${NC}"
echo "This is recommended to avoid download during first run."
read -p "Download now? (Y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo -e "${GREEN}Downloading Florence-2 model...${NC}"
    python3 << 'PYTHON_EOF'
import os
import sys

# Add project to path
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

print("Setting up flash_attn mock...")
# Setup flash_attn mock before importing transformers
from edge.src.detector import _setup_flash_attn_mock
_setup_flash_attn_mock()

print("Downloading Florence-2-base model...")
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_name = "microsoft/Florence-2-base"

print("Downloading processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print("✓ Processor downloaded")

print("Downloading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32
)
print("✓ Model downloaded")

print()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model cached at: ~/.cache/huggingface/hub/")
PYTHON_EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model downloaded successfully!${NC}"
    else
        echo -e "${RED}Model download failed. It will be downloaded on first run.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping model download. It will be downloaded on first run.${NC}"
fi

# Verify complete installation
echo ""
echo -e "${GREEN}Verifying installation...${NC}"
python3 << 'PYTHON_EOF'
import sys
import os

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

def check(name, import_name=None):
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'OK')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: NOT INSTALLED ({e})")
        return False

print("\nChecking dependencies:")
all_ok = True
all_ok &= check("PyTorch", "torch")
all_ok &= check("Transformers", "transformers")
all_ok &= check("OpenCV", "cv2")
all_ok &= check("PIL", "PIL")
all_ok &= check("NumPy", "numpy")
all_ok &= check("Requests", "requests")
all_ok &= check("PyYAML", "yaml")
all_ok &= check("Accelerate", "accelerate")
all_ok &= check("Timm", "timm")
all_ok &= check("Einops", "einops")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})")
else:
    print(f"  - CUDA: Not available (using CPU)")

# Test detector module
print("\nTesting detector module:")
try:
    from edge.src.detector import Florence2Detector, DetectorConfig
    print("  ✓ Detector module imports successfully")
except Exception as e:
    print(f"  ✗ Detector module failed: {e}")
    all_ok = False

if all_ok:
    print("\n✓ All dependencies installed successfully!")
else:
    print("\n⚠ Some components may have issues")
    sys.exit(1)
PYTHON_EOF

# Get local IP for config hint
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_LAPTOP_IP")

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit the config file to set your laptop's IP:"
echo -e "   ${CYAN}nano $PROJECT_ROOT/edge/config/config.yaml${NC}"
echo "   Change: url: \"http://YOUR_LAPTOP_IP:5000\""
echo ""
echo "2. On your laptop, start the dashboard:"
echo -e "   ${CYAN}./scripts/build_dashboard.sh${NC}"
echo ""
echo "3. Connect USB camera and run detection:"
echo -e "   ${CYAN}./scripts/build_edge.sh run${NC}"
echo ""
echo -e "Current device IP: ${CYAN}${LOCAL_IP}${NC}"
echo ""
