#!/bin/bash
# =============================================================================
# Jetson Orin Complete Setup Script
# One-command setup for Florence-2 Waste Bin Detection System
#
# This script handles ALL the following issues automatically:
# 1. PyTorch CUDA - Installs from Jetson AI Lab PyPI
# 2. flash_attn mock - Created automatically by detector module
# 3. cuSPARSELt library - Sets up LD_LIBRARY_PATH
# 4. All Python dependencies
# 5. Florence-2 model download
#
# Usage: ./scripts/setup_jetson.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "\n${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

print_header "Jetson Orin Florence-2 Setup"
echo ""
echo "This script will:"
echo "  1. Install system dependencies"
echo "  2. Create Python virtual environment"
echo "  3. Install PyTorch with CUDA from Jetson AI Lab"
echo "  4. Install all Python packages"
echo "  5. Set up library paths"
echo "  6. Download Florence-2 model (~500MB)"
echo "  7. Verify installation"
echo ""

# Check if running on Jetson
print_step "Detecting platform..."
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
    if echo "$MODEL" | grep -qi "jetson\|nvidia"; then
        print_success "Detected: $MODEL"
    else
        print_warning "This doesn't appear to be a Jetson device"
        print_warning "Model: $MODEL"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    print_warning "Could not detect device model"
fi

# Check Python
print_step "Checking Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    echo "Install with: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION"

# Install system dependencies
print_step "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    v4l-utils \
    libhdf5-dev \
    2>/dev/null || print_warning "Some packages may not be available"

# Create virtual environment
VENV_DIR="$PROJECT_ROOT/edge/venv"
print_step "Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Existing venv found. Removing for clean install..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
print_success "Created virtual environment"

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip wheel setuptools --quiet
print_success "pip upgraded"

# Install PyTorch from Jetson AI Lab
print_step "Installing PyTorch with CUDA from Jetson AI Lab..."
JETSON_PYPI="https://pypi.jetson-ai-lab.io/jp6/cu126"

pip install torch==2.8.0 torchvision==0.23.0 --index-url="$JETSON_PYPI" 2>&1 | while read line; do
    if [[ $line == *"Downloading"* ]] || [[ $line == *"Installing"* ]]; then
        echo -e "  ${CYAN}$line${NC}"
    fi
done

# Verify CUDA
print_step "Verifying CUDA availability..."
CUDA_CHECK=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA FAIL')
" 2>&1)

if [[ $CUDA_CHECK == *"CUDA OK"* ]]; then
    print_success "$CUDA_CHECK"
else
    print_error "CUDA not available!"
    print_error "PyTorch may not have installed correctly"
    echo ""
    echo "Try manually:"
    echo "  pip uninstall torch torchvision"
    echo "  pip install torch==2.8.0 torchvision==0.23.0 --index-url=$JETSON_PYPI"
    exit 1
fi

# Install other dependencies
print_step "Installing Python dependencies..."
pip install -r "$PROJECT_ROOT/edge/requirements/base.txt" --quiet

# Re-verify CUDA (some packages can break it)
CUDA_RECHECK=$(python3 -c "import torch; print('ok' if torch.cuda.is_available() else 'fail')" 2>/dev/null || echo "error")
if [ "$CUDA_RECHECK" != "ok" ]; then
    print_warning "CUDA broken by dependencies. Reinstalling PyTorch..."
    pip uninstall -y torch torchvision 2>/dev/null || true
    pip install torch==2.8.0 torchvision==0.23.0 --index-url="$JETSON_PYPI" --quiet
fi

# Set up cuSPARSELt library path
print_step "Setting up library paths..."
CUSPARSELT_LIB="$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib"

if [ -d "$CUSPARSELT_LIB" ]; then
    # Create activation script with library path
    cat > "$VENV_DIR/bin/activate_jetson.sh" << EOF
#!/bin/bash
# Jetson-specific activation with cuSPARSELt support
source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH="$CUSPARSELT_LIB:\$LD_LIBRARY_PATH"
echo "Jetson environment activated with cuSPARSELt support"
EOF
    chmod +x "$VENV_DIR/bin/activate_jetson.sh"
    print_success "Created activate_jetson.sh with LD_LIBRARY_PATH"

    # Add to venv's activate script
    if ! grep -q "cusparselt" "$VENV_DIR/bin/activate"; then
        echo "" >> "$VENV_DIR/bin/activate"
        echo "# Jetson cuSPARSELt support" >> "$VENV_DIR/bin/activate"
        echo "export LD_LIBRARY_PATH=\"$CUSPARSELT_LIB:\$LD_LIBRARY_PATH\"" >> "$VENV_DIR/bin/activate"
    fi
else
    print_warning "cuSPARSELt library not found (may not be needed)"
fi

# Download Florence-2 model
print_step "Downloading Florence-2 model..."
echo "This may take a few minutes..."

export PROJECT_ROOT
python3 << 'PYTHON_EOF'
import os
import sys
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

# Setup flash_attn mock
from edge.src.detector import _setup_flash_attn_mock
_setup_flash_attn_mock()

print("  Downloading processor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)
print("  ✓ Processor downloaded")

print("  Downloading model (this takes a while)...")
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print("  ✓ Model downloaded")

param_count = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {param_count:,}")
PYTHON_EOF

if [ $? -eq 0 ]; then
    print_success "Florence-2 model downloaded"
else
    print_warning "Model download failed (will retry on first run)"
fi

# Final verification
print_step "Running final verification..."
python3 << 'PYTHON_EOF'
import sys
import os
sys.path.insert(0, '.')

def check(name, import_name=None):
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'OK')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: FAILED ({e})")
        return False

print("\nDependency check:")
ok = True
ok &= check("PyTorch", "torch")
ok &= check("Transformers", "transformers")
ok &= check("OpenCV", "cv2")
ok &= check("PIL", "PIL")
ok &= check("NumPy", "numpy")
ok &= check("Requests", "requests")
ok &= check("Accelerate", "accelerate")

import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  ✗ CUDA: NOT AVAILABLE")
    ok = False

# Test detector import
print("\nDetector module:")
try:
    from edge.src.detector import Florence2Detector, DetectorConfig
    print("  ✓ Detector imports successfully")
except Exception as e:
    print(f"  ✗ Detector failed: {e}")
    ok = False

if ok:
    print("\n✓ All checks passed!")
    sys.exit(0)
else:
    print("\n⚠ Some checks failed")
    sys.exit(1)
PYTHON_EOF

VERIFY_STATUS=$?

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_LAPTOP_IP")

# Print summary
print_header "Setup Complete!"
echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    print_success "All components installed successfully"
else
    print_warning "Some components may need attention"
fi

echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo ""
echo "1. Configure the dashboard server IP:"
echo -e "   ${CYAN}nano $PROJECT_ROOT/edge/config/config.yaml${NC}"
echo "   Change: url: \"http://YOUR_LAPTOP_IP:5000\""
echo ""
echo "2. On your laptop, start the dashboard:"
echo -e "   ${CYAN}./scripts/build_dashboard.sh${NC}"
echo ""
echo "3. Run the detection system:"
echo -e "   ${CYAN}./scripts/build_edge.sh run${NC}"
echo ""
echo -e "This device IP: ${CYAN}${LOCAL_IP}${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
