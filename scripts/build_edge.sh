#!/bin/bash
# =============================================================================
# Run the Edge Detection System on Jetson Orin / Raspberry Pi 4
#
# This script ONLY runs detection. It does NOT install dependencies.
# For first-time setup, run: ./scripts/setup_edge.sh
# For Jetson-specific setup:  ./scripts/setup_jetson.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check if setup has been run
VENV_DIR="$PROJECT_ROOT/edge/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo ""
    echo "Please run first-time setup:"
    echo -e "  ${YELLOW}./scripts/setup_jetson.sh${NC}  (for Jetson Orin)"
    echo -e "  ${YELLOW}./scripts/setup_edge.sh${NC}    (for Raspberry Pi or generic)"
    echo ""
    exit 1
fi

# Detect platform
detect_platform() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
        if echo "$MODEL" | grep -qi "nvidia\|jetson"; then
            echo "jetson"
            return
        elif echo "$MODEL" | grep -qi "raspberry pi"; then
            echo "rpi"
            return
        fi
    fi
    echo "generic"
}

PLATFORM=$(detect_platform)

# Activate virtual environment
if [ "$PLATFORM" = "jetson" ] && [ -f "$VENV_DIR/bin/activate_jetson.sh" ]; then
    source "$VENV_DIR/bin/activate_jetson.sh"
else
    source "$VENV_DIR/bin/activate"
fi

# Set up library paths for Jetson (cuSPARSELt) - fallback if not in activate script
if [ "$PLATFORM" = "jetson" ]; then
    CUSPARSELT_LIB="$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib"
    if [ -d "$CUSPARSELT_LIB" ] && [[ ":$LD_LIBRARY_PATH:" != *":$CUSPARSELT_LIB:"* ]]; then
        export LD_LIBRARY_PATH="$CUSPARSELT_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Parse arguments
ACTION=${1:-run}
CONFIG=${2:-"edge/config/config.yaml"}

case $ACTION in
    run|start)
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Edge Detection System${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "${CYAN}Platform: ${PLATFORM}${NC}"
        echo ""

        # Quick CUDA check for Jetson
        if [ "$PLATFORM" = "jetson" ]; then
            CUDA_OK=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "error")
            if [ "$CUDA_OK" = "yes" ]; then
                echo -e "${GREEN}✓ CUDA available${NC}"
            else
                echo -e "${RED}✗ CUDA not available!${NC}"
                echo -e "${YELLOW}Run ./scripts/setup_jetson.sh to fix${NC}"
                exit 1
            fi
        fi

        # Check if config exists
        if [ ! -f "$CONFIG" ]; then
            echo -e "${YELLOW}Config not found at $CONFIG, using defaults${NC}"
            CONFIG=""
        fi

        echo ""
        echo -e "${GREEN}Starting detection...${NC}"
        echo ""

        # Run the edge detection system
        if [ -n "$CONFIG" ]; then
            python3 -m edge.src.main --config "$CONFIG" "${@:3}"
        else
            python3 -m edge.src.main "${@:2}"
        fi
        ;;

    test-camera)
        echo -e "${CYAN}Testing camera...${NC}"
        python3 -c "
from edge.src.camera import Camera, CameraConfig, list_available_cameras
import time

print('Available cameras:', list_available_cameras())

config = CameraConfig(device='/dev/video0', width=640, height=480, fps=5)
with Camera(config) as cam:
    for i in range(5):
        ret, frame = cam.read_frame()
        if ret:
            print(f'Frame {i+1}: {frame.shape}')
        else:
            print(f'Frame {i+1}: Failed')
        time.sleep(0.5)
print('Camera test complete!')
"
        ;;

    test-model)
        echo -e "${CYAN}Testing model loading...${NC}"
        python3 -c "
from edge.src.detector import Florence2Detector, DetectorConfig
from edge.src.platform import detect_platform, get_device

print(f'Platform: {detect_platform()}')
print(f'Device: {get_device()}')

config = DetectorConfig(device='auto')
detector = Florence2Detector(config)

print('Loading model...')
if detector.load():
    print('Model loaded successfully!')
    print(f'Running on: {detector.device}')
    detector.unload()
else:
    print('Failed to load model')
"
        ;;

    check)
        echo -e "${CYAN}Checking system status...${NC}"
        echo ""
        echo -e "${CYAN}Platform:${NC} $PLATFORM"
        echo -e "${CYAN}Python:${NC} $(python3 --version)"

        python3 << 'EOF'
import sys

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

print("\nDependencies:")
check("PyTorch", "torch")
check("Transformers", "transformers")
check("OpenCV", "cv2")
check("PIL", "PIL")

import torch
print(f"\nCUDA:")
if torch.cuda.is_available():
    print(f"  ✓ Available: {torch.cuda.get_device_name(0)}")
else:
    print("  ✗ Not available")

print("\nDetector module:")
try:
    from edge.src.detector import Florence2Detector
    print("  ✓ Imports successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
EOF
        ;;

    *)
        echo "Usage: $0 {run|test-camera|test-model|check}"
        echo ""
        echo "Commands:"
        echo "  run           - Run the edge detection system"
        echo "  test-camera   - Test camera capture"
        echo "  test-model    - Test model loading"
        echo "  check         - Check system status and dependencies"
        echo ""
        echo "First-time setup:"
        echo "  ./scripts/setup_jetson.sh  (Jetson Orin)"
        echo "  ./scripts/setup_edge.sh    (Raspberry Pi / generic)"
        exit 1
        ;;
esac
