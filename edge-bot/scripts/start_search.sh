#!/bin/bash
# start_search.sh - Main script to run TurtleBot4 Trash Bot mission
#
# This script handles the complete workflow:
#   1. Verifies connection to remote vision server (192.168.0.81:5000)
#   2. Launches all trash_bot ROS2 nodes
#   3. Triggers /find_bin service (which auto-undocks then searches)
#   4. Robot searches for bins, approaches, classifies, and returns to dock
#
# All vision inference runs on the remote server - NO local llama.cpp needed!
#
# Usage:
#   ./start_search.sh          # Start mission
#
# To stop: ./stop.sh
# To dock manually: ./dock.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Server configuration
VISION_SERVER="http://192.168.0.81:5000"

cd /home/g3ubuntu/ROS

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source /home/g3ubuntu/ROS/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  TurtleBot4 Trash Bot Mission${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo -e "${YELLOW}Mission flow:${NC}"
echo "  1. Connect to vision server (Florence-2 + SmolVLM2)"
echo "  2. Launch ROS2 nodes"
echo "  3. Auto-undock from charging station"
echo "  4. Search for trash bins using vision"
echo "  5. Approach and classify bin"
echo "  6. Send results to dashboard"
echo "  7. Return to dock"
echo ""

# Kill any existing processes
echo -e "${YELLOW}[1/4] Stopping existing processes...${NC}"
pkill -9 -f "trash_bot" 2>/dev/null || true
pkill -9 -f "navigation_vlm" 2>/dev/null || true
pkill -9 -f "classifier_node" 2>/dev/null || true
pkill -9 -f "camera_publisher" 2>/dev/null || true
pkill -9 -f "motion_subscriber" 2>/dev/null || true
pkill -9 -f "command_service" 2>/dev/null || true
sleep 2

# Check remote vision server
echo -e "${YELLOW}[2/4] Checking vision server at ${VISION_SERVER}...${NC}"
if curl -s --connect-timeout 5 "${VISION_SERVER}/api/vision/status" > /dev/null 2>&1; then
    echo -e "${GREEN}      Vision server is reachable${NC}"
    # Check if model is loaded
    STATUS=$(curl -s "${VISION_SERVER}/api/vision/status" 2>/dev/null)
    if echo "$STATUS" | grep -q '"loaded":true\|"loaded": true'; then
        echo -e "${GREEN}      Vision model is loaded and ready${NC}"
    else
        echo -e "${YELLOW}      Vision model may not be loaded - server will load on first request${NC}"
    fi
else
    echo -e "${RED}WARNING: Cannot reach vision server at ${VISION_SERVER}${NC}"
    echo -e "${RED}         Navigation will fail without the vision server!${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify camera symlinks
echo -e "${YELLOW}      Checking camera symlinks...${NC}"
if [ -L /dev/nav_camera ] && [ -L /dev/class_camera ]; then
    NAV_CAM=$(readlink -f /dev/nav_camera)
    CLASS_CAM=$(readlink -f /dev/class_camera)
    echo -e "${GREEN}      nav_camera -> ${NAV_CAM}${NC}"
    echo -e "${GREEN}      class_camera -> ${CLASS_CAM}${NC}"
else
    echo -e "${RED}WARNING: Camera symlinks not set up!${NC}"
    echo "      Run: sudo ln -sf /dev/video2 /dev/nav_camera"
    echo "           sudo ln -sf /dev/video0 /dev/class_camera"
fi

# Launch trash_bot nodes
echo ""
echo -e "${YELLOW}[3/4] Launching trash_bot ROS2 nodes...${NC}"
ros2 launch trash_bot trash_bot_launch.py > /tmp/trash_bot_launch.log 2>&1 &
LAUNCH_PID=$!
echo "      Launch PID: $LAUNCH_PID"
echo "      Log: /tmp/trash_bot_launch.log"

# Wait for nodes to initialize
echo "      Waiting for nodes to start (20s)..."
sleep 20

# Verify critical nodes
NODES_OK=true
if ros2 node list 2>/dev/null | grep -q "navigation_vlm_node"; then
    echo -e "${GREEN}      navigation_vlm_node: OK${NC}"
else
    echo -e "${RED}      navigation_vlm_node: FAILED${NC}"
    NODES_OK=false
fi

if ros2 node list 2>/dev/null | grep -q "classifier_node"; then
    echo -e "${GREEN}      classifier_node: OK${NC}"
else
    echo -e "${RED}      classifier_node: FAILED${NC}"
    NODES_OK=false
fi

if ros2 node list 2>/dev/null | grep -q "classification_camera_publisher"; then
    echo -e "${GREEN}      classification_camera_publisher: OK${NC}"
else
    echo -e "${YELLOW}      classification_camera_publisher: not running (optional)${NC}"
fi

if [ "$NODES_OK" = false ]; then
    echo -e "${RED}ERROR: Critical nodes failed to start${NC}"
    echo "Check logs: tail -100 /tmp/trash_bot_launch.log"
    exit 1
fi

# Start the mission (undock + search)
echo ""
echo -e "${YELLOW}[4/4] Starting mission (undock -> search -> classify -> dock)...${NC}"
echo ""

ros2 service call /find_bin std_srvs/srv/SetBool "{data: true}"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Mission started!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "The robot will now:"
echo "  1. Undock from charger"
echo "  2. Search for trash bins (using ${VISION_SERVER})"
echo "  3. Approach and classify"
echo "  4. Return to dock when done"
echo ""
echo -e "${CYAN}Monitor:${NC} tail -f /tmp/trash_bot_launch.log"
echo -e "${CYAN}Stop:${NC}    ./stop.sh"
echo -e "${CYAN}Dock:${NC}    ./dock.sh"
echo ""
