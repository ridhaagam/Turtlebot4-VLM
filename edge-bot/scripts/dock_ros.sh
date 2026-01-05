#!/bin/bash
# dock_ros.sh - Simple dock script using TurtleBot4 auto-dock action
# Just sends the dock action directly to the robot

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== TurtleBot4 Auto Dock ===${NC}"
echo ""

# Restart ROS2 daemon for clean state
echo "Restarting ROS2 daemon..."
ros2 daemon stop >/dev/null 2>&1
sleep 1
ros2 daemon start >/dev/null 2>&1
sleep 2

# Check if dock action is available
echo "Checking for dock action..."
if ! timeout 10 ros2 action list 2>/dev/null | grep -q "/_do_not_use/dock"; then
    echo -e "${RED}ERROR: Dock action not available${NC}"
    echo "  - Is the robot connected?"
    echo "  - Is the Create3 base powered on?"
    exit 1
fi

echo -e "${GREEN}Dock action available${NC}"
echo ""
echo -e "${YELLOW}Sending dock command...${NC}"
echo "  (Robot will search for dock IR beacon and navigate to it)"
echo "  (This may take up to 2 minutes)"
echo ""

# Send dock action
DOCK_OUTPUT=$(timeout 120 ros2 action send_goal /_do_not_use/dock irobot_create_msgs/action/Dock "{}" 2>&1)
echo "$DOCK_OUTPUT"

# Check result
if echo "$DOCK_OUTPUT" | grep -q "is_docked: true"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  DOCK SUCCESSFUL! Robot is docked.${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
elif echo "$DOCK_OUTPUT" | grep -q "is_docked: false"; then
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  DOCK FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Possible reasons:"
    echo "  - Robot cannot see dock IR beacon"
    echo "  - Robot is too far from dock"
    echo "  - Obstacle blocking path to dock"
    echo ""
    echo "Try:"
    echo "  1. Move robot closer to dock (within 1-2 meters)"
    echo "  2. Make sure robot is facing the dock"
    echo "  3. Run ./dock_ros.sh again"
    exit 1
else
    echo ""
    echo -e "${RED}Dock action failed or timed out${NC}"
    exit 1
fi
