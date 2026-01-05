#!/bin/bash
# Smart Dock Script - Docks the TurtleBot4
# Works both when navigation node is running or after stop.sh
# Logs all details to ~/ROS/logs/dock.log

source /opt/ros/jazzy/setup.bash
source /home/g3ubuntu/ROS/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_FILE="$HOME/ROS/logs/dock.log"
HISTORY_FILE="$HOME/ROS/logs/command_history.json"
mkdir -p "$(dirname "$LOG_FILE")"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo -e "$1"
}

# Function to retrace commands using Python script (properly waits for action completion)
do_retrace() {
    if [ ! -f "$HISTORY_FILE" ]; then
        log_msg "${YELLOW}No command history found - skipping retrace${NC}"
        return 0
    fi

    log_msg "${CYAN}Starting Python-based retrace (waits for action completion)...${NC}"
    echo ""

    # Use Python retrace script that properly waits for each action to complete
    # This is critical because bash-based timing doesn't work reliably for Create3 actions
    python3 /home/g3ubuntu/ROS/retrace.py 2>&1 | tee -a "$LOG_FILE"
    local result=${PIPESTATUS[0]}

    if [ $result -eq 0 ]; then
        log_msg "${GREEN}Retrace complete!${NC}"
        return 0
    else
        log_msg "${RED}Retrace failed (exit code: $result)${NC}"
        return 1
    fi
}

# Function to do direct dock action
do_direct_dock() {
    log_msg "Sending dock action to TurtleBot..."

    # Restart ROS2 daemon to ensure clean state
    ros2 daemon stop >/dev/null 2>&1
    sleep 1
    ros2 daemon start >/dev/null 2>&1
    sleep 3

    # Check if dock action is available
    if ! timeout 10 ros2 action list 2>/dev/null | grep -q "/_do_not_use/dock"; then
        log_msg "${RED}ERROR: Dock action not available - is robot connected?${NC}"
        return 1
    fi

    # Send dock action
    log_msg "Executing dock action (may take up to 90 seconds)..."
    DOCK_OUTPUT=$(timeout 90 ros2 action send_goal /_do_not_use/dock irobot_create_msgs/action/Dock "{}" 2>&1)
    local result=$?
    echo "$DOCK_OUTPUT"

    # Check if dock was successful (is_docked: true in output)
    if echo "$DOCK_OUTPUT" | grep -q "is_docked: true"; then
        log_msg "${GREEN}DOCK SUCCESSFUL! Robot is docked.${NC}"
        # Clear the history file after successful dock
        rm -f "$HISTORY_FILE" 2>/dev/null
        return 0
    elif echo "$DOCK_OUTPUT" | grep -q "is_docked: false"; then
        log_msg "${RED}DOCK FAILED: Robot could not dock (not close enough to dock)${NC}"
        log_msg "${YELLOW}Try manually moving robot closer to dock and run ./dock.sh again${NC}"
        return 1
    else
        log_msg "${RED}Dock action failed (exit code: $result)${NC}"
        return 1
    fi
}

echo -e "${GREEN}=== Smart Dock Script ===${NC}"
echo -e "Log file: ${CYAN}$LOG_FILE${NC}"
echo ""

# Add separator to log
echo "" >> "$LOG_FILE"
echo "================================================================" >> "$LOG_FILE"
log_msg "DOCK.SH INITIATED"
echo "================================================================" >> "$LOG_FILE"

# Check if navigation node is running
if timeout 5 ros2 node list 2>/dev/null | grep -q "navigation_vlm_node"; then
    log_msg "${GREEN}Navigation node is running${NC}"

    # Step 1: Call /dock_now service to stop search and save history
    log_msg "Calling /dock_now service to stop search..."
    DOCK_RESULT=$(ros2 service call /dock_now std_srvs/srv/Trigger "{}" 2>&1)
    echo "$DOCK_RESULT"

    if echo "$DOCK_RESULT" | grep -q "success=True"; then
        log_msg "${GREEN}Search stopped successfully${NC}"
    else
        log_msg "${YELLOW}dock_now service call may have failed, trying topic fallback...${NC}"
        ros2 topic pub --once /robot_cmd trash_bot/msg/RobotCommand "{command: 'dock_now', speed: 0.0, duration: 0.0}" 2>/dev/null
        sleep 2
    fi

    # Wait for history to be saved
    sleep 2
    log_msg "Checking for command history..."
fi

# Step 2: Check for command history and do retrace
if [ -f "$HISTORY_FILE" ]; then
    log_msg "${CYAN}Found command history - will retrace path${NC}"

    # Python retrace script uses action clients directly (no motion_subscriber needed)
    # It properly waits for each action to complete before sending the next
    do_retrace
else
    log_msg "${YELLOW}No command history found - skipping retrace${NC}"
    echo "  Robot position is unknown"
    echo "  Will attempt direct dock (may fail if not near dock)"
fi

# Step 3: Do direct dock action
echo ""
log_msg "Proceeding to dock action..."
do_direct_dock

echo ""
echo -e "Full log: ${CYAN}$LOG_FILE${NC}"
