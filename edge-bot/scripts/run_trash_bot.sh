#!/bin/bash
# TurtleBot4 Trash Detection System - Autonomous Runner
# This script manages the full trash detection workflow

set -e

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source /home/g3ubuntu/ROS/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "  TurtleBot4 Trash Detection System"
echo "============================================"

# Function to check if bringup is running
check_bringup() {
    if pgrep -f "turtlebot4_node" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to start bringup
start_bringup() {
    echo -e "${YELLOW}Starting TurtleBot4 bringup...${NC}"
    ros2 launch turtlebot4_bringup robot.launch.py model:=lite &
    sleep 15
    if check_bringup; then
        echo -e "${GREEN}TurtleBot4 bringup started successfully${NC}"
    else
        echo -e "${RED}Failed to start TurtleBot4 bringup${NC}"
        exit 1
    fi
}

# Function to undock robot
undock_robot() {
    echo -e "${YELLOW}Undocking robot...${NC}"
    timeout 30 ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}" 2>&1 | grep -E "Goal|Result|SUCCEEDED|ABORTED" || true
    echo -e "${GREEN}Undock command sent${NC}"
}

# Function to dock robot
dock_robot() {
    echo -e "${YELLOW}Docking robot (returning to charging station)...${NC}"
    timeout 120 ros2 action send_goal /dock irobot_create_msgs/action/Dock "{}" 2>&1 | grep -E "Goal|Result|SUCCEEDED|ABORTED" || true
    echo -e "${GREEN}Dock command sent${NC}"
}

# Function to drive forward
drive_forward() {
    local distance=${1:-0.3}
    local speed=${2:-0.2}
    echo -e "${YELLOW}Driving forward ${distance}m at ${speed}m/s...${NC}"
    timeout 30 ros2 action send_goal /drive_distance irobot_create_msgs/action/DriveDistance "{distance: ${distance}, max_translation_speed: ${speed}}" 2>&1 | grep -E "Goal|Result|SUCCEEDED" || true
}

# Function to rotate
rotate() {
    local angle=${1:-1.57}  # Default 90 degrees
    local speed=${2:-0.5}
    echo -e "${YELLOW}Rotating ${angle} radians...${NC}"
    timeout 20 ros2 action send_goal /rotate_angle irobot_create_msgs/action/RotateAngle "{angle: ${angle}, max_rotation_speed: ${speed}}" 2>&1 | grep -E "Goal|Result|SUCCEEDED" || true
}

# Function to start motion subscriber
start_motion_subscriber() {
    echo -e "${YELLOW}Starting motion subscriber...${NC}"
    ros2 run trash_bot motion_subscriber &
    MOTION_PID=$!
    sleep 2
    echo -e "${GREEN}Motion subscriber started (PID: $MOTION_PID)${NC}"
}

# Function to send robot command
send_command() {
    local cmd=$1
    local speed=${2:-0.3}
    local duration=${3:-2.0}
    echo -e "${YELLOW}Sending command: ${cmd}${NC}"
    ros2 topic pub --once /robot_cmd trash_bot/msg/RobotCommand "{command: '${cmd}', speed: ${speed}, duration: ${duration}}" 2>/dev/null
}

# Function for search pattern
search_pattern() {
    echo -e "${YELLOW}Executing search pattern...${NC}"

    # Move forward
    drive_forward 0.5 0.15
    sleep 1

    # Rotate 90 degrees left
    rotate 1.57 0.4
    sleep 1

    # Move forward
    drive_forward 0.3 0.15
    sleep 1

    # Rotate 90 degrees right
    rotate -1.57 0.4
    sleep 1

    # Move forward
    drive_forward 0.3 0.15

    echo -e "${GREEN}Search pattern complete${NC}"
}

# Main menu
show_menu() {
    echo ""
    echo "============================================"
    echo "  Available Commands:"
    echo "============================================"
    echo "  1. Start System (bringup + motion subscriber)"
    echo "  2. Undock Robot"
    echo "  3. Drive Forward"
    echo "  4. Rotate Left (90 deg)"
    echo "  5. Rotate Right (90 deg)"
    echo "  6. Execute Search Pattern"
    echo "  7. Dock Robot (return to charger)"
    echo "  8. Send Custom Command"
    echo "  9. Stop All"
    echo "  0. Exit"
    echo "============================================"
}

# Process command line arguments
case "$1" in
    start)
        if ! check_bringup; then
            start_bringup
        fi
        start_motion_subscriber
        ;;
    undock)
        undock_robot
        ;;
    dock)
        dock_robot
        ;;
    forward)
        drive_forward ${2:-0.3} ${3:-0.2}
        ;;
    rotate)
        rotate ${2:-1.57} ${3:-0.5}
        ;;
    search)
        search_pattern
        ;;
    auto)
        # Full autonomous routine (old search pattern version)
        echo -e "${GREEN}Starting autonomous trash detection routine (legacy)...${NC}"
        if ! check_bringup; then
            start_bringup
        fi
        start_motion_subscriber
        sleep 2
        undock_robot
        sleep 3
        search_pattern
        sleep 2
        dock_robot
        echo -e "${GREEN}Autonomous routine complete!${NC}"
        ;;
    mission)
        # Full mission with SmolVLM2 via llama.cpp
        echo -e "${GREEN}Starting FULL AUTONOMOUS MISSION with AI navigation...${NC}"
        echo -e "${YELLOW}This will:${NC}"
        echo -e "  1. Start SmolVLM2 vision server (llama.cpp)"
        echo -e "  2. Undock from charging station"
        echo -e "  3. Search for trash bin using SmolVLM2 (USB webcam)"
        echo -e "  4. Navigate toward detected bin"
        echo -e "  5. Classify bin contents using SmolVLM2"
        echo -e "  6. Send results to dashboard server"
        echo -e "  7. Return to charging dock"
        echo ""

        # Start vision server first
        echo -e "${YELLOW}Starting SmolVLM2 vision server...${NC}"
        /home/g3ubuntu/ROS/start_vision_server.sh
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to start vision server${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Launching trash_bot nodes...${NC}"
        ros2 launch trash_bot trash_bot_launch.py &
        LAUNCH_PID=$!
        sleep 10
        echo -e "${YELLOW}Starting full mission...${NC}"
        ros2 run trash_bot command_client.py full_mission
        echo -e "${GREEN}Mission complete. Stopping nodes...${NC}"
        kill $LAUNCH_PID 2>/dev/null || true
        ;;
    vision)
        # Start vision server only
        echo -e "${GREEN}Starting SmolVLM2 vision server...${NC}"
        /home/g3ubuntu/ROS/start_vision_server.sh
        ;;
    test-vision)
        # Test vision with webcam
        echo -e "${GREEN}Testing SmolVLM2 vision...${NC}"
        source /home/g3ubuntu/ROS/embedded-system-final/edge/venv/bin/activate
        timeout 180 python3 /home/g3ubuntu/ROS/test_llama_vision.py
        ;;
    interactive)
        while true; do
            show_menu
            read -p "Enter choice: " choice
            case $choice in
                1)
                    if ! check_bringup; then
                        start_bringup
                    fi
                    start_motion_subscriber
                    ;;
                2) undock_robot ;;
                3)
                    read -p "Distance (m): " dist
                    drive_forward ${dist:-0.3}
                    ;;
                4) rotate 1.57 ;;
                5) rotate -1.57 ;;
                6) search_pattern ;;
                7) dock_robot ;;
                8)
                    read -p "Command (forward/backward/left/right/stop): " cmd
                    read -p "Speed (0-1): " spd
                    read -p "Duration (seconds): " dur
                    send_command $cmd $spd $dur
                    ;;
                9)
                    pkill -f motion_subscriber 2>/dev/null || true
                    echo "Stopped motion subscriber"
                    ;;
                0)
                    echo "Exiting..."
                    exit 0
                    ;;
                *)
                    echo "Invalid choice"
                    ;;
            esac
        done
        ;;
    *)
        echo "Usage: $0 {start|undock|dock|forward|rotate|search|auto|mission|vision|test-vision|interactive}"
        echo ""
        echo "Commands:"
        echo "  start       - Start bringup and motion subscriber"
        echo "  undock      - Undock from charging station"
        echo "  dock        - Return to charging station"
        echo "  forward [dist] [speed] - Drive forward"
        echo "  rotate [angle] [speed] - Rotate (radians)"
        echo "  search      - Execute search pattern"
        echo "  auto        - Full autonomous routine (legacy)"
        echo "  vision      - Start SmolVLM2 vision server only"
        echo "  test-vision - Test SmolVLM2 with webcam"
        echo "  mission     - FULL AI MISSION with SmolVLM2"
        echo "  interactive - Interactive menu mode"
        echo ""
        echo "AI Mission Workflow (using llama.cpp + SmolVLM2):"
        echo "  1. Start SmolVLM2 vision server (llama.cpp)"
        echo "  2. Undock from charger"
        echo "  3. SmolVLM2 searches for trash bin (USB webcam)"
        echo "  4. Navigate and approach bin"
        echo "  5. SmolVLM2 classifies bin contents"
        echo "  6. Results sent to dashboard (192.168.0.81:5000)"
        echo "  7. Return to charging dock"
        echo ""
        echo "Quick test: $0 test-vision"
        ;;
esac
