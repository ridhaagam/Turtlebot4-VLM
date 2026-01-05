#!/bin/bash
# stop.sh - Stop all TurtleBot4 trash_bot processes
#
# Usage: ./stop.sh
#        ./stop.sh --all   # Also stop vision server

echo "Stopping trash_bot processes..."

pkill -9 -f "ros2 launch trash_bot" 2>/dev/null || true
pkill -9 -f "navigation_vlm" 2>/dev/null || true
pkill -9 -f "classifier_node" 2>/dev/null || true
pkill -9 -f "camera_publisher" 2>/dev/null || true
pkill -9 -f "motion_subscriber" 2>/dev/null || true
pkill -9 -f "command_service" 2>/dev/null || true
pkill -9 -f "map_publisher" 2>/dev/null || true
pkill -9 -f "bin_detector" 2>/dev/null || true

if [[ "$1" == "--all" ]]; then
    echo "Stopping vision server..."
    pkill -9 -f "llama-server" 2>/dev/null || true
fi

sleep 2
echo "All processes stopped"
echo ""
echo "To dock robot: ./dock.sh"
