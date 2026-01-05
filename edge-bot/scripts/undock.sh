#!/bin/bash
# Undock the TurtleBot4
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "Undocking robot..."
timeout 60 ros2 action send_goal /_do_not_use/undock irobot_create_msgs/action/Undock "{}"

if [ $? -eq 0 ]; then
    echo "Undock completed"
else
    echo "Undock failed"
fi
