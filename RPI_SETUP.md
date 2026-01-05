# Raspberry Pi 4 + ROS2 Jazzy Setup Guide

Complete guide for setting up the TurtleBot4 Lite robot with ROS2 Jazzy on Raspberry Pi 4.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Ubuntu Installation](#ubuntu-installation)
3. [ROS2 Jazzy Installation](#ros2-jazzy-installation)
4. [TurtleBot4 Packages](#turtlebot4-packages)
5. [Camera Setup](#camera-setup)
6. [Python Dependencies](#python-dependencies)
7. [Building trash_bot Package](#building-trash_bot-package)
8. [Network Configuration](#network-configuration)
9. [Running the Robot](#running-the-robot)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Hardware | Raspberry Pi 4 (4GB+ RAM recommended) |
| Robot | TurtleBot4 Lite (iRobot Create3 base) |
| OS | Ubuntu 24.04 LTS (64-bit ARM) |
| ROS2 | Jazzy Jalisco |
| Cameras | 2x USB cameras (Logitech C270 or similar) |
| Network | WiFi connection to same network as server |

---

## Ubuntu Installation

### Option 1: TurtleBot4 Pre-built Image (Recommended)

Download the official TurtleBot4 image for RPi4:
```bash
# Download from Clearpath Robotics
wget https://turtlebot4.public.clearpathrobotics.com/images/latest/turtlebot4-jammy-jazzy.img.xz

# Flash to SD card (replace /dev/sdX with your device)
xzcat turtlebot4-jammy-jazzy.img.xz | sudo dd of=/dev/sdX bs=4M status=progress
```

### Option 2: Fresh Ubuntu 24.04 Installation

1. Download Ubuntu 24.04 Server for ARM64:
   - https://ubuntu.com/download/raspberry-pi

2. Flash to SD card using Raspberry Pi Imager

3. Boot and complete initial setup

---

## ROS2 Jazzy Installation

### Set Locale
```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Setup Sources
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS2 Jazzy
```bash
sudo apt update
sudo apt upgrade

# Desktop install (recommended for development)
sudo apt install ros-jazzy-desktop

# OR minimal install (for production)
sudo apt install ros-jazzy-ros-base

# Development tools
sudo apt install ros-dev-tools
```

### Environment Setup
```bash
# Add to ~/.bashrc
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation
```bash
ros2 --version
# Should output: ros2 0.12.x
```

---

## TurtleBot4 Packages

### Install TurtleBot4 Packages
```bash
sudo apt install ros-jazzy-turtlebot4-bringup \
                 ros-jazzy-turtlebot4-navigation \
                 ros-jazzy-turtlebot4-msgs \
                 ros-jazzy-irobot-create-msgs
```

### Create3 Communication
```bash
# Install Create3 dependencies
sudo apt install ros-jazzy-irobot-create-common-bringup

# Set ROS_DOMAIN_ID to match Create3 (default is 0)
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### Verify TurtleBot4 Connection
```bash
# Should list Create3 topics
ros2 topic list | grep create

# Check dock status
ros2 topic echo /dock_status --once
```

---

## Camera Setup

### Check Connected Cameras
```bash
# List video devices
ls -la /dev/video*

# Get detailed info
v4l2-ctl --list-devices
```

### Set Permissions
```bash
# Add user to video group
sudo usermod -aG video $USER

# Create udev rules for consistent naming (optional)
sudo tee /etc/udev/rules.d/99-usb-cameras.rules << 'EOF'
SUBSYSTEM=="video4linux", ATTR{name}=="*C270*", ATTR{index}=="0", SYMLINK+="nav_camera"
SUBSYSTEM=="video4linux", ATTR{name}=="*C270*", ATTR{index}=="1", SYMLINK+="class_camera"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Test Camera Capture
```bash
# Install test tools
sudo apt install v4l-utils ffmpeg

# Capture test frame
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test.jpg

# View with Python
python3 << 'EOF'
import cv2
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
if ret:
    cv2.imwrite('/tmp/test.jpg', frame)
    print("Camera OK!")
else:
    print("Camera FAILED!")
cap.release()
EOF
```

---

## Python Dependencies

### Create Virtual Environment (Recommended)
```bash
cd ~/ROS
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip wheel

pip install \
    opencv-python-headless \
    requests \
    pyyaml \
    numpy
```

### Add to bashrc
```bash
echo "source ~/ROS/venv/bin/activate" >> ~/.bashrc
```

---

## Building trash_bot Package

### Clone Repository
```bash
mkdir -p ~/ROS/src
cd ~/ROS/src

# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/embedded-system-final.git

# Copy edge-bot package to src
cp -r embedded-system-final/edge-bot/src/trash_bot .
```

### Build Package
```bash
cd ~/ROS
source /opt/ros/jazzy/setup.bash

# Build
colcon build --packages-select trash_bot --symlink-install

# Source workspace
source install/setup.bash
```

### Verify Build
```bash
# Check package is found
ros2 pkg list | grep trash_bot

# Check launch file
ros2 launch trash_bot trash_bot_launch.py --show-args
```

---

## Network Configuration

### Set ROS2 Domain
```bash
# Add to ~/.bashrc
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# For discovery across networks
export ROS_LOCALHOST_ONLY=0
```

### Configure Server URL
```bash
# Edit robot config
nano ~/ROS/src/trash_bot/config/robot_config.yaml

# Set server URL to your dashboard server IP
server:
  url: "http://192.168.0.81:5000"
```

### Test Server Connection
```bash
curl http://192.168.0.81:5000/api/health
# Should return: {"status": "healthy", ...}
```

---

## Running the Robot

### Quick Start
```bash
# Source environment
source /opt/ros/jazzy/setup.bash
source ~/ROS/install/setup.bash
export ROS_DOMAIN_ID=0

# Launch all nodes
ros2 launch trash_bot trash_bot_launch.py \
    server_url:=http://192.168.0.81:5000 \
    navigation_camera_device:=/dev/video2 \
    classification_camera_device:=/dev/video0
```

### Start a Mission
```bash
# In another terminal
ros2 service call /find_bin std_srvs/srv/SetBool "{data: true}"
```

### Stop Robot
```bash
ros2 service call /robot_cmd trash_bot/srv/RobotCommand "{command: 'stop'}"
```

### Manual Dock
```bash
ros2 action send_goal /_do_not_use/dock irobot_create_msgs/action/Dock "{}"
```

---

## Troubleshooting

### Robot Not Moving

1. **Check TurtleBot4 connection:**
   ```bash
   ros2 topic list | grep cmd_vel
   ros2 topic echo /cmd_vel
   ```

2. **Check navigation node:**
   ```bash
   ros2 node list | grep navigation
   ros2 node info /navigation_vlm_node
   ```

3. **Check server connection:**
   ```bash
   curl -v http://192.168.0.81:5000/api/health
   ```

### Camera Not Found

1. **List devices:**
   ```bash
   ls -la /dev/video*
   v4l2-ctl --list-devices
   ```

2. **Check permissions:**
   ```bash
   groups $USER | grep video
   ```

3. **Check USB:**
   ```bash
   lsusb | grep -i cam
   dmesg | tail -20
   ```

### ROS2 Discovery Issues

1. **Check domain ID:**
   ```bash
   echo $ROS_DOMAIN_ID
   # Should match Create3 (default 0)
   ```

2. **Check network:**
   ```bash
   ros2 daemon stop
   ros2 daemon start
   ros2 topic list
   ```

3. **Firewall:**
   ```bash
   sudo ufw allow 7400:7500/udp  # DDS ports
   ```

### Build Errors

1. **Missing dependencies:**
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

2. **Clean build:**
   ```bash
   rm -rf build/ install/ log/
   colcon build --packages-select trash_bot
   ```

---

## Quick Reference

```bash
# Full setup sequence
source /opt/ros/jazzy/setup.bash
source ~/ROS/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Run
ros2 launch trash_bot trash_bot_launch.py

# Start mission
ros2 service call /find_bin std_srvs/srv/SetBool "{data: true}"

# Stop
ros2 service call /robot_cmd trash_bot/srv/RobotCommand "{command: 'stop'}"

# Dock
ros2 action send_goal /_do_not_use/dock irobot_create_msgs/action/Dock "{}"
```

---

## Version Information

Tested with:
- Ubuntu 24.04 LTS (ARM64)
- ROS2 Jazzy Jalisco
- TurtleBot4 Lite (Create3 base)
- Raspberry Pi 4 (4GB/8GB)
