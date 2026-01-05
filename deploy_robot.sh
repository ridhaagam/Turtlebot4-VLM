#!/bin/bash
# Deploy robot code to TurtleBot4 at 192.168.0.6
# Password: 021001

echo "Copying files to TurtleBot..."
scp -o StrictHostKeyChecking=no \
  /home/agam/Documents/embedded-system-final/edge-bot/src/trash_bot/trash_bot/classifier_node.py \
  /home/agam/Documents/embedded-system-final/edge-bot/src/trash_bot/trash_bot/llama_vision_client.py \
  /home/agam/Documents/embedded-system-final/edge-bot/src/trash_bot/trash_bot/navigation_vlm_node.py \
  g3ubuntu@192.168.0.6:~/ROS/src/trash_bot/trash_bot/

echo "Done! Now SSH to robot and build."
echo "Run: ssh g3ubuntu@192.168.0.6"
