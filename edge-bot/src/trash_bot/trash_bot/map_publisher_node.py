#!/usr/bin/env python3
"""
map_publisher_node.py
ROS2 node that subscribes to SLAM occupancy grid and publishes it to the dashboard.

Converts nav_msgs/OccupancyGrid to PNG image and POSTs to /api/map/update.
"""

import sys
import os

# Add paths for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2
import base64
import requests
import numpy as np
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

try:
    import tf2_ros
    from tf2_ros import TransformException
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False


class MapPublisherNode(Node):
    """ROS2 node for publishing SLAM map to dashboard server."""

    def __init__(self):
        super().__init__('map_publisher_node')

        # Parameters
        self.declare_parameter('server_url', 'http://192.168.0.81:5000')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('publish_rate', 0.2)  # Hz (every 5 seconds)
        self.declare_parameter('username', 'admin')
        self.declare_parameter('password', 'admin')

        self.server_url = self.get_parameter('server_url').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.username = self.get_parameter('username').get_parameter_value().string_value
        self.password = self.get_parameter('password').get_parameter_value().string_value

        # State
        self.token = None
        self.latest_map = None
        self.latest_map_time = None
        self.robot_x = None
        self.robot_y = None
        self.robot_theta = None
        self._lock = threading.Lock()

        # Callback group
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10,
            callback_group=self.callback_group
        )

        # TF listener for robot pose
        if HAS_TF2:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer for periodic publishing
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(
            period,
            self.publish_map,
            callback_group=self.callback_group
        )

        self.get_logger().info('='*50)
        self.get_logger().info('Map Publisher Node')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Server: {self.server_url}')
        self.get_logger().info(f'Map topic: {self.map_topic}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')

        # Get initial token
        self._get_token()

    def _get_token(self) -> bool:
        """Login and get JWT token."""
        try:
            response = requests.post(
                f'{self.server_url}/api/auth/login',
                json={'username': self.username, 'password': self.password},
                timeout=10
            )
            if response.ok:
                data = response.json()
                self.token = data.get('token')
                self.get_logger().info('Logged in successfully')
                return True
            else:
                self.get_logger().error(f'Login failed: {response.status_code}')
                return False
        except Exception as e:
            self.get_logger().error(f'Login error: {e}')
            return False

    def map_callback(self, msg: OccupancyGrid):
        """Handle incoming occupancy grid message."""
        with self._lock:
            self.latest_map = msg
            self.latest_map_time = datetime.utcnow()

    def _get_robot_pose(self):
        """Get robot pose from TF."""
        if not HAS_TF2:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y

            # Convert quaternion to yaw
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_theta = np.arctan2(siny_cosp, cosy_cosp)

        except Exception as e:
            # TF not available yet, that's ok
            pass

    def _occupancy_grid_to_image(self, msg: OccupancyGrid) -> np.ndarray:
        """Convert OccupancyGrid to grayscale image.

        Occupancy values:
        - -1: Unknown (gray)
        - 0: Free (white)
        - 100: Occupied (black)
        """
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

        # Create image: unknown=128 (gray), free=255 (white), occupied=0 (black)
        image = np.zeros((msg.info.height, msg.info.width), dtype=np.uint8)

        # Unknown (-1) -> gray (128)
        image[data == -1] = 128

        # Free (0) -> white (255)
        image[data == 0] = 255

        # Occupied (>0) -> scale to black (0-50 range)
        occupied_mask = data > 0
        image[occupied_mask] = np.clip(50 - (data[occupied_mask] // 2), 0, 50).astype(np.uint8)

        # Flip vertically (ROS map origin is bottom-left, image origin is top-left)
        image = np.flipud(image)

        return image

    def publish_map(self):
        """Convert latest map to image and publish to server."""
        with self._lock:
            if self.latest_map is None:
                return

            map_msg = self.latest_map

        try:
            # Get robot pose
            self._get_robot_pose()

            # Convert to image
            image = self._occupancy_grid_to_image(map_msg)

            # Encode as PNG
            _, buffer = cv2.imencode('.png', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            # Post to server
            if not self.token:
                self._get_token()

            if self.token:
                payload = {
                    'image': image_b64,
                    'width': map_msg.info.width,
                    'height': map_msg.info.height,
                    'resolution': map_msg.info.resolution,
                    'origin_x': map_msg.info.origin.position.x,
                    'origin_y': map_msg.info.origin.position.y,
                }

                # Add robot pose if available
                if self.robot_x is not None:
                    payload['robot_x'] = self.robot_x
                    payload['robot_y'] = self.robot_y
                    payload['robot_theta'] = self.robot_theta

                response = requests.post(
                    f'{self.server_url}/api/map/update',
                    headers={'Authorization': f'Bearer {self.token}'},
                    json=payload,
                    timeout=10
                )

                if response.status_code == 401:
                    # Token expired, refresh
                    self._get_token()
                elif not response.ok:
                    self.get_logger().warn(f'Failed to post map: {response.status_code}')

        except Exception as e:
            self.get_logger().error(f'Map publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
