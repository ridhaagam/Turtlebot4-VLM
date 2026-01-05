#!/usr/bin/env python3
"""
camera_publisher_node.py
ROS2 node that captures camera frames and publishes them to the dashboard server.

Captures from USB webcam at 0.5 Hz and POSTs to /api/camera/frame.
Pauses capture when navigation or classification is active to share camera resource.
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
import threading
from datetime import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool, String
from trash_bot.msg import BinDetection


class CameraPublisherNode(Node):
    """ROS2 node for publishing camera frames to dashboard server."""

    def __init__(self):
        super().__init__('camera_publisher_node')

        # Parameters
        self.declare_parameter('server_url', 'http://192.168.0.81:5000')
        self.declare_parameter('camera_device', '/dev/nav_camera')
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('publish_rate', 0.5)  # Hz (every 2 seconds)
        self.declare_parameter('username', 'admin')
        self.declare_parameter('password', 'admin')

        self.server_url = self.get_parameter('server_url').get_parameter_value().string_value
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().string_value
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.username = self.get_parameter('username').get_parameter_value().string_value
        self.password = self.get_parameter('password').get_parameter_value().string_value

        # Resolve symlinks for OpenCV - use full device PATH, not index!
        # On systems with many video devices, cv2.VideoCapture(0) != /dev/video0
        import os
        self.device_path = os.path.realpath(self.camera_device)

        # State
        self.token = None
        self.is_paused = False
        self.camera = None
        self._lock = threading.Lock()
        self.latest_detection = None  # Store latest BinDetection for bounding boxes
        self.detection_lock = threading.Lock()

        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to navigation state to pause during active navigation
        self.navigation_state_sub = self.create_subscription(
            String,
            '/navigation_state',
            self.navigation_state_callback,
            10,
            callback_group=self.callback_group
        )

        # Subscribe to bin detection for bounding boxes
        self.bin_detection_sub = self.create_subscription(
            BinDetection,
            '/bin_detected',
            self.bin_detection_callback,
            10,
            callback_group=self.callback_group
        )

        # Timer for periodic capture
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(
            period,
            self.capture_and_publish,
            callback_group=self.callback_group
        )

        self.get_logger().info('='*50)
        self.get_logger().info('Camera Publisher Node (Navigation Camera)')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Server: {self.server_url}')
        self.get_logger().info(f'Camera: {self.camera_device} -> {self.device_path}')
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

    def navigation_state_callback(self, msg: String):
        """Handle navigation state changes.

        Note: We no longer pause during navigation since dashboard camera (/dev/video2)
        is separate from navigation camera (/dev/video0).
        """
        pass  # Keep subscription for potential future use

    def bin_detection_callback(self, msg: BinDetection):
        """Store latest bin detection for bounding box overlay."""
        with self.detection_lock:
            self.latest_detection = msg

    def draw_bounding_boxes(self, frame: np.ndarray) -> tuple:
        """Draw bounding boxes on frame from latest detection.

        Returns:
            tuple: (annotated_frame, detections_list)
        """
        detections = []

        with self.detection_lock:
            detection = self.latest_detection

        if detection is None or not detection.bin_detected:
            return frame, detections

        # Colors for different detection types
        colors = {
            'trash_bin': (0, 255, 0),      # Green
            'recycling_bin': (255, 165, 0), # Orange
            'bin': (0, 255, 255),           # Yellow
            'object': (255, 0, 255),        # Magenta
        }
        default_color = (0, 255, 0)

        # Draw multiple bounding boxes from Florence-2
        if detection.num_detections > 0:
            for i in range(detection.num_detections):
                x = detection.bbox_xs[i] if i < len(detection.bbox_xs) else 0
                y = detection.bbox_ys[i] if i < len(detection.bbox_ys) else 0
                w = detection.bbox_widths[i] if i < len(detection.bbox_widths) else 0
                h = detection.bbox_heights[i] if i < len(detection.bbox_heights) else 0
                label = detection.bbox_labels[i] if i < len(detection.bbox_labels) else 'object'
                conf = detection.bbox_confidences[i] if i < len(detection.bbox_confidences) else 0.0

                if w > 0 and h > 0:
                    color = colors.get(label, default_color)

                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Draw label with confidence
                    label_text = f'{label}: {conf:.2f}'
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
                    cv2.putText(frame, label_text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
                    })

        # Fallback: draw primary bbox if no multi-detections
        elif detection.bbox_width > 0 and detection.bbox_height > 0:
            x, y = detection.bbox_x, detection.bbox_y
            w, h = detection.bbox_width, detection.bbox_height
            label = detection.label or 'bin'
            conf = detection.confidence

            color = colors.get(label, default_color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f'{label}: {conf:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
            cv2.putText(frame, label_text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
            })

        return frame, detections

    def capture_and_publish(self):
        """Capture frame from camera, draw bboxes, and publish to server."""
        if self.is_paused:
            return

        with self._lock:
            try:
                # Open camera on-demand using full device PATH (not index)
                camera = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                if not camera.isOpened():
                    self.get_logger().warn('Failed to open camera')
                    return

                # Clear buffer
                for _ in range(3):
                    camera.grab()

                ret, frame = camera.read()
                camera.release()

                if not ret:
                    self.get_logger().warn('Failed to capture frame')
                    return

                # Draw bounding boxes from Florence-2 detections
                annotated_frame, detections = self.draw_bounding_boxes(frame.copy())

                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_b64 = base64.b64encode(buffer).decode('utf-8')

                # Post to server
                if not self.token:
                    self._get_token()

                if self.token:
                    # Send to annotated frame endpoint with detections
                    payload = {
                        'image': image_b64,
                        'width': annotated_frame.shape[1],
                        'height': annotated_frame.shape[0],
                        'device_id': 'turtlebot4',
                        'detections': detections,
                        'has_detections': len(detections) > 0
                    }

                    # Try new endpoint first, fall back to original
                    response = requests.post(
                        f'{self.server_url}/api/camera/frame-detected',
                        headers={'Authorization': f'Bearer {self.token}'},
                        json=payload,
                        timeout=5
                    )

                    # Fall back to original endpoint if new one doesn't exist
                    if response.status_code == 404:
                        response = requests.post(
                            f'{self.server_url}/api/camera/frame',
                            headers={'Authorization': f'Bearer {self.token}'},
                            json={
                                'image': image_b64,
                                'width': annotated_frame.shape[1],
                                'height': annotated_frame.shape[0],
                                'device_id': 'turtlebot4'
                            },
                            timeout=5
                        )

                    if response.status_code == 401:
                        # Token expired, refresh
                        self._get_token()
                    elif not response.ok:
                        self.get_logger().warn(f'Failed to post frame: {response.status_code}')

            except Exception as e:
                self.get_logger().error(f'Capture error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
