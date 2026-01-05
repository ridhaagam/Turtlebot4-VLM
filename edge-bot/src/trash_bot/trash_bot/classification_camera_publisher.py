#!/usr/bin/env python3
"""
classification_camera_publisher.py
ROS2 node that captures frames from the classification camera and publishes them to the dashboard.

This is a dedicated camera publisher for the classification camera (/dev/class_camera),
which streams independently to provide a second camera feed on the dashboard.
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from trash_bot.msg import BinDetection


class ClassificationCameraPublisher(Node):
    """ROS2 node for publishing classification camera frames to dashboard server."""

    def __init__(self):
        super().__init__('classification_camera_publisher')

        # Parameters
        self.declare_parameter('server_url', 'http://192.168.0.81:5000')
        self.declare_parameter('camera_device', '/dev/class_camera')
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
        self.resolved_device = os.path.realpath(self.camera_device)
        # Use the full path for cv2.VideoCapture
        self.device_path = self.resolved_device

        # State
        self.token = None
        self.is_paused = False  # Pause when classifier is actively using camera
        self.pause_start_time = 0.0  # Track when pause started for auto-resume
        self.AUTO_RESUME_TIMEOUT = 10.0  # Auto-resume after 10 seconds if stuck
        self._lock = threading.Lock()
        self._upload_in_progress = False  # Track if upload is running

        # HTTP session with connection pooling and retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=2,
            pool_maxsize=2
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # QoS profile for reliable message delivery (don't drop messages!)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to classification_active to pause when classifier is capturing
        self.classification_active_sub = self.create_subscription(
            Bool,
            '/classification_active',
            self.classification_active_callback,
            reliable_qos,
            callback_group=self.callback_group
        )

        # Subscribe to bin classification results to display on dashboard
        self.bin_classified_sub = self.create_subscription(
            BinDetection,
            '/bin_classified',
            self.bin_classified_callback,
            10,
            callback_group=self.callback_group
        )

        # Track latest classification result
        self.latest_classification = None
        self.classification_lock = threading.Lock()

        # Timer for periodic capture
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(
            period,
            self.capture_and_publish,
            callback_group=self.callback_group
        )

        self.get_logger().info('='*50)
        self.get_logger().info('Classification Camera Publisher')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Server: {self.server_url}')
        self.get_logger().info(f'Camera: {self.camera_device} -> {self.resolved_device}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')

        # Get initial token
        self._get_token()

    def _get_token(self) -> bool:
        """Login and get JWT token."""
        try:
            response = self.session.post(
                f'{self.server_url}/api/auth/login',
                json={'username': self.username, 'password': self.password},
                timeout=15
            )
            if response.ok:
                data = response.json()
                self.token = data.get('token')
                self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                self.get_logger().info('Logged in successfully')
                return True
            else:
                self.get_logger().error(f'Login failed: {response.status_code}')
                return False
        except Exception as e:
            self.get_logger().error(f'Login error: {e}')
            return False

    def classification_active_callback(self, msg: Bool):
        """Handle classification_active signal from classifier_node.

        Pause camera capture when classifier is actively using the camera.
        """
        with self._lock:
            was_paused = self.is_paused
            self.is_paused = msg.data
            if self.is_paused and not was_paused:
                self.pause_start_time = time.time()
                self.get_logger().info('Classification active - pausing camera capture')
            elif not self.is_paused and was_paused:
                pause_duration = time.time() - self.pause_start_time
                self.get_logger().info(f'Classification done - resuming camera capture (paused for {pause_duration:.1f}s)')
                self.pause_start_time = 0.0

    def bin_classified_callback(self, msg: BinDetection):
        """Store latest classification result to send to dashboard."""
        with self.classification_lock:
            self.latest_classification = {
                'fullness_level': msg.label,
                'fullness_percent': int(msg.confidence * 100),
                'label': 'trash bin',
                'confidence': msg.confidence,
            }
            # Post classification result to dashboard
            self._post_classification_result()

    def _post_classification_result(self):
        """Post latest classification result to dashboard (non-blocking)."""
        with self.classification_lock:
            if self.latest_classification is None:
                return
            result = self.latest_classification.copy()

        def do_post():
            if not self.token:
                self._get_token()

            if self.token:
                try:
                    response = self.session.post(
                        f'{self.server_url}/api/camera/classification/result',
                        json=result,
                        timeout=15
                    )
                    if response.status_code == 401:
                        self._get_token()
                    elif response.ok:
                        self.get_logger().info('Classification result posted to dashboard')
                except Exception as e:
                    self.get_logger().warn(f'Failed to post classification result: {e}')

        # Run in background thread to avoid blocking
        threading.Thread(target=do_post, daemon=True).start()

    def capture_and_publish(self):
        """Capture frame from classification camera and publish to server."""
        # Check for auto-resume timeout (in case resume message was dropped)
        with self._lock:
            if self.is_paused and self.pause_start_time > 0:
                pause_duration = time.time() - self.pause_start_time
                if pause_duration > self.AUTO_RESUME_TIMEOUT:
                    self.get_logger().warn(f'AUTO-RESUME: Paused for {pause_duration:.1f}s (>{self.AUTO_RESUME_TIMEOUT}s) - forcing resume')
                    self.is_paused = False
                    self.pause_start_time = 0.0

            if self.is_paused:
                # Classification is happening - skip this capture cycle
                return

        # Skip if previous upload still in progress
        if self._upload_in_progress:
            return

        try:
            # Open camera on-demand with retry (in case classifier is using it)
            # Use full device PATH (not index) with V4L2 backend
            camera = None
            for attempt in range(3):
                camera = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                if camera.isOpened():
                    break
                else:
                    camera.release()
                    if attempt < 2:
                        time.sleep(0.2)  # Brief wait before retry

            if camera is None or not camera.isOpened():
                # Camera busy (classifier might be using it) - skip this frame
                return

            # Clear buffer
            for _ in range(3):
                camera.grab()

            ret, frame = camera.read()
            camera.release()

            if not ret:
                self.get_logger().warn('Failed to capture classification camera frame')
                return

            # Encode as JPEG with reduced quality for faster upload
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            # Build payload
            payload = {
                'image': image_b64,
                'width': frame.shape[1],
                'height': frame.shape[0],
                'device_id': 'classification',
            }

            # Upload in background thread to avoid blocking timer
            def do_upload():
                self._upload_in_progress = True
                try:
                    if not self.token:
                        self._get_token()

                    if self.token:
                        response = self.session.post(
                            f'{self.server_url}/api/camera/classification/frame',
                            json=payload,
                            timeout=15
                        )

                        if response.status_code == 401:
                            self._get_token()
                            # Retry once with new token
                            if self.token:
                                response = self.session.post(
                                    f'{self.server_url}/api/camera/classification/frame',
                                    json=payload,
                                    timeout=15
                                )

                        if not response.ok:
                            self.get_logger().warn(f'Failed to post classification frame: {response.status_code}')

                except requests.exceptions.Timeout:
                    self.get_logger().warn('Classification frame upload timed out - will retry next cycle')
                except Exception as e:
                    self.get_logger().error(f'Classification frame upload error: {e}')
                finally:
                    self._upload_in_progress = False

            threading.Thread(target=do_upload, daemon=True).start()

        except Exception as e:
            self.get_logger().error(f'Classification camera capture error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ClassificationCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
