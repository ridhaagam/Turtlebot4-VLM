#!/usr/bin/env python3
"""
classifier_node.py
ROS2 Bin Classification Service using SmolVLM2 via llama.cpp server.

This node captures from USB webcam and uses SmolVLM2 to classify bin contents.
Results are sent to the dashboard server.
"""

import sys
import os

# Add this script's directory to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

EMBEDDED_SYSTEM_PATH = '/home/g3ubuntu/ROS/embedded-system-final/edge/src'
VENV_SITE_PACKAGES = '/home/g3ubuntu/ROS/embedded-system-final/edge/venv/lib/python3.12/site-packages'
sys.path.insert(0, EMBEDDED_SYSTEM_PATH)
sys.path.insert(0, VENV_SITE_PACKAGES)

import cv2
import uuid
import logging
import requests
import base64
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from trash_bot.srv import ClassifyBin
from trash_bot.msg import BinDetection, RobotCommand
from std_msgs.msg import Bool

# Import llama vision client
try:
    from trash_bot.llama_vision_client import LlamaVisionClient, ClassificationResult
except ImportError:
    from llama_vision_client import LlamaVisionClient, ClassificationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierNode(Node):
    """ROS2 service node for bin classification using SmolVLM2 via llama.cpp."""

    def __init__(self):
        super().__init__('classifier_node')

        # Parameters
        self.declare_parameter('llama_server_url', 'http://192.168.0.81:5000')
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('server_url', 'http://192.168.0.81:5000')

        self.llama_server_url = self.get_parameter('llama_server_url').get_parameter_value().string_value
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().string_value
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.server_url = self.get_parameter('server_url').get_parameter_value().string_value

        self.callback_group = ReentrantCallbackGroup()

        # Service
        self.classify_service = self.create_service(
            ClassifyBin, '/classify_bin', self.classify_callback,
            callback_group=self.callback_group
        )

        # Publishers
        self.detection_publisher = self.create_publisher(BinDetection, '/bin_classified', 10)
        self.cmd_publisher = self.create_publisher(RobotCommand, '/robot_cmd', 10)

        # QoS profile for reliable message delivery (don't drop pause/resume messages!)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        # Publisher to signal classification_camera_publisher to pause/resume
        self.classification_active_pub = self.create_publisher(Bool, '/classification_active', reliable_qos)

        # Components
        self.vision_client = None
        self.camera = None
        self.initialized = False

        self.get_logger().info('='*50)
        self.get_logger().info('Classifier Node (llama.cpp SmolVLM2)')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Llama server: {self.llama_server_url}')
        self.get_logger().info(f'Camera: {self.camera_device}')
        self.get_logger().info(f'Dashboard: {self.server_url}')

        self._initialize()

    def _initialize(self):
        """Initialize vision client (camera opened on-demand)."""
        try:
            # Resolve symlinks and get full device path
            # Use FULL PATH for cv2.VideoCapture, not index (index doesn't map to /dev/videoN)
            import os
            self.device_path = os.path.realpath(self.camera_device)
            self.get_logger().info(f'Camera device path: {self.device_path}')

            # Don't open camera at startup - open on-demand to share with navigation node
            self.get_logger().info('Camera will be opened on-demand for classification')

            # Vision client
            self.vision_client = LlamaVisionClient(
                server_url=self.llama_server_url,
                timeout=30.0  # Reduced timeout - server GPU is much faster
            )

            self.initialized = True
            self.get_logger().info('Classifier node initialized')

        except Exception as e:
            self.get_logger().error(f'Initialization error: {e}')

    def _capture_frame(self):
        """Capture frame from camera (opens camera on-demand)."""
        try:
            # Signal that classification needs the camera
            self.get_logger().info('Signaling camera need - pausing classification_camera_publisher...')
            active_msg = Bool()
            active_msg.data = True
            self.classification_active_pub.publish(active_msg)

            # Wait for camera publisher to finish current capture and release camera
            # Camera publisher takes ~1 second per capture cycle with retries
            import time
            time.sleep(1.5)  # Give camera publisher time to finish current capture and release

            # Open camera on-demand using full device PATH (not index)
            self.get_logger().info(f'Opening camera device {self.device_path}...')
            camera = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set MJPG codec to avoid timeout issues
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            if not camera.isOpened():
                self.get_logger().error('Failed to open camera')
                # Signal done even on failure
                active_msg.data = False
                self.classification_active_pub.publish(active_msg)
                return None

            # Clear buffer and try to read with retries
            frame = None
            for attempt in range(5):
                camera.grab()  # Clear old frame
                ret, frame = camera.read()
                if ret and frame is not None:
                    break
                self.get_logger().warn(f'Camera read attempt {attempt+1} failed, retrying...')
                time.sleep(0.2)

            # Release camera immediately
            camera.release()
            self.get_logger().info('Camera released')

            # Signal that classification is done with camera
            active_msg.data = False
            self.classification_active_pub.publish(active_msg)
            self.get_logger().info('Camera available for classification_camera_publisher')

            if frame is None:
                self.get_logger().error('Failed to capture frame after retries')
                return None
            return frame

        except Exception as e:
            self.get_logger().error(f'Camera capture error: {e}')
            # Signal done even on error
            try:
                active_msg = Bool()
                active_msg.data = False
                self.classification_active_pub.publish(active_msg)
            except:
                pass
            return None

    def _send_to_dashboard(self, frame, result: ClassificationResult) -> bool:
        """Send classification result to dashboard server using authenticated API."""
        try:
            # Use vision client's send_detection which handles authentication
            inference_time = getattr(self, 'inference_time_ms', 100.0)
            return self.vision_client.send_detection(frame, result, inference_time)
        except Exception as e:
            self.get_logger().error(f'Dashboard send error: {e}')
            return False

    def _fullness_to_percent(self, fullness: str) -> int:
        """Convert fullness level to percentage."""
        return {'EMPTY': 0, 'PARTIALLY_FULL': 50, 'FULL': 100}.get(fullness, -1)

    def _ensure_camera_released(self):
        """Ensure camera is released and resume signal is sent (called in finally block)."""
        try:
            active_msg = Bool()
            active_msg.data = False
            # Publish multiple times to ensure delivery
            for _ in range(3):
                self.classification_active_pub.publish(active_msg)
                import time
                time.sleep(0.05)
            self.get_logger().info('Camera release signal sent (triple-published for reliability)')
        except Exception as e:
            self.get_logger().error(f'Failed to send camera release signal: {e}')

    def classify_callback(self, request, response):
        """Handle classification requests with bin verification."""
        self.get_logger().info('='*50)
        self.get_logger().info('Classification request received')

        if not self.initialized:
            response.success = False
            response.fullness_level = 'unknown'
            response.fullness_percent = -1
            response.label = ''
            response.confidence = 0.0
            response.status_summary = 'Classifier not initialized'
            response.sent_to_server = False
            return response

        try:
            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                response.success = False
                response.status_summary = 'Failed to capture frame'
                response.sent_to_server = False
                return response

            self.get_logger().info(f'Captured frame: {frame.shape}')
            self.get_logger().info('Running two-step classification (Florence-2 + SmolVLM)...')

            # Classify with two-step verification (track inference time)
            import time
            inference_start = time.time()
            result = self.vision_client.classify_bin(frame)
            self.inference_time_ms = (time.time() - inference_start) * 1000
            self.get_logger().info(f'Classification took {self.inference_time_ms:.1f}ms')

            # Check if bin was actually found
            if not result.bin_found:
                self.get_logger().warn('='*50)
                self.get_logger().warn('FALSE ARRIVAL: No bin detected!')
                self.get_logger().warn(f'Objects found: {result.objects_detected}')
                self.get_logger().warn('Triggering search resume...')
                self.get_logger().warn('='*50)

                # Trigger resume search
                self._trigger_resume_search()

                # Build failure response
                response.success = False
                response.fullness_level = 'NO_BIN'
                response.fullness_percent = -1
                response.label = 'false_arrival'
                response.confidence = 0.0
                response.status_summary = f'No bin found - resuming search. Objects: {result.objects_detected}'
                response.sent_to_server = False

                # Publish detection with bin_detected=False
                det_msg = BinDetection()
                det_msg.header.stamp = self.get_clock().now().to_msg()
                det_msg.bin_detected = False
                det_msg.fullness_level = 'NO_BIN'
                det_msg.fullness_percent = -1
                det_msg.label = 'false_arrival'
                det_msg.confidence = 0.0
                self.detection_publisher.publish(det_msg)

                return response

            # Bin found - log enhanced results
            self.get_logger().info('='*50)
            self.get_logger().info('BIN VERIFIED - Classification successful')
            self.get_logger().info(f'  Containers: {result.containers_count} ({result.containers_type})')
            self.get_logger().info(f'  Fill Level: {result.fill_level_percent}%')
            self.get_logger().info(f'  Fullness: {result.fullness}')
            self.get_logger().info(f'  Waste Type: {result.waste_type}')
            self.get_logger().info(f'  Action: {result.action}')
            self.get_logger().info(f'  Scene: {result.scene_description}')
            self.get_logger().info('='*50)

            # Build response with enhanced data
            response.success = True
            response.fullness_level = result.fullness
            response.fullness_percent = result.fill_level_percent  # Use actual percentage
            response.label = result.waste_type
            response.confidence = result.confidence
            response.status_summary = result.summary if result.summary else f'{result.fullness} - {result.waste_type}'

            # Publish detection
            det_msg = BinDetection()
            det_msg.header.stamp = self.get_clock().now().to_msg()
            det_msg.bin_detected = True
            det_msg.fullness_level = result.fullness
            det_msg.fullness_percent = result.fill_level_percent
            det_msg.label = result.waste_type
            det_msg.confidence = result.confidence
            self.detection_publisher.publish(det_msg)

            # Send to dashboard
            sent = self._send_to_dashboard(frame, result)
            response.sent_to_server = sent
            if sent:
                self.get_logger().info('Results sent to dashboard')
            else:
                self.get_logger().warn('Failed to send to dashboard')

            # TRIGGER DOCK: Classification complete - tell navigation to return to dock
            self._trigger_dock()

        except Exception as e:
            self.get_logger().error(f'Classification error: {e}')
            import traceback
            traceback.print_exc()

            response.success = False
            response.fullness_level = 'error'
            response.fullness_percent = -1
            response.status_summary = f'Error: {str(e)}'
            response.sent_to_server = False

        finally:
            # ALWAYS ensure camera is released, even if an exception occurred
            self._ensure_camera_released()

        return response

    def _trigger_resume_search(self):
        """Publish command to resume bin search after false arrival."""
        try:
            cmd = RobotCommand()
            cmd.command = 'resume_search'
            self.cmd_publisher.publish(cmd)
            self.get_logger().info('Published resume_search command to /robot_cmd')
        except Exception as e:
            self.get_logger().error(f'Failed to trigger resume search: {e}')

    def _trigger_dock(self):
        """Publish command to dock after successful classification."""
        try:
            self.get_logger().info('='*50)
            self.get_logger().info('Classification complete - triggering DOCK')
            self.get_logger().info('='*50)

            cmd = RobotCommand()
            cmd.command = 'dock_now'
            self.cmd_publisher.publish(cmd)
            self.get_logger().info('Published dock_now command to /robot_cmd')
        except Exception as e:
            self.get_logger().error(f'Failed to trigger dock: {e}')

    def destroy_node(self):
        # Camera is opened/closed on-demand, no cleanup needed
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
