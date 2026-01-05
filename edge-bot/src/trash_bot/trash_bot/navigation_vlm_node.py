#!/usr/bin/env python3
"""
navigation_vlm_node.py
ROS2 Node for SmolVLM2-based visual navigation via llama.cpp server.

This node captures from USB webcam and uses SmolVLM2 (via llama.cpp HTTP API)
to navigate toward detected trash bins.
"""

import sys
import os

# Add this script's directory to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

VENV_SITE_PACKAGES = '/home/g3ubuntu/ROS/embedded-system-final/edge/venv/lib/python3.12/site-packages'
if VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

import cv2
import numpy as np
import logging
from enum import Enum
from typing import Optional
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
import json

from trash_bot.msg import RobotCommand, BinDetection
from trash_bot.srv import ClassifyBin
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from irobot_create_msgs.action import Undock

# Import llama vision client
try:
    from trash_bot.llama_vision_client import LlamaVisionClient, NavigationCommand, NavigationResult
except ImportError:
    from llama_vision_client import LlamaVisionClient, NavigationCommand, NavigationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    APPROACHING = "approaching"
    ARRIVED = "arrived"
    COMPLETED = "completed"


class NavigationVLMNode(Node):
    """ROS2 node for visual navigation using SmolVLM2 via llama.cpp."""

    def __init__(self):
        super().__init__('navigation_vlm_node')

        # Parameters
        self.declare_parameter('llama_server_url', 'http://192.168.0.81:5000')
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('camera_topic', '/oak/rgb/image_raw')  # OAK-D camera topic
        self.declare_parameter('use_ros_camera', False)  # Use USB camera by default
        self.declare_parameter('inference_rate', 0.5)  # Hz - inference every 2 seconds (fast with server GPU)
        self.declare_parameter('forward_speed', 1.0)  # Speed factor 0-1 (1.0 = max_linear_speed)
        self.declare_parameter('turn_speed', 0.5)  # Speed factor 0-1 (0.5 = 50% of max_angular_speed)
        self.declare_parameter('search_timeout', 300.0)  # 5 minutes
        self.declare_parameter('use_florence2', True)  # Use Florence-2 for object detection

        self.llama_server_url = self.get_parameter('llama_server_url').get_parameter_value().string_value
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.use_ros_camera = self.get_parameter('use_ros_camera').get_parameter_value().bool_value
        self.inference_rate = self.get_parameter('inference_rate').get_parameter_value().double_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.turn_speed = self.get_parameter('turn_speed').get_parameter_value().double_value
        self.search_timeout = self.get_parameter('search_timeout').get_parameter_value().double_value
        self.use_florence2 = self.get_parameter('use_florence2').get_parameter_value().bool_value

        self.callback_group = ReentrantCallbackGroup()

        # State
        self.state = SearchState.IDLE
        self.state_lock = threading.Lock()
        self.search_start_time = 0.0
        self.consecutive_not_found = 0
        self.search_rotation_count = 0
        self.inference_in_progress = False  # Prevent concurrent inferences
        self.latest_result = None  # Store latest inference result for camera publisher
        self.command_cooldown_until = 0.0  # Don't send new commands until this time
        self.last_command_time = 0.0  # Track when last command was sent
        self.watchdog_last_activity = 0.0  # Watchdog: last time we made progress
        self.watchdog_timeout = 60.0  # Reset if no progress in 60 seconds

        # BIN LOCK: Prevent oscillation between forward and search
        # When bin is detected, maintain approach mode for several frames
        self.bin_lock_until = 0.0  # Timestamp until which we stay in approach mode
        self.bin_lock_duration = 6.0  # Stay in approach mode for 6 seconds after last detection (covers 3 inference cycles)
        self.min_consecutive_miss = 5  # Require 5 consecutive misses before switching to search

        # DETECTION HISTORY: Smooth detection across multiple frames
        # Prevents single-frame false negatives from causing oscillation
        self.detection_history = []  # List of recent detection results (1.0 = detected, 0.0 = not)
        self.detection_history_size = 5  # Track last 5 inferences
        self.smoothed_detection_threshold = 0.4  # At least 2 of 5 detections to stay in approach

        # RECOVERY SEARCH: Small rotations to re-find bin before full search
        self.last_bin_direction = 'center'  # Track last known bin position
        self.recovery_mode = False  # Whether in recovery search mode
        self.recovery_attempts = 0  # Current recovery attempt count
        self.max_recovery_attempts = 3  # Max small rotations before full search
        self.recovery_rotation_angle = 0.3  # ~17 degrees for small search

        # PROXIMITY ARRIVAL: Track last known bin size to detect arrival when too close
        # When robot gets very close, Florence-2 can't recognize the bin anymore
        # If we were approaching ANY bin and lose detection, we've probably arrived
        # Classification camera will verify if it's actually a bin
        self.last_bin_size = None  # Track: 'small', 'medium', 'large'
        self.last_bin_detected_time = 0.0  # When we last saw the bin
        self.proximity_arrival_timeout = 5.0  # Extended to 5s - if we had any bin within 5s and lost it, we arrived

        # SETTLE TIME: Wait for robot to stop before running inference
        # This prevents blurry/motion-affected images
        self.command_active_until = 0.0  # When current command ends (robot stops)
        self.settle_time = 1.0  # Wait 1 second after movement stops before inference (better image stability)

        # AUTO-CLASSIFICATION: Trigger classification periodically when searching
        # This catches cases where Florence-2 doesn't recognize the bin
        self.auto_classify_interval = 15.0  # Try classification every 15s of searching
        self.auto_classify_min_search_time = 10.0  # Wait at least 10s before first auto-classify
        self.last_auto_classify_time = 0.0  # When we last triggered auto-classification
        self.auto_classify_enabled = True  # Enable/disable auto-classification

        # DUAL-CAMERA CONFIRMATION: Both cameras must agree before ARRIVED
        # Front camera (Florence-2): detects large bin
        # Classification camera (SmolVLM): confirms it's actually a bin
        self.dual_confirm_enabled = True  # Enable dual-camera confirmation
        self.classification_confirmed = False  # Has classification camera confirmed bin?
        self.classification_confirm_time = 0.0  # When classification confirmed
        self.classification_confirm_timeout = 15.0  # Confirmation valid for 15 seconds
        self.continuous_classify_interval = 1.5  # Check classification every 1.5s during approach (faster confirmation)
        self.last_continuous_classify_time = 0.0  # Last continuous classification check
        self.classification_in_progress = False  # Prevent concurrent classifications
        self.front_camera_arrived = False  # Front camera says ARRIVED (large bin detected)
        self.front_camera_arrived_time = 0.0  # When front camera triggered ARRIVED
        self.max_approach_without_confirm = 30.0  # Max time to approach without classification confirm

        # Camera (USB fallback)
        self.camera = None

        # ROS camera (OAK-D)
        self.cv_bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_timestamp = 0.0

        # Publishers
        self.cmd_publisher = self.create_publisher(RobotCommand, '/robot_cmd', 10)
        self.detection_publisher = self.create_publisher(BinDetection, '/bin_detected', 10)
        self.state_publisher = self.create_publisher(String, '/navigation_state', 10)
        self.search_complete_publisher = self.create_publisher(Bool, '/search_complete', 10)

        # Service
        self.search_service = self.create_service(
            SetBool, '/find_bin', self.find_bin_callback,
            callback_group=self.callback_group
        )

        # Dock now service - stops search and saves command history
        from std_srvs.srv import Trigger
        self.dock_service = self.create_service(
            Trigger, '/dock_now', self.dock_now_callback,
            callback_group=self.callback_group
        )

        # Service client for classification (dual-camera confirmation)
        self.classify_client = self.create_client(
            ClassifyBin, '/classify_bin',
            callback_group=self.callback_group
        )

        # Action client for undocking
        self.undock_client = ActionClient(
            self, Undock, '/_do_not_use/undock',
            callback_group=self.callback_group
        )

        # Command history for retracing path back to dock
        self.command_history = []  # [(command, speed, duration, timestamp), ...]
        self.command_history_file = os.path.expanduser('~/ROS/logs/command_history.json')
        os.makedirs(os.path.dirname(self.command_history_file), exist_ok=True)

        # Flag to prevent resume after dock_now
        self.docking_requested = False

        # Clear any old command history on startup
        self._clear_command_history()

        # Subscribe to robot commands for resume_search from classifier
        self.cmd_subscription = self.create_subscription(
            RobotCommand,
            '/robot_cmd',
            self._robot_cmd_callback,
            10,
            callback_group=self.callback_group
        )

        # ROS camera subscription (OAK-D)
        if self.use_ros_camera:
            self.image_subscription = self.create_subscription(
                Image,
                self.camera_topic,
                self._image_callback,
                10,
                callback_group=self.callback_group
            )
            self.get_logger().info(f'Subscribed to camera topic: {self.camera_topic}')

        # Inference timer
        inference_period = 1.0 / self.inference_rate if self.inference_rate > 0 else 10.0
        self.inference_timer = self.create_timer(
            inference_period, self.inference_callback,
            callback_group=self.callback_group
        )

        # Initialize llama client
        self.vision_client = None
        self.client_ready = False

        self.get_logger().info('='*50)
        self.get_logger().info('Navigation VLM Node (llama.cpp)')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Llama server: {self.llama_server_url}')
        self.get_logger().info(f'Camera device: {self.camera_device}')
        self.get_logger().info(f'Inference rate: {self.inference_rate} Hz')
        self.get_logger().info(f'Dual-camera confirm: {self.dual_confirm_enabled}')
        if self.dual_confirm_enabled:
            self.get_logger().info(f'  - Continuous classify interval: {self.continuous_classify_interval}s')
            self.get_logger().info(f'  - Confirm timeout: {self.classification_confirm_timeout}s')

        # Initialize in background
        threading.Thread(target=self._initialize, daemon=True).start()

    def _initialize(self):
        """Initialize vision client (camera opened on-demand)."""
        try:
            # Initialize llama client first (doesn't require camera)
            self.get_logger().info('Initializing llama.cpp vision client...')
            self.vision_client = LlamaVisionClient(
                server_url=self.llama_server_url,
                timeout=30.0  # Reduced timeout - server GPU is much faster
            )

            # Mark as ready - camera will be opened on-demand
            self.client_ready = True
            self.get_logger().info('='*50)
            self.get_logger().info('NAVIGATION NODE READY')
            if self.use_ros_camera:
                self.get_logger().info(f'Using ROS camera topic: {self.camera_topic}')
            else:
                self.get_logger().info(f'Using USB camera device: {self.camera_device}')
            self.get_logger().info('='*50)

        except Exception as e:
            self.get_logger().error(f'Initialization error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _image_callback(self, msg: Image):
        """Callback for ROS camera topic (OAK-D)."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
                self.frame_timestamp = time.time()
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def _open_camera(self):
        """Open camera if not already open."""
        if self.camera is not None and self.camera.isOpened():
            return True

        self.get_logger().info(f'Opening camera device {self.camera_device}...')
        # Use V4L2 backend directly to avoid GStreamer issues
        self.camera = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.camera.isOpened():
            self.get_logger().info('Camera opened successfully')
            return True
        else:
            self.get_logger().error('Failed to open camera')
            self.camera = None
            return False

    def _capture_frame(self):
        """Capture frame from camera (ROS topic or USB fallback)."""
        if self.use_ros_camera:
            # Get frame from ROS topic (OAK-D camera)
            with self.frame_lock:
                if self.latest_frame is not None:
                    # Check if frame is recent (within 5 seconds)
                    age = time.time() - self.frame_timestamp
                    if age < 5.0:
                        return self.latest_frame.copy()
                    else:
                        self.get_logger().warn(f'ROS camera frame is stale ({age:.1f}s old)')
                        return None
                else:
                    self.get_logger().warn('No frame from ROS camera topic yet')
                    return None
        else:
            # USB camera fallback
            if not self._open_camera():
                return None

            # Clear buffer
            for _ in range(3):
                self.camera.grab()

            ret, frame = self.camera.read()
            return frame if ret else None

    def _trigger_continuous_classification(self):
        """
        Trigger classification asynchronously during approach.

        This runs the classification camera to verify bin presence while
        the front camera is navigating. Called periodically during APPROACHING state.
        """
        if self.classification_in_progress:
            self.get_logger().debug('Classification already in progress, skipping')
            return

        if not self.classify_client.service_is_ready():
            self.get_logger().warn('ClassifyBin service not ready')
            return

        self.classification_in_progress = True
        self.get_logger().info('='*50)
        self.get_logger().info('DUAL-CONFIRM: Triggering classification check')
        self.get_logger().info('='*50)

        # Create async request
        request = ClassifyBin.Request()
        future = self.classify_client.call_async(request)
        future.add_done_callback(self._classification_done_callback)

    def _classification_done_callback(self, future):
        """Handle classification result from async service call."""
        self.classification_in_progress = False
        now = time.time()

        try:
            response = future.result()

            if response.success:
                # Classification camera found a bin!
                self.get_logger().info('='*50)
                self.get_logger().info('DUAL-CONFIRM: Classification camera CONFIRMED BIN!')
                self.get_logger().info(f'  Fullness: {response.fullness_level}')
                self.get_logger().info(f'  Waste type: {response.label}')
                self.get_logger().info(f'  Confidence: {response.confidence:.2f}')
                self.get_logger().info('='*50)

                self.classification_confirmed = True
                self.classification_confirm_time = now

                # Check if front camera also triggered ARRIVED
                if self.front_camera_arrived:
                    front_age = now - self.front_camera_arrived_time
                    if front_age < self.classification_confirm_timeout:
                        self.get_logger().info('='*50)
                        self.get_logger().info('DUAL-CONFIRM SUCCESS: Both cameras agree!')
                        self.get_logger().info('Transitioning to ARRIVED state')
                        self.get_logger().info('='*50)
                        self._publish_command('stop')
                        self._stop_search(success=True)
                    else:
                        self.get_logger().info(f'Front camera ARRIVED was too old ({front_age:.1f}s ago)')
            else:
                # Classification camera did NOT find a bin
                self.get_logger().warn('='*50)
                self.get_logger().warn('DUAL-CONFIRM: Classification camera found NO BIN')
                self.get_logger().warn(f'  Status: {response.status_summary}')
                self.get_logger().warn('='*50)

                # Invalidate any previous confirmation
                self.classification_confirmed = False

                # If front camera had triggered ARRIVED, this was a false positive
                if self.front_camera_arrived:
                    self.get_logger().warn('Front camera ARRIVED was FALSE POSITIVE - continuing search')
                    self.front_camera_arrived = False
                    self.front_camera_arrived_time = 0.0

        except Exception as e:
            self.get_logger().error(f'Classification callback error: {e}')

    def _is_classification_valid(self) -> bool:
        """Check if classification confirmation is still valid (not expired)."""
        if not self.classification_confirmed:
            return False
        age = time.time() - self.classification_confirm_time
        return age < self.classification_confirm_timeout

    def inference_callback(self):
        """Run inference at regular intervals.

        KEY FIX: Inference runs continuously, but commands are only sent
        after previous command completes (cooldown). This prevents the robot
        from appearing stuck while still processing camera frames.
        """
        with self.state_lock:
            current_state = self.state

        if current_state not in [SearchState.SEARCHING, SearchState.APPROACHING]:
            return

        if not self.client_ready:
            return

        # Prevent concurrent inferences
        if self.inference_in_progress:
            return

        self.inference_in_progress = True
        now = time.time()

        # SETTLE TIME: Wait for robot to stop moving before inference
        # This ensures we get a stable, non-blurry image
        if self.command_active_until > 0:
            if now < self.command_active_until:
                # Command is still executing - robot is moving
                remaining = self.command_active_until - now
                self.get_logger().debug(f'Robot moving, waiting for stop ({remaining:.1f}s remaining)')
                self.inference_in_progress = False
                return
            elif now < self.command_active_until + self.settle_time:
                # Command just finished - wait for settle
                settle_remaining = (self.command_active_until + self.settle_time) - now
                self.get_logger().debug(f'Settling after stop ({settle_remaining:.2f}s remaining)')
                self.inference_in_progress = False
                return
            # else: settle time passed, proceed with inference

        # Check search timeout
        elapsed = now - self.search_start_time
        if elapsed > self.search_timeout:
            self.get_logger().warn(f'Search timeout after {elapsed:.1f}s')
            self._stop_search(success=False)
            self.inference_in_progress = False
            return

        # Watchdog: Check if we're stuck (no progress for 60s)
        if self.watchdog_last_activity > 0 and (now - self.watchdog_last_activity) > self.watchdog_timeout:
            self.get_logger().warn(f'WATCHDOG: No progress for {self.watchdog_timeout}s, resetting state')
            self._publish_command('stop')
            self.command_cooldown_until = 0.0
            self.watchdog_last_activity = now

        # Capture frame
        frame = self._capture_frame()
        if frame is None:
            self.get_logger().warn('No frame available')
            self.inference_in_progress = False
            return

        try:
            # Use higher image quality during APPROACHING for better detection accuracy
            image_quality = 80 if current_state == SearchState.APPROACHING else 65

            if self.use_florence2:
                self.get_logger().debug(f'Running Florence-2 + SmolVLM inference (elapsed: {elapsed:.1f}s, quality={image_quality})...')
                result = self.vision_client.get_navigation_with_detection(frame, quality=image_quality)
            else:
                self.get_logger().debug(f'Running SmolVLM2 inference (elapsed: {elapsed:.1f}s)...')
                result = self.vision_client.get_navigation_command(frame)

            bbox_info = f', bboxes={len(result.bboxes)}' if result.bboxes else ''
            self.get_logger().info(
                f'INFERENCE: cmd={result.command.value}, detected={result.bin_detected}, '
                f'pos={result.bin_position}, size={result.bin_size}{bbox_info}'
            )

            # Store latest result for camera publisher
            self.latest_result = result

            # KEY FIX: Only send commands if cooldown has expired
            # This allows inference to run continuously while robot executes commands
            if now >= self.command_cooldown_until:
                self._process_result(result)
                self.watchdog_last_activity = now  # Update watchdog on successful command
            else:
                remaining = self.command_cooldown_until - now
                self.get_logger().debug(f'Cooldown active, skipping command ({remaining:.1f}s remaining)')
                # Still publish detection for dashboard even if not sending command
                self._publish_detection(result)

            # CONTINUOUS CLASSIFICATION: Trigger classification periodically during approach
            # This allows the top camera (classification) to pre-confirm bin presence
            # before front camera triggers ARRIVED, speeding up the confirmation process
            if self.dual_confirm_enabled and current_state == SearchState.APPROACHING:
                time_since_last_classify = now - self.last_continuous_classify_time
                if time_since_last_classify > self.continuous_classify_interval:
                    self.get_logger().info(f'CONTINUOUS CLASSIFY: {time_since_last_classify:.1f}s since last check')
                    self.last_continuous_classify_time = now
                    self._trigger_continuous_classification()

        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            import traceback
            self.get_logger().debug(traceback.format_exc())
        finally:
            self.inference_in_progress = False

    def _process_result(self, result: NavigationResult):
        """Process navigation result with bin lock, detection smoothing, and recovery search."""
        now = time.time()

        # DETECTION HISTORY: Add result to history buffer for smoothing
        self.detection_history.append(1.0 if result.bin_detected else 0.0)
        if len(self.detection_history) > self.detection_history_size:
            self.detection_history.pop(0)

        # Calculate smoothed detection score (average of recent detections)
        smoothed_score = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0.0

        if result.bin_detected:
            # BIN DETECTED - extend bin lock, reset miss counter, exit recovery mode
            self.consecutive_not_found = 0
            self.bin_lock_until = now + self.bin_lock_duration
            self.recovery_mode = False
            self.recovery_attempts = 0

            # Track bin size and direction for proximity arrival and recovery search
            self.last_bin_size = result.bin_size
            self.last_bin_detected_time = now
            if result.bin_position:
                self.last_bin_direction = result.bin_position

            self.get_logger().info(f'BIN DETECTED: size={result.bin_size}, pos={result.bin_position}, smoothed={smoothed_score:.2f}')

            if result.command == NavigationCommand.ARRIVED:
                # DUAL-CAMERA CONFIRMATION: Front camera says ARRIVED
                # But we need classification camera to also confirm before stopping
                if self.dual_confirm_enabled:
                    self.get_logger().info('='*50)
                    self.get_logger().info('FRONT CAMERA: ARRIVED signal (large bin detected)')
                    self.front_camera_arrived = True
                    self.front_camera_arrived_time = now

                    # Check if classification camera has already confirmed
                    if self._is_classification_valid():
                        self.get_logger().info('Classification camera already confirmed!')
                        self.get_logger().info('DUAL-CONFIRM SUCCESS: Both cameras agree!')
                        self.get_logger().info('='*50)
                        self._publish_command('stop')
                        self._stop_search(success=True)
                        return
                    else:
                        # Need classification confirmation - DON'T STOP, keep approaching slowly
                        # Classification camera may not see bin yet if we're too far
                        self.get_logger().info('Classification camera not yet confirmed...')
                        self.get_logger().info('KEEP APPROACHING SLOWLY while triggering classification')
                        self.get_logger().info('='*50)
                        # Move forward slowly to get closer for classification camera
                        self._publish_command('forward', self.forward_speed * 0.5, 1.5)
                        self._trigger_continuous_classification()
                        # Don't return - let robot keep approaching
                        return
                else:
                    # Dual-confirm disabled - original behavior
                    self.get_logger().info('=== ARRIVED at bin! ===')
                    self._publish_command('stop')
                    self._stop_search(success=True)
                    return

            elif result.command == NavigationCommand.FORWARD:
                self._transition_state(SearchState.APPROACHING)
                # STANDARDIZED: All movements are 2 seconds
                self._publish_command('forward', self.forward_speed, 2.0)

            elif result.command == NavigationCommand.LEFT:
                self._transition_state(SearchState.APPROACHING)
                # STANDARDIZED: All movements are 2 seconds
                self._publish_command('left', self.turn_speed, 2.0)

            elif result.command == NavigationCommand.RIGHT:
                self._transition_state(SearchState.APPROACHING)
                # STANDARDIZED: All movements are 2 seconds
                self._publish_command('right', self.turn_speed, 2.0)

            elif result.command == NavigationCommand.BACKWARD:
                # Obstacle detected - back up to avoid collision
                self.get_logger().warn('OBSTACLE - Backing up!')
                self._publish_command('backward', self.forward_speed * 0.5, 2.0)

        else:
            # NO BIN DETECTED - check for proximity arrival first
            self.consecutive_not_found += 1

            # AUTO-CLASSIFICATION CHECK:
            # If we've been searching for a while without finding a bin,
            # trigger classification to check if we're actually at a bin
            # (Florence-2 might not recognize it)
            if self.auto_classify_enabled and self.state == SearchState.SEARCHING:
                search_duration = now - self.search_start_time
                time_since_last_classify = now - self.last_auto_classify_time

                if search_duration > self.auto_classify_min_search_time and time_since_last_classify > self.auto_classify_interval:
                    self.get_logger().info('='*50)
                    self.get_logger().info(f'AUTO-CLASSIFICATION: Searching for {search_duration:.0f}s without finding bin')
                    self.get_logger().info('Triggering classification to verify if at bin')
                    self.get_logger().info('='*50)
                    self.last_auto_classify_time = now

                    # Don't stop the search - just trigger classification async
                    # Set front_camera_arrived so callback knows to complete if bin found
                    self.front_camera_arrived = True
                    self.front_camera_arrived_time = now
                    self._trigger_continuous_classification()
                    # Continue searching while classification runs
                    # Callback will handle stop_search if bin is found

            # PROXIMITY ARRIVAL CHECK:
            # If we recently saw ANY bin and now can't detect anything,
            # we're probably too close - the bin fills the camera and isn't recognized
            # This is AGGRESSIVE - better to arrive falsely (classification will verify) than miss the bin
            time_since_last_bin = now - self.last_bin_detected_time if self.last_bin_detected_time > 0 else float('inf')

            if (self.last_bin_size is not None and  # Any bin size triggers this now
                time_since_last_bin < self.proximity_arrival_timeout and
                self.state == SearchState.APPROACHING):

                self.get_logger().info('='*50)
                self.get_logger().info(
                    f'PROXIMITY ARRIVAL: Lost {self.last_bin_size} bin after {time_since_last_bin:.1f}s'
                )

                # DUAL-CAMERA CONFIRMATION for proximity arrival
                if self.dual_confirm_enabled:
                    self.front_camera_arrived = True
                    self.front_camera_arrived_time = now

                    if self._is_classification_valid():
                        self.get_logger().info('Classification camera already confirmed!')
                        self.get_logger().info('DUAL-CONFIRM SUCCESS: Both cameras agree!')
                        self.get_logger().info('='*50)
                        self._publish_command('stop')
                        self._stop_search(success=True)
                        return
                    else:
                        # Don't stop - keep creeping forward for classification camera
                        self.get_logger().info('Classification not yet confirmed - keep approaching slowly')
                        self.get_logger().info('='*50)
                        self._publish_command('forward', self.forward_speed * 0.3, 1.0)
                        self._trigger_continuous_classification()
                        return
                else:
                    self.get_logger().info('Assuming arrived - classification will verify')
                    self.get_logger().info('='*50)
                    self._publish_command('stop')
                    self._stop_search(success=True)
                    return

            # BIN LOCK CHECK: If we recently saw the bin, keep going forward
            # Also check smoothed detection score for extra stability
            if now < self.bin_lock_until and self.consecutive_not_found < self.min_consecutive_miss:
                remaining_lock = self.bin_lock_until - now
                self.get_logger().info(
                    f'BIN LOCK ACTIVE: Continuing forward despite no detection '
                    f'(miss {self.consecutive_not_found}/{self.min_consecutive_miss}, lock {remaining_lock:.1f}s, smoothed={smoothed_score:.2f})'
                )
                # Keep approaching - the bin is probably still there
                # STANDARDIZED: All movements are 2 seconds
                self._transition_state(SearchState.APPROACHING)
                self._publish_command('forward', self.forward_speed, 2.0)
            elif smoothed_score >= self.smoothed_detection_threshold:
                # Smoothed detection says bin is still there - keep approaching
                self.get_logger().info(
                    f'SMOOTHED DETECTION: Continuing approach (smoothed={smoothed_score:.2f} >= {self.smoothed_detection_threshold})'
                )
                self._transition_state(SearchState.APPROACHING)
                self._publish_command('forward', self.forward_speed, 2.0)
            else:
                # Bin lock expired AND smoothed score low - try recovery search first
                if self.bin_lock_until > 0:
                    self.get_logger().info(
                        f'BIN LOCK EXPIRED: Entering recovery search mode '
                        f'(miss {self.consecutive_not_found}, smoothed={smoothed_score:.2f}, last_dir={self.last_bin_direction})'
                    )
                    self.bin_lock_until = 0.0  # Clear the lock
                    self.recovery_mode = True
                    self.recovery_attempts = 0

                # RECOVERY SEARCH: Try small rotations before full search
                if self.recovery_mode and self.recovery_attempts < self.max_recovery_attempts:
                    self.recovery_attempts += 1
                    # Rotate in last known direction first, then alternate
                    if self.recovery_attempts == 1 and self.last_bin_direction == 'left':
                        direction = 'left'
                    elif self.recovery_attempts == 1 and self.last_bin_direction == 'right':
                        direction = 'right'
                    else:
                        direction = 'left' if self.recovery_attempts % 2 == 0 else 'right'

                    self.get_logger().info(
                        f'RECOVERY SEARCH: Small rotation {direction} (attempt {self.recovery_attempts}/{self.max_recovery_attempts})'
                    )
                    # Small rotation to re-find bin
                    self._publish_command(direction, self.turn_speed * 0.5, 1.0)
                    return  # Don't transition to full search yet

                # Recovery failed - switch to full search
                self.recovery_mode = False
                self.recovery_attempts = 0
                self._transition_state(SearchState.SEARCHING)

                # Increment search counter for forward movement
                self.search_rotation_count += 1

                # Every 4th command, move forward to explore (not just rotate in place)
                if self.search_rotation_count % 4 == 0:
                    self.get_logger().info('Searching: moving FORWARD to explore new area')
                    self._publish_command('forward', self.forward_speed, 2.0)
                elif result.command == NavigationCommand.BACKWARD:
                    # Obstacle detected - back up to avoid collision
                    # STANDARDIZED: All movements are 2 seconds
                    self.get_logger().warn('OBSTACLE DETECTED - Backing up!')
                    self._publish_command('backward', self.forward_speed * 0.5, 2.0)
                elif result.command == NavigationCommand.SEARCH_LEFT:
                    self.get_logger().info('Searching: rotating LEFT as directed by server')
                    self._publish_command('left', self.turn_speed, 2.0)
                elif result.command == NavigationCommand.SEARCH_RIGHT:
                    self.get_logger().info('Searching: rotating RIGHT as directed by server')
                    self._publish_command('right', self.turn_speed, 2.0)
                else:
                    # Fallback for NOT_FOUND or SEARCHING (legacy)
                    self.get_logger().info('Searching: using fallback rotation pattern')
                    self._execute_search_rotation()

        self._publish_detection(result)

    def _execute_search_rotation(self):
        """Execute search rotation."""
        self.search_rotation_count += 1

        # STANDARDIZED: All movements are 2 seconds
        if self.search_rotation_count % 4 == 0:
            self._publish_command('forward', self.forward_speed, 2.0)
        else:
            direction = 'left' if self.search_rotation_count % 2 == 0 else 'right'
            self._publish_command(direction, self.turn_speed, 2.0)

    def _transition_state(self, new_state: SearchState):
        """Transition state."""
        with self.state_lock:
            if self.state != new_state:
                self.get_logger().info(f'State: {self.state.value} -> {new_state.value}')
                self.state = new_state
                msg = String()
                msg.data = new_state.value
                self.state_publisher.publish(msg)

    def _publish_command(self, command: str, speed: float = 0.3, duration: float = 0.0):
        """Publish robot command and set cooldown to wait for completion."""
        msg = RobotCommand()
        msg.command = command
        msg.speed = speed
        msg.duration = duration
        self.cmd_publisher.publish(msg)

        now = time.time()

        # Record command to history for retracing path back to dock
        self._record_command(command, speed, duration)

        # Track when this command will end (for settle time)
        if duration > 0:
            self.command_active_until = now + duration
            # Cooldown = command duration + settle time
            self.command_cooldown_until = now + duration + self.settle_time
            self.get_logger().info(f'>>> COMMAND: {command} speed={speed:.1f} duration={duration:.1f}s')
        else:
            self.command_active_until = now  # Immediate commands (like stop)
            self.command_cooldown_until = now + self.settle_time
            self.get_logger().info(f'>>> COMMAND: {command} (immediate)')

    def _robot_cmd_callback(self, msg: RobotCommand):
        """Handle robot commands, including resume_search and dock_now from classifier/dock.sh."""
        if msg.command == 'resume_search':
            # Don't resume if docking was requested
            if self.docking_requested:
                self.get_logger().warn('RESUME SEARCH ignored - docking in progress')
                return

            self.get_logger().info('='*50)
            self.get_logger().info('RESUME SEARCH: False arrival detected, restarting search')
            self.get_logger().info('='*50)

            with self.state_lock:
                # Only resume if we were in COMPLETED or ARRIVED state
                if self.state in [SearchState.COMPLETED, SearchState.ARRIVED]:
                    self.state = SearchState.SEARCHING

            # Reset bin tracking for fresh search
            self.consecutive_not_found = 0
            self.search_rotation_count = 0
            self.bin_lock_until = 0.0
            self.last_bin_size = None
            self.last_bin_detected_time = 0.0
            self.command_cooldown_until = 0.0
            self.command_active_until = 0.0
            self.watchdog_last_activity = time.time()

            # Reset detection history and recovery search
            self.detection_history = []
            self.last_bin_direction = 'center'
            self.recovery_mode = False
            self.recovery_attempts = 0

            # Reset dual-camera confirmation state
            self.classification_confirmed = False
            self.classification_confirm_time = 0.0
            self.last_continuous_classify_time = 0.0
            self.classification_in_progress = False
            self.front_camera_arrived = False
            self.front_camera_arrived_time = 0.0

            # Restart inference timer if it was cancelled
            if self.inference_timer is not None:
                self.inference_timer.reset()
                self.get_logger().info('Inference timer restarted for resumed search')

        elif msg.command == 'dock_now':
            # Call the service handler directly
            self._do_dock_now()

    def _do_dock_now(self):
        """Stop search and save command history for docking."""
        # Check if already docking (prevent double-call race)
        if self.docking_requested:
            self.get_logger().debug('dock_now called but already docking - ignoring')
            return

        self.get_logger().info('='*50)
        self.get_logger().info('DOCK NOW: Stopping search and saving command history')
        self.get_logger().info('='*50)

        # Set flag FIRST to prevent resume_search and dual-confirm race
        self.docking_requested = True

        # Stop the search - this is SUCCESS because classification completed!
        self._stop_search(success=True)
        self._save_command_history()  # Save history for retrace

        self.get_logger().info(f'Search stopped. Saved {len(self.command_history)} commands.')
        self.get_logger().info('dock.sh can now retrace and dock.')

    def dock_now_callback(self, request, response):
        """Handle /dock_now service call to stop search and prepare for docking."""
        self._do_dock_now()
        response.success = True
        response.message = f'Stopped search, saved {len(self.command_history)} commands for retrace'
        return response

    def _publish_detection(self, result: NavigationResult):
        """Publish detection message with Florence-2 bounding boxes."""
        msg = BinDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.bin_detected = result.bin_detected
        msg.label = result.bin_size or 'unknown'
        msg.confidence = result.confidence
        msg.distance = {'large': 0.3, 'medium': 0.8, 'small': 1.5}.get(result.bin_size, 2.0)
        msg.angle = {'left': 0.4, 'right': -0.4}.get(result.bin_position, 0.0)

        # Include Florence-2 bounding boxes
        if result.bboxes:
            msg.num_detections = len(result.bboxes)
            msg.bbox_labels = [b.get('label', 'object') for b in result.bboxes]
            msg.bbox_confidences = [float(b.get('confidence', 0.0)) for b in result.bboxes]
            msg.bbox_xs = [int(b.get('x', 0)) for b in result.bboxes]
            msg.bbox_ys = [int(b.get('y', 0)) for b in result.bboxes]
            msg.bbox_widths = [int(b.get('w', 0)) for b in result.bboxes]
            msg.bbox_heights = [int(b.get('h', 0)) for b in result.bboxes]

            # Set primary bbox to first detection
            if len(result.bboxes) > 0:
                msg.bbox_x = int(result.bboxes[0].get('x', 0))
                msg.bbox_y = int(result.bboxes[0].get('y', 0))
                msg.bbox_width = int(result.bboxes[0].get('w', 0))
                msg.bbox_height = int(result.bboxes[0].get('h', 0))
        else:
            msg.num_detections = 0

        self.detection_publisher.publish(msg)

    def _stop_search(self, success: bool):
        """Stop search and release camera for classifier."""
        # Check if already stopped (prevent duplicate completion messages)
        with self.state_lock:
            if self.state == SearchState.COMPLETED or self.state == SearchState.IDLE:
                # Already stopped - don't publish another completion message
                self.get_logger().debug(f'_stop_search called but already in {self.state.name} state')
                return
            self.state = SearchState.COMPLETED if success else SearchState.IDLE

        self._publish_command('stop')

        # Cancel inference timer to save CPU when not searching
        if self.inference_timer is not None:
            self.inference_timer.cancel()
            self.get_logger().debug('Inference timer cancelled')

        # Release camera so classifier can use it
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.get_logger().info('Camera released for classifier')

        msg = Bool()
        msg.data = success
        self.search_complete_publisher.publish(msg)
        self.get_logger().info(f'Search {"completed" if success else "failed"}')

        # Save command history when search completes successfully
        if success:
            self._save_command_history()

    def _record_command(self, command: str, speed: float, duration: float):
        """Record a command to history for retracing path back to dock."""
        # Only record movement commands (not stop)
        if command in ['forward', 'backward', 'left', 'right']:
            entry = {
                'command': command,
                'speed': speed,
                'duration': duration,
                'timestamp': time.time()
            }
            self.command_history.append(entry)
            self.get_logger().debug(f'Recorded command: {command} speed={speed} duration={duration}')

    def _save_command_history(self):
        """Save command history to JSON file for dock.sh to use."""
        try:
            data = {
                'commands': self.command_history,
                'total_commands': len(self.command_history),
                'saved_at': time.time()
            }
            with open(self.command_history_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'Saved {len(self.command_history)} commands to {self.command_history_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to save command history: {e}')

    def _clear_command_history(self):
        """Clear command history (on new search start)."""
        self.command_history = []
        try:
            if os.path.exists(self.command_history_file):
                os.remove(self.command_history_file)
                self.get_logger().debug('Cleared old command history file')
        except Exception as e:
            self.get_logger().warn(f'Could not clear command history file: {e}')

    def _undock_robot(self) -> bool:
        """Undock the robot before starting search. Returns True if successful.

        Uses a separate thread to avoid blocking the main executor.
        """
        self.get_logger().info('='*50)
        self.get_logger().info('UNDOCKING: Robot must undock before searching')
        self.get_logger().info('='*50)

        # Wait for undock action server (with timeout)
        server_ready = self.undock_client.wait_for_server(timeout_sec=5.0)
        if not server_ready:
            self.get_logger().warn('Undock action server not available - robot may already be undocked')
            return True  # Continue anyway, robot might already be undocked

        # Send undock goal using synchronous approach in separate thread
        goal_msg = Undock.Goal()
        self.get_logger().info('Sending undock goal...')

        undock_result = {'success': False, 'done': False}

        def undock_thread():
            try:
                # Use synchronous call pattern with executor
                future = self.undock_client.send_goal_async(goal_msg)

                # Wait for goal acceptance with timeout
                start = time.time()
                while not future.done() and (time.time() - start) < 10.0:
                    time.sleep(0.1)

                if not future.done():
                    self.get_logger().warn('Undock goal submission timed out')
                    undock_result['done'] = True
                    return

                goal_handle = future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn('Undock goal rejected - robot may already be undocked')
                    undock_result['success'] = True
                    undock_result['done'] = True
                    return

                self.get_logger().info('Undock goal accepted, waiting for result...')

                # Wait for result with timeout
                result_future = goal_handle.get_result_async()
                start = time.time()
                while not result_future.done() and (time.time() - start) < 60.0:
                    time.sleep(0.5)

                if result_future.done():
                    result = result_future.result()
                    self.get_logger().info(f'Undock completed with status: {result.status}')
                    undock_result['success'] = True
                else:
                    self.get_logger().warn('Undock result timed out - assuming success')
                    undock_result['success'] = True

            except Exception as e:
                self.get_logger().error(f'Undock thread error: {e}')
                undock_result['success'] = True  # Continue anyway
            finally:
                undock_result['done'] = True

        # Run undock in separate thread to avoid blocking executor
        thread = threading.Thread(target=undock_thread, daemon=True)
        thread.start()

        # Wait for undock to complete (with overall timeout)
        thread.join(timeout=70.0)  # 70s to allow for 60s action + 10s goal acceptance

        if not undock_result['done']:
            self.get_logger().warn('Undock overall timeout - continuing anyway')
            return True

        return undock_result.get('success', True)

    def find_bin_callback(self, request, response):
        """Handle find_bin service."""
        self.get_logger().info(f'find_bin called: {request.data}')

        if request.data:
            if not self.client_ready:
                response.success = False
                response.message = 'Vision client not ready - wait for initialization'
                return response

            with self.state_lock:
                if self.state in [SearchState.SEARCHING, SearchState.APPROACHING]:
                    response.success = False
                    response.message = 'Search already in progress'
                    return response

            # STEP 1: Clear command history for fresh path tracking
            self._clear_command_history()
            self.docking_requested = False  # Reset docking flag for new search
            self.get_logger().info('Command history cleared for new search')

            # STEP 2: Undock robot before starting search
            self.get_logger().info('='*40)
            self.get_logger().info('UNDOCKING BEFORE SEARCH')
            self.get_logger().info('='*40)
            undock_success = self._undock_robot()
            if not undock_success:
                self.get_logger().warn('Undock may have failed, but continuing with search')

            # STEP 3: Set state to searching (after undock completes)
            with self.state_lock:
                self.state = SearchState.SEARCHING

            self.consecutive_not_found = 0
            self.search_rotation_count = 0
            self.search_start_time = time.time()
            self.watchdog_last_activity = time.time()  # Initialize watchdog
            self.command_cooldown_until = 0.0  # Clear any stale cooldown
            self.command_active_until = 0.0  # Clear any active command
            self.bin_lock_until = 0.0  # Clear bin lock for fresh search
            self.last_bin_size = None  # Reset proximity tracking
            self.last_bin_detected_time = 0.0

            # Reset detection history and recovery search
            self.detection_history = []
            self.last_bin_direction = 'center'
            self.recovery_mode = False
            self.recovery_attempts = 0

            # Reset dual-camera confirmation state
            self.classification_confirmed = False
            self.classification_confirm_time = 0.0
            self.last_continuous_classify_time = 0.0
            self.classification_in_progress = False
            self.front_camera_arrived = False
            self.front_camera_arrived_time = 0.0

            # Restart inference timer if it was cancelled
            if self.inference_timer is not None:
                self.inference_timer.reset()
                self.get_logger().debug('Inference timer restarted')

            self.get_logger().info('='*40)
            self.get_logger().info('STARTING BIN SEARCH')
            self.get_logger().info('='*40)

            response.success = True
            response.message = 'Undocked and search started'
        else:
            self._stop_search(success=False)
            response.success = True
            response.message = 'Search stopped'

        return response

    def destroy_node(self):
        if self.camera:
            self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationVLMNode()
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
