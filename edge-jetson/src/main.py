"""
Main entry point for the edge detection system.
Orchestrates camera capture, inference, and API communication.
"""

import sys
import signal
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

try:
    from .device_platform import detect_platform, get_platform_config_overrides, Platform
    from .camera import Camera, CameraConfig
    from .detector import Florence2Detector, DetectorConfig
    from .api_client import APIClient, ServerConfig
except ImportError:
    from device_platform import detect_platform, get_platform_config_overrides, Platform
    from camera import Camera, CameraConfig
    from detector import Florence2Detector, DetectorConfig
    from api_client import APIClient, ServerConfig

logger = logging.getLogger(__name__)


class EdgeDetectionSystem:
    """
    Main system class that orchestrates all components.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.camera: Optional[Camera] = None
        self.detector: Optional[Florence2Detector] = None
        self.api_client: Optional[APIClient] = None
        self._running = False
        self._frame_count = 0
        self._detection_count = 0

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file and apply platform overrides."""
        # Default configuration
        config = {
            "camera": {
                "device": "/dev/video0",
                "width": 640,
                "height": 480,
                "fps": 5,
            },
            "model": {
                "name": "microsoft/Florence-2-base",
                "device": "auto",
                "task": "<OD>",
                "confidence_threshold": 0.3,
            },
            "server": {
                "protocol": "rest",
                "url": "http://localhost:5000",
                "endpoints": {
                    "detections": "/api/detections",
                    "health": "/api/health"
                },
                "send_images": True,
                "image_quality": 85,
            },
            "inference": {
                "target_fps": 5,
                "skip_frames": 0,
            },
            "logging": {
                "level": "INFO",
            }
        }

        # Load from config file if provided
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config = self._deep_merge(config, file_config)
                logger.info(f"Loaded config from {config_path}")

        # Auto-detect and load platform-specific config
        platform = detect_platform()
        platform_overrides = get_platform_config_overrides()
        config = self._deep_merge(config, platform_overrides)

        # Load platform-specific config file if exists
        if config_path:
            config_dir = Path(config_path).parent
            platform_config_path = config_dir / f"{platform.value}.yaml"
            if platform_config_path.exists():
                with open(platform_config_path, 'r') as f:
                    platform_config = yaml.safe_load(f)
                    if platform_config:
                        config = self._deep_merge(config, platform_config)
                logger.info(f"Loaded platform config from {platform_config_path}")

        return config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Edge Detection System...")
        logger.info(f"Platform: {detect_platform().value}")

        # Initialize camera
        camera_config = CameraConfig(
            device=self.config["camera"]["device"],
            width=self.config["camera"]["width"],
            height=self.config["camera"]["height"],
            fps=self.config["camera"]["fps"],
        )
        self.camera = Camera(camera_config)

        if not self.camera.open():
            logger.error("Failed to initialize camera")
            return False

        # Initialize detector
        detector_config = DetectorConfig(
            model_name=self.config["model"]["name"],
            device=self.config["model"]["device"],
            task=self.config["model"]["task"],
            confidence_threshold=self.config["model"]["confidence_threshold"],
        )
        self.detector = Florence2Detector(detector_config)

        logger.info("Loading detection model (this may take a moment)...")
        if not self.detector.load():
            logger.error("Failed to initialize detector")
            return False

        # Initialize API client
        # Merge config endpoints with default auth endpoints
        endpoints = {
            "detections": "/api/detections",
            "health": "/api/health",
            "login": "/api/auth/login",
            "verify": "/api/auth/verify",
        }
        if "endpoints" in self.config["server"]:
            endpoints.update(self.config["server"]["endpoints"])

        server_config = ServerConfig(
            protocol=self.config["server"]["protocol"],
            url=self.config["server"]["url"],
            endpoints=endpoints,
            send_images=self.config["server"]["send_images"],
            image_quality=self.config["server"]["image_quality"],
            username=self.config["server"].get("username", "admin"),
            password=self.config["server"].get("password", "admin"),
        )
        self.api_client = APIClient(server_config)

        logger.info("All components initialized successfully")
        return True

    def _save_debug_image(self, frame, result):
        """Save annotated debug image locally."""
        import cv2
        try:
            debug_dir = Path("debug_output")
            debug_dir.mkdir(exist_ok=True)

            annotated = frame.copy()

            # Draw bboxes
            for det in result.detections:
                bbox = det.bbox
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                is_content = det.is_content

                # Color: green for bins, blue for content
                if is_content:
                    color = (255, 100, 100)  # Light blue for content
                elif det.bin_fullness:
                    if "0-25" in det.bin_fullness:
                        color = (0, 255, 0)  # Green
                    elif "25-75" in det.bin_fullness:
                        color = (0, 255, 255)  # Yellow
                    elif "75-90" in det.bin_fullness:
                        color = (0, 165, 255)  # Orange
                    else:
                        color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Default green

                thickness = 2 if is_content else 3
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

                # Label
                label = det.label[:30]  # Truncate long labels
                if det.bin_fullness_percent is not None:
                    label += f" {det.bin_fullness_percent}%"
                cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save
            filename = debug_dir / f"frame_{self._frame_count:04d}.jpg"
            cv2.imwrite(str(filename), annotated)
            logger.info(f"  [DEBUG] Saved: {filename}")

        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")

    def run(self, wait_for_server: bool = True):
        """
        Run the main detection loop.

        Args:
            wait_for_server: Whether to wait for server before starting.
        """
        if not self.initialize():
            logger.error("Failed to initialize system")
            return

        # Optionally wait for server
        if wait_for_server:
            logger.info("Waiting for dashboard server...")
            if not self.api_client.wait_for_server(timeout=60):
                logger.warning("Server not available, will retry during operation")

        self._running = True
        target_fps = self.config["inference"]["target_fps"]
        frame_interval = 1.0 / target_fps
        skip_frames = self.config["inference"]["skip_frames"]

        logger.info(f"Starting detection loop (target: {target_fps} FPS)")

        while self._running:
            loop_start = time.time()

            try:
                # Skip frames if configured
                for _ in range(skip_frames):
                    self.camera.read_frame()

                # Capture frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                self._frame_count += 1

                # Generate frame ID and timestamp
                frame_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat() + "Z"

                # Run detection
                result = self.detector.detect(frame, frame_id, timestamp)

                self._detection_count += len(result.detections)

                # Log detection info with bin status
                if result.detections:
                    labels = [d.label for d in result.detections]
                    bin_info = ""
                    if result.bin_detected:
                        bin_info = f" | Bins: {result.bin_count}, Fullness: {result.overall_fullness_percent}%"
                    logger.info(
                        f"Frame {self._frame_count}: {len(result.detections)} objects "
                        f"({', '.join(labels)}) in {result.inference_time_ms:.1f}ms{bin_info}"
                    )
                else:
                    logger.info(
                        f"Frame {self._frame_count}: No detections "
                        f"({result.inference_time_ms:.1f}ms)"
                    )

                # Log summary
                if result.status_summary:
                    logger.info(f"  Summary: {result.status_summary}")

                # Debug: Save annotated image locally every 10 frames
                if self._frame_count % 10 == 1 and result.detections:
                    self._save_debug_image(frame, result)

                # Send to server
                logger.info(f"  Sending to server: {self.config['server']['url']}...")
                send_success = self.api_client.send_detection(result.to_dict(), frame)

                if send_success:
                    logger.info(f"  -> Sent successfully! Ready for next frame.")
                else:
                    logger.warning("  -> Failed to send detection to server")

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.shutdown()

    def shutdown(self):
        """Clean up and shutdown all components."""
        logger.info("Shutting down Edge Detection System...")

        if self.camera:
            self.camera.close()

        if self.detector:
            self.detector.unload()

        if self.api_client:
            self.api_client.close()

        logger.info(
            f"Shutdown complete. Processed {self._frame_count} frames, "
            f"detected {self._detection_count} objects."
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge Detection System for Jetson Orin / Raspberry Pi"
    )
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server before starting"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Find config file
    config_path = args.config
    if not Path(config_path).exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config" / "config.yaml"
        if not config_path.exists():
            config_path = None
            logger.info("No config file found, using defaults")

    # Run system
    system = EdgeDetectionSystem(str(config_path) if config_path else None)
    system.run(wait_for_server=not args.no_wait)


if __name__ == "__main__":
    main()
