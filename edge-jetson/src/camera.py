"""
USB Camera capture module using OpenCV.
Supports /dev/video0 and configurable resolution/FPS.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import time
from dataclasses import dataclass
from threading import Thread, Lock
import queue

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 5
    buffer_size: int = 1
    auto_reconnect: bool = True
    reconnect_delay: float = 2.0


class Camera:
    """USB Camera capture class with frame buffering and auto-reconnect."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.config.buffer_size)
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0
        self._frame_count: int = 0

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        with self._lock:
            if self.cap is not None:
                self.cap.release()

            # Try to open camera by device path or index
            device = self.config.device
            if device.startswith("/dev/video"):
                try:
                    device_index = int(device.replace("/dev/video", ""))
                except ValueError:
                    device_index = 0
            else:
                try:
                    device_index = int(device)
                except ValueError:
                    device_index = 0

            logger.info(f"Opening camera device: {self.config.device} (index: {device_index})")

            # Try V4L2 backend first (better for Linux)
            self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                logger.warning("V4L2 backend failed, trying default backend")
                self.cap = cv2.VideoCapture(device_index)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera: {self.config.device}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            return True

    def close(self):
        """Close the camera device."""
        self.stop_continuous_capture()

        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logger.info("Camera closed")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is BGR format numpy array.
        """
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                if self.config.auto_reconnect:
                    logger.warning("Camera disconnected, attempting reconnect...")
                    time.sleep(self.config.reconnect_delay)
                    if not self.open():
                        return False, None
                else:
                    return False, None

            ret, frame = self.cap.read()

            if ret:
                self._last_frame = frame.copy()
                self._last_frame_time = time.time()
                self._frame_count += 1
            else:
                logger.warning("Failed to read frame from camera")

            return ret, frame

    def capture_screenshot(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single screenshot from the camera.
        Alias for read_frame() for semantic clarity.

        Returns:
            Tuple of (success, frame).
        """
        return self.read_frame()

    def get_last_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the last captured frame without reading a new one.

        Returns:
            Tuple of (frame, timestamp). Frame may be None if no frame captured yet.
        """
        return self._last_frame, self._last_frame_time

    def start_continuous_capture(self):
        """Start continuous frame capture in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Started continuous capture")

    def stop_continuous_capture(self):
        """Stop continuous frame capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Stopped continuous capture")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        frame_interval = 1.0 / self.config.fps

        while self._running:
            start_time = time.time()

            ret, frame = self.read_frame()

            if ret and frame is not None:
                # Clear old frames and add new one
                try:
                    while not self._frame_queue.empty():
                        self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame_from_queue(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a frame from the continuous capture queue.

        Args:
            timeout: Maximum time to wait for a frame.

        Returns:
            Tuple of (success, frame).
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    @property
    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        with self._lock:
            return self.cap is not None and self.cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def list_available_cameras(max_cameras: int = 10) -> list:
    """
    List available camera devices.

    Args:
        max_cameras: Maximum number of cameras to check.

    Returns:
        List of available camera indices.
    """
    available = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()

    return available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # List available cameras
    cameras = list_available_cameras()
    print(f"Available cameras: {cameras}")

    if cameras:
        # Test camera capture
        config = CameraConfig(
            device=f"/dev/video{cameras[0]}",
            width=640,
            height=480,
            fps=5
        )

        with Camera(config) as cam:
            print("Capturing 5 test frames...")
            for i in range(5):
                ret, frame = cam.read_frame()
                if ret:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Frame {i+1}: Failed to capture")
                time.sleep(0.2)
