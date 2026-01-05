"""
API Client module for communicating with the dashboard server.
Supports REST API with optional WebSocket and MQTT backends.
Includes JWT authentication and HMAC-SHA256 request signing.
"""

import requests
import json
import base64
import cv2
import numpy as np
import hmac
import hashlib
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from threading import Thread, Event
import queue

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    protocol: str = "rest"  # "rest", "websocket", "mqtt"
    url: str = "http://localhost:5000"
    endpoints: Dict[str, str] = None
    timeout: float = 10.0
    retry_count: int = 3
    retry_delay: float = 1.0
    send_images: bool = True
    image_quality: int = 85  # JPEG quality
    # Authentication credentials
    username: str = "admin"
    password: str = "admin"

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = {
                "detections": "/api/detections",
                "health": "/api/health",
                "login": "/api/auth/login",
                "verify": "/api/auth/verify"
            }


class BaseClient(ABC):
    """Abstract base class for API clients."""

    @abstractmethod
    def send_detection(self, data: Dict[str, Any]) -> bool:
        """Send detection data to server."""
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """Check server health."""
        pass

    @abstractmethod
    def close(self):
        """Close the client connection."""
        pass


class RESTClient(BaseClient):
    """REST API client using HTTP POST requests with JWT auth and HMAC signing."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.session = requests.Session()
        self._connected = False
        self._auth_token: Optional[str] = None
        self._hmac_secret: Optional[str] = None
        self._token_expiry: float = 0

    def _generate_hmac(self, data: str) -> str:
        """Generate HMAC-SHA256 signature for request body."""
        if not self._hmac_secret:
            return ""
        return hmac.new(
            self._hmac_secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def login(self) -> bool:
        """
        Authenticate with the server and obtain JWT token + HMAC secret.

        Returns:
            True if login was successful.
        """
        url = f"{self.config.url}{self.config.endpoints['login']}"

        try:
            response = self.session.post(
                url,
                json={
                    "username": self.config.username,
                    "password": self.config.password
                },
                timeout=self.config.timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self._auth_token = data.get("token")
                    self._hmac_secret = data.get("hmac_secret")
                    # Token valid for 24 hours, refresh after 23 hours
                    self._token_expiry = time.time() + (23 * 60 * 60)
                    logger.info(f"Authenticated as '{self.config.username}'")
                    return True
                else:
                    logger.error(f"Login failed: {data.get('error', 'Unknown error')}")
            else:
                logger.error(f"Login failed with status {response.status_code}")

        except Exception as e:
            logger.error(f"Login error: {e}")

        return False

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid auth token, login if needed."""
        if self._auth_token and time.time() < self._token_expiry:
            return True

        logger.info("Token expired or missing, re-authenticating...")
        return self.login()

    def send_frame_for_inference(self, image_base64: str, width: int = 640, height: int = 480, device_id: str = "turtlebot4") -> Optional[Dict[str, Any]]:
        """
        Send camera frame to server for inference and receive navigation command.
        Uses server-side Florence-2 + SmolVLM2 on RTX 5090.

        Args:
            image_base64: Base64 encoded JPEG image
            width: Image width
            height: Image height
            device_id: Device identifier

        Returns:
            Navigation result dict with command, or None if failed:
            {
                "success": True,
                "command": "forward|left|right|arrived|not_found",
                "bin_detected": True,
                "position": "center",
                "size": "medium",
                "confidence": 0.9,
                "bboxes": [...],
                "inference_time_ms": 123.4
            }
        """
        if not self._ensure_authenticated():
            logger.error("Cannot send frame: authentication failed")
            return None

        url = f"{self.config.url}{self.config.endpoints.get('frame_infer', '/api/camera/frame-infer')}"

        payload = {
            "image": image_base64,
            "width": width,
            "height": height,
            "device_id": device_id
        }

        payload_json = json.dumps(payload, sort_keys=True)
        hmac_signature = self._generate_hmac(payload_json)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._auth_token}",
            "X-HMAC-Signature": hmac_signature
        }

        for attempt in range(self.config.retry_count):
            try:
                response = self.session.post(
                    url,
                    data=payload_json,
                    timeout=self.config.timeout,
                    headers=headers
                )

                if response.status_code == 200:
                    self._connected = True
                    result = response.json()
                    if result.get("success"):
                        logger.info(
                            f"Server inference: command={result.get('command')} "
                            f"bin={result.get('bin_detected')} "
                            f"pos={result.get('position')} "
                            f"time={result.get('inference_time_ms', 0):.1f}ms"
                        )
                        return result
                    else:
                        logger.warning(f"Server inference failed: {result.get('error')}")
                        return result

                elif response.status_code == 401:
                    logger.warning("Auth token rejected, re-authenticating...")
                    self._auth_token = None
                    if self._ensure_authenticated():
                        headers["Authorization"] = f"Bearer {self._auth_token}"
                        hmac_signature = self._generate_hmac(payload_json)
                        headers["X-HMAC-Signature"] = hmac_signature
                        continue
                elif response.status_code == 503:
                    logger.warning("Vision model not loaded on server")
                    return {"success": False, "command": "not_found", "error": "Model not loaded"}
                else:
                    logger.warning(f"Server returned {response.status_code}: {response.text[:200]}")

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                self._connected = False
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

            if attempt < self.config.retry_count - 1:
                time.sleep(self.config.retry_delay)

        return None

    def send_detection(self, data: Dict[str, Any]) -> bool:
        """
        Send detection data to the server via HTTP POST.
        Includes JWT authentication and HMAC signature.

        Args:
            data: Detection data dictionary.

        Returns:
            True if sent successfully.
        """
        # Ensure we're authenticated before sending
        if not self._ensure_authenticated():
            logger.error("Cannot send detection: authentication failed")
            return False

        url = f"{self.config.url}{self.config.endpoints['detections']}"

        # Calculate payload size
        has_image = 'image_base64' in data
        image_size_kb = len(data.get('image_base64', '')) / 1024 if has_image else 0

        # Serialize payload for HMAC
        payload_json = json.dumps(data, sort_keys=True)
        hmac_signature = self._generate_hmac(payload_json)

        # Build headers with auth token and HMAC signature
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._auth_token}",
            "X-HMAC-Signature": hmac_signature
        }

        for attempt in range(self.config.retry_count):
            try:
                response = self.session.post(
                    url,
                    data=payload_json,  # Use serialized JSON for consistent HMAC
                    timeout=self.config.timeout,
                    headers=headers
                )

                if response.status_code in (200, 201):
                    self._connected = True
                    resp_data = response.json() if response.text else {}
                    logger.info(
                        f"Server accepted: ID={resp_data.get('id', 'N/A')[:8]}... "
                        f"objects={resp_data.get('object_count', 0)} "
                        f"(image: {image_size_kb:.1f}KB, HMAC: verified)"
                    )
                    return True
                elif response.status_code == 401:
                    # Token might be expired, try to re-authenticate
                    logger.warning("Auth token rejected, re-authenticating...")
                    self._auth_token = None
                    if self._ensure_authenticated():
                        # Update headers with new token
                        headers["Authorization"] = f"Bearer {self._auth_token}"
                        hmac_signature = self._generate_hmac(payload_json)
                        headers["X-HMAC-Signature"] = hmac_signature
                        continue
                else:
                    logger.warning(
                        f"Server returned {response.status_code}: {response.text[:200]}"
                    )

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                self._connected = False
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

            if attempt < self.config.retry_count - 1:
                time.sleep(self.config.retry_delay)

        return False

    def check_health(self) -> bool:
        """Check if the server is healthy."""
        url = f"{self.config.url}{self.config.endpoints['health']}"

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            self._connected = response.status_code == 200
            return self._connected
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            self._connected = False
            return False

    def close(self):
        """Close the session."""
        self.session.close()

    @property
    def is_connected(self) -> bool:
        return self._connected


class WebSocketClient(BaseClient):
    """WebSocket client for real-time streaming (optional)."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._ws = None
        self._connected = False
        self._running = False
        self._thread: Optional[Thread] = None
        self._send_queue: queue.Queue = queue.Queue()

    def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            import websockets
            import asyncio

            # Convert HTTP URL to WebSocket URL
            ws_url = self.config.url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = f"{ws_url}/ws/detections"

            logger.info(f"Connecting to WebSocket: {ws_url}")

            # Note: Full async implementation would be needed for production
            self._connected = True
            return True

        except ImportError:
            logger.error("websockets package not installed")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    def send_detection(self, data: Dict[str, Any]) -> bool:
        """Send detection via WebSocket."""
        if not self._connected:
            if not self.connect():
                return False

        try:
            self._send_queue.put(data)
            return True
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")
            return False

    def check_health(self) -> bool:
        return self._connected

    def close(self):
        self._running = False
        self._connected = False


class MQTTClient(BaseClient):
    """MQTT client for IoT-style messaging (optional)."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt

            # Parse URL for MQTT
            # Expected format: mqtt://host:port
            url = self.config.url.replace("mqtt://", "").replace("http://", "")
            parts = url.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 1883

            self._client = mqtt.Client()
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(host, port, 60)
            self._client.loop_start()

            return True

        except ImportError:
            logger.error("paho-mqtt package not installed")
            return False
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning("Disconnected from MQTT broker")

    def send_detection(self, data: Dict[str, Any]) -> bool:
        """Publish detection to MQTT topic."""
        if not self._connected:
            if not self.connect():
                return False

        try:
            topic = "detections/frames"
            payload = json.dumps(data)

            result = self._client.publish(topic, payload, qos=1)
            return result.rc == 0

        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")
            return False

    def check_health(self) -> bool:
        return self._connected

    def close(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()


class APIClient:
    """
    High-level API client that wraps protocol-specific implementations.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self._client: Optional[BaseClient] = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on protocol."""
        protocol = self.config.protocol.lower()

        if protocol == "rest":
            self._client = RESTClient(self.config)
        elif protocol == "websocket":
            self._client = WebSocketClient(self.config)
        elif protocol == "mqtt":
            self._client = MQTTClient(self.config)
        else:
            logger.warning(f"Unknown protocol '{protocol}', defaulting to REST")
            self._client = RESTClient(self.config)

        logger.info(f"Initialized {protocol.upper()} client")

    def _draw_bboxes(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image with different styles for bins vs content.
        - Bins: solid colored rectangles with corner accents, color based on fullness
        - Content: semi-transparent filled regions with 50% opacity

        Args:
            image: BGR image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        annotated = image.copy()

        # Distinct color palette for bins (BGR format)
        bin_colors = [
            (94, 197, 34),    # Green
            (246, 130, 59),   # Blue
            (0, 191, 255),    # Deep sky blue
            (255, 144, 30),   # Dodger blue
            (180, 105, 255),  # Hot pink
            (0, 215, 255),    # Gold
        ]

        # Content object color palette (BGR format)
        content_colors = [
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light red
            (255, 255, 100),  # Cyan
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Yellow
        ]

        bin_index = 0
        content_index = 0

        for det in detections:
            bbox = det.get("bbox", {})
            x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
            label = det.get("label", "object")
            is_content = det.get("is_content", False)
            fullness = det.get("bin_fullness", "")
            fullness_pct = det.get("bin_fullness_percent")

            if is_content:
                # Content objects: use distinct colors from palette, cycling through
                color = content_colors[content_index % len(content_colors)]
                content_index += 1
                thickness = 2

                # Simple bbox for content (no segmentation)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

                # Label
                label_text = label

            else:
                # Bins: color based on fullness level
                if "0-25" in (fullness or ""):
                    color = (94, 197, 34)    # Green - empty
                elif "25-75" in (fullness or ""):
                    color = (22, 204, 132)   # Yellow-green - partial
                elif "75-90" in (fullness or ""):
                    color = (11, 158, 245)   # Orange - mostly full
                elif "90-100" in (fullness or ""):
                    color = (68, 68, 239)    # Red - full
                else:
                    # Use cycling color if no fullness info
                    color = bin_colors[bin_index % len(bin_colors)]

                bin_index += 1
                thickness = 3

                # Solid rectangle for bins
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

                # Draw corner accents for bins
                corner_len = min(w, h) // 5
                accent_thickness = thickness + 2
                # Top-left
                cv2.line(annotated, (x, y), (x + corner_len, y), color, accent_thickness)
                cv2.line(annotated, (x, y), (x, y + corner_len), color, accent_thickness)
                # Top-right
                cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, accent_thickness)
                cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, accent_thickness)
                # Bottom-left
                cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, accent_thickness)
                cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, accent_thickness)
                # Bottom-right
                cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, accent_thickness)
                cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, accent_thickness)

                # Label with fullness percentage
                label_text = label
                if fullness_pct is not None:
                    label_text += f" {fullness_pct}%"

            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # Label background
            label_y = y - text_h - 10 if y > text_h + 10 else y + h + 5
            cv2.rectangle(annotated, (x, label_y), (x + text_w + 10, label_y + text_h + 8), color, -1)
            cv2.putText(annotated, label_text, (x + 5, label_y + text_h + 3), font, font_scale, (255, 255, 255), font_thickness)

        return annotated

    def send_frame_for_inference(
        self,
        image: np.ndarray,
        device_id: str = "turtlebot4"
    ) -> Optional[Dict[str, Any]]:
        """
        Send camera frame to server for inference and receive navigation command.
        Uses server-side Florence-2 + SmolVLM2 on RTX 5090.

        Args:
            image: BGR image (numpy array)
            device_id: Device identifier

        Returns:
            Navigation result dict with command, or None if failed
        """
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality]
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            height, width = image.shape[:2]

            # Only RESTClient supports this method
            if isinstance(self._client, RESTClient):
                return self._client.send_frame_for_inference(
                    image_base64=image_base64,
                    width=width,
                    height=height,
                    device_id=device_id
                )
            else:
                logger.warning("Server inference only supported with REST protocol")
                return None

        except Exception as e:
            logger.error(f"Failed to encode image for server inference: {e}")
            return None

    def send_detection(
        self,
        detection_data: Dict[str, Any],
        image: Optional[np.ndarray] = None
    ) -> bool:
        """
        Send detection data to the server.
        Includes both original and bbox-annotated images.

        Args:
            detection_data: Detection result dictionary.
            image: Optional BGR image to include.

        Returns:
            True if sent successfully.
        """
        data = detection_data.copy()

        # Encode images if provided and configured to send
        if image is not None and self.config.send_images:
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality]

                # Original image
                _, buffer = cv2.imencode('.jpg', image, encode_params)
                data['image_base64'] = base64.b64encode(buffer).decode('utf-8')

                # Draw bboxes and encode annotated image
                detections = detection_data.get('detections', [])
                if detections:
                    annotated = self._draw_bboxes(image, detections)
                    _, buffer_annotated = cv2.imencode('.jpg', annotated, encode_params)
                    data['image_bbox_base64'] = base64.b64encode(buffer_annotated).decode('utf-8')

            except Exception as e:
                logger.warning(f"Failed to encode image: {e}")

        return self._client.send_detection(data)

    def check_health(self) -> bool:
        """Check if the server is reachable and healthy."""
        return self._client.check_health()

    def wait_for_server(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """
        Wait for server to become available.

        Args:
            timeout: Maximum time to wait in seconds.
            interval: Check interval in seconds.

        Returns:
            True if server became available within timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.check_health():
                logger.info("Server is available")
                return True

            logger.info(f"Waiting for server at {self.config.url}...")
            time.sleep(interval)

        logger.error(f"Server not available after {timeout}s")
        return False

    def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test REST client
    config = ServerConfig(
        protocol="rest",
        url="http://localhost:5000",
        send_images=True
    )

    client = APIClient(config)

    # Check health
    print(f"Server healthy: {client.check_health()}")

    # Send test detection
    test_data = {
        "timestamp": "2025-12-01T10:30:00Z",
        "frame_id": "test-frame-001",
        "detections": [
            {
                "label": "person",
                "confidence": 0.95,
                "bbox": {"x": 100, "y": 150, "width": 200, "height": 300}
            }
        ],
        "inference_time_ms": 45.2,
        "image_width": 640,
        "image_height": 480
    }

    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = client.send_detection(test_data, test_image)
    print(f"Detection sent: {result}")

    client.close()
