#!/usr/bin/env python3
"""
llama_vision_client.py
HTTP client for server-side SmolVLM2 vision inference.

Sends images to central server (192.168.0.81:5000) for GPU-accelerated
inference instead of running locally on RPi4.
"""

import base64
import logging
import requests
from typing import Optional, List, Dict
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class NavigationCommand(Enum):
    """Navigation commands output by SmolVLM."""
    FORWARD = "forward"
    BACKWARD = "backward"  # Back up - too close to obstacle
    LEFT = "left"
    RIGHT = "right"
    ARRIVED = "arrived"
    NOT_FOUND = "not_found"
    SEARCHING = "searching"
    SEARCH_LEFT = "search_left"
    SEARCH_RIGHT = "search_right"


@dataclass
class NavigationResult:
    """Result of navigation inference."""
    command: NavigationCommand
    confidence: float
    raw_response: str
    bin_detected: bool
    bin_position: Optional[str] = None
    bin_size: Optional[str] = None
    bboxes: List[Dict] = field(default_factory=list)  # Florence-2 detections: [{x, y, w, h, label, conf}, ...]


@dataclass
class ClassificationResult:
    """Result of bin classification with enhanced details."""
    bin_found: bool = True  # Whether a bin was actually detected
    fullness: str = "UNKNOWN"  # EMPTY, PARTIALLY_FULL, FULL
    waste_type: str = "UNKNOWN"  # RECYCLABLE, ORGANIC, GENERAL, MIXED
    confidence: float = 0.0
    raw_response: str = ""
    # Enhanced fields
    fill_level_percent: int = 0  # 0-100%
    action: str = ""  # Action recommendation
    scene_description: str = ""  # Scene description
    containers_count: int = 0  # Number of containers
    containers_type: str = ""  # Type of containers
    objects_detected: List[str] = field(default_factory=list)  # Florence-2 detected objects
    bboxes: List[Dict] = field(default_factory=list)  # Florence-2 bounding boxes: [{x, y, w, h, label, confidence}, ...]
    summary: str = ""  # Formatted summary string


class LlamaVisionClient:
    """
    Client for server-side SmolVLM2 vision inference.

    Replaces local llama.cpp calls with HTTP requests to central server
    running GPU-accelerated inference.
    """

    def __init__(
        self,
        server_url: str = "http://192.168.0.81:5000",
        timeout: float = 30.0,
        username: str = "admin",
        password: str = "admin",
        verify_connection: bool = True
    ):
        """
        Initialize the vision client.

        Args:
            server_url: Base URL of the vision server
            timeout: Request timeout in seconds
            username: Authentication username
            password: Authentication password
            verify_connection: Check server health on init
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.username = username
        self.password = password
        self._token = None
        self._session = requests.Session()

        # Configure session for keep-alive
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

        # Verify connection at startup
        if verify_connection:
            self._verify_server_connection()

    def _verify_server_connection(self, max_retries: int = 5, retry_delay: float = 2.0):
        """
        Verify server is reachable and responding.
        Uses exponential backoff for retries.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Checking server connection (attempt {attempt + 1}/{max_retries})...")
                status = self.check_server_status()

                if "error" not in status:
                    logger.info(f"Server connected: {self.server_url}")
                    if status.get("loaded", False):
                        logger.info("Vision model is loaded and ready")
                    else:
                        logger.warning("Vision model not yet loaded on server")
                    return True

                logger.warning(f"Server check failed: {status.get('error')}")

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.1f}s...")
                import time
                time.sleep(wait_time)

        logger.error(f"Failed to connect to server after {max_retries} attempts")
        logger.error(f"Server URL: {self.server_url}")
        logger.error("Navigation will NOT work until server is available!")
        return False

    def _get_token(self) -> Optional[str]:
        """Get or refresh authentication token."""
        if self._token:
            return self._token

        try:
            logger.debug(f"Logging in to {self.server_url}...")
            response = self._session.post(
                f"{self.server_url}/api/auth/login",
                json={"username": self.username, "password": self.password},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self._token = data.get("token")
                logger.debug("Login successful")
                return self._token
            else:
                logger.error(f"Auth failed: {response.status_code} - {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Auth error: {e}")
            return None

    def _image_to_base64(self, image, quality: int = 65) -> str:
        """Convert numpy array (BGR) to base64 JPEG.

        Args:
            image: BGR numpy array
            quality: JPEG quality 1-100 (default 65 for fast transfer, use 80 for better detection)

        Uses lower quality (65%) for faster WiFi transfer while maintaining
        sufficient detail for Florence-2 detection. Higher quality (80%) can be
        used during approach for better detection accuracy.
        """
        import cv2
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _make_request(self, endpoint: str, image_base64: str) -> Optional[dict]:
        """Make authenticated request to vision endpoint."""
        token = self._get_token()
        if not token:
            logger.error("Failed to get authentication token")
            return None

        headers = {"Authorization": f"Bearer {token}"}

        try:
            logger.debug(f"Sending request to {self.server_url}{endpoint}...")
            response = self._session.post(
                f"{self.server_url}{endpoint}",
                json={"image": image_base64},
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 401:
                # Token expired, refresh and retry
                logger.debug("Token expired, refreshing...")
                self._token = None
                token = self._get_token()
                if token:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = self._session.post(
                        f"{self.server_url}{endpoint}",
                        json={"image": image_base64},
                        headers=headers,
                        timeout=self.timeout
                    )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return None

        except requests.Timeout:
            logger.error(f"Request timeout after {self.timeout}s")
            return None
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def get_navigation_command(self, image) -> NavigationResult:
        """
        Analyze image for trash bin and return navigation command.

        Args:
            image: BGR numpy array from camera

        Returns:
            NavigationResult with command and metadata
        """
        try:
            image_base64 = self._image_to_base64(image)
            result = self._make_request("/api/vision/navigate", image_base64)

            if result and result.get("success"):
                # Map command string to enum
                cmd_str = result.get("command", "not_found")
                try:
                    command = NavigationCommand(cmd_str)
                except ValueError:
                    command = NavigationCommand.NOT_FOUND

                return NavigationResult(
                    command=command,
                    confidence=result.get("confidence", 0.0),
                    raw_response=result.get("raw_response", ""),
                    bin_detected=result.get("bin_detected", False),
                    bin_position=result.get("position"),
                    bin_size=result.get("size")
                )
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response from server"
                return NavigationResult(
                    command=NavigationCommand.NOT_FOUND,
                    confidence=0.0,
                    raw_response=f"Error: {error_msg}",
                    bin_detected=False
                )

        except Exception as e:
            logger.error(f"Navigation inference error: {e}")
            return NavigationResult(
                command=NavigationCommand.NOT_FOUND,
                confidence=0.0,
                raw_response=f"Error: {str(e)}",
                bin_detected=False
            )

    def get_navigation_with_detection(self, image, quality: int = 65) -> NavigationResult:
        """
        Analyze image using the frame-infer endpoint for navigation commands.

        This method sends the image to /api/camera/frame-infer which handles
        both inference and frame storage for dashboard display.

        Args:
            image: BGR numpy array from camera
            quality: JPEG quality 1-100 (default 65, use 80 during approach for better detection)

        Returns:
            NavigationResult with command, metadata, and bounding boxes
        """
        try:
            import cv2
            image_base64 = self._image_to_base64(image, quality=quality)
            height, width = image.shape[:2]

            # Get auth token
            token = self._get_token()
            if not token:
                logger.error("Failed to get authentication token")
                return NavigationResult(
                    command=NavigationCommand.NOT_FOUND,
                    confidence=0.0,
                    raw_response="Auth failed",
                    bin_detected=False
                )

            # Call the new frame-infer endpoint
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "image": image_base64,
                "width": width,
                "height": height,
                "device_id": "turtlebot4"
            }

            response = self._session.post(
                f"{self.server_url}/api/camera/frame-infer",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 401:
                # Token expired, refresh and retry
                self._token = None
                token = self._get_token()
                if token:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = self._session.post(
                        f"{self.server_url}/api/camera/frame-infer",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout
                    )

            if response.status_code == 200:
                result = response.json()

                # Map command string to enum
                cmd_str = result.get("command", "not_found")
                try:
                    command = NavigationCommand(cmd_str)
                except ValueError:
                    command = NavigationCommand.NOT_FOUND

                # Extract bounding boxes if present
                bboxes = result.get("bboxes", [])

                return NavigationResult(
                    command=command,
                    confidence=result.get("confidence", 0.0),
                    raw_response=result.get("raw_response", ""),
                    bin_detected=result.get("bin_detected", False),
                    bin_position=result.get("position"),
                    bin_size=result.get("size"),
                    bboxes=bboxes
                )
            else:
                logger.error(f"frame-infer failed: {response.status_code} - {response.text}")
                return NavigationResult(
                    command=NavigationCommand.NOT_FOUND,
                    confidence=0.0,
                    raw_response=f"Error: {response.status_code}",
                    bin_detected=False
                )

        except requests.Timeout:
            logger.error(f"Request timeout after {self.timeout}s")
            return NavigationResult(
                command=NavigationCommand.NOT_FOUND,
                confidence=0.0,
                raw_response="Timeout",
                bin_detected=False
            )
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return NavigationResult(
                command=NavigationCommand.NOT_FOUND,
                confidence=0.0,
                raw_response=f"Error: {str(e)}",
                bin_detected=False
            )

    def classify_bin(self, image) -> ClassificationResult:
        """
        Classify trash bin contents with two-step validation.

        Server first verifies bin presence with Florence-2, then uses SmolVLM
        for detailed analysis. Returns bin_found=False if no bin detected.

        Args:
            image: BGR numpy array from camera

        Returns:
            ClassificationResult with enhanced details and bin_found flag
        """
        try:
            image_base64 = self._image_to_base64(image)
            result = self._make_request("/api/vision/classify", image_base64)

            if result and result.get("success"):
                # Parse enhanced response
                bin_found = result.get("bin_found", True)

                # Parse nested containers object
                containers = result.get("containers", {})
                containers_count = containers.get("count", 0) if isinstance(containers, dict) else 0
                containers_type = containers.get("type", "") if isinstance(containers, dict) else ""

                # Parse nested fill_level object
                fill_level = result.get("fill_level", {})
                fill_percent = fill_level.get("percent", 0) if isinstance(fill_level, dict) else 0

                # Parse bboxes from Florence-2 detections (now includes is_content, bin_fullness, etc.)
                bboxes = result.get("bboxes", [])
                # Ensure all bbox fields are preserved
                for bbox in bboxes:
                    if "is_content" not in bbox:
                        bbox["is_content"] = False
                    if "is_bin" not in bbox:
                        bbox["is_bin"] = True
                    if "bin_fullness" not in bbox:
                        bbox["bin_fullness"] = None
                    if "bin_fullness_percent" not in bbox:
                        bbox["bin_fullness_percent"] = fill_percent

                return ClassificationResult(
                    bin_found=bin_found,
                    fullness=result.get("fullness", "UNKNOWN"),
                    waste_type=result.get("waste_type", "UNKNOWN"),
                    confidence=result.get("confidence", 0.0),
                    raw_response=result.get("raw_response", ""),
                    fill_level_percent=fill_percent,
                    action=result.get("action", ""),
                    scene_description=result.get("scene", ""),
                    containers_count=containers_count,
                    containers_type=containers_type,
                    objects_detected=result.get("objects_detected", []),
                    bboxes=bboxes,
                    summary=result.get("summary", "")
                )
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response from server"
                return ClassificationResult(
                    bin_found=False,
                    fullness="UNKNOWN",
                    waste_type="UNKNOWN",
                    confidence=0.0,
                    raw_response=f"Error: {error_msg}"
                )

        except Exception as e:
            logger.error(f"Classification inference error: {e}")
            return ClassificationResult(
                bin_found=False,
                fullness="UNKNOWN",
                waste_type="UNKNOWN",
                confidence=0.0,
                raw_response=f"Error: {str(e)}"
            )

    def check_server_status(self) -> dict:
        """
        Check if the vision server is available and model is loaded.

        Returns:
            Server status dict with 'loaded', 'device', etc.
        """
        try:
            response = self._session.get(
                f"{self.server_url}/api/vision/status",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status check failed: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}

    def _draw_bboxes(self, image, detections: List[Dict]):
        """
        Draw bounding boxes on image with different styles for bins vs content.
        Matches edge-jetson/src/api_client.py drawing style.

        Args:
            image: BGR image (numpy array)
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        import cv2
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
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", bbox.get("w", 0))
            h = bbox.get("height", bbox.get("h", 0))
            label = det.get("label", "object")
            is_content = det.get("is_content", False)
            is_bin = det.get("is_bin", not is_content)
            fullness = det.get("bin_fullness", "")
            fullness_pct = det.get("bin_fullness_percent")

            if is_content:
                # Content objects: use distinct colors from palette, draw simpler
                color = content_colors[content_index % len(content_colors)]
                content_index += 1
                thickness = 2

                # Simple bbox for content with dashed appearance (draw as solid for simplicity)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
                label_text = label

            elif is_bin:
                # Bins: color based on fullness level
                if "0-25" in (fullness or ""):
                    color = (94, 197, 34)    # Green - empty
                elif "25-75" in (fullness or ""):
                    color = (22, 204, 132)   # Yellow-green - partial
                elif "75-90" in (fullness or ""):
                    color = (11, 158, 245)   # Orange - mostly full
                elif "90-100" in (fullness or "") or "75-100" in (fullness or ""):
                    color = (68, 68, 239)    # Red - full
                else:
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

            else:
                # Other objects (neither bin nor content)
                color = (200, 200, 200)  # Gray
                thickness = 1
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
                label_text = label

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

    def send_detection(self, image, result: ClassificationResult, inference_time_ms: float = 0.0) -> bool:
        """
        Send classification result to dashboard with proper authentication.
        Matches edge-jetson/src/api_client.py format.

        Args:
            image: BGR numpy array of the classified image
            result: ClassificationResult with classification details
            inference_time_ms: Time taken for inference in milliseconds

        Returns:
            True if successfully sent, False otherwise
        """
        import uuid
        import cv2
        from datetime import datetime, timezone

        try:
            token = self._get_token()
            if not token:
                logger.error("Failed to get auth token for detection send")
                return False

            # Get image dimensions
            height, width = image.shape[:2]

            # Map fullness to percentage range (matching edge-jetson format)
            fullness_map = {
                'EMPTY': '0-25%',
                'PARTIALLY_FULL': '25-75%',
                'FULL': '75-100%',
                'UNKNOWN': '0-25%'
            }

            # Build detections list from Florence-2 bboxes
            # Server now returns is_content, is_bin, bin_fullness, bin_fullness_percent per bbox
            detections_list = []
            if result.bboxes:
                for bbox in result.bboxes:
                    # Use server-provided fields, fallback to defaults
                    is_content = bbox.get('is_content', False)
                    is_bin = bbox.get('is_bin', not is_content)
                    bin_fullness = bbox.get('bin_fullness') or fullness_map.get(result.fullness, '0-25%')
                    bin_fullness_pct = bbox.get('bin_fullness_percent')
                    if bin_fullness_pct is None:
                        bin_fullness_pct = result.fill_level_percent

                    detections_list.append({
                        'label': bbox.get('label', 'object'),
                        'confidence': bbox.get('confidence', result.confidence),
                        'is_content': is_content,
                        'is_bin': is_bin,
                        'bin_fullness': bin_fullness,
                        'bin_fullness_percent': bin_fullness_pct,
                        'parent_bin_id': bbox.get('parent_bin_id'),
                        'bbox': {
                            'x': bbox.get('x', 0),
                            'y': bbox.get('y', 0),
                            'width': bbox.get('w', 0),
                            'height': bbox.get('h', 0)
                        }
                    })

            # If no bboxes from Florence-2, add a placeholder
            if not detections_list:
                detections_list.append({
                    'label': 'waste_bin',
                    'confidence': result.confidence,
                    'is_content': False,
                    'is_bin': True,
                    'bin_fullness': fullness_map.get(result.fullness, '0-25%'),
                    'bin_fullness_percent': result.fill_level_percent,
                    'bbox': {'x': 50, 'y': 50, 'width': width - 100, 'height': height - 100}
                })

            # Encode original image (higher quality for storage)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # Draw bboxes and encode annotated image (like edge-jetson)
            annotated = self._draw_bboxes(image, detections_list)
            _, bbox_buffer = cv2.imencode('.jpg', annotated, encode_params)
            image_bbox_b64 = base64.b64encode(bbox_buffer.tobytes()).decode("utf-8")

            # Build detailed status summary (multiline format)
            contents_str = ', '.join(result.objects_detected[:5]) if result.objects_detected else 'None detected'

            # Determine status level
            fill_pct = result.fill_level_percent
            if fill_pct >= 90:
                status = "CRITICAL"
                action = "Immediate collection required!"
            elif fill_pct >= 75:
                status = "HIGH"
                action = "Schedule collection soon."
            elif fill_pct >= 50:
                status = "MODERATE"
                action = "Monitor regularly."
            else:
                status = "LOW"
                action = "Good capacity available."

            # Use scene from server, with better fallback
            scene_text = result.scene_description
            if not scene_text or scene_text == "":
                if result.objects_detected:
                    scene_text = f"Detected: {contents_str}"
                else:
                    scene_text = f"Bin analyzed - {result.fullness.lower()}"

            status_summary = f"""Containers: {result.containers_count if result.containers_count > 0 else 1} ({result.containers_type or 'waste bin'})
Fill Level: {fill_pct}% [{status}]
Action: {result.action or action}
Contents: {contents_str}
Scene: {scene_text}"""

            # Build payload matching edge-jetson DetectionResult.to_dict() format
            payload = {
                'timestamp': datetime.now(timezone.utc).isoformat() + "Z",
                'frame_id': str(uuid.uuid4()),
                'detections': detections_list,
                'inference_time_ms': inference_time_ms if inference_time_ms > 0 else 100.0,
                'image_width': width,
                'image_height': height,
                'bin_detected': result.bin_found,
                'bin_count': result.containers_count if result.containers_count > 0 else 1,
                'overall_fullness': fullness_map.get(result.fullness, '0-25%'),
                'overall_fullness_percent': fill_pct,
                'status_summary': status_summary,
                'image_base64': image_b64,
                'image_bbox_base64': image_bbox_b64
            }

            headers = {"Authorization": f"Bearer {token}"}
            response = self._session.post(
                f"{self.server_url}/api/detections",
                json=payload,
                headers=headers,
                timeout=15
            )

            if response.status_code == 401:
                # Token expired, retry
                self._token = None
                token = self._get_token()
                if token:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = self._session.post(
                        f"{self.server_url}/api/detections",
                        json=payload,
                        headers=headers,
                        timeout=15
                    )

            if response.status_code in (200, 201):
                resp_data = response.json() if response.text else {}
                logger.info(
                    f"Detection sent: ID={resp_data.get('id', 'N/A')[:8]}... "
                    f"objects={len(detections_list)} fullness={fill_pct}%"
                )
                return True
            else:
                logger.error(f"Detection send failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Detection send error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Test code
if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.INFO)

    print("Testing VisionClient (Server-side inference)...")
    print("Server: http://192.168.0.81:5000")

    client = LlamaVisionClient(server_url="http://192.168.0.81:5000")

    # Check server status
    print("\nChecking server status...")
    status = client.check_server_status()
    print(f"  Status: {status}")

    if not status.get("loaded", False):
        print("WARNING: Vision model not loaded on server!")
        print("You may need to call /api/vision/load first")

    # Capture test image
    print("\nCapturing from webcam...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Cannot capture from webcam")
        exit(1)

    print(f"Captured: {frame.shape}")

    # Test navigation
    print("\nTesting navigation...")
    nav_result = client.get_navigation_command(frame)
    print(f"  Command: {nav_result.command.value}")
    print(f"  Bin detected: {nav_result.bin_detected}")
    print(f"  Position: {nav_result.bin_position}")
    print(f"  Size: {nav_result.bin_size}")
    print(f"  Confidence: {nav_result.confidence}")

    # Test classification
    print("\nTesting classification...")
    class_result = client.classify_bin(frame)
    print(f"  Fullness: {class_result.fullness}")
    print(f"  Waste type: {class_result.waste_type}")
    print(f"  Confidence: {class_result.confidence}")

    print("\nTest complete!")
