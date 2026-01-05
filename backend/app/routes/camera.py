"""
Live camera feed endpoints for robot camera streaming.

The robot posts frames periodically, and the dashboard polls for the latest frame.
"""

import base64
import logging
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, current_app

from ..routes.auth import token_required

logger = logging.getLogger(__name__)

camera_bp = Blueprint("camera", __name__)

# In-memory storage for latest camera frame (ephemeral, not database)
_camera_lock = threading.Lock()
_latest_frame = {
    "data": None,  # bytes (JPEG)
    "timestamp": None,  # datetime
    "width": 0,
    "height": 0,
    "device_id": None,
    "detections": [],  # Florence-2 detection bounding boxes
    "has_detections": False,
}

# In-memory storage for classification camera frame (separate from navigation camera)
_classification_lock = threading.Lock()
_classification_frame = {
    "data": None,  # bytes (JPEG)
    "timestamp": None,  # datetime
    "width": 0,
    "height": 0,
    "device_id": "classification",
    "classification_result": None,  # Latest classification result
}


@camera_bp.route("/api/camera/frame", methods=["POST"])
@token_required
def post_camera_frame():
    """
    Robot posts latest camera frame.

    Request JSON:
    {
        "image": "base64_encoded_jpeg",
        "width": 640,
        "height": 480,
        "device_id": "turtlebot4"  (optional)
    }
    """
    global _latest_frame

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode base64 image
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        with _camera_lock:
            _latest_frame["data"] = image_bytes
            _latest_frame["timestamp"] = datetime.utcnow()
            _latest_frame["width"] = data.get("width", 640)
            _latest_frame["height"] = data.get("height", 480)
            _latest_frame["device_id"] = data.get("device_id", "default")
            # Clear detections - new frame without inference
            _latest_frame["detections"] = []
            _latest_frame["has_detections"] = False

        return jsonify({
            "success": True,
            "timestamp": _latest_frame["timestamp"].isoformat() + "Z"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/camera/frame-detected", methods=["POST"])
@token_required
def post_camera_frame_detected():
    """
    Robot posts camera frame with Florence-2 detection bounding boxes.

    Request JSON:
    {
        "image": "base64_encoded_jpeg",
        "width": 640,
        "height": 480,
        "device_id": "turtlebot4",
        "detections": [
            {"label": "trash_bin", "confidence": 0.95, "bbox": {"x": 100, "y": 150, "width": 200, "height": 180}}
        ],
        "has_detections": true
    }
    """
    global _latest_frame

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode base64 image
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        # Get incoming detections - ALWAYS update (don't preserve old)
        incoming_detections = data.get("detections", [])
        incoming_has_detections = data.get("has_detections", False)

        with _camera_lock:
            _latest_frame["data"] = image_bytes
            _latest_frame["timestamp"] = datetime.utcnow()
            _latest_frame["width"] = data.get("width", 640)
            _latest_frame["height"] = data.get("height", 480)
            _latest_frame["device_id"] = data.get("device_id", "default")
            # Always update detections with current frame's data
            _latest_frame["detections"] = incoming_detections
            _latest_frame["has_detections"] = incoming_has_detections

        detection_count = len(incoming_detections)
        return jsonify({
            "success": True,
            "timestamp": _latest_frame["timestamp"].isoformat() + "Z",
            "detections_received": detection_count
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/camera/latest", methods=["GET"])
def get_latest_frame():
    """
    Dashboard polls for latest camera frame.
    Returns JPEG image directly for easy display in <img> tag.
    """
    with _camera_lock:
        if _latest_frame["data"] is None:
            return jsonify({"error": "No camera frame available"}), 404

        # Check if frame is stale (older than 30 seconds)
        if _latest_frame["timestamp"]:
            age = (datetime.utcnow() - _latest_frame["timestamp"]).total_seconds()
            if age > 30:
                return jsonify({"error": "Camera frame stale", "age_seconds": age}), 404

        return Response(
            _latest_frame["data"],
            mimetype="image/jpeg",
            headers={
                "X-Frame-Timestamp": _latest_frame["timestamp"].isoformat() + "Z",
                "X-Frame-Width": str(_latest_frame["width"]),
                "X-Frame-Height": str(_latest_frame["height"]),
                "Cache-Control": "no-cache, no-store, must-revalidate",
            }
        )


@camera_bp.route("/api/camera/latest/base64", methods=["GET"])
def get_latest_frame_base64():
    """
    Get latest frame as base64 JSON (alternative to raw JPEG).
    """
    with _camera_lock:
        if _latest_frame["data"] is None:
            return jsonify({"error": "No camera frame available"}), 404

        # Check if frame is stale
        if _latest_frame["timestamp"]:
            age = (datetime.utcnow() - _latest_frame["timestamp"]).total_seconds()
            if age > 30:
                return jsonify({"error": "Camera frame stale", "age_seconds": age}), 404

        return jsonify({
            "image": base64.b64encode(_latest_frame["data"]).decode("utf-8"),
            "timestamp": _latest_frame["timestamp"].isoformat() + "Z",
            "width": _latest_frame["width"],
            "height": _latest_frame["height"],
            "device_id": _latest_frame["device_id"],
        })


@camera_bp.route("/api/camera/status", methods=["GET"])
def camera_status():
    """
    Get camera status (has frame, timestamp, resolution, detections).
    """
    with _camera_lock:
        has_frame = _latest_frame["data"] is not None
        timestamp = _latest_frame["timestamp"]
        age = None

        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()

        return jsonify({
            "has_frame": has_frame,
            "timestamp": timestamp.isoformat() + "Z" if timestamp else None,
            "age_seconds": age,
            "width": _latest_frame["width"] if has_frame else None,
            "height": _latest_frame["height"] if has_frame else None,
            "device_id": _latest_frame["device_id"],
            "is_live": has_frame and age is not None and age < 10,
            "has_detections": _latest_frame.get("has_detections", False),
            "detection_count": len(_latest_frame.get("detections", [])),
        })


@camera_bp.route("/api/camera/detections", methods=["GET"])
def get_camera_detections():
    """
    Get latest detection bounding boxes for overlay rendering.

    Returns JSON with detection info for frontend to render bounding boxes.
    """
    with _camera_lock:
        if _latest_frame["data"] is None:
            return jsonify({"error": "No camera frame available", "detections": []}), 404

        timestamp = _latest_frame["timestamp"]
        age = None
        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()

        return jsonify({
            "timestamp": timestamp.isoformat() + "Z" if timestamp else None,
            "age_seconds": age,
            "width": _latest_frame["width"],
            "height": _latest_frame["height"],
            "has_detections": _latest_frame.get("has_detections", False),
            "detections": _latest_frame.get("detections", []),
        })


@camera_bp.route("/api/camera/frame-infer", methods=["POST"])
@token_required
def post_camera_frame_with_inference():
    """
    Robot posts camera frame and server runs Florence-2 + SmolVLM inference.

    This endpoint:
    1. Accepts a camera frame
    2. Runs Florence-2 object detection
    3. Runs SmolVLM navigation inference
    4. Updates the camera frame storage with bounding boxes
    5. Saves navigation result to database
    6. Returns navigation command

    Request JSON:
    {
        "image": "base64_encoded_jpeg",
        "width": 640,
        "height": 480,
        "device_id": "turtlebot4"  (optional)
    }

    Response JSON:
    {
        "success": true,
        "command": "forward|left|right|arrived|not_found",
        "bin_detected": true,
        "position": "center",
        "size": "medium",
        "confidence": 0.9,
        "bboxes": [...],
        "inference_time_ms": 1234.5,
        "history_id": "uuid"
    }
    """
    import time
    from .. import db
    from ..services.vision_model import vision_service
    from ..models.navigation import NavigationHistory

    global _latest_frame

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data", "success": False}), 400

    try:
        # Decode base64 image for storage
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64_clean = image_b64.split(",")[1]
        else:
            image_b64_clean = image_b64

        image_bytes = base64.b64decode(image_b64_clean)
        device_id = data.get("device_id", "turtlebot4")
        width = data.get("width", 640)
        height = data.get("height", 480)

        # Check if vision model is loaded
        if not vision_service.is_loaded:
            # Store frame without detections
            with _camera_lock:
                _latest_frame = {
                    "data": image_bytes,
                    "timestamp": datetime.utcnow(),
                    "width": width,
                    "height": height,
                    "device_id": device_id,
                    "detections": [],
                    "has_detections": False,
                }
            return jsonify({
                "error": "Vision model not loaded",
                "success": False,
                "command": "not_found"
            }), 503

        # Run Florence-2 + SmolVLM inference
        start_time = time.time()
        result = vision_service.navigate_with_detection(image_b64)
        inference_time = (time.time() - start_time) * 1000

        # Convert bboxes to frontend format
        detections = []
        for bbox in result.bboxes:
            detections.append({
                "label": bbox.get("label", "object"),
                "confidence": bbox.get("confidence", 0.9),
                "bbox": {
                    "x": bbox.get("x", 0),
                    "y": bbox.get("y", 0),
                    "width": bbox.get("w", 0),
                    "height": bbox.get("h", 0),
                }
            })

        # Update camera frame storage with detections
        with _camera_lock:
            _latest_frame["data"] = image_bytes
            _latest_frame["timestamp"] = datetime.utcnow()
            _latest_frame["width"] = width
            _latest_frame["height"] = height
            _latest_frame["device_id"] = device_id
            _latest_frame["detections"] = list(detections)  # Copy to avoid reference issues
            _latest_frame["has_detections"] = len(detections) > 0

        # Save to database
        nav_history = NavigationHistory.from_navigation_result(
            result,
            inference_time_ms=inference_time,
            device_id=device_id
        )
        nav_history.model_used = "florence2+smolvlm"
        db.session.add(nav_history)
        db.session.commit()

        # Build response
        response = result.to_dict()
        response["success"] = True
        response["inference_time_ms"] = round(inference_time, 2)
        response["history_id"] = nav_history.id
        response["detections_count"] = len(detections)

        # Log the command being sent to robot (print with flush for Docker)
        client_ip = request.remote_addr or 'unknown'
        print(f"=== ROBOT INFERENCE ===", flush=True)
        print(f"  From: {client_ip} ({device_id})", flush=True)
        print(f"  Command: {result.command.value}", flush=True)
        print(f"  Bin detected: {result.bin_detected}", flush=True)
        if result.bin_detected:
            print(f"  Position: {result.bin_position}, Size: {result.bin_size}", flush=True)
        print(f"  Bboxes: {len(detections)}, Time: {inference_time:.0f}ms", flush=True)

        return jsonify(response)

    except Exception as e:
        import traceback
        current_app.logger.error(f"Frame-infer error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


# ============================================================================
# CLASSIFICATION CAMERA ENDPOINTS (Second camera for bin classification)
# ============================================================================

@camera_bp.route("/api/camera/classification/frame", methods=["POST"])
@token_required
def post_classification_frame():
    """
    Robot posts classification camera frame (separate from navigation camera).

    Request JSON:
    {
        "image": "base64_encoded_jpeg",
        "width": 640,
        "height": 480,
        "device_id": "classification"  (optional)
    }
    """
    global _classification_frame

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode base64 image
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        with _classification_lock:
            _classification_frame["data"] = image_bytes
            _classification_frame["timestamp"] = datetime.utcnow()
            _classification_frame["width"] = data.get("width", 640)
            _classification_frame["height"] = data.get("height", 480)
            _classification_frame["device_id"] = data.get("device_id", "classification")

        return jsonify({
            "success": True,
            "timestamp": _classification_frame["timestamp"].isoformat() + "Z"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/camera/classification/latest", methods=["GET"])
def get_classification_frame():
    """
    Dashboard polls for latest classification camera frame.
    Returns JPEG image directly for easy display in <img> tag.
    """
    with _classification_lock:
        if _classification_frame["data"] is None:
            return jsonify({"error": "No classification camera frame available"}), 404

        # Check if frame is stale (older than 30 seconds)
        if _classification_frame["timestamp"]:
            age = (datetime.utcnow() - _classification_frame["timestamp"]).total_seconds()
            if age > 30:
                return jsonify({"error": "Classification camera frame stale", "age_seconds": age}), 404

        return Response(
            _classification_frame["data"],
            mimetype="image/jpeg",
            headers={
                "X-Frame-Timestamp": _classification_frame["timestamp"].isoformat() + "Z",
                "X-Frame-Width": str(_classification_frame["width"]),
                "X-Frame-Height": str(_classification_frame["height"]),
                "Cache-Control": "no-cache, no-store, must-revalidate",
            }
        )


@camera_bp.route("/api/camera/classification/status", methods=["GET"])
def classification_camera_status():
    """
    Get classification camera status.
    """
    with _classification_lock:
        has_frame = _classification_frame["data"] is not None
        timestamp = _classification_frame["timestamp"]
        age = None

        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()

        return jsonify({
            "has_frame": has_frame,
            "timestamp": timestamp.isoformat() + "Z" if timestamp else None,
            "age_seconds": age,
            "width": _classification_frame["width"] if has_frame else None,
            "height": _classification_frame["height"] if has_frame else None,
            "device_id": _classification_frame["device_id"],
            "is_live": has_frame and age is not None and age < 10,
            "classification_result": _classification_frame.get("classification_result"),
        })


@camera_bp.route("/api/camera/classification/result", methods=["POST"])
@token_required
def post_classification_result():
    """
    Robot posts classification result to associate with classification camera feed.

    Request JSON:
    {
        "fullness_level": "25-75%",
        "fullness_percent": 50,
        "label": "trash bin",
        "confidence": 0.95,
        "waste_type": "GENERAL"
    }
    """
    global _classification_frame

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing classification data"}), 400

    try:
        with _classification_lock:
            _classification_frame["classification_result"] = {
                "fullness_level": data.get("fullness_level", "unknown"),
                "fullness_percent": data.get("fullness_percent", 0),
                "label": data.get("label", "unknown"),
                "confidence": data.get("confidence", 0.0),
                "waste_type": data.get("waste_type", "UNKNOWN"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        return jsonify({
            "success": True,
            "message": "Classification result stored"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
