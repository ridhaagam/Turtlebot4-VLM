"""
SLAM map endpoints for occupancy grid visualization.

The robot posts map updates from SLAM, and the dashboard polls for display.
"""

import base64
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

from ..routes.auth import token_required

maps_bp = Blueprint("maps", __name__)

# In-memory storage for latest map (ephemeral, not database)
_map_lock = threading.Lock()
_latest_map = {
    "data": None,  # bytes (JPEG/PNG)
    "timestamp": None,  # datetime
    "width": 0,
    "height": 0,
    "resolution": 0.05,  # meters per pixel
    "origin_x": 0.0,
    "origin_y": 0.0,
    "robot_x": None,  # robot position in map frame
    "robot_y": None,
    "robot_theta": None,
}


@maps_bp.route("/api/map/update", methods=["POST"])
@token_required
def post_map_update():
    """
    Robot posts occupancy grid map as image.

    Request JSON:
    {
        "image": "base64_encoded_image",
        "width": 384,
        "height": 384,
        "resolution": 0.05,
        "origin_x": -10.0,
        "origin_y": -10.0,
        "robot_x": 0.5,  (optional)
        "robot_y": 1.2,  (optional)
        "robot_theta": 0.0  (optional)
    }
    """
    global _latest_map

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode base64 image
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        with _map_lock:
            _latest_map = {
                "data": image_bytes,
                "timestamp": datetime.utcnow(),
                "width": data.get("width", 0),
                "height": data.get("height", 0),
                "resolution": data.get("resolution", 0.05),
                "origin_x": data.get("origin_x", 0.0),
                "origin_y": data.get("origin_y", 0.0),
                "robot_x": data.get("robot_x"),
                "robot_y": data.get("robot_y"),
                "robot_theta": data.get("robot_theta"),
            }

        return jsonify({
            "success": True,
            "timestamp": _latest_map["timestamp"].isoformat() + "Z"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@maps_bp.route("/api/map/latest", methods=["GET"])
def get_latest_map():
    """
    Dashboard polls for latest SLAM map.
    Returns image directly for easy display.
    """
    with _map_lock:
        if _latest_map["data"] is None:
            return jsonify({"error": "No map available"}), 404

        # Check if map is stale (older than 60 seconds)
        if _latest_map["timestamp"]:
            age = (datetime.utcnow() - _latest_map["timestamp"]).total_seconds()
            if age > 60:
                return jsonify({"error": "Map data stale", "age_seconds": age}), 404

        return Response(
            _latest_map["data"],
            mimetype="image/png",
            headers={
                "X-Map-Timestamp": _latest_map["timestamp"].isoformat() + "Z",
                "X-Map-Width": str(_latest_map["width"]),
                "X-Map-Height": str(_latest_map["height"]),
                "X-Map-Resolution": str(_latest_map["resolution"]),
                "X-Map-Origin-X": str(_latest_map["origin_x"]),
                "X-Map-Origin-Y": str(_latest_map["origin_y"]),
                "Cache-Control": "no-cache, no-store, must-revalidate",
            }
        )


@maps_bp.route("/api/map/latest/json", methods=["GET"])
def get_latest_map_json():
    """
    Get latest map as base64 JSON with metadata.
    """
    with _map_lock:
        if _latest_map["data"] is None:
            return jsonify({"error": "No map available"}), 404

        # Check if map is stale
        if _latest_map["timestamp"]:
            age = (datetime.utcnow() - _latest_map["timestamp"]).total_seconds()
            if age > 60:
                return jsonify({"error": "Map data stale", "age_seconds": age}), 404

        return jsonify({
            "image": base64.b64encode(_latest_map["data"]).decode("utf-8"),
            "timestamp": _latest_map["timestamp"].isoformat() + "Z",
            "width": _latest_map["width"],
            "height": _latest_map["height"],
            "resolution": _latest_map["resolution"],
            "origin_x": _latest_map["origin_x"],
            "origin_y": _latest_map["origin_y"],
            "robot_x": _latest_map["robot_x"],
            "robot_y": _latest_map["robot_y"],
            "robot_theta": _latest_map["robot_theta"],
        })


@maps_bp.route("/api/map/status", methods=["GET"])
def map_status():
    """
    Get map status (has map, timestamp, dimensions).
    """
    with _map_lock:
        has_map = _latest_map["data"] is not None
        timestamp = _latest_map["timestamp"]
        age = None

        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()

        return jsonify({
            "has_map": has_map,
            "timestamp": timestamp.isoformat() + "Z" if timestamp else None,
            "age_seconds": age,
            "width": _latest_map["width"] if has_map else None,
            "height": _latest_map["height"] if has_map else None,
            "resolution": _latest_map["resolution"] if has_map else None,
            "origin_x": _latest_map["origin_x"] if has_map else None,
            "origin_y": _latest_map["origin_y"] if has_map else None,
            "robot_x": _latest_map["robot_x"],
            "robot_y": _latest_map["robot_y"],
            "robot_theta": _latest_map["robot_theta"],
            "is_live": has_map and age is not None and age < 30,
        })
