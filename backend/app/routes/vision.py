"""
Vision API endpoints for SmolVLM2 inference.

Provides navigation and classification endpoints for the TurtleBot4 RPi4.
"""

import time
import logging
from flask import Blueprint, jsonify, request, current_app

from .. import db
from ..services.vision_model import vision_service
from ..models.navigation import NavigationHistory
from .auth import token_required

logger = logging.getLogger(__name__)

vision_bp = Blueprint("vision", __name__)


@vision_bp.route("/api/vision/status", methods=["GET"])
def vision_status():
    """
    Get vision model status.

    Returns:
        JSON with model status including device, dtype, GPU info
    """
    return jsonify(vision_service.get_status())


@vision_bp.route("/api/vision/navigate", methods=["POST"])
@token_required
def navigate():
    """
    Analyze image for bin detection and return navigation command.

    Request JSON:
    {
        "image": "base64_encoded_jpeg_image"
    }

    Response JSON:
    {
        "success": true,
        "command": "forward|left|right|arrived|not_found",
        "bin_detected": true,
        "position": "left|center|right|null",
        "size": "small|medium|large|null",
        "confidence": 0.9,
        "inference_time_ms": 1234.5,
        "raw_response": "..."
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided", "success": False}), 400

    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided", "success": False}), 400

    if not vision_service.is_loaded:
        return jsonify({
            "error": "Vision model not loaded. Call /api/vision/load first.",
            "success": False
        }), 503

    try:
        start_time = time.time()
        result = vision_service.navigate(image_base64)
        inference_time = (time.time() - start_time) * 1000

        response = result.to_dict()
        response["success"] = True
        response["inference_time_ms"] = round(inference_time, 2)

        current_app.logger.info(
            f"Navigation: cmd={result.command.value}, "
            f"bin={result.bin_detected}, pos={result.bin_position}, "
            f"size={result.bin_size}, time={inference_time:.0f}ms"
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Navigation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@vision_bp.route("/api/vision/classify", methods=["POST"])
@token_required
def classify():
    """
    Classify bin contents from image with two-step validation.

    Uses Florence-2 to verify bin presence, then SmolVLM for detailed analysis.
    Returns bin_found=false if no bin detected, enabling robot to resume search.

    Request JSON:
    {
        "image": "base64_encoded_jpeg_image"
    }

    Response JSON:
    {
        "success": true,
        "bin_found": true,
        "containers": {"count": 1, "type": "plastic bag"},
        "fill_level": {"percent": 12, "label": "LOW"},
        "action": "Excellent capacity available",
        "waste_type": "GENERAL",
        "scene": "A black trash bin with...",
        "objects_detected": ["trash bin", "plastic bag"],
        "summary": "Situation Summary\\nContainers: 1...",
        "confidence": 0.8,
        "inference_time_ms": 1234.5,
        "raw_response": "...",
        "fullness": "EMPTY"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided", "success": False}), 400

    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided", "success": False}), 400

    if not vision_service.is_loaded:
        return jsonify({
            "error": "Vision model not loaded. Call /api/vision/load first.",
            "success": False
        }), 503

    try:
        start_time = time.time()
        # Use new two-step classification with bin verification
        result = vision_service.classify_with_validation(image_base64)
        inference_time = (time.time() - start_time) * 1000

        response = result.to_dict()
        response["success"] = True
        response["inference_time_ms"] = round(inference_time, 2)

        # Detailed logging
        if result.bin_found:
            current_app.logger.info(
                f"Classification: bin_found=True, fullness={result.fullness}, "
                f"fill_level={result.fill_level_percent}%, type={result.waste_type}, "
                f"action={result.action}, time={inference_time:.0f}ms"
            )
        else:
            current_app.logger.warning(
                f"Classification: bin_found=False, objects={result.objects_detected}, "
                f"time={inference_time:.0f}ms - Robot should resume search"
            )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Classification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@vision_bp.route("/api/vision/navigate-florence2", methods=["POST"])
@token_required
def navigate_florence2():
    """
    Analyze image using Florence-2 for object detection + SmolVLM for navigation direction.

    This endpoint uses Florence-2 to detect bins, then SmolVLM to determine direction.
    Results are saved to database for history tracking.

    Request JSON:
    {
        "image": "base64_encoded_jpeg_image",
        "device_id": "turtlebot4" (optional)
    }

    Response JSON:
    {
        "success": true,
        "command": "forward|left|right|arrived|not_found",
        "bin_detected": true,
        "position": "left|center|right|null",
        "size": "small|medium|large|null",
        "confidence": 0.9,
        "bboxes": [{"x": 100, "y": 150, "w": 200, "h": 180, "label": "trash_bin", "confidence": 0.95}],
        "inference_time_ms": 1234.5,
        "raw_response": "...",
        "history_id": "uuid"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided", "success": False}), 400

    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided", "success": False}), 400

    device_id = data.get("device_id", "turtlebot4")

    if not vision_service.is_loaded:
        return jsonify({
            "error": "Vision model not loaded. Call /api/vision/load first.",
            "success": False
        }), 503

    try:
        start_time = time.time()
        result = vision_service.navigate_with_detection(image_base64)
        inference_time = (time.time() - start_time) * 1000

        # Save to database
        nav_history = NavigationHistory.from_navigation_result(
            result,
            inference_time_ms=inference_time,
            device_id=device_id
        )
        nav_history.model_used = "florence2+smolvlm"
        db.session.add(nav_history)
        db.session.commit()

        response = result.to_dict()
        response["success"] = True
        response["inference_time_ms"] = round(inference_time, 2)
        response["history_id"] = nav_history.id

        bbox_count = len(result.bboxes) if result.bboxes else 0
        current_app.logger.info(
            f"Florence-2 Navigation: cmd={result.command.value}, "
            f"bin={result.bin_detected}, pos={result.bin_position}, "
            f"size={result.bin_size}, bboxes={bbox_count}, time={inference_time:.0f}ms"
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Florence-2 navigation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@vision_bp.route("/api/vision/load", methods=["POST"])
@token_required
def load_model():
    """
    Manually load or reload the vision model.

    Request JSON (optional):
    {
        "model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    }

    Response JSON:
    {
        "success": true,
        "status": {...model status...}
    }
    """
    data = request.get_json() or {}
    model_name = data.get("model_name", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    current_app.logger.info(f"Loading vision model: {model_name}")

    try:
        success = vision_service.load_model(model_name)
        return jsonify({
            "success": success,
            "message": "Model loaded successfully" if success else "Failed to load model",
            "status": vision_service.get_status()
        })
    except Exception as e:
        logger.error(f"Model load error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False,
            "status": vision_service.get_status()
        }), 500


@vision_bp.route("/api/vision/navigation/latest", methods=["GET"])
def get_latest_navigation():
    """
    Get the latest navigation inference result.

    Only returns data if it's recent (within 5 seconds). Otherwise returns
    empty data so frontend clears stale overlays.

    Response JSON:
    {
        "id": "uuid",
        "timestamp": "...",
        "command": "forward",
        "confidence": 0.9,
        "bin_detected": true,
        "bin_position": "center",
        "bin_size": "medium",
        "bboxes": [...],
        "inference_time_ms": 123.4,
        "is_stale": false
    }
    """
    from datetime import datetime, timedelta

    try:
        latest = NavigationHistory.query.order_by(
            NavigationHistory.timestamp.desc()
        ).first()

        if not latest:
            return jsonify({
                "error": "No navigation history",
                "command": "search_left",
                "bin_detected": False,
                "bboxes": [],
                "is_stale": True
            }), 404

        # Check if data is stale (older than 5 seconds)
        age_seconds = (datetime.utcnow() - latest.timestamp).total_seconds()
        is_stale = age_seconds > 5.0

        result = latest.to_dict()
        result["age_seconds"] = round(age_seconds, 1)
        result["is_stale"] = is_stale

        # If stale, clear bboxes so frontend removes old overlays
        if is_stale:
            result["bboxes"] = []
            result["bin_detected"] = False

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching latest navigation: {e}")
        return jsonify({"error": str(e)}), 500


@vision_bp.route("/api/vision/navigation/history", methods=["GET"])
def get_navigation_history():
    """
    Get navigation history with pagination.

    Query params:
    - page: Page number (default 1)
    - per_page: Items per page (default 20, max 100)

    Response JSON:
    {
        "history": [...],
        "page": 1,
        "pages": 5,
        "total": 100,
        "has_next": true,
        "has_prev": false
    }
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)

        pagination = NavigationHistory.query.order_by(
            NavigationHistory.timestamp.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)

        return jsonify({
            "history": [h.to_dict() for h in pagination.items],
            "page": pagination.page,
            "pages": pagination.pages,
            "total": pagination.total,
            "has_next": pagination.has_next,
            "has_prev": pagination.has_prev
        })

    except Exception as e:
        logger.error(f"Error fetching navigation history: {e}")
        return jsonify({"error": str(e)}), 500


@vision_bp.route("/api/vision/navigation/stats", methods=["GET"])
def get_navigation_stats():
    """
    Get navigation statistics.

    Response JSON:
    {
        "total_inferences": 100,
        "bins_detected": 75,
        "commands": {"forward": 30, "left": 20, "right": 20, "arrived": 5},
        "avg_inference_time_ms": 123.4,
        "avg_confidence": 0.85
    }
    """
    try:
        from sqlalchemy import func

        total = NavigationHistory.query.count()
        bins_detected = NavigationHistory.query.filter_by(bin_detected=True).count()

        # Command distribution
        command_stats = db.session.query(
            NavigationHistory.command,
            func.count(NavigationHistory.id)
        ).group_by(NavigationHistory.command).all()

        commands = {cmd: count for cmd, count in command_stats}

        # Average inference time and confidence
        avg_stats = db.session.query(
            func.avg(NavigationHistory.inference_time_ms),
            func.avg(NavigationHistory.confidence)
        ).first()

        return jsonify({
            "total_inferences": total,
            "bins_detected": bins_detected,
            "commands": commands,
            "avg_inference_time_ms": round(avg_stats[0] or 0, 2),
            "avg_confidence": round(avg_stats[1] or 0, 3)
        })

    except Exception as e:
        logger.error(f"Error fetching navigation stats: {e}")
        return jsonify({"error": str(e)}), 500
