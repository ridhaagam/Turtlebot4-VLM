"""
Detection CRUD endpoints with authentication.
"""

import os
import base64
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import desc

from .. import db, socketio
from ..models import Detection, DetectionObject
from .auth import auth_required, token_required

detections_bp = Blueprint("detections", __name__)


def _draw_bboxes_on_image(image_base64: str, detections: list) -> str:
    """
    Draw bounding boxes on an image and return as base64.

    Args:
        image_base64: Base64 encoded image
        detections: List of detection dicts with bbox info

    Returns:
        Base64 encoded annotated image
    """
    try:
        # Decode image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        # Color scheme based on object type
        COLORS = {
            'bin_empty': (34, 197, 94),      # Green - 0-25%
            'bin_partial': (132, 204, 22),    # Lime - 25-75%
            'bin_mostly': (245, 158, 11),     # Amber - 75-90%
            'bin_full': (239, 68, 68),        # Red - 90-100%
            'content': (59, 130, 246),        # Blue - content objects
            'object': (156, 163, 175),        # Gray - regular objects
        }

        for det in detections:
            bbox = det.get('bbox', {})
            x = bbox.get('x', det.get('bbox_x', 0))
            y = bbox.get('y', det.get('bbox_y', 0))
            w = bbox.get('width', det.get('bbox_width', 0))
            h = bbox.get('height', det.get('bbox_height', 0))

            if w == 0 or h == 0:
                continue

            label = det.get('label', 'object')
            is_bin = det.get('is_bin', False)
            is_content = det.get('is_content', False)
            fullness_percent = det.get('bin_fullness_percent')

            # Determine color
            if is_bin:
                if fullness_percent is not None:
                    if fullness_percent <= 25:
                        color = COLORS['bin_empty']
                    elif fullness_percent <= 75:
                        color = COLORS['bin_partial']
                    elif fullness_percent <= 90:
                        color = COLORS['bin_mostly']
                    else:
                        color = COLORS['bin_full']
                else:
                    color = COLORS['bin_empty']
                thickness = 3
            elif is_content:
                color = COLORS['content']
                thickness = 2
            else:
                color = COLORS['object']
                thickness = 2

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Draw label background
            label_text = label
            if is_bin and fullness_percent is not None:
                label_text = f"{label} ({fullness_percent}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, 1)

            label_y = max(y - 5, text_h + 5)
            cv2.rectangle(image, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 3), color, -1)
            cv2.putText(image, label_text, (x + 5, label_y), font, font_scale, (255, 255, 255), 1)

        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        current_app.logger.error(f"Error drawing bboxes: {e}")
        return None


@detections_bp.route("/api/detections", methods=["POST"])
@auth_required  # Requires token + HMAC for edge devices
def create_detection():
    """
    Receive detection data from edge device.

    Expected JSON:
    {
        "timestamp": "2025-12-01T10:30:00Z",
        "frame_id": "uuid",
        "detections": [
            {
                "label": "person",
                "confidence": 0.95,
                "bbox": {"x": 100, "y": 150, "width": 200, "height": 300}
            }
        ],
        "inference_time_ms": 45.2,
        "image_width": 640,
        "image_height": 480,
        "bin_detected": true,
        "bin_count": 1,
        "overall_fullness": "0-25%",
        "overall_fullness_percent": 20,
        "status_summary": "1 bin detected - Nearly empty",
        "image_base64": "..."  # Optional
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        current_app.logger.info(
            f"Received detection: frame={data.get('frame_id', 'N/A')[:8]}... "
            f"objects={len(data.get('detections', []))} "
            f"bins={data.get('bin_count', 0)} "
            f"fullness={data.get('overall_fullness', 'N/A')}"
        )

        # Create detection record
        detection = Detection.from_dict(data)
        db.session.add(detection)

        # IMPORTANT: Flush to generate the detection.id before using it as FK
        db.session.flush()

        # Handle images if provided (original + bbox annotated)
        image_base64 = data.get("image_base64")
        if image_base64:
            image_path = _save_image(image_base64, detection.frame_id, suffix="")
            detection.image_path = image_path
            current_app.logger.info(f"  Original image saved: {image_path}")

        image_bbox_base64 = data.get("image_bbox_base64")

        # Auto-generate bbox image if not provided but we have detections
        if not image_bbox_base64 and image_base64 and data.get("detections"):
            current_app.logger.info(f"  Generating bbox image from {len(data.get('detections', []))} detections...")
            image_bbox_base64 = _draw_bboxes_on_image(image_base64, data.get("detections", []))
            if image_bbox_base64:
                current_app.logger.info(f"  BBox image generated successfully")

        if image_bbox_base64:
            bbox_image_path = _save_image(image_bbox_base64, detection.frame_id, suffix="_bbox")
            detection.image_bbox_path = bbox_image_path
            current_app.logger.info(f"  BBox image saved: {bbox_image_path}")

        # Create detection objects (now detection.id is guaranteed to exist)
        for obj_data in data.get("detections", []):
            obj = DetectionObject.from_dict(obj_data, detection.id)
            db.session.add(obj)

        db.session.commit()

        current_app.logger.info(
            f"  Stored: id={detection.id[:8]}... | "
            f"Summary: {detection.status_summary or 'No bins'}"
        )

        # Emit WebSocket event for real-time updates
        detection_dict = detection.to_dict()
        socketio.emit("new_detection", detection_dict)
        current_app.logger.info(f"  WebSocket: Emitted new_detection event")

        return jsonify({
            "success": True,
            "id": detection.id,
            "object_count": len(data.get("detections", []))
        }), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Failed to create detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@detections_bp.route("/api/detections", methods=["GET"])
def list_detections():
    """
    List detections with pagination and filtering.

    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - label: Filter by object label
    - device_id: Filter by device ID
    - start_date: Filter by start date (ISO format)
    - end_date: Filter by end date (ISO format)
    """
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    label = request.args.get("label")
    device_id = request.args.get("device_id")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    query = Detection.query

    # Apply filters
    if device_id:
        query = query.filter(Detection.device_id == device_id)

    if start_date:
        try:
            from dateutil import parser
            start = parser.parse(start_date)
            query = query.filter(Detection.timestamp >= start)
        except ValueError:
            pass

    if end_date:
        try:
            from dateutil import parser
            end = parser.parse(end_date)
            query = query.filter(Detection.timestamp <= end)
        except ValueError:
            pass

    if label:
        query = query.join(DetectionObject).filter(DetectionObject.label == label)

    # Order and paginate
    query = query.order_by(desc(Detection.created_at))
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "detections": [d.to_dict() for d in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "per_page": pagination.per_page,
        "pages": pagination.pages,
        "has_next": pagination.has_next,
        "has_prev": pagination.has_prev,
    })


@detections_bp.route("/api/detections/latest", methods=["GET"])
def get_latest_detection():
    """Get the most recent detection."""
    detection = Detection.query.order_by(desc(Detection.created_at)).first()

    if not detection:
        return jsonify({"error": "No detections found"}), 404

    return jsonify(detection.to_dict())


@detections_bp.route("/api/detections/<detection_id>", methods=["GET"])
def get_detection(detection_id: str):
    """Get a specific detection by ID."""
    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({"error": "Detection not found"}), 404

    return jsonify(detection.to_dict())


@detections_bp.route("/api/detections/<detection_id>", methods=["DELETE"])
def delete_detection(detection_id: str):
    """Delete a detection and its associated objects."""
    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({"error": "Detection not found"}), 404

    try:
        # Delete image file if exists
        if detection.image_path:
            try:
                Path(detection.image_path).unlink(missing_ok=True)
            except Exception:
                pass

        db.session.delete(detection)
        db.session.commit()

        return jsonify({"success": True, "deleted_id": detection_id})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@detections_bp.route("/api/detections/stats", methods=["GET"])
def get_stats():
    """Get detection statistics including bin fullness data."""
    from sqlalchemy import func

    total_detections = Detection.query.count()
    total_objects = DetectionObject.query.count()

    # Bin-specific stats
    bins_detected = Detection.query.filter(Detection.bin_detected == True).count()

    # Get fullness distribution
    fullness_distribution = db.session.query(
        Detection.overall_fullness,
        func.count(Detection.id).label("count")
    ).filter(
        Detection.overall_fullness.isnot(None)
    ).group_by(Detection.overall_fullness).all()

    # Get average fullness from recent detections
    avg_fullness = db.session.query(
        func.avg(Detection.overall_fullness_percent)
    ).filter(
        Detection.overall_fullness_percent.isnot(None)
    ).scalar() or 0

    # Get label counts
    label_counts = db.session.query(
        DetectionObject.label,
        func.count(DetectionObject.id).label("count")
    ).group_by(DetectionObject.label).order_by(desc("count")).limit(10).all()

    # Get recent activity (detections per hour for last 24 hours)
    recent_activity = db.session.query(
        func.date_format(Detection.created_at, "%Y-%m-%d %H:00:00").label("hour"),
        func.count(Detection.id).label("count")
    ).filter(
        Detection.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
    ).group_by("hour").all()

    # Get latest bin status
    latest_bin = Detection.query.filter(
        Detection.bin_detected == True
    ).order_by(desc(Detection.created_at)).first()

    latest_bin_status = None
    if latest_bin:
        latest_bin_status = {
            "fullness": latest_bin.overall_fullness,
            "percent": latest_bin.overall_fullness_percent,
            "summary": latest_bin.status_summary,
            "timestamp": latest_bin.timestamp.isoformat() + "Z" if latest_bin.timestamp else None,
        }

    return jsonify({
        "total_detections": total_detections,
        "total_objects": total_objects,
        "bins_detected": bins_detected,
        "average_fullness_percent": round(avg_fullness, 1),
        "fullness_distribution": [{"level": l or "unknown", "count": c} for l, c in fullness_distribution],
        "label_counts": [{"label": l, "count": c} for l, c in label_counts],
        "recent_activity": [{"hour": h, "count": c} for h, c in recent_activity],
        "latest_bin_status": latest_bin_status,
    })


@detections_bp.route("/api/detections/export", methods=["GET"])
def export_detections():
    """
    Export detections as JSON.

    Query parameters:
    - start_date: Start date (ISO format)
    - end_date: End date (ISO format)
    - limit: Maximum number of records (default: 1000)
    """
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = min(request.args.get("limit", 1000, type=int), 10000)

    query = Detection.query

    if start_date:
        try:
            from dateutil import parser
            start = parser.parse(start_date)
            query = query.filter(Detection.timestamp >= start)
        except ValueError:
            pass

    if end_date:
        try:
            from dateutil import parser
            end = parser.parse(end_date)
            query = query.filter(Detection.timestamp <= end)
        except ValueError:
            pass

    detections = query.order_by(desc(Detection.created_at)).limit(limit).all()

    return jsonify({
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "count": len(detections),
        "detections": [d.to_dict() for d in detections]
    })


def _save_image(image_base64: str, frame_id: str, suffix: str = "") -> str:
    """
    Save base64 encoded image to disk.

    Args:
        image_base64: Base64 encoded image data.
        frame_id: Frame ID for filename.
        suffix: Optional suffix for filename (e.g., "_bbox").

    Returns:
        Path to saved image.
    """
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
    Path(upload_folder).mkdir(parents=True, exist_ok=True)

    # Create date-based subdirectory
    date_folder = datetime.utcnow().strftime("%Y/%m/%d")
    full_path = Path(upload_folder) / date_folder
    full_path.mkdir(parents=True, exist_ok=True)

    # Save image with optional suffix
    filename = f"{frame_id}{suffix}.jpg"
    filepath = full_path / filename

    try:
        image_data = base64.b64decode(image_base64)
        with open(filepath, "wb") as f:
            f.write(image_data)
        return str(filepath)
    except Exception as e:
        current_app.logger.error(f"Failed to save image: {e}")
        return None


@detections_bp.route("/api/images/<path:image_path>", methods=["GET"])
def get_image(image_path: str):
    """Serve detection images."""
    from flask import send_file

    upload_folder = current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
    full_path = Path(upload_folder) / image_path

    if not full_path.exists():
        return jsonify({"error": "Image not found"}), 404

    # Security check - ensure path is within upload folder
    try:
        full_path.resolve().relative_to(Path(upload_folder).resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 403

    return send_file(full_path, mimetype="image/jpeg")


@detections_bp.route("/api/detections/<detection_id>/image", methods=["GET"])
def get_detection_image(detection_id: str):
    """Get the original image for a specific detection."""
    from flask import send_file

    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({"error": "Detection not found"}), 404

    if not detection.image_path:
        return jsonify({"error": "No image available for this detection"}), 404

    image_path = Path(detection.image_path)

    if not image_path.exists():
        return jsonify({"error": "Image file not found"}), 404

    return send_file(image_path, mimetype="image/jpeg")


@detections_bp.route("/api/detections/<detection_id>/image/bbox", methods=["GET"])
def get_detection_bbox_image(detection_id: str):
    """Get the bbox-annotated image for a specific detection."""
    from flask import send_file

    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({"error": "Detection not found"}), 404

    # Try bbox image first, fall back to original
    if detection.image_bbox_path:
        image_path = Path(detection.image_bbox_path)
        if image_path.exists():
            return send_file(image_path, mimetype="image/jpeg")

    # Fallback to original image
    if detection.image_path:
        image_path = Path(detection.image_path)
        if image_path.exists():
            return send_file(image_path, mimetype="image/jpeg")

    return jsonify({"error": "No image available for this detection"}), 404
