"""
Navigation history model for tracking inference results.

Records each navigation inference with command, detections, and bounding boxes.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, CHAR, Boolean

from .. import db


class NavigationHistory(db.Model):
    """Records navigation inference history for dashboard display."""

    __tablename__ = "navigation_history"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Navigation command result
    command = Column(String(32), nullable=False)  # forward, left, right, arrived, not_found
    confidence = Column(Float, default=0.0)

    # Bin detection info
    bin_detected = Column(Boolean, default=False)
    bin_position = Column(String(32), nullable=True)  # left, center, right
    bin_size = Column(String(32), nullable=True)  # small, medium, large

    # Bounding boxes as JSON
    bboxes_json = Column(Text, nullable=True)
    detection_count = Column(Integer, default=0)

    # Raw model response
    raw_response = Column(Text, nullable=True)

    # Performance
    inference_time_ms = Column(Float, nullable=True)

    # Image reference (optional base64 thumbnail)
    image_thumbnail = Column(Text, nullable=True)

    # Device info
    device_id = Column(String(64), default="turtlebot4")

    # Model info
    model_used = Column(String(64), default="florence2+smolvlm")

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        import json

        bboxes = []
        if self.bboxes_json:
            try:
                bboxes = json.loads(self.bboxes_json)
            except:
                pass

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "command": self.command,
            "confidence": self.confidence,
            "bin_detected": self.bin_detected,
            "bin_position": self.bin_position,
            "bin_size": self.bin_size,
            "bboxes": bboxes,
            "detection_count": self.detection_count,
            "raw_response": self.raw_response,
            "inference_time_ms": self.inference_time_ms,
            "device_id": self.device_id,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }

    @classmethod
    def from_navigation_result(cls, result, inference_time_ms=None, device_id="turtlebot4"):
        """Create from NavigationResult."""
        import json

        nav = cls()
        nav.command = result.command.value if hasattr(result.command, 'value') else str(result.command)
        nav.confidence = result.confidence
        nav.bin_detected = result.bin_detected
        nav.bin_position = result.bin_position
        nav.bin_size = result.bin_size
        nav.raw_response = result.raw_response

        if result.bboxes:
            nav.bboxes_json = json.dumps(result.bboxes)
            nav.detection_count = len(result.bboxes)

        nav.inference_time_ms = inference_time_ms
        nav.device_id = device_id

        return nav
