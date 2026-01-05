"""
Detection database models for waste bin monitoring system.
"""

from datetime import datetime
import uuid
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy import event

from .. import db


def generate_uuid():
    return str(uuid.uuid4())


class Detection(db.Model):
    """
    Represents a single detection frame with associated objects and bin status.
    """
    __tablename__ = "detections"

    id = db.Column(CHAR(36), primary_key=True, default=generate_uuid)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    frame_id = db.Column(db.String(64), nullable=False, unique=True)
    image_path = db.Column(db.String(256), nullable=True)  # Original image
    image_bbox_path = db.Column(db.String(256), nullable=True)  # Annotated image with bboxes
    image_width = db.Column(db.Integer, nullable=True)
    image_height = db.Column(db.Integer, nullable=True)
    inference_time_ms = db.Column(db.Float, nullable=True)
    device_id = db.Column(db.String(64), nullable=True, default="default")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Bin status fields
    bin_detected = db.Column(db.Boolean, default=False)
    bin_count = db.Column(db.Integer, default=0)
    overall_fullness = db.Column(db.String(32), nullable=True)  # "0-25%", "25-75%", "75-90%", "90-100%"
    overall_fullness_percent = db.Column(db.Integer, nullable=True)
    status_summary = db.Column(db.Text, nullable=True)  # Text type for longer VLM summaries

    # Relationships
    objects = db.relationship(
        "DetectionObject",
        backref="detection",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def to_dict(self, include_objects: bool = True) -> dict:
        """Convert to dictionary representation."""
        data = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "image_bbox_path": self.image_bbox_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": self.inference_time_ms,
            "device_id": self.device_id,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "object_count": self.objects.count(),
            # Bin status
            "bin_detected": self.bin_detected,
            "bin_count": self.bin_count,
            "overall_fullness": self.overall_fullness,
            "overall_fullness_percent": self.overall_fullness_percent,
            "status_summary": self.status_summary,
        }

        if include_objects:
            data["detections"] = [obj.to_dict() for obj in self.objects.all()]

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Detection":
        """Create from dictionary data."""
        from dateutil import parser

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = parser.parse(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)

        return cls(
            frame_id=data.get("frame_id", generate_uuid()),
            timestamp=timestamp or datetime.utcnow(),
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            inference_time_ms=data.get("inference_time_ms"),
            device_id=data.get("device_id", "default"),
            # Bin status from edge device
            bin_detected=data.get("bin_detected", False),
            bin_count=data.get("bin_count", 0),
            overall_fullness=data.get("overall_fullness"),
            overall_fullness_percent=data.get("overall_fullness_percent"),
            status_summary=data.get("status_summary", ""),
        )


class DetectionObject(db.Model):
    """
    Represents a single detected object within a frame.
    Includes bin fullness classification for waste containers.
    """
    __tablename__ = "detection_objects"

    id = db.Column(CHAR(36), primary_key=True, default=generate_uuid)
    detection_id = db.Column(
        CHAR(36),
        db.ForeignKey("detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    label = db.Column(db.String(128), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    bbox_x = db.Column(db.Integer, nullable=False)
    bbox_y = db.Column(db.Integer, nullable=False)
    bbox_width = db.Column(db.Integer, nullable=False)
    bbox_height = db.Column(db.Integer, nullable=False)

    # Bin classification fields (only populated for waste containers)
    is_bin = db.Column(db.Boolean, default=False)  # True if this object is a waste bin
    bin_fullness = db.Column(db.String(32), nullable=True)  # "0-25%", "25-75%", "75-90%", "90-100%"
    bin_fullness_percent = db.Column(db.Integer, nullable=True)

    # Content object fields (for objects detected inside bins)
    is_content = db.Column(db.Boolean, default=False)  # True if detected inside a bin
    parent_bin_id = db.Column(db.Integer, nullable=True)  # Index of parent bin in detection

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": {
                "x": self.bbox_x,
                "y": self.bbox_y,
                "width": self.bbox_width,
                "height": self.bbox_height,
            },
            "is_bin": self.is_bin,
            "is_content": self.is_content,
        }
        # Include bin fullness if this is a bin
        if self.bin_fullness is not None:
            result["bin_fullness"] = self.bin_fullness
            result["bin_fullness_percent"] = self.bin_fullness_percent
        # Include parent bin id for content objects
        if self.parent_bin_id is not None:
            result["parent_bin_id"] = self.parent_bin_id

        return result

    @classmethod
    def from_dict(cls, data: dict, detection_id: str) -> "DetectionObject":
        """Create from dictionary data."""
        # Handle both nested bbox format and flat format
        bbox = data.get("bbox", {})
        bbox_x = bbox.get("x") if bbox else data.get("bbox_x", 0)
        bbox_y = bbox.get("y") if bbox else data.get("bbox_y", 0)
        bbox_width = bbox.get("width") if bbox else data.get("bbox_width", 0)
        bbox_height = bbox.get("height") if bbox else data.get("bbox_height", 0)

        return cls(
            detection_id=detection_id,
            label=data.get("label", "unknown"),
            confidence=data.get("confidence", 0.0),
            bbox_x=bbox_x or 0,
            bbox_y=bbox_y or 0,
            bbox_width=bbox_width or 0,
            bbox_height=bbox_height or 0,
            # Bin classification
            is_bin=data.get("is_bin", False),
            bin_fullness=data.get("bin_fullness"),
            bin_fullness_percent=data.get("bin_fullness_percent"),
            # Content object fields
            is_content=data.get("is_content", False),
            parent_bin_id=data.get("parent_bin_id"),
        )
