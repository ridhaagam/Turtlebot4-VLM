"""
Mission model for tracking autonomous mission history.

Records each mission run with steps, status, and related detections.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, CHAR
from sqlalchemy.orm import relationship

from .. import db


class Mission(db.Model):
    """Records each autonomous mission run."""

    __tablename__ = "missions"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Mission status
    status = Column(String(32), default="running")  # running, completed, failed, cancelled

    # Mission type
    mission_type = Column(String(64), default="full_mission")  # full_mission, find_bin, classify

    # Step results stored as JSON string
    steps_json = Column(Text, nullable=True)  # JSON array of step outcomes

    # Summary statistics
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)

    # Error information if failed
    error_message = Column(Text, nullable=True)

    # Device info
    device_id = Column(String(64), default="turtlebot4")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """Convert mission to dictionary."""
        import json

        steps = []
        if self.steps_json:
            try:
                steps = json.loads(self.steps_json)
            except:
                pass

        return {
            "id": self.id,
            "start_time": self.start_time.isoformat() + "Z" if self.start_time else None,
            "end_time": self.end_time.isoformat() + "Z" if self.end_time else None,
            "status": self.status,
            "mission_type": self.mission_type,
            "steps": steps,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "device_id": self.device_id,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data):
        """Create mission from dictionary."""
        import json
        from dateutil.parser import parse

        mission = cls()
        mission.id = data.get("id", str(uuid.uuid4()))

        if data.get("start_time"):
            mission.start_time = parse(data["start_time"])
        if data.get("end_time"):
            mission.end_time = parse(data["end_time"])

        mission.status = data.get("status", "running")
        mission.mission_type = data.get("mission_type", "full_mission")

        if data.get("steps"):
            mission.steps_json = json.dumps(data["steps"])

        mission.total_steps = data.get("total_steps", 0)
        mission.completed_steps = data.get("completed_steps", 0)
        mission.duration_seconds = data.get("duration_seconds")
        mission.error_message = data.get("error_message")
        mission.device_id = data.get("device_id", "turtlebot4")

        return mission

    def add_step(self, step_name: str, success: bool, message: str = None, duration_ms: float = None):
        """Add a step result to the mission."""
        import json

        steps = []
        if self.steps_json:
            try:
                steps = json.loads(self.steps_json)
            except:
                pass

        step = {
            "name": step_name,
            "success": success,
            "message": message,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        steps.append(step)

        self.steps_json = json.dumps(steps)
        self.total_steps = len(steps)
        self.completed_steps = sum(1 for s in steps if s.get("success", False))

    def complete(self, success: bool = True, error_message: str = None):
        """Mark mission as completed."""
        self.end_time = datetime.utcnow()
        self.status = "completed" if success else "failed"
        self.error_message = error_message

        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class MapSnapshot(db.Model):
    """Snapshot of SLAM map at a point in time."""

    __tablename__ = "map_snapshots"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Image storage
    image_path = Column(String(256), nullable=True)

    # Map metadata
    width = Column(Integer)
    height = Column(Integer)
    resolution = Column(Float)  # meters per pixel
    origin_x = Column(Float)
    origin_y = Column(Float)

    # Robot position when snapshot was taken
    robot_x = Column(Float, nullable=True)
    robot_y = Column(Float, nullable=True)
    robot_theta = Column(Float, nullable=True)

    # Mission association
    mission_id = Column(CHAR(36), nullable=True, index=True)

    # Device info
    device_id = Column(String(64), default="turtlebot4")

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_theta": self.robot_theta,
            "mission_id": self.mission_id,
            "device_id": self.device_id,
        }
