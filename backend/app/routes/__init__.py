"""
API Routes.
"""

from .health import health_bp
from .detections import detections_bp

__all__ = ["health_bp", "detections_bp"]
