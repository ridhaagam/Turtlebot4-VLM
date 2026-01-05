"""
Health check endpoints.
"""

from flask import Blueprint, jsonify
from datetime import datetime

from .. import db

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Returns server status and database connectivity.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": "unknown",
    }

    # Check database connectivity
    try:
        db.session.execute(db.text("SELECT 1"))
        status["database"] = "connected"
    except Exception as e:
        status["status"] = "degraded"
        status["database"] = f"error: {str(e)}"

    return jsonify(status), 200 if status["status"] == "healthy" else 503


@health_bp.route("/api/health/ready", methods=["GET"])
def readiness_check():
    """
    Readiness check for Kubernetes/Docker health probes.
    """
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify({"ready": True}), 200
    except Exception:
        return jsonify({"ready": False}), 503


@health_bp.route("/api/health/live", methods=["GET"])
def liveness_check():
    """
    Liveness check - always returns OK if the server is running.
    """
    return jsonify({"alive": True}), 200
