"""
Authentication routes with JWT and HMAC verification.
"""

import os
import hmac
import hashlib
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify, current_app, g

from .. import db
from ..models.user import User

auth_bp = Blueprint("auth", __name__)

# Secret keys - in production these should be environment variables
JWT_SECRET = os.environ.get("JWT_SECRET", "waste-bin-monitor-jwt-secret-2024")
HMAC_SECRET = os.environ.get("HMAC_SECRET", "waste-bin-monitor-hmac-secret-2024")
JWT_EXPIRY_HOURS = 24


def generate_token(user: User) -> str:
    """Generate JWT token for user."""
    payload = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def generate_hmac(data: str) -> str:
    """Generate HMAC signature for data."""
    return hmac.new(
        HMAC_SECRET.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()


def verify_hmac(data: str, signature: str) -> bool:
    """Verify HMAC signature."""
    expected = generate_hmac(data)
    return hmac.compare_digest(expected, signature)


def token_required(f):
    """Decorator to require valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from header
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]

        if not token:
            return jsonify({"error": "Authentication token required"}), 401

        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Store user info in g for route access
        g.current_user = payload
        return f(*args, **kwargs)

    return decorated


def hmac_required(f):
    """Decorator to require valid HMAC signature for data integrity."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get HMAC signature from header
        signature = request.headers.get("X-HMAC-Signature")

        if not signature:
            return jsonify({"error": "HMAC signature required"}), 401

        # Get request body as string
        data = request.get_data(as_text=True)

        if not verify_hmac(data, signature):
            return jsonify({"error": "Invalid HMAC signature"}), 401

        return f(*args, **kwargs)

    return decorated


def auth_required(f):
    """Combined decorator requiring both token and HMAC."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Token verification
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]

        if not token:
            return jsonify({"error": "Authentication token required"}), 401

        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.current_user = payload

        # HMAC verification for POST/PUT/PATCH requests
        if request.method in ["POST", "PUT", "PATCH"]:
            signature = request.headers.get("X-HMAC-Signature")
            if signature:
                data = request.get_data(as_text=True)
                if not verify_hmac(data, signature):
                    return jsonify({"error": "Invalid HMAC signature"}), 401

        return f(*args, **kwargs)

    return decorated


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """
    Login endpoint.

    Request:
        {"username": "admin", "password": "admin"}

    Response:
        {"token": "...", "user": {...}, "hmac_secret": "..."}
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 401

    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()

    # Generate token
    token = generate_token(user)

    # Log with client info for debugging (print with flush for Docker)
    client_ip = request.remote_addr or 'unknown'
    user_agent = request.headers.get('User-Agent', 'unknown')[:50]
    print(f"=== ROBOT LOGIN ===", flush=True)
    print(f"  User: '{username}'", flush=True)
    print(f"  IP: {client_ip}", flush=True)
    print(f"  Agent: {user_agent}", flush=True)
    print(f"  Token expires in: {JWT_EXPIRY_HOURS}h", flush=True)

    return jsonify({
        "success": True,
        "token": token,
        "user": user.to_dict(),
        "hmac_secret": HMAC_SECRET,  # Share HMAC secret with authenticated clients
        "expires_in": JWT_EXPIRY_HOURS * 3600,
    })


@auth_bp.route("/api/auth/verify", methods=["GET"])
@token_required
def verify():
    """Verify token is valid."""
    return jsonify({
        "valid": True,
        "user": g.current_user,
    })


@auth_bp.route("/api/auth/logout", methods=["POST"])
@token_required
def logout():
    """Logout endpoint (client should discard token)."""
    current_app.logger.info(f"User '{g.current_user.get('username')}' logged out")
    return jsonify({"success": True, "message": "Logged out successfully"})


@auth_bp.route("/api/auth/hmac-secret", methods=["GET"])
@token_required
def get_hmac_secret():
    """Get HMAC secret for authenticated devices."""
    return jsonify({
        "hmac_secret": HMAC_SECRET,
    })
