"""
User model for authentication.
"""

from datetime import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.mysql import CHAR

from .. import db


def generate_uuid():
    return str(uuid.uuid4())


class User(db.Model):
    """User model for authentication."""
    __tablename__ = "users"

    id = db.Column(CHAR(36), primary_key=True, default=generate_uuid)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(32), default="user")  # admin, user, device
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify password."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "last_login": self.last_login.isoformat() + "Z" if self.last_login else None,
        }

    @classmethod
    def create_default_admin(cls):
        """Create default admin user if not exists."""
        admin = cls.query.filter_by(username="admin").first()
        if not admin:
            admin = cls(username="admin", role="admin")
            admin.set_password("admin")
            db.session.add(admin)
            db.session.commit()
            return admin
        return None
