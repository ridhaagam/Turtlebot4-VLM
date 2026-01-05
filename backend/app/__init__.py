"""
Flask application factory for the Detection Dashboard backend.
"""

from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_migrate import Migrate

db = SQLAlchemy()
socketio = SocketIO()
migrate = Migrate()


def create_app(config_name: str = "development"):
    """
    Application factory.

    Args:
        config_name: Configuration to use (development, production, testing)

    Returns:
        Flask application instance.
    """
    app = Flask(__name__)

    # Load configuration
    from .config import config
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    socketio.init_app(app, cors_allowed_origins="*")

    # Register blueprints
    from .routes.health import health_bp
    from .routes.detections import detections_bp
    from .routes.auth import auth_bp
    from .routes.vision import vision_bp
    from .routes.camera import camera_bp
    from .routes.maps import maps_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(detections_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(vision_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(maps_bp)

    # Initialize database
    with app.app_context():
        import logging
        import os
        logger = logging.getLogger(__name__)

        # Import all models to ensure they're registered
        from .models.user import User
        from .models.detection import Detection, DetectionObject
        from .models.mission import Mission, MapSnapshot
        from .models.navigation import NavigationHistory

        # Check if migrations folder exists
        migrations_path = os.path.join(os.path.dirname(__file__), '..', 'migrations')
        has_migrations = os.path.exists(migrations_path)

        if has_migrations:
            try:
                from flask_migrate import upgrade, stamp
                upgrade()
                logger.info("Database migrations applied successfully")
            except Exception as e:
                error_str = str(e).lower()
                if "no such table" in error_str or "doesn't exist" in error_str:
                    logger.info("Fresh database detected, creating tables...")
                    db.create_all()
                    try:
                        stamp()
                    except:
                        pass
                else:
                    logger.warning(f"Migration warning (continuing): {e}")
                    db.create_all()
        else:
            # No migrations folder - just use create_all
            logger.info("No migrations folder - using db.create_all()")
            db.create_all()

        # Create default admin user
        User.create_default_admin()

    # Load vision models in background thread (doesn't block startup)
    import os
    if os.environ.get("LOAD_VISION_MODEL", "true").lower() == "true":
        import threading
        from .services.vision_model import vision_service

        def load_vision_models():
            import logging
            import time
            logger = logging.getLogger(__name__)

            # Load SmolVLM2 model
            logger.info("=" * 50)
            logger.info("Starting vision model loading...")
            logger.info("=" * 50)

            model_name = os.environ.get(
                "VISION_MODEL",
                "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
            )

            start = time.time()
            logger.info(f"Loading SmolVLM2 model: {model_name}")
            success = vision_service.load_model(model_name)
            elapsed = time.time() - start

            if success:
                logger.info(f"SmolVLM2 model loaded in {elapsed:.1f}s")
            else:
                logger.error(f"Failed to load SmolVLM2 model after {elapsed:.1f}s")
                return

            # Load Florence-2 model for object detection
            florence_model = os.environ.get(
                "FLORENCE_MODEL",
                "microsoft/Florence-2-base"
            )

            start = time.time()
            logger.info(f"Loading Florence-2 model: {florence_model}")
            florence_success = vision_service.load_florence2(florence_model)
            elapsed = time.time() - start

            if florence_success:
                logger.info(f"Florence-2 model loaded in {elapsed:.1f}s")
            else:
                logger.warning(f"Failed to load Florence-2 model after {elapsed:.1f}s (will retry on first use)")

            logger.info("=" * 50)
            logger.info("Vision model loading complete!")
            logger.info(f"Status: {vision_service.get_status()}")
            logger.info("=" * 50)

        threading.Thread(target=load_vision_models, daemon=True).start()

    return app
