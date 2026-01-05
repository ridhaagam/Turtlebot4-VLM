"""
Application entry point.
"""

import os
import sys
import logging
from app import create_app, socketio

# Configure logging to stdout for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set Flask and Werkzeug loggers
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('flask').setLevel(logging.INFO)
logging.getLogger('app').setLevel(logging.INFO)

# Determine configuration
config_name = os.environ.get("FLASK_ENV", "development")

# Create application
app = create_app(config_name)

# Configure Flask app logger to use stdout
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(handler)

# Log startup info
app.logger.info("=" * 50)
app.logger.info("Flask Backend Starting")
app.logger.info(f"Config: {config_name}")
app.logger.info("=" * 50)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = config_name == "development"

    # Use socketio.run for WebSocket support
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True
    )
