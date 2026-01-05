"""
Platform detection module for cross-platform compatibility.
Supports: NVIDIA Jetson Orin, Raspberry Pi 4, and generic x86/ARM systems.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Platform(Enum):
    JETSON = "jetson"
    RPI = "rpi"
    GENERIC = "generic"


_detected_platform: Optional[Platform] = None


def detect_platform() -> Platform:
    """
    Auto-detect the hardware platform at runtime.

    Returns:
        Platform enum indicating the detected platform.
    """
    global _detected_platform

    if _detected_platform is not None:
        return _detected_platform

    device_tree_path = Path("/proc/device-tree/model")

    if device_tree_path.exists():
        try:
            model = device_tree_path.read_text().lower()

            if "nvidia" in model or "jetson" in model:
                _detected_platform = Platform.JETSON
                logger.info(f"Detected platform: NVIDIA Jetson ({model.strip()})")
                return _detected_platform

            if "raspberry pi" in model:
                _detected_platform = Platform.RPI
                logger.info(f"Detected platform: Raspberry Pi ({model.strip()})")
                return _detected_platform

        except Exception as e:
            logger.warning(f"Failed to read device tree: {e}")

    _detected_platform = Platform.GENERIC
    logger.info("Detected platform: Generic (no specific hardware detected)")
    return _detected_platform


def get_device() -> str:
    """
    Get the appropriate torch device based on platform capabilities.

    Returns:
        "cuda" for Jetson with CUDA support, "cpu" otherwise.
    """
    platform = detect_platform()

    if platform == Platform.JETSON:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA is available, using GPU")
                return "cuda"
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")

    logger.info("Using CPU for inference")
    return "cpu"


def get_platform_config_overrides() -> dict:
    """
    Get platform-specific configuration overrides.

    Returns:
        Dictionary of configuration overrides for the detected platform.
    """
    platform = detect_platform()

    if platform == Platform.JETSON:
        return {
            "model": {
                "device": "cuda",
                "use_fp16": True,
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 5,
            },
            "inference": {
                "batch_size": 1,
                "target_fps": 5,
            }
        }

    elif platform == Platform.RPI:
        return {
            "model": {
                "device": "cpu",
                "use_fp16": False,
                "use_quantization": True,
            },
            "camera": {
                "width": 320,
                "height": 240,
                "fps": 2,
            },
            "inference": {
                "batch_size": 1,
                "target_fps": 1,
            }
        }

    # Generic platform
    return {
        "model": {
            "device": "cpu",
            "use_fp16": False,
        },
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 5,
        },
        "inference": {
            "batch_size": 1,
            "target_fps": 3,
        }
    }


def get_recommended_model() -> str:
    """
    Get the recommended model based on platform capabilities.

    Returns:
        HuggingFace model identifier.
    """
    platform = detect_platform()

    if platform == Platform.JETSON:
        return "microsoft/Florence-2-base"
    elif platform == Platform.RPI:
        # Smaller model for RPi due to memory constraints
        return "microsoft/Florence-2-base"
    else:
        return "microsoft/Florence-2-base"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Platform: {detect_platform()}")
    print(f"Device: {get_device()}")
    print(f"Recommended model: {get_recommended_model()}")
    print(f"Config overrides: {get_platform_config_overrides()}")
