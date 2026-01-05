#!/usr/bin/env python3
"""
smolvlm_navigator.py
SmolVLM-256M-Instruct based visual navigation for trash bin detection.

This module provides visual navigation commands using the SmolVLM-256M-Instruct
model to identify trash bins and determine movement commands (FORWARD, LEFT,
RIGHT, ARRIVED, NOT_FOUND) based on bin position in the image.
"""

import sys
import os
import logging
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
from PIL import Image

# Add venv packages to path
VENV_SITE_PACKAGES = '/home/g3ubuntu/ROS/embedded-system-final/edge/venv/lib/python3.12/site-packages'
if VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

logger = logging.getLogger(__name__)


class NavigationCommand(Enum):
    """Navigation commands output by SmolVLM."""
    FORWARD = "forward"      # Bin is centered, move toward it
    LEFT = "left"            # Bin is on the left side, turn left
    RIGHT = "right"          # Bin is on the right side, turn right
    ARRIVED = "arrived"      # Bin is very close, filling most of frame
    NOT_FOUND = "not_found"  # No bin visible
    SEARCHING = "searching"  # Actively searching (rotating/moving)


class PositionVerification(Enum):
    """Position verification results for classification."""
    READY = "ready"          # Positioned correctly to see inside bin
    ADJUST_LEFT = "adjust_left"
    ADJUST_RIGHT = "adjust_right"
    ADJUST_FORWARD = "adjust_forward"
    ADJUST_BACK = "adjust_back"
    NOT_A_BIN = "not_a_bin"  # No bin in view


@dataclass
class NavigationResult:
    """Result of navigation inference."""
    command: NavigationCommand
    confidence: float
    raw_response: str
    bin_detected: bool
    bin_position: Optional[str] = None  # "left", "center", "right"
    bin_size: Optional[str] = None      # "small", "medium", "large"


@dataclass
class SmolVLMConfig:
    """Configuration for SmolVLM navigator."""
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    device: str = "cpu"  # RPi4 uses CPU
    max_new_tokens: int = 50
    temperature: float = 0.1  # Low temperature for consistent outputs
    # Navigation thresholds
    arrival_size_threshold: float = 0.4  # Bin fills 40%+ of frame = arrived
    center_tolerance: float = 0.2        # Within 20% of center = forward


class SmolVLMNavigator:
    """
    SmolVLM-based visual navigation for bin detection and approach.

    Uses SmolVLM-256M-Instruct to analyze images from the OAK-D camera
    and output navigation commands to guide the robot toward trash bins.
    """

    # Navigation prompt - asks model to identify bin and its position
    NAVIGATION_PROMPT = """Look at this image carefully. Is there a trash bin, garbage can, or waste container visible?

If YES, describe where it is:
- Is it on the LEFT side, CENTER, or RIGHT side of the image?
- Is it SMALL (far away), MEDIUM, or LARGE (close/filling frame)?

If NO bin is visible, say "NO BIN".

Answer in this format only:
BIN: [YES/NO]
POSITION: [LEFT/CENTER/RIGHT/NONE]
SIZE: [SMALL/MEDIUM/LARGE/NONE]"""

    # Position verification prompt for classification
    POSITION_VERIFY_PROMPT = """Look at this image. Can you see inside a trash bin or waste container from this angle?

Answer with one of:
- READY: Yes, I can see inside the bin clearly
- ADJUST_LEFT: Move left to see inside better
- ADJUST_RIGHT: Move right to see inside better
- ADJUST_FORWARD: Move closer to see inside
- ADJUST_BACK: Move back, too close
- NOT_A_BIN: No bin visible in this image

Answer with just ONE word from the options above."""

    def __init__(self, config: Optional[SmolVLMConfig] = None):
        self.config = config or SmolVLMConfig()
        self.model = None
        self.processor = None
        self._loaded = False
        self._device = self.config.device

    def load(self) -> bool:
        """
        Load the SmolVLM-256M-Instruct model.

        Returns:
            True if model loaded successfully.
        """
        if self._loaded:
            return True

        try:
            logger.info(f"Loading SmolVLM model: {self.config.model_name}")
            logger.info(f"Device: {self._device}")

            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            # Load model - use float32 for CPU
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Move to device
            self.model = self.model.to(self._device)
            self.model.eval()

            self._loaded = True
            logger.info("SmolVLM model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load SmolVLM model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_inference(self, pil_image: Image.Image, prompt: str) -> str:
        """
        Run inference on an image with the given prompt.

        Args:
            pil_image: PIL Image to analyze
            prompt: Text prompt for the model

        Returns:
            Model's text response
        """
        import torch

        # Format input for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=input_text,
            images=[pil_image],
            return_tensors="pt"
        ).to(self._device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=False
            )

        # Decode response
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Extract just the assistant's response (after the prompt)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()

        return response

    def _parse_navigation_response(self, response: str) -> NavigationResult:
        """
        Parse the model's response to extract navigation command.

        Args:
            response: Raw text response from model

        Returns:
            NavigationResult with parsed command
        """
        response_upper = response.upper()

        # Check if bin detected
        bin_detected = "BIN: YES" in response_upper or "YES" in response_upper.split('\n')[0]

        if not bin_detected or "NO BIN" in response_upper:
            return NavigationResult(
                command=NavigationCommand.NOT_FOUND,
                confidence=0.8,
                raw_response=response,
                bin_detected=False
            )

        # Parse position
        position = None
        if "POSITION: LEFT" in response_upper or "LEFT SIDE" in response_upper:
            position = "left"
        elif "POSITION: RIGHT" in response_upper or "RIGHT SIDE" in response_upper:
            position = "right"
        elif "POSITION: CENTER" in response_upper or "CENTER" in response_upper:
            position = "center"

        # Parse size
        size = None
        if "SIZE: LARGE" in response_upper or "LARGE" in response_upper:
            size = "large"
        elif "SIZE: MEDIUM" in response_upper or "MEDIUM" in response_upper:
            size = "medium"
        elif "SIZE: SMALL" in response_upper or "SMALL" in response_upper:
            size = "small"

        # Determine navigation command
        if size == "large":
            command = NavigationCommand.ARRIVED
            confidence = 0.9
        elif position == "left":
            command = NavigationCommand.LEFT
            confidence = 0.85
        elif position == "right":
            command = NavigationCommand.RIGHT
            confidence = 0.85
        elif position == "center":
            command = NavigationCommand.FORWARD
            confidence = 0.9
        else:
            # Bin detected but position unclear - default to searching
            command = NavigationCommand.FORWARD
            confidence = 0.6

        return NavigationResult(
            command=command,
            confidence=confidence,
            raw_response=response,
            bin_detected=True,
            bin_position=position,
            bin_size=size
        )

    def get_navigation_command(self, image: np.ndarray) -> NavigationResult:
        """
        Analyze an image and return navigation command for bin approach.

        Args:
            image: BGR numpy array from OpenCV (e.g., from OAK-D camera)

        Returns:
            NavigationResult with command and metadata
        """
        if not self._loaded:
            if not self.load():
                return NavigationResult(
                    command=NavigationCommand.NOT_FOUND,
                    confidence=0.0,
                    raw_response="Model not loaded",
                    bin_detected=False
                )

        try:
            # Convert BGR to RGB and then to PIL
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Resize if too large (SmolVLM works with smaller images)
            max_size = 384
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Run inference
            response = self._run_inference(pil_image, self.NAVIGATION_PROMPT)

            logger.debug(f"SmolVLM response: {response}")

            # Parse response
            result = self._parse_navigation_response(response)

            return result

        except Exception as e:
            logger.error(f"Navigation inference error: {e}")
            return NavigationResult(
                command=NavigationCommand.NOT_FOUND,
                confidence=0.0,
                raw_response=f"Error: {str(e)}",
                bin_detected=False
            )

    def verify_position(self, image: np.ndarray) -> Tuple[PositionVerification, str]:
        """
        Verify if robot is correctly positioned to see inside bin.

        Used before classification to ensure good viewing angle.

        Args:
            image: BGR numpy array from USB webcam

        Returns:
            Tuple of (PositionVerification result, raw response)
        """
        if not self._loaded:
            if not self.load():
                return PositionVerification.NOT_A_BIN, "Model not loaded"

        try:
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Resize if needed
            max_size = 384
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Run inference
            response = self._run_inference(pil_image, self.POSITION_VERIFY_PROMPT)
            response_upper = response.upper().strip()

            logger.debug(f"Position verification response: {response}")

            # Parse response
            if "READY" in response_upper:
                return PositionVerification.READY, response
            elif "ADJUST_LEFT" in response_upper or "LEFT" in response_upper:
                return PositionVerification.ADJUST_LEFT, response
            elif "ADJUST_RIGHT" in response_upper or "RIGHT" in response_upper:
                return PositionVerification.ADJUST_RIGHT, response
            elif "ADJUST_FORWARD" in response_upper or "CLOSER" in response_upper:
                return PositionVerification.ADJUST_FORWARD, response
            elif "ADJUST_BACK" in response_upper or "BACK" in response_upper:
                return PositionVerification.ADJUST_BACK, response
            else:
                return PositionVerification.NOT_A_BIN, response

        except Exception as e:
            logger.error(f"Position verification error: {e}")
            return PositionVerification.NOT_A_BIN, f"Error: {str(e)}"

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("SmolVLM model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing SmolVLM Navigator...")
    print(f"Model: SmolVLM-256M-Instruct")

    config = SmolVLMConfig(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        device="cpu"
    )

    navigator = SmolVLMNavigator(config)

    print("\nLoading model (this may take a moment on RPi4)...")
    if navigator.load():
        print("Model loaded successfully!")
        print(f"Device: {navigator._device}")

        # Test with a simple image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        print("\nTesting navigation command...")
        result = navigator.get_navigation_command(test_image)
        print(f"Command: {result.command.value}")
        print(f"Confidence: {result.confidence}")
        print(f"Bin detected: {result.bin_detected}")
        print(f"Raw response: {result.raw_response[:200]}...")

        print("\nTesting position verification...")
        pos_result, pos_response = navigator.verify_position(test_image)
        print(f"Position: {pos_result.value}")
        print(f"Response: {pos_response[:200]}...")

        navigator.unload()
        print("\nModel unloaded.")
    else:
        print("Failed to load model!")
