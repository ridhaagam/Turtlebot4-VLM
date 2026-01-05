"""
Vision model service for SmolVLM2 inference with GPU acceleration.

Provides navigation and classification inference for the TurtleBot4.
"""

import base64
import io
import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NavigationCommand(Enum):
    """Navigation commands output by SmolVLM2."""
    FORWARD = "forward"
    BACKWARD = "backward"  # Back up - too close to obstacle
    LEFT = "left"
    RIGHT = "right"
    ARRIVED = "arrived"
    SEARCH_LEFT = "search_left"    # No bin found, rotate left to search
    SEARCH_RIGHT = "search_right"  # No bin found, rotate right to search


@dataclass
class NavigationResult:
    """Result of navigation inference."""
    command: NavigationCommand
    confidence: float
    raw_response: str
    bin_detected: bool
    bin_position: Optional[str] = None
    bin_size: Optional[str] = None
    bboxes: List[Dict] = field(default_factory=list)  # Florence-2 detections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command.value,
            "bin_detected": self.bin_detected,
            "position": self.bin_position,
            "size": self.bin_size,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "bboxes": self.bboxes
        }


@dataclass
class ClassificationResult:
    """Result of bin classification with enhanced details."""
    bin_found: bool = True
    containers_count: int = 0
    containers_type: str = ""
    fill_level_percent: int = 0
    waste_type: str = "UNKNOWN"
    action: str = ""
    scene_description: str = ""
    objects_detected: List[str] = field(default_factory=list)
    bboxes: List[Dict] = field(default_factory=list)  # Florence-2 bounding boxes
    confidence: float = 0.0
    raw_response: str = ""
    # Legacy fields for backward compatibility
    fullness: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        # Compute fill level label
        if self.fill_level_percent <= 25:
            fill_label = "LOW"
        elif self.fill_level_percent <= 50:
            fill_label = "MEDIUM"
        elif self.fill_level_percent <= 75:
            fill_label = "HIGH"
        else:
            fill_label = "FULL"

        # Build summary string
        summary = f"""Situation Summary
Containers: {self.containers_count} ({self.containers_type})
Fill Level: {self.fill_level_percent}% [{fill_label}]
Waste Type: {self.waste_type}
Action: {self.action}
Scene: {self.scene_description}"""

        return {
            "bin_found": self.bin_found,
            "containers": {
                "count": self.containers_count,
                "type": self.containers_type
            },
            "fill_level": {
                "percent": self.fill_level_percent,
                "label": fill_label
            },
            "waste_type": self.waste_type,
            "action": self.action,
            "scene": self.scene_description,
            "objects_detected": self.objects_detected,
            "bboxes": self.bboxes,  # Include Florence-2 bounding boxes
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "summary": summary,
            # Legacy fields
            "fullness": self.fullness
        }


class VisionModelService:
    """
    Singleton service for SmolVLM2 model management.

    Loads model once at startup and handles concurrent inference requests
    using a lock to prevent GPU memory issues.
    """

    _instance = None
    _lock = threading.Lock()

    # Expanded bin labels for Florence-2 detection - COMPREHENSIVE list to NEVER miss bins
    BIN_LABELS = [
        # Standard bin names (highest priority)
        'bin', 'trash', 'trash bin', 'trash can', 'garbage', 'garbage bin',
        'garbage can', 'waste', 'waste bin', 'dustbin', 'wastebasket',
        'recycling', 'recycling bin', 'rubbish', 'rubbish bin', 'litter bin',
        'wheelie bin', 'dumpster', 'skip', 'compactor',
        # Container types that could be bins
        'container', 'bucket', 'barrel', 'can', 'receptacle', 'pail',
        'tub', 'basin', 'vessel', 'cylinder', 'basket', 'hamper',
        # Florence-2 misidentifications (bins often detected as these)
        'sink', 'bowl', 'box', 'crate', 'drum',
        # Color-specific bins (common in offices/public areas)
        'green bin', 'blue bin', 'yellow bin', 'red bin', 'black bin',
        'recycle', 'compost', 'landfill',
        # Generic large objects that might be bins
        'large container', 'plastic container', 'metal container',
        'storage bin', 'utility bin', 'industrial bin',
        # Cart/trolley style bins
        'cart', 'trolley', 'wheeled bin', 'rolling bin',
    ]

    # Label corrections - Florence-2 sometimes misidentifies bins
    LABEL_CORRECTIONS = {
        'sink': 'plastic bin',
        'bowl': 'waste container',
        'box': 'container',
        'crate': 'storage bin',
        'drum': 'industrial bin',
        'cart': 'wheeled bin',
    }

    # Items that indicate we're looking INSIDE a bin (bin contents)
    # If classification camera sees these, we found the bin!
    BIN_CONTENTS_LABELS = [
        # Recyclables
        'bottle', 'plastic bottle', 'water bottle', 'soda bottle',
        'bottle cap', 'cap', 'lid',  # Common bin items
        'can', 'soda can', 'beer can', 'aluminum can', 'tin can',
        'paper', 'newspaper', 'magazine', 'cardboard',
        'plastic', 'plastic bag', 'wrapper', 'packaging',
        # Food waste
        'food', 'food waste', 'banana', 'apple', 'fruit', 'vegetable',
        'cup', 'paper cup', 'coffee cup', 'styrofoam',
        # General trash
        'trash', 'garbage', 'waste', 'litter', 'debris',
        'tissue', 'napkin', 'paper towel',
        'box', 'carton', 'container',
    ]

    # Items that should NOT be considered bin contents (specific false positives only)
    NOT_BIN_CONTENTS = [
        # Specific non-bin items that might match generic terms
        'shower cap', 'swim cap', 'baseball cap', 'hat',
        'watering can', 'spray can',  # Not trash cans
        'pot lid', 'toilet lid',  # Not bin lids
        'shipping container', 'storage container',
        'gift box', 'cardboard box',  # Might be on desk
        'trophy cup',  # Not a drinking cup
    ]

    # Objects that are DEFINITELY NOT bins - never trigger ARRIVED on these
    NOT_BIN_BLACKLIST = [
        # Furniture
        'chair', 'table', 'desk', 'couch', 'sofa', 'bed', 'bench', 'stool',
        'cabinet', 'shelf', 'drawer', 'wardrobe', 'dresser', 'furniture',
        'ottoman', 'armchair', 'loveseat', 'nightstand', 'bookcase',
        # People/Animals
        'person', 'human', 'man', 'woman', 'child', 'dog', 'cat', 'animal',
        'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe',
        # Vehicles and parts
        'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'vehicle', 'airplane',
        'boat', 'train', 'skateboard', 'surfboard', 'scooter',
        'wheel', 'tire', 'bumper', 'headlight', 'taillight',
        # Electronics
        'tv', 'television', 'monitor', 'computer', 'laptop', 'phone', 'cell phone',
        'keyboard', 'mouse', 'remote', 'microphone', 'speaker', 'camera',
        # Household items that are NOT bins
        'broom', 'mop', 'vacuum', 'duster', 'brush',
        'umbrella', 'handbag', 'purse', 'wallet',
        'shoe', 'boot', 'sandal', 'sneaker',
        'hat', 'cap', 'helmet', 'glasses', 'sunglasses',
        'bottle', 'cup', 'mug', 'glass', 'bowl', 'plate', 'fork', 'knife', 'spoon',
        'vase', 'potted plant', 'pot', 'pan', 'kettle',
        # Structural elements
        'door', 'window', 'wall', 'floor', 'ceiling', 'stairs', 'railing',
        'pillar', 'column', 'beam', 'fence', 'gate',
        # Nature
        'plant', 'tree', 'flower', 'grass', 'rock', 'stone', 'sand', 'water',
        # Bags and boxes (NOT bins!)
        'bag', 'backpack', 'suitcase', 'luggage', 'briefcase', 'handbag',
        'box', 'cardboard', 'package', 'parcel', 'crate',
        'cart', 'trolley', 'wagon', 'wheelbarrow', 'dolly',
        # Signs and fixtures
        'sign', 'traffic light', 'stop sign', 'street sign',
        'pole', 'lamp', 'light', 'lantern', 'chandelier',
        'mirror', 'picture', 'frame', 'clock', 'painting', 'poster',
        # Appliances
        'refrigerator', 'fridge', 'oven', 'microwave', 'appliance',
        'washing machine', 'dryer', 'dishwasher', 'toaster', 'blender',
        'air conditioner', 'heater', 'fan', 'radiator', 'mechanical fan',
        # Electrical
        'power plugs', 'power plugs and sockets', 'socket', 'outlet', 'plug', 'switch',
        'wire', 'cable', 'cord', 'extension cord',
        # Bathroom items (sink removed - often misidentified as plastic bins)
        'toilet', 'bathtub', 'shower', 'towel', 'soap',
        # Paper products
        'book', 'paper', 'newspaper', 'magazine', 'notebook', 'document',
        # Sports equipment
        'ball', 'bat', 'racket', 'frisbee', 'kite', 'skis', 'snowboard',
        'tennis racket', 'baseball bat', 'golf club',
        # Misc
        'teddy bear', 'toy', 'doll', 'stuffed animal',
        'pillow', 'blanket', 'curtain', 'rug', 'carpet', 'mat',
        'rope', 'chain', 'wire', 'cable', 'hose', 'pipe',
        'fire hydrant', 'parking meter', 'bench', 'fountain',
        'statue', 'sculpture', 'monument'
    ]

    # Thresholds for bin detection - balanced for detection + approach
    ARRIVED_AREA_THRESHOLD = 0.35  # 35% of frame = ARRIVED (bin must be closer)
    LARGE_AREA_THRESHOLD = 0.20    # 20% = large bin (keep approaching)
    MEDIUM_AREA_THRESHOLD = 0.08   # 8% = medium bin (detect smaller bins)
    SMALL_AREA_THRESHOLD = 0.03    # 3% = small bin (detect distant bins)
    CENTER_MIN = 0.25              # Center zone starts at 25% (wider zone)
    CENTER_MAX = 0.75              # Center zone ends at 75% (wider zone)

    NAVIGATION_PROMPT = """Is there a trash bin, garbage can, recycling bin, or waste container in this image?
Also look for bins with plastic bags inside.
Answer YES if you see any bin or waste container, NO if not.
If YES, is it on the LEFT, CENTER, or RIGHT?"""

    CLASSIFICATION_PROMPT = """Look inside this trash bin. Analyze its contents.

Answer these questions:
1. FULLNESS: Is the bin EMPTY, PARTIALLY_FULL, or FULL?
2. WASTE_TYPE: What type of waste? RECYCLABLE, ORGANIC, GENERAL, or MIXED?

Answer format:
FULLNESS: [EMPTY/PARTIALLY_FULL/FULL]
WASTE_TYPE: [RECYCLABLE/ORGANIC/GENERAL/MIXED]"""

    # Simple YES/NO prompt - SmolVLM only verifies bin presence
    # All other data (fill level, scene, waste type) comes from Florence-2 detections
    ENHANCED_CLASSIFICATION_PROMPT = """Is there a trash bin, garbage can, or waste container in this image?
Answer only YES or NO."""

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # SmolVLM model
        self.model = None
        self.processor = None
        self.device = None
        self.torch_dtype = None
        self._model_lock = threading.Lock()
        self._initialized = True
        self._loaded = False

        # Florence-2 model for object detection
        self.florence_model = None
        self.florence_processor = None
        self._florence_loaded = False

        # Search direction alternator (for when no bin is found)
        self._search_direction_left = True  # Start with left

    def load_model(self, model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct") -> bool:
        """
        Load SmolVLM2 model with GPU acceleration.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            True if loaded successfully
        """
        if self._loaded:
            logger.info("Model already loaded")
            return True

        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Loading SmolVLM2 model: {model_name}")

            # Determine device and dtype
            if torch.cuda.is_available():
                self.device = "cuda"
                self.torch_dtype = torch.float16
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = "cpu"
                self.torch_dtype = torch.float32
                logger.warning("CUDA not available, falling back to CPU")

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model with appropriate dtype
            logger.info("Loading model (this may take a minute)...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._loaded = True

            logger.info(f"SmolVLM2 model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _decode_image(self, image_base64: str):
        """Decode base64 image to PIL Image."""
        from PIL import Image

        # Handle data URI prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _run_inference(self, image, prompt: str, max_tokens: int = 100) -> str:
        """
        Run inference on image with prompt.

        Thread-safe with model lock.
        """
        import torch

        with self._model_lock:
            # Format messages for SmolVLM2
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
                images=[image],
                return_tensors="pt"
            )

            # Move to device with correct dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=False
                )

            # Decode response
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Extract assistant response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            elif prompt in response:
                response = response.replace(prompt, "").strip()

            return response

    def navigate(self, image_base64: str) -> NavigationResult:
        """
        Analyze image for bin detection and return navigation command.
        """
        if not self._loaded:
            return NavigationResult(
                command=NavigationCommand.SEARCH_LEFT,  # Default search direction when model not loaded
                confidence=0.0,
                raw_response="Model not loaded",
                bin_detected=False
            )

        try:
            image = self._decode_image(image_base64)

            # Resize if too large (save GPU memory)
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))

            response = self._run_inference(image, self.NAVIGATION_PROMPT, max_tokens=80)
            return self._parse_navigation_response(response)

        except Exception as e:
            logger.error(f"Navigation inference error: {e}")
            import traceback
            traceback.print_exc()
            return NavigationResult(
                command=NavigationCommand.SEARCH_LEFT,  # Default search direction on error
                confidence=0.0,
                raw_response=f"Error: {str(e)}",
                bin_detected=False
            )

    def classify(self, image_base64: str) -> ClassificationResult:
        """
        Classify bin contents from image.
        """
        if not self._loaded:
            return ClassificationResult(
                fullness="UNKNOWN",
                waste_type="UNKNOWN",
                confidence=0.0,
                raw_response="Model not loaded"
            )

        try:
            image = self._decode_image(image_base64)

            # Resize if too large
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))

            response = self._run_inference(image, self.CLASSIFICATION_PROMPT, max_tokens=80)
            return self._parse_classification_response(response)

        except Exception as e:
            logger.error(f"Classification inference error: {e}")
            import traceback
            traceback.print_exc()
            return ClassificationResult(
                fullness="UNKNOWN",
                waste_type="UNKNOWN",
                confidence=0.0,
                raw_response=f"Error: {str(e)}"
            )

    def classify_with_validation(self, image_base64: str) -> ClassificationResult:
        """
        Two-step classification with bin verification.

        1. Run Florence-2 to detect if there's actually a bin
        2. If bin found, run SmolVLM for detailed analysis
        3. If no bin, return bin_found=False with scene description

        Args:
            image_base64: Base64 encoded JPEG image

        Returns:
            ClassificationResult with enhanced details
        """
        import time

        logger.info("=== CLASSIFICATION INFERENCE START ===")
        start_time = time.time()

        if not self._loaded:
            logger.error("Model not loaded for classification")
            return ClassificationResult(
                bin_found=False,
                scene_description="Model not loaded",
                raw_response="Model not loaded"
            )

        try:
            image = self._decode_image(image_base64)
            original_width, original_height = image.size
            # Save original image for content detection (before any resizing)
            original_image = image.copy()
            logger.info(f"  Image size: {original_width}x{original_height}")

            # Step 1: Run Florence-2 to verify this is actually a bin
            florence_start = time.time()
            detections = self._detect_objects_florence2(image)
            florence_time = (time.time() - florence_start) * 1000

            florence_labels = [d["label"] for d in detections]
            logger.info(f"  Florence-2 labels: {florence_labels}")
            logger.info(f"  Florence-2 time: {florence_time:.0f}ms")

            # CHECK FOR BIN CONTENTS - if we see trash items, we're looking INSIDE a bin!
            # This is a SUCCESS case - the classification camera found the bin contents
            bin_contents_found = []
            for label in florence_labels:
                label_lower = label.lower().strip()

                # Skip if label is in the NOT_BIN_CONTENTS blacklist
                is_blacklisted = False
                for blacklisted in self.NOT_BIN_CONTENTS:
                    if blacklisted == label_lower or label_lower == blacklisted:
                        is_blacklisted = True
                        logger.debug(f"  Skipping blacklisted content: {label}")
                        break

                if is_blacklisted:
                    continue

                # Check for bin contents - use more precise matching
                for content_label in self.BIN_CONTENTS_LABELS:
                    # Exact match or label starts with content_label
                    if label_lower == content_label or label_lower.startswith(content_label + ' '):
                        bin_contents_found.append(label)
                        break
                    # Or content_label is contained as a complete word
                    elif f' {content_label}' in f' {label_lower} ':
                        bin_contents_found.append(label)
                        break

            # Log bin contents if found (but don't return early - let SmolVLM verify)
            if bin_contents_found:
                logger.info(f"  BIN CONTENTS DETECTED: {bin_contents_found}")
                logger.info("  Will use SmolVLM to verify and get detailed classification...")

            # Check for bin indicators (bin label detected)
            confirmed_bins = [d for d in detections if d.get("is_bin", False)]
            uncertain_bins = [d for d in detections if d.get("needs_verification", False)]
            has_bin = bool(confirmed_bins or uncertain_bins)

            # ALSO check for any LARGE object - when camera is very close,
            # Florence-2 might not label it as "bin" but it could still be one
            # BUT exclude blacklisted objects - they are NEVER bins!
            img_area = original_width * original_height
            large_objects = []
            for d in detections:
                # Skip blacklisted objects - they can never be bins
                if d.get("is_blacklisted", False):
                    continue
                obj_area = d["w"] * d["h"]
                area_ratio = obj_area / img_area
                # Use MEDIUM_AREA_THRESHOLD (8%) to detect smaller objects
                if area_ratio > self.MEDIUM_AREA_THRESHOLD:
                    large_objects.append({
                        "label": d["label"],
                        "area_ratio": area_ratio
                    })

            has_large_object = bool(large_objects)
            has_bin_contents = bool(bin_contents_found)

            logger.info(f"  Bin indicators found: {has_bin}")
            logger.info(f"  Bin contents found: {has_bin_contents}")
            logger.info(f"  Large objects (>{self.MEDIUM_AREA_THRESHOLD:.0%}): {large_objects}")

            # Decision: Proceed with SmolVLM if we have:
            # - Bin indicators (bin label detected)
            # - Bin contents (bottle, can, etc. - we're looking inside)
            # - Large objects (could be a bin)
            # SmolVLM will verify and provide detailed classification
            should_classify = has_bin or has_bin_contents or has_large_object

            if not should_classify:
                # No bin, no contents, no large objects - nothing to classify
                total_time = (time.time() - start_time) * 1000
                logger.warning("  NO BIN INDICATORS - returning bin_found=false")
                logger.info(f"  Total time: {total_time:.0f}ms")
                logger.info("=== CLASSIFICATION INFERENCE END ===")

                return ClassificationResult(
                    bin_found=False,
                    objects_detected=florence_labels,
                    scene_description=f"No bin detected. Objects found: {', '.join(florence_labels) if florence_labels else 'none'}",
                    raw_response=f"Florence-2 labels: {florence_labels}",
                    confidence=0.8
                )

            # Log why we're proceeding
            if has_bin:
                logger.info("  Proceeding with SmolVLM: bin indicators found")
            elif has_bin_contents:
                logger.info(f"  Proceeding with SmolVLM: bin contents found ({bin_contents_found})")
            else:
                logger.info(f"  Proceeding with SmolVLM: large object detected")

            # Step 2: Bin found - run SmolVLM for detailed analysis
            # Resize for SmolVLM memory optimization
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))
                logger.info(f"  Resized for SmolVLM: {image.size[0]}x{image.size[1]}")

            smolvlm_start = time.time()
            response = self._run_inference(image, self.ENHANCED_CLASSIFICATION_PROMPT, max_tokens=300)
            smolvlm_time = (time.time() - smolvlm_start) * 1000

            # Log full response for debugging
            logger.info(f"  SmolVLM FULL response:\n{response}")
            logger.info(f"  SmolVLM time: {smolvlm_time:.0f}ms")

            # Parse SmolVLM response - pass detections with bboxes and original image for content detection
            result = self._parse_enhanced_classification_response(response, florence_labels, detections, original_image)

            total_time = (time.time() - start_time) * 1000
            logger.info(f"  Bin verified: {result.bin_found}")
            logger.info(f"  Fill level: {result.fill_level_percent}%")
            logger.info(f"  Waste type: {result.waste_type}")
            logger.info(f"  Action: {result.action}")
            logger.info(f"  Total time: {total_time:.0f}ms")
            logger.info("=== CLASSIFICATION INFERENCE END ===")

            return result

        except Exception as e:
            logger.error(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return ClassificationResult(
                bin_found=False,
                scene_description=f"Error: {str(e)}",
                raw_response=f"Error: {str(e)}"
            )

    def _parse_enhanced_classification_response(self, response: str, detected_objects: List[str], detections: List[Dict] = None, original_image=None) -> ClassificationResult:
        """Parse classification using Florence-2 detections + SmolVLM YES/NO verification.

        SmolVLM only provides YES/NO bin verification.
        All other data (fill level, scene, waste type, containers) comes from Florence-2.

        Args:
            response: SmolVLM raw response text (just YES/NO)
            detected_objects: List of label strings from Florence-2
            detections: Full detection dicts with bboxes from Florence-2
            original_image: Original PIL image for content detection
        """
        response_upper = response.upper().strip()

        # === SmolVLM: Simple YES/NO bin verification ===
        smolvlm_verified = False
        if "YES" in response_upper and "NO" not in response_upper[:10]:
            smolvlm_verified = True
            logger.info("SmolVLM CONFIRMED: Bin present (YES)")
        elif response_upper.startswith("NO") or response_upper == "NO":
            smolvlm_verified = False
            logger.info("SmolVLM says: No bin (NO)")
        else:
            # Check for affirmative phrases
            if any(phrase in response_upper for phrase in ["YES", "THERE IS", "I SEE", "VISIBLE"]):
                smolvlm_verified = True
                logger.info("SmolVLM CONFIRMED: Bin present (affirmative phrase)")
            # Check for bin-related keywords (SmolVLM sometimes lists bin types instead of YES)
            elif any(kw in response_upper for kw in ["BIN", "TRASH", "GARBAGE", "WASTE", "CONTAINER", "CAN", "RECYCLE", "RUBBISH"]):
                smolvlm_verified = True
                logger.info(f"SmolVLM CONFIRMED: Bin present (mentioned bin-related terms: '{response[:50]}')")
            else:
                logger.warning(f"SmolVLM unclear response: '{response[:50]}', checking Florence-2 fallback...")

        # === Florence-2: Get all classification data ===

        # Count bins from Florence-2 detections
        bin_detections = [d for d in (detections or []) if d.get("is_bin", False)]

        # SAFETY NET: If Florence-2 detected bins, trust it even if SmolVLM said NO
        # This ensures bins are NEVER missed - Florence-2 is more reliable for object detection
        florence2_found_bins = len(bin_detections) > 0

        if florence2_found_bins and not smolvlm_verified:
            logger.info(f"  SAFETY NET: Florence-2 detected {len(bin_detections)} bin(s) - overriding SmolVLM NO")
            bin_verified = True
        else:
            bin_verified = smolvlm_verified

        # Also check if we found bin contents (bottles, cans, etc.) - likely means bin is present
        if not bin_verified and detected_objects:
            content_indicators = sum(1 for obj in detected_objects
                                    if any(c in obj.lower() for c in self.BIN_CONTENTS_LABELS))
            if content_indicators >= 3:
                logger.info(f"  SAFETY NET: Found {content_indicators} bin content indicators - assuming bin present")
                bin_verified = True

        containers_count = len(bin_detections) if bin_detections else (1 if bin_verified else 0)

        # Get container type from first bin label (apply correction)
        containers_type = "waste bin"
        if bin_detections:
            original_type = bin_detections[0].get("label", "waste bin")
            containers_type = self.LABEL_CORRECTIONS.get(original_type.lower(), original_type)

        logger.info(f"  Florence-2 containers: {containers_count} ({containers_type})")

        # Detect content objects and estimate fill level
        content_objects = []
        content_bboxes = []

        if detections and original_image is not None and bin_verified:
            for idx, d in enumerate(detections):
                if d.get("is_bin", False):
                    logger.info(f"  Detecting contents inside bin '{d.get('label')}'...")
                    contents = self._detect_bin_contents(
                        original_image, d, idx, 0, "0-25%"  # Placeholder, will update
                    )
                    content_bboxes.extend(contents)
                    content_objects.extend([c.get("label", "") for c in contents])

        # Estimate fill level from content count
        content_count = len(content_objects)
        if content_count == 0:
            fill_level = 10  # Empty or nearly empty
        elif content_count <= 2:
            fill_level = 25
        elif content_count <= 5:
            fill_level = 50
        elif content_count <= 8:
            fill_level = 70
        else:
            fill_level = 85

        logger.info(f"  Florence-2 content objects: {content_count} -> fill_level={fill_level}%")

        # Determine waste type from detected content labels
        waste_type = self._classify_waste_type(detected_objects + content_objects)
        logger.info(f"  Waste type from labels: {waste_type}")

        # Compute fullness labels BEFORE scene description (needed for scene text)
        if fill_level <= 25:
            fullness = "EMPTY"
            fullness_label = "0-25%"
        elif fill_level <= 75:
            fullness = "PARTIALLY_FULL"
            fullness_label = "25-75%"
        elif fill_level <= 90:
            fullness = "FULL"
            fullness_label = "75-90%"
        else:
            fullness = "FULL"
            fullness_label = "90-100%"

        # Build detailed scene description paragraph from Florence-2 detections
        # Apply label corrections for misidentified objects
        corrected_objects = []
        for label in detected_objects:
            corrected = self.LABEL_CORRECTIONS.get(label.lower(), label)
            corrected_objects.append(corrected)

        corrected_contents = []
        for label in content_objects:
            corrected = self.LABEL_CORRECTIONS.get(label.lower(), label)
            corrected_contents.append(corrected)

        all_labels = list(set(corrected_objects + corrected_contents))

        # Build detailed paragraph description
        if all_labels:
            # Group by type
            bins = [l for l in all_labels if any(b in l.lower() for b in self.BIN_LABELS)]
            contents = [l for l in all_labels if l not in bins]

            # Build descriptive paragraph
            bin_desc = bins[0] if bins else "waste container"
            num_contents = len(contents)

            if num_contents == 0:
                scene = (
                    f"The image shows a {bin_desc} that appears to be mostly empty. "
                    f"The container is estimated to be {fill_level}% full with {fullness.lower().replace('_', ' ')} capacity. "
                    f"No significant waste items are visible inside the bin. "
                    f"The waste type is classified as {waste_type.lower()}. "
                    f"This bin has excellent capacity available for additional waste disposal."
                )
            elif num_contents <= 3:
                content_list = ', '.join(contents[:3])
                scene = (
                    f"The image shows a {bin_desc} containing a few items including {content_list}. "
                    f"The container is estimated to be {fill_level}% full, classified as {fullness.lower().replace('_', ' ')}. "
                    f"Based on the visible contents, the waste type is identified as {waste_type.lower()}. "
                    f"There is still adequate space available in this bin for additional items."
                )
            else:
                content_list = ', '.join(contents[:5])
                scene = (
                    f"The image shows a {bin_desc} with multiple items visible including {content_list}. "
                    f"The container appears to be {fill_level}% full, categorized as {fullness.lower().replace('_', ' ')}. "
                    f"The predominant waste type detected is {waste_type.lower()} based on {num_contents} identified items. "
                    f"{'Attention may be needed soon as the bin is filling up.' if fill_level > 50 else 'The bin still has room for more waste.'}"
                )
        else:
            scene = (
                f"A waste container has been detected and analyzed in the image. "
                f"The bin appears to be {fill_level}% full with {fullness.lower().replace('_', ' ')} status. "
                f"The waste is classified as {waste_type.lower()} type. "
                f"Monitoring will continue to track fill levels and optimize collection schedules."
            )

        logger.info(f"  Scene: {scene[:100]}...")

        # Compute action recommendation
        action = self._get_action_recommendation(fill_level)

        # Build bboxes with fullness info (apply label corrections)
        bboxes = []

        if detections:
            for idx, d in enumerate(detections):
                is_bin = d.get("is_bin", False)
                original_label = d.get("label", "object")
                # Apply label correction for misidentified objects
                corrected_label = self.LABEL_CORRECTIONS.get(original_label.lower(), original_label)

                bbox_entry = {
                    "x": d.get("x", 0),
                    "y": d.get("y", 0),
                    "w": d.get("w", 0),
                    "h": d.get("h", 0),
                    "label": corrected_label,
                    "confidence": d.get("confidence", 0.5),
                    "is_bin": is_bin,
                    "is_content": False,
                    "bin_fullness": fullness_label if is_bin else None,
                    "bin_fullness_percent": fill_level if is_bin else None
                }
                bboxes.append(bbox_entry)

        # Update content bboxes with correct fullness
        for cb in content_bboxes:
            cb["bin_fullness"] = fullness_label
            cb["bin_fullness_percent"] = fill_level
            bboxes.append(cb)

        logger.info(f"  Total bboxes: {len(bboxes)} ({len(content_bboxes)} content objects)")

        return ClassificationResult(
            bin_found=bin_verified,
            containers_count=containers_count,
            containers_type=containers_type,
            fill_level_percent=fill_level,
            waste_type=waste_type,
            action=action,
            scene_description=scene,
            objects_detected=detected_objects,
            bboxes=bboxes,  # Include Florence-2 bounding boxes
            confidence=0.8 if bin_verified else 0.5,
            raw_response=response,
            fullness=fullness
        )

    def _get_action_recommendation(self, fill_level: int) -> str:
        """Get 4-tier action recommendation based on fill level."""
        if fill_level <= 25:
            return "Excellent capacity available"
        elif fill_level <= 50:
            return "Good capacity available"
        elif fill_level <= 75:
            return "Getting full - monitor soon"
        else:
            return "Needs emptying"

    def _classify_waste_type(self, labels: List[str]) -> str:
        """Classify waste type from Florence-2 detected labels."""
        labels_lower = [l.lower() for l in labels]
        labels_str = " ".join(labels_lower)

        # Recyclable indicators
        recyclable_keywords = ["bottle", "plastic", "can", "aluminum", "tin", "glass", "paper", "cardboard", "carton"]
        recyclable_count = sum(1 for kw in recyclable_keywords if kw in labels_str)

        # Organic indicators
        organic_keywords = ["food", "fruit", "vegetable", "banana", "apple", "orange", "leaf", "plant", "organic"]
        organic_count = sum(1 for kw in organic_keywords if kw in labels_str)

        # General waste indicators
        general_keywords = ["wrapper", "bag", "tissue", "napkin", "styrofoam", "trash", "garbage", "waste"]
        general_count = sum(1 for kw in general_keywords if kw in labels_str)

        # Determine type based on highest count
        if recyclable_count > organic_count and recyclable_count > general_count:
            return "RECYCLABLE"
        elif organic_count > recyclable_count and organic_count > general_count:
            return "ORGANIC"
        elif recyclable_count > 0 and organic_count > 0:
            return "MIXED"
        else:
            return "GENERAL"

    def _is_potential_bin(self, bbox: Dict, frame_width: int, frame_height: int) -> tuple:
        """
        Detect potential bin based on size and position heuristics.
        Even without a 'bin' label, large centered objects could be bins.

        Args:
            bbox: dict with 'x', 'y', 'w', 'h' keys
            frame_width: image width (e.g., 640)
            frame_height: image height (e.g., 480)

        Returns:
            tuple: (is_potential_bin: bool, position: str, size: str, arrived: bool)
        """
        x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0)

        # Calculate metrics
        bbox_area = w * h
        frame_area = frame_width * frame_height
        area_ratio = bbox_area / frame_area if frame_area > 0 else 0

        # Center of bbox (normalized 0-1)
        center_x = (x + w / 2) / frame_width if frame_width > 0 else 0.5
        center_y = (y + h / 2) / frame_height if frame_height > 0 else 0.5

        # ASPECT RATIO: Bins are usually taller than wide or roughly square
        aspect_ratio = h / w if w > 0 else 1
        is_bin_shape = 0.4 <= aspect_ratio <= 3.0

        # POSITION: In lower portion of frame (bins are on ground)
        is_ground_level = center_y > 0.25

        # Determine position for navigation (use class thresholds)
        if center_x < self.CENTER_MIN:
            position = 'left'
        elif center_x > self.CENTER_MAX:
            position = 'right'
        elif abs(center_x - 0.5) < 0.10:
            position = 'center'
        elif center_x < 0.5:
            position = 'slight_left'
        else:
            position = 'slight_right'

        # Determine size category (use lowered thresholds for better detection)
        if area_ratio >= self.ARRIVED_AREA_THRESHOLD:
            size = 'large'
        elif area_ratio >= self.LARGE_AREA_THRESHOLD:
            size = 'medium'
        elif area_ratio >= self.MEDIUM_AREA_THRESHOLD:
            size = 'small'
        else:
            size = 'tiny'

        # Check if centered (for ARRIVED determination)
        is_centered = self.CENTER_MIN <= center_x <= self.CENTER_MAX

        # ARRIVED: Large + centered + ground level
        arrived = (area_ratio >= self.ARRIVED_AREA_THRESHOLD and
                   is_centered and is_ground_level)

        # Is this potentially a bin? Large enough and reasonable shape
        # Use SMALL_AREA_THRESHOLD (2%) to detect distant/small bins
        is_potential = (area_ratio >= self.SMALL_AREA_THRESHOLD and
                        is_bin_shape and is_ground_level)

        logger.debug(f"Heuristics: area={area_ratio:.1%}, center_x={center_x:.2f}, "
                     f"shape={aspect_ratio:.1f}, potential={is_potential}, arrived={arrived}")

        return is_potential, position, size, arrived

    def _parse_navigation_response(self, response: str) -> NavigationResult:
        """Parse model response to extract navigation command."""
        response_upper = response.upper()

        bin_detected = "BIN: YES" in response_upper or (
            "YES" in response_upper and "NO BIN" not in response_upper and "NO" not in response_upper[:20]
        )

        if not bin_detected or "NO BIN" in response_upper:
            # No bin found - return alternating search direction
            return NavigationResult(
                command=self._get_search_command(),
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

        # Determine command based on position and size
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

    def _parse_classification_response(self, response: str) -> ClassificationResult:
        """Parse model response to extract classification."""
        response_upper = response.upper()

        # Parse fullness
        fullness = "UNKNOWN"
        if "EMPTY" in response_upper and "PARTIALLY" not in response_upper:
            fullness = "EMPTY"
        elif "PARTIALLY_FULL" in response_upper or "PARTIALLY FULL" in response_upper or "PARTIAL" in response_upper:
            fullness = "PARTIALLY_FULL"
        elif "FULL" in response_upper:
            fullness = "FULL"

        # Parse waste type
        waste_type = "GENERAL"
        if "RECYCLABLE" in response_upper or "RECYCLE" in response_upper:
            waste_type = "RECYCLABLE"
        elif "ORGANIC" in response_upper or "COMPOST" in response_upper:
            waste_type = "ORGANIC"
        elif "MIXED" in response_upper:
            waste_type = "MIXED"

        confidence = 0.8 if fullness != "UNKNOWN" else 0.5

        return ClassificationResult(
            fullness=fullness,
            waste_type=waste_type,
            confidence=confidence,
            raw_response=response
        )

    def load_florence2(self, model_name: str = "microsoft/Florence-2-base") -> bool:
        """
        Load Florence-2 model for object detection.

        Args:
            model_name: HuggingFace model identifier (base or large)

        Returns:
            True if loaded successfully
        """
        if self._florence_loaded:
            logger.info("Florence-2 model already loaded")
            return True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            from transformers.modeling_utils import PreTrainedModel

            logger.info(f"Loading Florence-2 model: {model_name}")

            # Patch for Florence-2 compatibility with newer transformers
            # Add _supports_sdpa attribute if missing
            if not hasattr(PreTrainedModel, '_supports_sdpa'):
                PreTrainedModel._supports_sdpa = False
                logger.info("Applied _supports_sdpa patch for Florence-2 compatibility")

            # Load Florence-2 processor
            logger.info("Loading Florence-2 processor...")
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load Florence-2 model with attn_implementation to avoid SDPA issues
            logger.info("Loading Florence-2 model...")
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype if self.torch_dtype else torch.float16,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation="eager"  # Avoid SDPA compatibility issues
            )

            if self.device == "cpu":
                self.florence_model = self.florence_model.to(self.device)

            self.florence_model.eval()
            self._florence_loaded = True

            logger.info(f"Florence-2 model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detect_objects_florence2(self, image) -> List[Dict]:
        """
        Run Florence-2 object detection on image.

        Args:
            image: PIL Image

        Returns:
            List of detections: [{"x": int, "y": int, "w": int, "h": int, "label": str, "confidence": float}]
        """
        import torch

        if not self._florence_loaded:
            # Try to load Florence-2 on first use
            if not self.load_florence2():
                return []

        try:
            with self._model_lock:
                # Use object detection task
                task_prompt = "<OD>"

                inputs = self.florence_processor(
                    text=task_prompt,
                    images=image,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if "pixel_values" in inputs and self.torch_dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)

                # Generate with greedy decoding and no cache (fixes compatibility issues)
                with torch.no_grad():
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=1,
                        use_cache=False,  # Disable cache to avoid past_key_values issues
                    )

                # Decode response
                generated_text = self.florence_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                # Parse Florence-2 output
                parsed = self.florence_processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image.width, image.height)
                )

                # Extract bounding boxes
                detections = []
                if task_prompt in parsed:
                    result = parsed[task_prompt]
                    bboxes = result.get("bboxes", [])
                    labels = result.get("labels", [])

                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        label = labels[i] if i < len(labels) else "object"
                        label_lower = label.lower()

                        # Check if this is a BLACKLISTED object (definitely NOT a bin)
                        is_blacklisted = any(bl in label_lower for bl in self.NOT_BIN_BLACKLIST)

                        # Use class-level BIN_LABELS for matching
                        # Check if label matches any bin keyword
                        is_bin = any(bl in label_lower for bl in self.BIN_LABELS) and not is_blacklisted

                        # Strong bin indicators - high confidence (expanded list)
                        strong_bin_keywords = [
                            "trash", "garbage", "waste", "rubbish", "dustbin", "recycling",
                            "bin", "dumpster", "wastebasket", "litter", "recycle", "compost"
                        ]
                        is_strong_bin = any(kw in label_lower for kw in strong_bin_keywords) and not is_blacklisted

                        # Calculate area ratio for heuristics
                        det_w = x2 - x1
                        det_h = y2 - y1
                        det_area = det_w * det_h
                        img_area = image.width * image.height
                        area_ratio = det_area / img_area if img_area > 0 else 0

                        # needs_verification should ONLY be true for objects that could plausibly be bins
                        # NOT just any large object - must have a bin-like label
                        plausible_bin_labels = ["container", "bucket", "barrel", "can", "cylinder", "basket", "receptacle", "pail", "tub"]
                        could_be_bin = any(pl in label_lower for pl in plausible_bin_labels) and not is_blacklisted

                        # DEBUG: Log the classification decision for each detection
                        logger.info(f"  Detection '{label}': is_bin={is_bin or is_strong_bin}, is_strong={is_strong_bin}, blacklisted={is_blacklisted}, needs_verify={could_be_bin and not is_bin}")

                        detections.append({
                            "x": x1,
                            "y": y1,
                            "w": det_w,
                            "h": det_h,
                            "label": label,
                            "confidence": 0.95 if is_strong_bin else (0.85 if is_bin else 0.6),
                            "is_bin": is_bin or is_strong_bin,
                            "is_blacklisted": is_blacklisted,
                            "needs_verification": could_be_bin and not is_bin,  # Only plausible bins need verification
                            "area_ratio": area_ratio
                        })

                return detections

        except Exception as e:
            logger.error(f"Florence-2 detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _detect_bin_contents(self, image, bin_bbox: Dict, bin_index: int, fullness_pct: int, fullness_label: str) -> List[Dict]:
        """
        Detect objects inside a bin by cropping and running Florence-2.
        Similar to edge-jetson's _detect_content_objects().

        Args:
            image: PIL Image (full image)
            bin_bbox: Bounding box of the bin {x, y, w, h, label}
            bin_index: Index of parent bin
            fullness_pct: Bin fullness percentage
            fullness_label: Bin fullness label (0-25%, 25-75%, etc.)

        Returns:
            List of content detections with is_content=True
        """
        content_detections = []

        try:
            x, y, w, h = bin_bbox["x"], bin_bbox["y"], bin_bbox["w"], bin_bbox["h"]
            img_w, img_h = image.size

            # Crop with slight padding for context
            pad = int(max(w, h) * 0.02)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            bin_crop = image.crop((x1, y1, x2, y2))
            crop_w, crop_h = bin_crop.size

            logger.info(f"  [CONTENT] Detecting contents in bin crop ({crop_w}x{crop_h})")

            # Run Florence-2 object detection on cropped bin region
            crop_detections = self._detect_objects_florence2(bin_crop)

            for det in crop_detections:
                label = det.get("label", "").lower()

                # Skip if it's the bin/container itself
                if any(bl in label for bl in self.BIN_LABELS):
                    continue

                # Map coordinates back to original image
                cx, cy, cw, ch = det["x"], det["y"], det["w"], det["h"]
                orig_x = int(x1 + cx)
                orig_y = int(y1 + cy)
                orig_w = cw
                orig_h = ch

                # Ensure bbox stays within parent bin bounds
                orig_x = max(x, min(orig_x, x + w - 10))
                orig_y = max(y, min(orig_y, y + h - 10))
                orig_w = min(orig_w, (x + w) - orig_x)
                orig_h = min(orig_h, (y + h) - orig_y)

                if orig_w > 5 and orig_h > 5:  # Minimum size filter
                    content_det = {
                        "x": orig_x,
                        "y": orig_y,
                        "w": orig_w,
                        "h": orig_h,
                        "label": det.get("label", "object"),
                        "confidence": 0.80,
                        "is_content": True,
                        "is_bin": False,
                        "parent_bin_id": bin_index,
                        "bin_fullness": fullness_label,
                        "bin_fullness_percent": fullness_pct
                    }
                    content_detections.append(content_det)
                    logger.info(f"    -> Content: '{det.get('label')}' at ({orig_x},{orig_y},{orig_w},{orig_h})")

            logger.info(f"  [CONTENT] Found {len(content_detections)} content objects in bin {bin_index}")

        except Exception as e:
            logger.warning(f"Content detection failed: {e}")

        return content_detections

    def navigate_with_detection(self, image_base64: str) -> NavigationResult:
        """
        Analyze image using Florence-2 for object detection + SmolVLM for verification.

        This method:
        1. Runs Florence-2 to detect objects
        2. Uses SmolVLM to VERIFY if detected objects are actually trash bins
        3. Only navigates toward confirmed bins
        4. Returns navigation command with bounding boxes

        Args:
            image_base64: Base64 encoded JPEG image

        Returns:
            NavigationResult with command, metadata, and bounding boxes
        """
        if not self._loaded:
            return NavigationResult(
                command=NavigationCommand.SEARCH_LEFT,  # Default search direction when model not loaded
                confidence=0.0,
                raw_response="Model not loaded",
                bin_detected=False,
                bboxes=[]
            )

        try:
            image = self._decode_image(image_base64)

            # Store ORIGINAL image dimensions - Florence-2 should use original
            original_width, original_height = image.size
            logger.info(f"Original image size: {original_width}x{original_height}")

            # Step 1: Run Florence-2 object detection on ORIGINAL image
            # This ensures bbox coordinates match the original image dimensions
            detections = self._detect_objects_florence2(image)

            # Step 2: Resize for SmolVLM (memory optimization)
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))
                logger.info(f"Resized image for SmolVLM: {image.size[0]}x{image.size[1]}")

            # Log what Florence-2 detected
            if detections:
                labels = [d["label"] for d in detections]
                logger.info(f"Florence-2 detected: {labels}")

            # Use all detections for bboxes
            bboxes = [{
                "x": d["x"],
                "y": d["y"],
                "w": d["w"],
                "h": d["h"],
                "label": d["label"],
                "confidence": d["confidence"]
            } for d in detections]

            florence_labels = [d['label'] for d in detections]
            blacklisted_labels = [d['label'] for d in detections if d.get('is_blacklisted', False)]
            logger.info(f"Florence-2 labels: {florence_labels}")
            if blacklisted_labels:
                logger.info(f"Blacklisted (ignored): {blacklisted_labels}")

            # === DUAL DETECTION: Labels + Heuristics ===
            # Method 1: Label-based detection (Florence-2 recognized it as bin-like)
            # EXCLUDE blacklisted objects
            confirmed_bins = [d for d in detections if d.get("is_bin", False) and not d.get("is_blacklisted", False)]
            uncertain_bins = [d for d in detections if d.get("needs_verification", False) and not d.get("is_blacklisted", False)]

            # Method 2: Heuristics-based detection (large centered object)
            # ONLY apply to non-blacklisted objects
            heuristic_bins = []
            for det in detections:
                # Skip blacklisted objects entirely
                if det.get('is_blacklisted', False):
                    logger.debug(f"Skipping blacklisted object: {det['label']}")
                    continue

                is_potential, h_pos, h_size, h_arrived = self._is_potential_bin(
                    det, original_width, original_height
                )
                if is_potential:
                    det['heuristic_position'] = h_pos
                    det['heuristic_size'] = h_size
                    det['heuristic_arrived'] = h_arrived
                    heuristic_bins.append(det)

            logger.info(f"Label-based bins: {len(confirmed_bins)}, Uncertain: {len(uncertain_bins)}, Heuristic: {len(heuristic_bins)}")

            # DEBUG: Log the actual bins found
            if confirmed_bins:
                logger.info(f"  Confirmed bins: {[(b['label'], b['is_bin'], b['is_blacklisted']) for b in confirmed_bins]}")
            if uncertain_bins:
                logger.info(f"  Uncertain bins: {[(b['label'], b['needs_verification']) for b in uncertain_bins]}")

            # DEBUG: Log ALL detections with their flags
            logger.info(f"  ALL detections with flags:")
            for d in detections:
                logger.info(f"    - '{d['label']}': is_bin={d.get('is_bin')}, blacklisted={d.get('is_blacklisted')}, verify={d.get('needs_verification')}")

            # Detection strategy:
            # 1. Label-based: Florence-2 recognized a bin label -> ARRIVED
            # 2. Size-based: Very large centered object (>35%) -> ARRIVED (classification will verify)
            # This prevents endless searching when bin isn't labeled correctly

            label_based_bin_detected = bool(confirmed_bins or uncertain_bins)

            # Check for LARGE centered objects that should trigger ARRIVED
            # Even without bin label - classification will verify if it's a bin
            size_based_arrived = False
            largest_object = None

            if not label_based_bin_detected and detections:
                # Find the largest non-blacklisted object
                for det in detections:
                    if det.get('is_blacklisted', False):
                        continue
                    area_ratio = det.get('area_ratio', 0)
                    # Check if large AND centered
                    det_center_x = (det['x'] + det['w'] / 2) / original_width
                    is_centered = self.CENTER_MIN <= det_center_x <= self.CENTER_MAX

                    if area_ratio >= self.ARRIVED_AREA_THRESHOLD and is_centered:
                        size_based_arrived = True
                        largest_object = det
                        logger.info(f"SIZE-BASED ARRIVED: '{det['label']}' fills {area_ratio:.1%} of frame, centered at {det_center_x:.2f}")
                        break

            if not label_based_bin_detected and not size_based_arrived:
                # No bin label AND no large centered object - continue searching
                logger.info("No bin LABEL and no large object - searching")
                search_cmd = self._get_search_command()
                return NavigationResult(
                    command=search_cmd,
                    confidence=0.8,
                    raw_response=f"No bin found. Search {search_cmd.value}. Florence-2={florence_labels}",
                    bin_detected=False,
                    bboxes=bboxes
                )

            # If we get here, either label-based OR size-based detection triggered
            if size_based_arrived and largest_object:
                # Size-based arrival - classification will verify
                logger.info(f"ARRIVED via SIZE (classification will verify): {largest_object['label']}")
                return NavigationResult(
                    command=NavigationCommand.ARRIVED,
                    confidence=0.7,  # Lower confidence since not label-based
                    raw_response=f"Large object detected ({largest_object['label']}). Classification will verify if bin.",
                    bin_detected=True,  # Assume it might be a bin
                    bin_position="center",
                    bin_size="large",
                    bboxes=bboxes
                )

            # Select best bin candidate from LABEL-BASED detections only
            # Heuristics are NOT used for bin selection
            all_candidates = confirmed_bins + uncertain_bins
            main_bin = max(all_candidates, key=lambda d: d["w"] * d["h"])
            logger.info(f"BIN SELECTED via LABEL: {main_bin['label']} (area_ratio={main_bin.get('area_ratio', 0):.1%})")

            # Calculate position and size using heuristics if available, otherwise compute
            if main_bin:

                # IMPORTANT: Use ORIGINAL image dimensions for position calculation
                # Florence-2 returns bbox coordinates in original image coordinate space
                img_width = original_width
                img_height = original_height

                bin_center_x = main_bin["x"] + main_bin["w"] / 2
                bin_area = main_bin["w"] * main_bin["h"]
                img_area = img_width * img_height

                # Calculate zone boundaries using class thresholds
                left_threshold = img_width * self.CENTER_MIN   # 35% = 224 for 640px
                right_threshold = img_width * self.CENTER_MAX  # 65% = 416 for 640px
                img_center = img_width / 2

                # Normalized center position (0-1)
                norm_center_x = bin_center_x / img_width

                # Calculate how far off-center the bin is (as percentage)
                offset_from_center = bin_center_x - img_center
                offset_percent = abs(offset_from_center) / (img_width / 2) * 100

                # Debug logging for position calculation
                logger.info(f"=== POSITION CALCULATION ===")
                logger.info(f"  Image dimensions: {img_width}x{img_height}")
                logger.info(f"  Bin bbox: x={main_bin['x']}, w={main_bin['w']}")
                logger.info(f"  Bin center_x: {bin_center_x:.1f} (norm={norm_center_x:.2f}), offset: {offset_from_center:+.1f}px ({offset_percent:.0f}%)")
                logger.info(f"  Zone thresholds: left < {self.CENTER_MIN:.0%} | center | right > {self.CENTER_MAX:.0%}")

                # Determine position from Florence-2 bbox using normalized coordinates
                if norm_center_x < self.CENTER_MIN:
                    position = "left"
                    logger.info(f"  Position: LEFT (< {self.CENTER_MIN:.0%})")
                elif norm_center_x > self.CENTER_MAX:
                    position = "right"
                    logger.info(f"  Position: RIGHT (> {self.CENTER_MAX:.0%})")
                elif abs(norm_center_x - 0.5) < 0.10:
                    # Within 10% of center - well centered
                    position = "center"
                    logger.info(f"  Position: CENTER (well centered, offset {offset_percent:.0f}%)")
                elif norm_center_x < 0.5:
                    position = "slight_left"
                    logger.info(f"  Position: SLIGHT_LEFT (offset {offset_percent:.0f}%)")
                else:
                    position = "slight_right"
                    logger.info(f"  Position: SLIGHT_RIGHT (offset {offset_percent:.0f}%)")

                # Determine size based on area ratio using class thresholds
                area_ratio = bin_area / img_area
                if area_ratio >= self.ARRIVED_AREA_THRESHOLD:  # 20%+ = large/arrived
                    size = "large"
                elif area_ratio >= self.MEDIUM_AREA_THRESHOLD:  # 8%+ = medium
                    size = "medium"
                else:
                    size = "small"

                logger.info(f"  Size: {size} (area_ratio={area_ratio:.1%})")

                raw_response = f"Verified bin at {position}, size={size} (area={area_ratio:.1%}), label={main_bin['label']}"

            # NOTE: We always have Florence-2 bbox at this point (required for bin_detected=True)
            # Safety defaults in case logic somehow fails (should never happen)
            else:
                logger.error("UNEXPECTED: bin_detected=True but no Florence-2 detections!")
                position = "center"
                size = "small"
                raw_response = "Fallback: bin detected but no bbox"

            # Determine navigation command
            # STRATEGY: Prefer FORWARD when possible, only turn when really needed
            logger.info(f"=== COMMAND DETERMINATION ===")
            logger.info(f"  Position: {position}, Size: {size}")

            # ARRIVED: Large bin in center
            if size == "large" and position in ["center", "slight_left", "slight_right"]:
                command = NavigationCommand.ARRIVED
                confidence = 0.9
                logger.info(f"  → Command: ARRIVED (large bin, close enough to center)")

            # Large bin but far off-center - need to turn
            elif size == "large" and position == "left":
                command = NavigationCommand.LEFT
                confidence = 0.85
                logger.info(f"  → Command: LEFT (large bin far left)")
            elif size == "large" and position == "right":
                command = NavigationCommand.RIGHT
                confidence = 0.85
                logger.info(f"  → Command: RIGHT (large bin far right)")

            # Medium bin - if well centered, trigger ARRIVED (close enough)
            elif size == "medium":
                if position == "center":
                    # Well centered medium bin - we've arrived!
                    command = NavigationCommand.ARRIVED
                    confidence = 0.85
                    logger.info(f"  → Command: ARRIVED (medium bin, well centered)")
                elif position in ["slight_left", "slight_right"]:
                    # Slightly off-center - go forward to get closer
                    command = NavigationCommand.FORWARD
                    confidence = 0.8
                    logger.info(f"  → Command: FORWARD (medium bin, slightly off-center)")
                elif position == "left":
                    command = NavigationCommand.LEFT
                    confidence = 0.8
                    logger.info(f"  → Command: LEFT (medium bin far left)")
                else:
                    command = NavigationCommand.RIGHT
                    confidence = 0.8
                    logger.info(f"  → Command: RIGHT (medium bin far right)")

            # Small bin - need to be more precise about centering
            elif position == "left":
                command = NavigationCommand.LEFT
                confidence = 0.85
                logger.info(f"  → Command: LEFT (bin on left)")
            elif position == "right":
                command = NavigationCommand.RIGHT
                confidence = 0.85
                logger.info(f"  → Command: RIGHT (bin on right)")
            elif position in ["center", "slight_left", "slight_right"]:
                # Any center-ish position - go forward
                command = NavigationCommand.FORWARD
                confidence = 0.85
                logger.info(f"  → Command: FORWARD (bin roughly centered)")
            else:
                command = NavigationCommand.FORWARD
                confidence = 0.7
                logger.info(f"  → Command: FORWARD (default)")

            return NavigationResult(
                command=command,
                confidence=confidence,
                raw_response=raw_response,
                bin_detected=True,
                bin_position=position,
                bin_size=size,
                bboxes=bboxes
            )

        except Exception as e:
            logger.error(f"Navigation with detection error: {e}")
            import traceback
            traceback.print_exc()
            return NavigationResult(
                command=NavigationCommand.SEARCH_LEFT,  # Default search direction on error
                confidence=0.0,
                raw_response=f"Error: {str(e)}",
                bin_detected=False,
                bboxes=[]
            )

    def _get_search_command(self) -> NavigationCommand:
        """Get alternating search command (left/right) for when no bin is found."""
        if self._search_direction_left:
            self._search_direction_left = False
            return NavigationCommand.SEARCH_LEFT
        else:
            self._search_direction_left = True
            return NavigationCommand.SEARCH_RIGHT

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_status(self) -> Dict[str, Any]:
        """Get model status information."""
        status = {
            "loaded": self._loaded,
            "florence2_loaded": self._florence_loaded,
            "device": str(self.device) if self.device else None,
            "dtype": str(self.torch_dtype) if self.torch_dtype else None,
        }

        try:
            import torch
            status["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except:
            status["cuda_available"] = False

        return status


# Global singleton instance
vision_service = VisionModelService()
