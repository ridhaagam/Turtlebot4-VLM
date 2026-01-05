"""
Florence-2 Waste Bin Detection and Fullness Classification module.
Detects waste containers and classifies their fullness level (0-25%, 25-75%, 75-90%, 90%+).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import re

try:
    from .device_platform import get_device, get_recommended_model, detect_platform, Platform
except ImportError:
    from device_platform import get_device, get_recommended_model, detect_platform, Platform

logger = logging.getLogger(__name__)


def _setup_flash_attn_mock():
    """
    Create a mock flash_attn module if not available.
    Florence-2 requires flash_attn for import but can run without it.
    """
    try:
        import flash_attn
        return  # Already installed
    except ImportError:
        pass

    # Create mock package
    import importlib.util
    mock_dir = "/tmp/mock_flash_attn"
    os.makedirs(mock_dir, exist_ok=True)

    # Write mock __init__.py
    init_code = '''
__version__ = "2.0.0"
def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available")
def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available")
'''
    with open(f"{mock_dir}/__init__.py", "w") as f:
        f.write(init_code)

    with open(f"{mock_dir}/flash_attn_interface.py", "w") as f:
        f.write(init_code)

    # Load as proper module
    sys.path.insert(0, "/tmp")
    spec = importlib.util.spec_from_file_location("flash_attn", f"{mock_dir}/__init__.py")
    flash_attn = importlib.util.module_from_spec(spec)
    sys.modules['flash_attn'] = flash_attn
    spec.loader.exec_module(flash_attn)

    spec2 = importlib.util.spec_from_file_location(
        "flash_attn.flash_attn_interface",
        f"{mock_dir}/flash_attn_interface.py"
    )
    flash_attn_interface = importlib.util.module_from_spec(spec2)
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    spec2.loader.exec_module(flash_attn_interface)

    logger.info("Created mock flash_attn module (using eager attention)")


class BinFullness(Enum):
    """Bin fullness classification levels."""
    EMPTY = "0-25%"        # Nearly empty
    PARTIAL = "25-75%"     # Partially full
    MOSTLY_FULL = "75-90%" # Mostly full
    FULL = "90-100%"       # Full/Overflowing
    UNKNOWN = "unknown"    # Could not determine


@dataclass
class Detection:
    """Single detection result with optional bin classification."""
    label: str
    confidence: float
    bbox: Dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    bin_fullness: Optional[str] = None  # Fullness level if this is a bin
    bin_fullness_percent: Optional[int] = None  # Estimated percentage
    is_content: bool = False  # True if this is an object detected inside a bin
    parent_bin_id: Optional[int] = None  # Index of the parent bin if is_content

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "is_content": self.is_content
        }
        if self.bin_fullness is not None:
            result["bin_fullness"] = self.bin_fullness
            result["bin_fullness_percent"] = self.bin_fullness_percent
        if self.parent_bin_id is not None:
            result["parent_bin_id"] = self.parent_bin_id
        return result


@dataclass
class DetectionResult:
    """Complete detection result for a frame."""
    timestamp: str
    frame_id: str
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    # Aggregated bin status
    bin_detected: bool = False
    bin_count: int = 0
    overall_fullness: Optional[str] = None
    overall_fullness_percent: Optional[int] = None
    status_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": self.inference_time_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "bin_detected": self.bin_detected,
            "bin_count": self.bin_count,
            "overall_fullness": self.overall_fullness,
            "overall_fullness_percent": self.overall_fullness_percent,
            "status_summary": self.status_summary
        }


@dataclass
class DetectorConfig:
    model_name: str = "microsoft/Florence-2-base"
    device: str = "auto"  # "auto", "cuda", "cpu"
    task: str = "<OD>"  # Object Detection task
    confidence_threshold: float = 0.3
    use_fp16: bool = True
    max_new_tokens: int = 1024
    # Bin detection settings
    bin_labels: List[str] = field(default_factory=lambda: [
        "bin", "trash", "garbage", "waste", "container", "trash can",
        "garbage bin", "waste bin", "waste container", "dustbin",
        "rubbish bin", "trash bin", "recycling bin", "dumpster"
    ])
    classify_fullness: bool = True


class Florence2Detector:
    """
    Florence-2 based waste bin detector with fullness classification.

    Capabilities:
    1. Object detection to find bins in the image
    2. VQA-based fullness classification for each detected bin
    3. Overall status summary generation
    """

    # Keywords indicating bin/container objects (expanded for better detection)
    BIN_KEYWORDS = {
        # Primary bin terms
        "bin", "trash", "garbage", "waste", "dustbin",
        "rubbish", "recycling", "dumpster", "receptacle",
        "trash can", "garbage can", "waste bin", "recycle bin",
        # Bag-based containers (common in waste management)
        "trash bag", "garbage bag", "waste bag",
        # Other container types
        "bucket", "basket", "barrel", "drum", "crate",
        "pail"
    }

    # Labels that are NOT bins (content/waste items)
    NOT_BIN_LABELS = {
        "tin can", "tincan", "tin", "can",  # cans are content
        "soda can", "beer can", "drink can", "aluminum can",
        "bottle", "water bottle", "plastic bottle",
        "cup", "paper", "tissue", "wrapper", "food",
        "box", "cardboard", "container"  # small containers are content
    }

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model = None
        self.processor = None
        self._device = None
        self._torch_dtype = torch.float32
        self._loaded = False

    def load(self) -> bool:
        """
        Load the Florence-2 model and processor.

        Returns:
            True if model loaded successfully.
        """
        if self._loaded:
            return True

        try:
            # Setup flash_attn mock before importing transformers
            _setup_flash_attn_mock()

            from transformers import AutoProcessor, AutoModelForCausalLM

            # Determine device
            if self.config.device == "auto":
                self._device = get_device()
            else:
                self._device = self.config.device

            logger.info(f"Loading model: {self.config.model_name}")
            logger.info(f"Using device: {self._device}")

            # Determine dtype based on platform
            platform = detect_platform()
            if self.config.use_fp16 and self._device == "cuda":
                self._torch_dtype = torch.float16
            else:
                self._torch_dtype = torch.float32

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self._torch_dtype,
                trust_remote_code=True
            ).to(self._device)

            # Set to eval mode
            self.model.eval()

            self._loaded = True
            logger.info("Model loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _is_bin_label(self, label: str) -> bool:
        """Check if a label refers to a bin/waste container (not content items)."""
        label_lower = label.lower()

        # First check if it's explicitly NOT a bin (content item)
        for not_bin in self.NOT_BIN_LABELS:
            if not_bin in label_lower:
                return False

        # Then check if it matches bin keywords
        return any(keyword in label_lower for keyword in self.BIN_KEYWORDS)

    def _run_inference(self, pil_image: Image.Image, task_prompt: str) -> str:
        """Run model inference with given task prompt."""
        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self._device)

        # Cast pixel_values to match model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=3
            )

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

    def _classify_bin_fullness(self, pil_image: Image.Image, bbox: Dict[str, int]) -> Tuple[str, int]:
        """
        Classify the fullness level of a detected bin using VQA.

        Args:
            pil_image: Full PIL image
            bbox: Bounding box of the bin {"x", "y", "width", "height"}

        Returns:
            Tuple of (fullness_level, percentage)
        """
        try:
            # Crop the bin region with some padding
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            img_w, img_h = pil_image.size

            # Add 10% padding
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            bin_crop = pil_image.crop((x1, y1, x2, y2))

            # Use VQA to ask about fullness
            # Florence-2 VQA task
            vqa_prompt = "<VQA>How full is this bin or container? Answer with: empty, quarter full, half full, mostly full, or completely full."

            response = self._run_inference(bin_crop, vqa_prompt)

            # Parse the response to determine fullness
            response_lower = response.lower()

            # Map response to fullness levels
            if any(word in response_lower for word in ["empty", "0%", "nothing", "vacant"]):
                return BinFullness.EMPTY.value, 12
            elif any(word in response_lower for word in ["quarter", "25%", "little", "some", "low"]):
                return BinFullness.EMPTY.value, 20
            elif any(word in response_lower for word in ["half", "50%", "middle", "moderate", "partial"]):
                return BinFullness.PARTIAL.value, 50
            elif any(word in response_lower for word in ["mostly", "75%", "three quarter", "high", "almost"]):
                return BinFullness.MOSTLY_FULL.value, 80
            elif any(word in response_lower for word in ["full", "100%", "complete", "overflow", "filled"]):
                return BinFullness.FULL.value, 95
            else:
                # Default to using visual analysis based on bbox aspect ratio and position
                return self._estimate_fullness_visual(bin_crop)

        except Exception as e:
            logger.warning(f"Failed to classify bin fullness via VQA: {e}")
            return BinFullness.UNKNOWN.value, 50

    def _estimate_fullness_visual(self, bin_crop: Image.Image) -> Tuple[str, int]:
        """
        Estimate bin fullness using visual analysis when VQA fails.
        Analyzes color distribution and edge density in upper portion of bin.
        """
        try:
            # Convert to numpy for analysis
            img_array = np.array(bin_crop)
            h, w = img_array.shape[:2]

            # Analyze upper 40% of the bin (where contents would be visible if full)
            upper_region = img_array[:int(h * 0.4), :, :]
            lower_region = img_array[int(h * 0.6):, :, :]

            # Calculate color variance (full bins tend to have more varied colors)
            upper_std = np.std(upper_region)
            lower_std = np.std(lower_region)

            # If upper region has high variance relative to lower, bin is likely full
            if upper_std > lower_std * 1.5:
                return BinFullness.FULL.value, 90
            elif upper_std > lower_std * 1.2:
                return BinFullness.MOSTLY_FULL.value, 75
            elif upper_std > lower_std * 0.8:
                return BinFullness.PARTIAL.value, 50
            else:
                return BinFullness.EMPTY.value, 20

        except Exception as e:
            logger.warning(f"Visual fullness estimation failed: {e}")
            return BinFullness.UNKNOWN.value, 50

    def _clean_vlm_response(self, response: str) -> str:
        """Clean up Florence-2 response by removing special tokens and prompt echoes."""
        # Remove common special tokens
        tokens_to_remove = [
            "<s>", "</s>", "<pad>", "</pad>",
            "<CAPTION>", "</CAPTION>",
            "<DETAILED_CAPTION>", "</DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>", "</MORE_DETAILED_CAPTION>",
            "<OD>", "</OD>",
            "<VQA>", "</VQA>",
        ]
        cleaned = response
        for token in tokens_to_remove:
            cleaned = cleaned.replace(token, "")

        # Remove location tokens like <loc_123>
        import re
        cleaned = re.sub(r'<loc_\d+>', '', cleaned)

        # Remove echoed prompts (Florence-2 sometimes echoes the question)
        prompt_patterns = [
            r"What items or objects can you see.*?List the visible items\.?",
            r"How full is this bin or container.*?completely full\.?",
            r"QA>.*?(items|visible|container|full).*?\.",
        ]
        for pattern in prompt_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    def _detect_bin_contents(self, pil_image: Image.Image, bbox: Dict[str, int]) -> List[str]:
        """
        Detect contents inside a bin using Florence-2 VQA.

        Args:
            pil_image: Full image
            bbox: Bounding box of the bin

        Returns:
            List of detected content items
        """
        try:
            # Crop the bin region
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            img_w, img_h = pil_image.size

            # Add padding
            pad = int(max(w, h) * 0.05)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            bin_crop = pil_image.crop((x1, y1, x2, y2))

            # Ask VQA what items are in the bin
            vqa_prompt = "<VQA>What items or objects can you see inside this container? List the visible items."
            response = self._run_inference(bin_crop, vqa_prompt)
            cleaned = self._clean_vlm_response(response)

            logger.info(f"[CONTENTS] Bin contents: {cleaned}")

            # Parse items from response
            items = []
            # Common waste items to look for
            waste_keywords = [
                "plastic", "paper", "tissue", "wrapper", "bottle", "bag",
                "food", "cardboard", "can", "cup", "napkin", "container",
                "box", "packaging", "trash", "garbage", "waste", "cloth",
                "textile", "organic", "leaf", "leaves", "debris"
            ]

            response_lower = cleaned.lower()
            for keyword in waste_keywords:
                if keyword in response_lower:
                    items.append(keyword)

            return items[:5]  # Return top 5 items

        except Exception as e:
            logger.warning(f"Content detection failed: {e}")
            return []

    def _bbox_iou(self, box1: Dict[str, int], box2: Dict[str, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bboxes."""
        x1, y1, w1, h1 = box1["x"], box1["y"], box1["width"], box1["height"]
        x2, y2, w2, h2 = box2["x"], box2["y"], box2["width"], box2["height"]

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _bbox_contains(self, outer: Dict[str, int], inner: Dict[str, int], threshold: float = 0.8) -> bool:
        """Check if outer bbox contains most of inner bbox."""
        ox, oy, ow, oh = outer["x"], outer["y"], outer["width"], outer["height"]
        ix, iy, iw, ih = inner["x"], inner["y"], inner["width"], inner["height"]

        # Calculate intersection
        xi1 = max(ox, ix)
        yi1 = max(oy, iy)
        xi2 = min(ox + ow, ix + iw)
        yi2 = min(oy + oh, iy + ih)

        if xi2 <= xi1 or yi2 <= yi1:
            return False

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        inner_area = iw * ih

        # If most of inner is inside outer, it's contained
        return (inter_area / inner_area) >= threshold if inner_area > 0 else False

    def _deduplicate_detections(self, detections: List[Detection], iou_threshold: float = 0.25) -> List[Detection]:
        """
        Remove duplicate/overlapping detections based on IoU threshold or containment.
        Uses low threshold (0.25) and containment check for aggressive deduplication.
        """
        if len(detections) <= 1:
            return detections

        # Sort by area (larger first), then by confidence
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.bbox["width"] * d.bbox["height"], d.confidence),
            reverse=True
        )
        kept = []

        for det in sorted_dets:
            is_duplicate = False
            det_area = det.bbox["width"] * det.bbox["height"]

            for kept_det in kept:
                iou = self._bbox_iou(det.bbox, kept_det.bbox)

                # Check if this detection is mostly inside an already-kept detection
                is_contained = self._bbox_contains(kept_det.bbox, det.bbox, threshold=0.6)

                if iou > iou_threshold or is_contained:
                    logger.info(f"[DEDUP] Removing '{det.label}' (IoU={iou:.2f}, contained={is_contained}) - keeping '{kept_det.label}'")
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(det)

        if len(kept) < len(detections):
            logger.info(f"[DEDUP] Reduced from {len(detections)} to {len(kept)} detections")

        return kept

    def _detect_content_objects(
        self,
        pil_image: Image.Image,
        bbox: Dict[str, int],
        bin_index: int
    ) -> List[Detection]:
        """
        Detect objects inside a bin using phrase grounding on common waste items.
        Returns Detection objects with bounding boxes mapped to original image coordinates.

        Args:
            pil_image: Full PIL image
            bbox: Bounding box of the parent bin
            bin_index: Index of the parent bin in the detections list

        Returns:
            List of Detection objects for content items with is_content=True
        """
        content_detections = []

        try:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            img_w, img_h = pil_image.size

            # Crop with slight padding for context
            pad = int(max(w, h) * 0.02)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            bin_crop = pil_image.crop((x1, y1, x2, y2))
            crop_w, crop_h = bin_crop.size

            # Use phrase grounding to find specific items
            # Common waste items to look for
            search_items = ["bottle", "lid", "cap", "paper", "tissue", "wrapper", "cup", "can"]

            logger.info(f"[CONTENT_OD] Searching for items in bin crop ({crop_w}x{crop_h})")

            for item in search_items:
                try:
                    # Use caption to phrase grounding task
                    grounding_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{item}"
                    generated_text = self._run_inference(bin_crop, grounding_prompt)

                    # Parse the grounding output
                    if hasattr(self.processor, 'post_process_generation'):
                        parsed = self.processor.post_process_generation(
                            generated_text,
                            task="<CAPTION_TO_PHRASE_GROUNDING>",
                            image_size=(crop_w, crop_h)
                        )

                        task_key = "<CAPTION_TO_PHRASE_GROUNDING>"
                        if task_key in parsed:
                            result = parsed[task_key]
                            bboxes = result.get('bboxes', [])
                            labels = result.get('labels', [])

                            for det_bbox, label in zip(bboxes, labels):
                                if len(det_bbox) >= 4:
                                    # Get coordinates relative to crop
                                    cx1, cy1, cx2, cy2 = det_bbox[:4]

                                    # Map back to original image coordinates
                                    orig_x = int(x1 + cx1)
                                    orig_y = int(y1 + cy1)
                                    orig_w = int(cx2 - cx1)
                                    orig_h = int(cy2 - cy1)

                                    # Ensure bbox stays within parent bin bounds
                                    orig_x = max(x, min(orig_x, x + w - 10))
                                    orig_y = max(y, min(orig_y, y + h - 10))
                                    orig_w = min(orig_w, (x + w) - orig_x)
                                    orig_h = min(orig_h, (y + h) - orig_y)

                                    if orig_w > 5 and orig_h > 5:  # Minimum size filter
                                        # Use the search item as label if parsed label is generic
                                        final_label = label.strip() if label and label.strip() else item

                                        content_det = Detection(
                                            label=final_label,
                                            confidence=0.80,
                                            bbox={
                                                "x": orig_x,
                                                "y": orig_y,
                                                "width": orig_w,
                                                "height": orig_h
                                            },
                                            is_content=True,
                                            parent_bin_id=bin_index
                                        )
                                        content_detections.append(content_det)
                                        logger.info(f"  -> Content: '{final_label}' at ({orig_x},{orig_y},{orig_w},{orig_h})")

                except Exception as e:
                    logger.debug(f"Grounding for '{item}' failed: {e}")
                    continue

            # Fallback: Also try standard OD and keep non-bin objects
            if not content_detections:
                logger.info(f"[CONTENT_OD] Fallback: Running standard OD")
                generated_text = self._run_inference(bin_crop, "<OD>")

                if hasattr(self.processor, 'post_process_generation'):
                    parsed = self.processor.post_process_generation(
                        generated_text,
                        task="<OD>",
                        image_size=(crop_w, crop_h)
                    )

                    if "<OD>" in parsed:
                        result = parsed["<OD>"]
                        bboxes = result.get('bboxes', [])
                        labels = result.get('labels', [])

                        logger.info(f"[CONTENT_OD] OD found {len(bboxes)} objects")

                        for det_bbox, label in zip(bboxes, labels):
                            if len(det_bbox) >= 4:
                                label_clean = label.strip().lower() if label else ""

                                # Skip if it's the bin/container itself
                                if self._is_bin_label(label_clean):
                                    logger.info(f"  -> Skipping bin label: '{label}'")
                                    continue

                                cx1, cy1, cx2, cy2 = det_bbox[:4]
                                orig_x = int(x1 + cx1)
                                orig_y = int(y1 + cy1)
                                orig_w = int(cx2 - cx1)
                                orig_h = int(cy2 - cy1)

                                orig_x = max(x, min(orig_x, x + w - 10))
                                orig_y = max(y, min(orig_y, y + h - 10))
                                orig_w = min(orig_w, (x + w) - orig_x)
                                orig_h = min(orig_h, (y + h) - orig_y)

                                if orig_w > 5 and orig_h > 5:
                                    content_det = Detection(
                                        label=label.strip() if label else "object",
                                        confidence=0.85,
                                        bbox={
                                            "x": orig_x,
                                            "y": orig_y,
                                            "width": orig_w,
                                            "height": orig_h
                                        },
                                        is_content=True,
                                        parent_bin_id=bin_index
                                    )
                                    content_detections.append(content_det)
                                    logger.info(f"  -> Content (OD): '{label}' at ({orig_x},{orig_y},{orig_w},{orig_h})")

        except Exception as e:
            logger.warning(f"Content object detection failed: {e}")

        # Deduplicate overlapping detections (IoU > 0.5 means same object)
        if len(content_detections) > 1:
            before_count = len(content_detections)
            content_detections = self._deduplicate_detections(content_detections, iou_threshold=0.5)
            if before_count != len(content_detections):
                logger.info(f"[CONTENT_OD] Deduplicated: {before_count} -> {len(content_detections)} detections")

        return content_detections

    def _generate_vlm_summary(self, pil_image: Image.Image, bin_detections: List[Detection], content_items: List[str] = None) -> str:
        """
        Generate a structured natural language summary using Florence-2.

        Args:
            pil_image: The full scene image
            bin_detections: List of detected bins with fullness info
            content_items: Optional list of detected content items

        Returns:
            Multi-line structured summary
        """
        try:
            if not bin_detections:
                return "No waste containers detected in the current frame."

            # Build comprehensive summary
            bin_count = len(bin_detections)
            fullness_percents = [d.bin_fullness_percent for d in bin_detections if d.bin_fullness_percent]
            avg_fullness = sum(fullness_percents) // len(fullness_percents) if fullness_percents else 0

            # Get bin labels
            bin_labels = list(set(d.label for d in bin_detections))
            bin_types = ", ".join(bin_labels[:3])

            # Determine status
            if avg_fullness >= 90:
                status = "CRITICAL"
                action = "Immediate collection required!"
            elif avg_fullness >= 75:
                status = "HIGH"
                action = "Schedule collection soon."
            elif avg_fullness >= 50:
                status = "MODERATE"
                action = "Monitor regularly."
            else:
                status = "LOW"
                action = "Good capacity available."

            # Build structured summary with line breaks
            lines = []
            lines.append(f"Containers: {bin_count} ({bin_types})")
            lines.append(f"Fill Level: {avg_fullness}% [{status}]")
            lines.append(f"Action: {action}")

            # Add content info if available
            if content_items:
                items_str = ", ".join(content_items[:4])
                lines.append(f"Contents: {items_str}")

            # Add detailed scene description
            try:
                caption_prompt = "<MORE_DETAILED_CAPTION>"
                scene_caption = self._run_inference(pil_image, caption_prompt)
                cleaned_caption = self._clean_vlm_response(scene_caption)

                # Truncate scene description to fit DB column (max ~400 chars for scene)
                if cleaned_caption and len(cleaned_caption) > 10:
                    if len(cleaned_caption) > 400:
                        cleaned_caption = cleaned_caption[:397] + "..."
                    lines.append(f"Scene: {cleaned_caption}")
            except Exception as e:
                logger.debug(f"Caption generation skipped: {e}")

            summary = "\n".join(lines)

            # Ensure total summary fits in DB Text column (safe limit)
            if len(summary) > 800:
                # Keep first lines, truncate scene
                summary = summary[:797] + "..."

            logger.info(f"[VLM] Generated summary:\n{summary}")
            return summary

        except Exception as e:
            logger.warning(f"VLM summary generation failed: {e}")
            bin_count = len(bin_detections)
            fullness_percents = [d.bin_fullness_percent for d in bin_detections if d.bin_fullness_percent]
            avg_fullness = sum(fullness_percents) // len(fullness_percents) if fullness_percents else 0
            return f"Containers: {bin_count}\nFill Level: {avg_fullness}%"

    def detect(
        self,
        image: np.ndarray,
        frame_id: str = "",
        timestamp: str = ""
    ) -> DetectionResult:
        """
        Run object detection and bin fullness classification on an image.

        Args:
            image: BGR numpy array from OpenCV.
            frame_id: Unique identifier for this frame.
            timestamp: ISO format timestamp.

        Returns:
            DetectionResult with bounding boxes, labels, and bin classifications.
        """
        if not self._loaded:
            if not self.load():
                return DetectionResult(
                    timestamp=timestamp,
                    frame_id=frame_id,
                    detections=[]
                )

        start_time = time.time()

        try:
            # Convert BGR to RGB and then to PIL
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            image_height, image_width = image.shape[:2]

            # Step 1: Run object detection
            task_prompt = self.config.task
            generated_text = self._run_inference(pil_image, task_prompt)

            # Parse the output to extract bounding boxes
            detections = self._parse_detection_output(
                generated_text,
                image_width,
                image_height
            )

            # Filter by confidence threshold
            detections = [
                d for d in detections
                if d.confidence >= self.config.confidence_threshold
            ]

            # Deduplicate overlapping detections
            detections = self._deduplicate_detections(detections, iou_threshold=0.5)

            logger.info(f"[DETECT] Found {len(detections)} objects after confidence filter and deduplication")
            for det in detections:
                logger.info(f"  -> Object: '{det.label}' (conf: {det.confidence:.2f})")

            # Step 2: Classify bin fullness for detected bins
            bin_detections = []
            content_detections = []

            for idx, detection in enumerate(detections):
                is_bin = self._is_bin_label(detection.label)
                logger.info(f"[CLASSIFY] '{detection.label}' is_bin={is_bin}")

                if is_bin:
                    if self.config.classify_fullness:
                        logger.info(f"  -> Running VQA fullness classification...")
                        fullness, percent = self._classify_bin_fullness(
                            pil_image,
                            detection.bbox
                        )
                        detection.bin_fullness = fullness
                        detection.bin_fullness_percent = percent
                        logger.info(f"  -> Fullness: {fullness} ({percent}%)")
                    else:
                        detection.bin_fullness = BinFullness.UNKNOWN.value
                        detection.bin_fullness_percent = 50
                        logger.info(f"  -> Fullness classification disabled")

                    bin_detections.append(detection)

                    # Step 2.5: Detect objects INSIDE this bin
                    logger.info(f"  -> Running content object detection...")
                    bin_contents = self._detect_content_objects(pil_image, detection.bbox, idx)
                    content_detections.extend(bin_contents)
                    logger.info(f"  -> Found {len(bin_contents)} content objects")

            # Add content detections to main detections list
            detections.extend(content_detections)

            # Calculate overall bin status
            bin_detected = len(bin_detections) > 0
            bin_count = len(bin_detections)

            overall_fullness = None
            overall_fullness_percent = None
            status_summary = "No bins detected"

            logger.info(f"[SUMMARY] Bins detected: {bin_count}")

            if bin_detected:
                # Calculate average fullness
                fullness_percents = [
                    d.bin_fullness_percent for d in bin_detections
                    if d.bin_fullness_percent is not None
                ]
                if fullness_percents:
                    avg_percent = sum(fullness_percents) // len(fullness_percents)
                    overall_fullness_percent = avg_percent

                    # Map to category
                    if avg_percent <= 25:
                        overall_fullness = BinFullness.EMPTY.value
                    elif avg_percent <= 75:
                        overall_fullness = BinFullness.PARTIAL.value
                    elif avg_percent <= 90:
                        overall_fullness = BinFullness.MOSTLY_FULL.value
                    else:
                        overall_fullness = BinFullness.FULL.value

                    logger.info(f"[SUMMARY] Overall fullness: {overall_fullness} ({avg_percent}%)")

                    # Step 3: Detect contents inside bins
                    all_contents = []
                    for bin_det in bin_detections:
                        logger.info(f"[CONTENTS] Analyzing contents of '{bin_det.label}'...")
                        contents = self._detect_bin_contents(pil_image, bin_det.bbox)
                        all_contents.extend(contents)

                    # Remove duplicates
                    unique_contents = list(dict.fromkeys(all_contents))
                    if unique_contents:
                        logger.info(f"[CONTENTS] Found items: {', '.join(unique_contents)}")

                    # Generate VLM-based summary with content info
                    logger.info("[SUMMARY] Generating VLM summary...")
                    status_summary = self._generate_vlm_summary(pil_image, bin_detections, unique_contents)

            logger.info(f"[SUMMARY] Status: {status_summary}")

            inference_time = (time.time() - start_time) * 1000
            logger.info(f"[DONE] Total inference time: {inference_time:.1f}ms")

            return DetectionResult(
                timestamp=timestamp,
                frame_id=frame_id,
                detections=detections,
                inference_time_ms=inference_time,
                image_width=image_width,
                image_height=image_height,
                bin_detected=bin_detected,
                bin_count=bin_count,
                overall_fullness=overall_fullness,
                overall_fullness_percent=overall_fullness_percent,
                status_summary=status_summary
            )

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return DetectionResult(
                timestamp=timestamp,
                frame_id=frame_id,
                detections=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                status_summary=f"Detection error: {str(e)}"
            )

    def _parse_detection_output(
        self,
        output: str,
        image_width: int,
        image_height: int
    ) -> List[Detection]:
        """
        Parse Florence-2 output to extract bounding boxes.

        Florence-2 outputs in format like:
        <OD><loc_x1><loc_y1><loc_x2><loc_y2>label<loc_x1>...

        Args:
            output: Raw model output text.
            image_width: Original image width.
            image_height: Original image height.

        Returns:
            List of Detection objects.
        """
        detections = []

        try:
            # Use processor's post-processing if available
            if hasattr(self.processor, 'post_process_generation'):
                parsed = self.processor.post_process_generation(
                    output,
                    task=self.config.task,
                    image_size=(image_width, image_height)
                )

                if self.config.task in parsed:
                    result = parsed[self.config.task]

                    bboxes = result.get('bboxes', [])
                    labels = result.get('labels', [])

                    for bbox, label in zip(bboxes, labels):
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]

                            detection = Detection(
                                label=label.strip() if label else "object",
                                confidence=0.9,  # Florence-2 doesn't output confidence
                                bbox={
                                    "x": int(x1),
                                    "y": int(y1),
                                    "width": int(x2 - x1),
                                    "height": int(y2 - y1)
                                }
                            )
                            detections.append(detection)

                return detections

            # Fallback: Manual parsing
            # Pattern for location tokens: <loc_XXX>
            loc_pattern = r'<loc_(\d+)>'
            locations = re.findall(loc_pattern, output)

            # Extract labels between location groups
            label_pattern = r'<loc_\d+><loc_\d+><loc_\d+><loc_\d+>([^<]+)'
            labels = re.findall(label_pattern, output)

            # Process in groups of 4 (x1, y1, x2, y2)
            for i in range(0, len(locations) - 3, 4):
                if i // 4 < len(labels):
                    # Florence-2 uses 1000-scale coordinates
                    x1 = int(locations[i]) * image_width // 1000
                    y1 = int(locations[i+1]) * image_height // 1000
                    x2 = int(locations[i+2]) * image_width // 1000
                    y2 = int(locations[i+3]) * image_height // 1000

                    label = labels[i // 4].strip()

                    detection = Detection(
                        label=label if label else "object",
                        confidence=0.9,
                        bbox={
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1
                        }
                    )
                    detections.append(detection)

        except Exception as e:
            logger.warning(f"Failed to parse detection output: {e}")
            logger.debug(f"Raw output: {output}")

        return detections

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device or "cpu"


# Import cv2 here to avoid circular imports
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not installed, some features may not work")
    cv2 = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test detector
    config = DetectorConfig(
        model_name="microsoft/Florence-2-base",
        device="auto",
        confidence_threshold=0.3,
        classify_fullness=True
    )

    detector = Florence2Detector(config)

    if detector.load():
        print("Model loaded successfully")
        print(f"Device: {detector.device}")

        # Create a test image (random noise)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        from datetime import datetime
        import uuid

        result = detector.detect(
            test_image,
            frame_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        print(f"Inference time: {result.inference_time_ms:.1f}ms")
        print(f"Detections: {len(result.detections)}")
        print(f"Bin detected: {result.bin_detected}")
        print(f"Status: {result.status_summary}")

        for det in result.detections:
            print(f"  - {det.label}: {det.bbox}")
            if det.bin_fullness:
                print(f"    Fullness: {det.bin_fullness} ({det.bin_fullness_percent}%)")

        detector.unload()
