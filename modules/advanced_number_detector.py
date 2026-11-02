"""
Multi-Modal Number Detection Module.
Combines multi-pass OCR digit extraction, vertical numeral contour shape detection (for marquee,
balloon, and floral numbers), and Vision-Language semantic scoring to reliably identify decoration numbers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from modules.semantic_vision import SemanticVisionModel

logger = logging.getLogger(__name__)

CANONICAL_NUMBERS = {
    "1": ["1", "one", "first", "1st", "one!", "|", "l"],
    "2": ["2", "two", "second", "2nd"],
    "3": ["3", "three", "third", "3rd"],
    "4": ["4", "four", "fourth", "4th"],
    "5": ["5", "five", "fifth", "5th"]
}


class AdvancedNumberDetector:
    def __init__(self, semantic_model: Optional[SemanticVisionModel] = None):
        self.semantic_model = semantic_model or SemanticVisionModel.get_instance()

    def detect_number(self, 
                      image_bgr: np.ndarray, 
                      target_numbers: List[str], 
                      ocr_results: Dict[str, Any],
                      pil_image: Optional[Image.Image] = None) -> Tuple[bool, Optional[str], float, List[Dict[str, Any]]]:
        """
        Runs multi-modal number verification for target numerals (e.g. ['1', 'ONE']).
        Returns:
            found (bool)
            canonical_number (str or None)
            confidence (float 0.0 to 1.0)
            evidence_records (List[Dict])
        """
        if not target_numbers:
            return True, None, 1.0, []

        evidence_records: List[Dict[str, Any]] = []

        # Canonicalize target query set
        canonical_targets = set()
        for t in target_numbers:
            t_str = str(t).lower().strip()
            matched_canon = None
            for canon_key, variants in CANONICAL_NUMBERS.items():
                if t_str == canon_key or t_str in variants:
                    matched_canon = canon_key
                    break
            canonical_targets.add(matched_canon or t_str)

        # ---------------- Method 1: OCR Multi-Pass Tokens ----------------
        ocr_numbers = ocr_results.get("detected_numbers", [])
        for num_rec in ocr_numbers:
            c = num_rec.get("canonical")
            if c in canonical_targets:
                evidence_records.append({
                    "method": num_rec.get("method", "OCR"),
                    "canonical": c,
                    "confidence": num_rec.get("confidence", 0.85),
                    "bbox": num_rec.get("bbox"),
                    "region": num_rec.get("region", "ocr_token"),
                    "details": f"OCR identified '{num_rec.get('raw_text')}' as number {c}"
                })

        # ---------------- Method 2: Contour Shape Analysis for Number 1 ----------------
        # Freestanding marquee "1", foil balloon "1", or floral foam "1"
        if "1" in canonical_targets and image_bgr is not None and image_bgr.size > 0:
            contour_matches = self._find_number_one_contours(image_bgr)
            for cm in contour_matches:
                evidence_records.append({
                    "method": "CONTOUR_SHAPE_ANALYSIS",
                    "canonical": "1",
                    "confidence": cm["confidence"],
                    "bbox": cm["bbox"],
                    "region": cm["region"],
                    "details": f"Vertical aspect-ratio contour ({cm['aspect_ratio']:.1f}:1) matching numeral '1' shape"
                })

        # ---------------- Method 3: Vision-Language Model Verification ----------------
        if "1" in canonical_targets and pil_image is not None and self.semantic_model.is_available:
            vlm_eval = self.semantic_model.evaluate_scene(pil_image)
            prob_num1 = vlm_eval.get("has_number_one_prob", 0.0)
            prob_bday1 = vlm_eval.get("is_first_birthday_prob", 0.0)

            if prob_num1 >= 0.55 or prob_bday1 >= 0.55:
                evidence_records.append({
                    "method": "VISION_LANGUAGE_SEMANTIC",
                    "canonical": "1",
                    "confidence": round(max(prob_num1, prob_bday1), 3),
                    "bbox": None,
                    "region": "scene_level",
                    "details": f"OpenCLIP zero-shot scene confidence (Number 1: {prob_num1:.2f}, 1st BDay: {prob_bday1:.2f})"
                })

        if not evidence_records:
            return False, None, 0.0, []

        # Find best evidence
        evidence_records.sort(key=lambda x: x["confidence"], reverse=True)
        best_rec = evidence_records[0]
        # Multi-evidence boost: if multiple distinct methods agree, boost confidence
        unique_methods = set(e["method"] for e in evidence_records)
        boosted_conf = min(0.98, best_rec["confidence"] + (0.08 * (len(unique_methods) - 1)))

        return True, best_rec["canonical"], round(boosted_conf, 3), evidence_records

    def _find_number_one_contours(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects tall vertical freestanding structures matching numeral 1 aspect ratio.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        min_height = int(h * 0.10)  # At least 10% of image height
        max_height = int(h * 0.65)  # At most 65% of image height

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if min_height <= ch <= max_height and cw > 15:
                aspect_ratio = float(ch) / float(cw)
                # The number 1 typically has an aspect ratio between 2.2 and 5.0
                if 2.1 <= aspect_ratio <= 5.2:
                    # Check region
                    cx = x + cw // 2
                    if cx < w * 0.40:
                        region = "left_flank"
                    elif cx > w * 0.60:
                        region = "right_flank"
                    else:
                        region = "center_stage"

                    # Calculate confidence based on closeness to ideal ratio 3.2
                    ratio_diff = abs(aspect_ratio - 3.2)
                    conf = max(0.50, min(0.85, 0.85 - 0.10 * ratio_diff))

                    candidates.append({
                        "bbox": [x, y, x + cw, y + ch],
                        "aspect_ratio": aspect_ratio,
                        "confidence": round(conf, 2),
                        "region": region
                    })

        # Return top 2 strongest contour candidates
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates[:2]
