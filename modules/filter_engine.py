"""
Smart Filter Execution Engine with Multi-Evidence Fusion.
Coordinates Advanced OCR, Multi-Modal Number Detection, OpenCLIP Semantic Vision,
YOLOv8 Human Detection, Laplacian Blur Sharpness, and Watermark analysis.
Implements calibrated decision logic that avoids false rejections and provides full diagnostic evidence.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image

from database.models import CategoryRule, ImageMetadata
from database.db import DatabaseManager
from modules.image_loader import ImageLoader
from modules.duplicate_detector import DuplicateDetector
from modules.blur_detector import BlurDetector
from modules.human_detector import HumanDetector
from modules.watermark_detector import WatermarkDetector
from modules.angle_detector import AngleDetector
from modules.advanced_ocr import AdvancedOCREngine
from modules.advanced_number_detector import AdvancedNumberDetector
from modules.semantic_vision import SemanticVisionModel

logger = logging.getLogger(__name__)


class FilterEngine:
    def __init__(self, db: DatabaseManager, debug_dir: Optional[str] = None):
        self.db = db
        self.duplicate_detector = DuplicateDetector(db=db)
        self.blur_detector = BlurDetector()
        self.human_detector = HumanDetector(model_type="auto")
        self.watermark_detector = WatermarkDetector()
        self.angle_detector = AngleDetector()
        self.ocr_engine = AdvancedOCREngine()
        self.semantic_vision = SemanticVisionModel.get_instance()
        self.number_detector = AdvancedNumberDetector(semantic_model=self.semantic_vision)

        # Setup debug log output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.debug_dir = debug_dir or os.path.join(base_dir, "logs", "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

    def process_image(self, 
                      filepath: str, 
                      category_rule: CategoryRule, 
                      batch_id: str) -> ImageMetadata:
        """
        Executes multi-evidence fusion pipeline.
        Produces full diagnostic evidence and calibrated final classification.
        """
        rejection_reasons: List[str] = []
        review_reasons: List[str] = []
        evidence_summary: Dict[str, Any] = {
            "image_path": filepath,
            "filename": os.path.basename(filepath),
            "batch_id": batch_id,
            "category": category_rule.name
        }

        # 1. Structural File Inspection
        info = ImageLoader.inspect_image_file(filepath)
        if not info:
            return ImageMetadata(
                original_filename=os.path.basename(filepath),
                file_size_bytes=0,
                width=0,
                height=0,
                file_format="UNKNOWN",
                sha256="",
                phash="",
                source_path=filepath,
                category=category_rule.name,
                upload_batch_id=batch_id,
                status="REJECTED",
                rejection_reasons=["Corrupted or unreadable image file."]
            )

        # 2. Check Global Duplicates
        is_dup, dup_type, dup_rec, dup_sim = self.duplicate_detector.check_duplicate(
            filepath=filepath,
            sha256=info["sha256"],
            phash_str=info["phash"],
            category=category_rule.name
        )

        if is_dup:
            dup_id = dup_rec.get("id") if dup_rec else None
            dup_orig_name = dup_rec.get("original_filename") if dup_rec else "Existing record"
            return ImageMetadata(
                original_filename=info["filename"],
                file_size_bytes=info["file_size_bytes"],
                width=info["width"],
                height=info["height"],
                file_format=info["file_format"],
                sha256=info["sha256"],
                phash=info["phash"],
                source_path=filepath,
                category=category_rule.name,
                upload_batch_id=batch_id,
                status="DUPLICATE",
                rejection_reasons=[f"Duplicate of image #{dup_id} ({dup_orig_name}) via {dup_type} (Similarity: {dup_sim:.2f})"],
                duplicate_of_id=dup_id,
                duplicate_similarity=dup_sim
            )

        # 3. Resolution and File Size Checks
        if info["width"] < category_rule.min_width or info["height"] < category_rule.min_height:
            rejection_reasons.append(
                f"Resolution too low: {info['width']}x{info['height']} (min required: {category_rule.min_width}x{category_rule.min_height})"
            )

        file_size_kb = info["file_size_bytes"] / 1024.0
        if file_size_kb < category_rule.min_file_size_kb:
            rejection_reasons.append(
                f"File size too small: {file_size_kb:.1f} KB (min: {category_rule.min_file_size_kb} KB)"
            )
        elif file_size_kb > category_rule.max_file_size_kb:
            rejection_reasons.append(
                f"File size too large: {file_size_kb:.1f} KB (max: {category_rule.max_file_size_kb} KB)"
            )

        # Load OpenCV and PIL representations safely
        cv_img = ImageLoader.load_cv2_image(filepath)
        pil_img = ImageLoader.load_pil_image(filepath)

        if cv_img is None or pil_img is None:
            return ImageMetadata(
                original_filename=info["filename"],
                file_size_bytes=info["file_size_bytes"],
                width=info["width"],
                height=info["height"],
                file_format=info["file_format"],
                sha256=info["sha256"],
                phash=info["phash"],
                source_path=filepath,
                category=category_rule.name,
                upload_batch_id=batch_id,
                status="REJECTED",
                rejection_reasons=["Failed to decode image pixels for computer vision analysis."]
            )

        # 4. Blur Sharpness Analysis
        blur_score = 0.0
        if category_rule.blur_detection_enabled:
            is_blurry, blur_score, blur_conf = self.blur_detector.evaluate(
                cv_img, threshold=category_rule.blur_threshold
            )
            evidence_summary["blur"] = {"score": blur_score, "threshold": category_rule.blur_threshold, "is_blurry": is_blurry}
            if is_blurry:
                # If severe blur (< 60% of threshold), hard reject; if borderline, flag for review
                if blur_score < (category_rule.blur_threshold * 0.70):
                    rejection_reasons.append(
                        f"Image is blurry: sharpness {blur_score:.1f} < threshold {category_rule.blur_threshold}"
                    )
                else:
                    review_reasons.append(
                        f"Borderline sharpness: {blur_score:.1f} (threshold {category_rule.blur_threshold}) - verify clarity"
                    )

        # 5. OpenCLIP Semantic Vision Evaluation
        vlm_results = self.semantic_vision.evaluate_scene(pil_img)
        evidence_summary["semantic_vlm"] = vlm_results

        # 6. Advanced Multi-Stage OCR Execution
        needs_ocr = bool(category_rule.required_text or category_rule.required_numbers or category_rule.custom_keywords)
        ocr_results = {
            "full_text": "",
            "raw_tokens": [],
            "detected_words": [],
            "detected_numbers": [],
            "highest_confidence": 0.0
        }
        if needs_ocr:
            ocr_results = self.ocr_engine.extract_all_text(cv_img)
            evidence_summary["ocr"] = {
                "detected_text": ocr_results["full_text"],
                "token_count": len(ocr_results["raw_tokens"]),
                "highest_conf": ocr_results["highest_confidence"]
            }

        # 7. Context-Aware Keyword & Phrase Matching
        text_passed = True
        matched_text_evidence = []
        if category_rule.required_text:
            text_passed, matched_text_evidence, missing_keywords = self.ocr_engine.match_required_keywords(
                ocr_tokens=ocr_results["raw_tokens"],
                full_text=ocr_results["full_text"],
                required_keywords=category_rule.required_text
            )
            evidence_summary["required_text"] = {
                "passed": text_passed,
                "evidence": matched_text_evidence,
                "missing": missing_keywords
            }

            if not text_passed:
                # If VLM strongly asserts birthday theme (>= 0.70) but OCR slightly missed decorative cursive text,
                # send to REVIEW_REQUIRED instead of harsh rejection!
                if vlm_results.get("is_general_birthday_prob", 0.0) >= 0.70:
                    review_reasons.append(
                        f"Missing exact OCR text: {', '.join(missing_keywords)}, but visual scene strongly matches celebration theme"
                    )
                else:
                    rejection_reasons.append(f"Missing required text: {', '.join(missing_keywords)}")

        # 8. Multi-Modal Number Detection (OCR + Contours + VLM)
        number_passed = True
        detected_number_str = ""
        canonical_detected_numbers: List[str] = []

        if category_rule.required_numbers:
            number_passed, detected_number_str, num_conf, num_evidence = self.number_detector.detect_number(
                image_bgr=cv_img,
                target_numbers=category_rule.required_numbers,
                ocr_results=ocr_results,
                pil_image=pil_img
            )
            evidence_summary["required_numbers"] = {
                "passed": number_passed,
                "detected": detected_number_str,
                "confidence": num_conf,
                "evidence": num_evidence
            }

            if detected_number_str:
                canonical_detected_numbers.append(detected_number_str)

            if not number_passed:
                # If required text is found and scene is first birthday, send to review rather than hard rejecting
                if text_passed and vlm_results.get("is_first_birthday_prob", 0.0) >= 0.50:
                    review_reasons.append(
                        f"Could not conclusively verify number {', '.join(category_rule.required_numbers)} - verify manual review"
                    )
                else:
                    rejection_reasons.append(f"Missing required number: {', '.join(category_rule.required_numbers)}")

        # 9. Watermark Detection
        watermark_detected = False
        watermark_conf = 0.0
        if category_rule.watermark_detection_enabled:
            watermark_detected, watermark_conf, wm_reason = self.watermark_detector.detect_watermark(
                cv_img, ocr_text=ocr_results["full_text"]
            )
            evidence_summary["watermark"] = {"detected": watermark_detected, "confidence": watermark_conf, "reason": wm_reason}
            if watermark_detected:
                if watermark_conf >= 0.80:
                    rejection_reasons.append(f"Watermark detected: {wm_reason}")
                else:
                    review_reasons.append(f"Possible watermark ({wm_reason}) - needs review")

        # 10. Human Detection (YOLOv8 + Semantic VLM)
        human_detected = False
        human_count = 0
        human_conf = 0.0
        if category_rule.human_detection_enabled:
            human_detected, human_count, human_conf, model_used = self.human_detector.detect_humans(cv_img)
            # If YOLO detects person or VLM strongly detects people (> 0.70)
            if not human_detected and vlm_results.get("has_people_prob", 0.0) > 0.75:
                human_detected = True
                human_count = 1
                human_conf = vlm_results["has_people_prob"]
                model_used = "Semantic-VLM"

            evidence_summary["human"] = {
                "detected": human_detected,
                "count": human_count,
                "confidence": human_conf,
                "model": model_used
            }

            if human_detected:
                msg = f"Detected {human_count} person(s) in photo ({model_used}, conf {human_conf:.2f})"
                if category_rule.human_action == "REJECT":
                    rejection_reasons.append(msg)
                else:
                    review_reasons.append(msg)

        # 11. Camera Angle Detection (Perspective + OpenCLIP Viewpoint)
        detected_angle = "UNKNOWN"
        angle_conf = 0.0
        if category_rule.angle_filter_enabled:
            detected_angle, angle_conf, angle_details = self.angle_detector.evaluate_angle(cv_img, pil_image=pil_img)
            pref_angles = category_rule.preferred_angles or ["FRONT"]
            evidence_summary["angle"] = {"detected": detected_angle, "confidence": angle_conf, "details": angle_details}

            if detected_angle not in pref_angles:
                if angle_conf >= 0.75:
                    rejection_reasons.append(
                        f"Non-preferred camera angle '{detected_angle}' (preferred: {', '.join(pref_angles)})"
                    )
                else:
                    review_reasons.append(
                        f"Borderline camera angle '{detected_angle}' (conf {angle_conf:.2f}) - verify viewpoint"
                    )

        # 12. Custom AI Rule / Keywords
        if category_rule.custom_keywords:
            kw_ok, kw_ev, missing_kw = self.ocr_engine.match_required_keywords(
                ocr_tokens=ocr_results["raw_tokens"],
                full_text=ocr_results["full_text"],
                required_keywords=category_rule.custom_keywords
            )
            if not kw_ok:
                rejection_reasons.append(f"Missing custom keyword: {', '.join(missing_kw)}")

        if category_rule.custom_ai_prompt:
            review_reasons.append(f"Custom AI Rule: '{category_rule.custom_ai_prompt}'")

        # ---------------- Intelligent Evidence Fusion Decision ----------------
        if rejection_reasons:
            final_status = "REJECTED"
            final_reasons = rejection_reasons
        elif review_reasons:
            final_status = "REVIEW_REQUIRED"
            final_reasons = review_reasons
        else:
            final_status = "APPROVED"
            final_reasons = ["All mandatory criteria and quality filters passed successfully."]

        evidence_summary["final_status"] = final_status
        evidence_summary["decision_reasons"] = final_reasons

        # Persist structured debug evidence file
        self._save_debug_record(info["sha256"][:12], evidence_summary)

        return ImageMetadata(
            original_filename=info["filename"],
            file_size_bytes=info["file_size_bytes"],
            width=info["width"],
            height=info["height"],
            file_format=info["file_format"],
            sha256=info["sha256"],
            phash=info["phash"],
            source_path=filepath,
            category=category_rule.name,
            upload_batch_id=batch_id,
            status=final_status,
            rejection_reasons=final_reasons,
            blur_score=blur_score,
            detected_angle=detected_angle,
            angle_confidence=angle_conf,
            human_detected=human_detected,
            human_count=human_count,
            watermark_detected=watermark_detected,
            watermark_confidence=watermark_conf,
            detected_text=ocr_results.get("full_text", ""),
            detected_numbers=canonical_detected_numbers
        )

    def _save_debug_record(self, short_hash: str, evidence: Dict[str, Any]):
        """Saves structured JSON file for OCR and AI evidence audit."""
        try:
            fn = f"debug_{short_hash}.json"
            fp = os.path.join(self.debug_dir, fn)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(evidence, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed writing debug JSON: {e}")
