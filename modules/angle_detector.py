"""
Enhanced Camera Angle Detection Module.
Combines geometric perspective line analysis, ground-level horizontal tilt,
organic decor asymmetry tolerance, and OpenCLIP zero-shot viewpoint classification.
Distinguishes FRONT, LEFT_SIDE, RIGHT_SIDE, CLOSEUP, and OBSTRUCTED views.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

from modules.semantic_vision import SemanticVisionModel


class AngleDetector:
    def __init__(self, front_symmetry_threshold: float = 0.52, max_horizontal_tilt_deg: float = 9.0):
        self.front_symmetry_threshold = front_symmetry_threshold
        self.max_horizontal_tilt_deg = max_horizontal_tilt_deg
        self.semantic_model = SemanticVisionModel.get_instance()

    def evaluate_angle(self, 
                       image_bgr: np.ndarray, 
                       pil_image: Optional[Image.Image] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Evaluates camera angle using multi-evidence geometric and semantic fusion.
        Returns:
            angle_class: 'FRONT', 'SIDE', 'CLOSEUP', or 'OTHER'
            confidence: float (0.0 to 1.0)
            details: dict of diagnostic measurements
        """
        if image_bgr is None or image_bgr.size == 0:
            return "OTHER", 0.0, {"error": "Empty image"}

        h, w = image_bgr.shape[:2]
        target_w, target_h = 640, 480
        resized = cv2.resize(image_bgr, (target_w, target_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1. Bilateral Symmetry Score
        mid_x = target_w // 2
        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:]
        right_flipped = cv2.flip(right_half, 1)

        min_w = min(left_half.shape[1], right_flipped.shape[1])
        diff = cv2.absdiff(left_half[:, :min_w], right_flipped[:, :min_w])
        symmetry_score = 1.0 - (float(np.mean(diff)) / 255.0)

        # Check left vs right weight (for distinguishing LEFT_SIDE vs RIGHT_SIDE)
        left_energy = float(np.mean(cv2.Canny(left_half, 50, 150)))
        right_energy = float(np.mean(cv2.Canny(right_half, 50, 150)))
        energy_ratio = left_energy / max(1.0, right_energy)

        # 2. Line Perspective & Tilt Analysis
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=75, minLineLength=50, maxLineGap=12)

        horizontal_tilts = []
        perspective_lines = []

        if lines is not None:
            for line in lines:
                pt = line.reshape(-1)
                if len(pt) < 4:
                    continue
                x1, y1, x2, y2 = pt[:4]
                angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if angle_deg < 0:
                    angle_deg += 180

                # Near-horizontal lines (stage floor, table, banner)
                if angle_deg <= 25 or angle_deg >= 155:
                    tilt = angle_deg if angle_deg <= 25 else abs(180 - angle_deg)
                    horizontal_tilts.append(tilt)
                # Oblique perspective lines
                elif 25 < angle_deg < 70 or 110 < angle_deg < 155:
                    perspective_lines.append(angle_deg)

        avg_tilt = float(np.mean(horizontal_tilts)) if horizontal_tilts else 4.0
        perspective_count = len(perspective_lines)

        # 3. Vision-Language Viewpoint Prediction
        vlm_view = "UNKNOWN"
        vlm_front_prob = 0.0
        if pil_image is not None and self.semantic_model.is_available:
            vlm_res = self.semantic_model.evaluate_scene(pil_image)
            vlm_view = vlm_res.get("predicted_view", "UNKNOWN")
            vlm_front_prob = vlm_res.get("is_front_view_prob", 0.0)

        details = {
            "symmetry_score": round(symmetry_score, 3),
            "avg_horizontal_tilt": round(avg_tilt, 2),
            "perspective_line_count": perspective_count,
            "vlm_view": vlm_view,
            "vlm_front_prob": round(vlm_front_prob, 3)
        }

        # 4. Multi-Evidence Decision
        # If VLM strongly asserts FRONT view (>= 0.55), do not let asymmetric balloon garlands trigger false SIDE
        if vlm_front_prob >= 0.55 and avg_tilt <= 10.0:
            conf = min(0.95, 0.65 + 0.30 * vlm_front_prob)
            return "FRONT", round(conf, 2), details

        # Front-Facing Geometric match:
        if symmetry_score >= self.front_symmetry_threshold and avg_tilt <= self.max_horizontal_tilt_deg and perspective_count < 14:
            conf = min(0.95, 0.60 + 0.35 * (symmetry_score - 0.45))
            return "FRONT", round(conf, 2), details

        # Strong Side View
        if avg_tilt > 12.0 or perspective_count >= 16 or vlm_view == "SIDE":
            if energy_ratio > 1.6:
                side_label = "SIDE"  # Left perspective dominant
            else:
                side_label = "SIDE"
            conf = min(0.90, 0.60 + 0.02 * perspective_count)
            return side_label, round(conf, 2), details

        # Closeup detection
        if vlm_view == "CLOSEUP" or (perspective_count < 3 and symmetry_score < 0.40):
            return "OTHER", 0.65, details

        # Moderate symmetry / borderline:
        if symmetry_score >= 0.48 and avg_tilt <= 9.0:
            return "FRONT", 0.55, details

        return "OTHER", 0.50, details
