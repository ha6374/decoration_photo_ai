"""
Blur & Sharpness Analysis Module.
Uses Laplacian variance and edge frequency distribution to evaluate photo clarity.
Allows user-configurable sharpness thresholds per category.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class BlurDetector:
    def __init__(self, default_threshold: float = 100.0):
        self.default_threshold = default_threshold

    def calculate_sharpness(self, image_bgr: np.ndarray) -> float:
        """
        Calculates Laplacian variance score.
        Higher variance indicates sharper edges; low variance indicates blur.
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.0

        # Convert to grayscale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        # Optional resize to standard 1080p scale if extremely large to maintain consistent scale
        h, w = gray.shape
        if max(h, w) > 1920:
            scale = 1920.0 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 64-bit float Laplacian to avoid overflow
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(laplacian.var())
        return round(score, 2)

    def evaluate(self, image_bgr: np.ndarray, threshold: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Evaluates image sharpness against a threshold.
        Returns:
            is_blurry (bool)
            sharpness_score (float)
            confidence (float between 0.0 and 1.0)
        """
        thresh = threshold if threshold is not None else self.default_threshold
        score = self.calculate_sharpness(image_bgr)
        is_blurry = score < thresh

        # Confidence is higher the further the score is from the threshold
        if thresh > 0:
            diff_ratio = abs(score - thresh) / thresh
            confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(1.0, diff_ratio)))
        else:
            confidence = 1.0

        return is_blurry, score, round(confidence, 2)
