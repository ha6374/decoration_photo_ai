"""
Enhanced Watermark & Unwanted Overlay Detection Module.
Distinguishes actual stock watermarks (Shutterstock, Getty, Alamy, iStock, copyright stamps)
from decorative celebration backdrops, floral borders, and balloon cluster textures.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np

# Specific stock agency, copyright, and watermark indicators
STRICT_WATERMARK_KEYWORDS = [
    "watermark", "shutterstock", "getty", "alamy", "istock", "depositphotos",
    "dreamstime", "123rf", "adobe stock", "stock photo", "copyright",
    "all rights reserved", "rights reserved", "photographer stamp", "proof copy"
]

# Words that must NEVER be treated as watermarks
SAFE_DECOR_KEYWORDS = [
    "happy birthday", "birthday", "welcome", "baby shower", "wedding",
    "haldi", "anniversary", "first", "one", "two", "balloon", "celebration"
]


class WatermarkDetector:
    def __init__(self, edge_density_threshold: float = 0.12, corner_margin_ratio: float = 0.15):
        self.edge_density_threshold = edge_density_threshold
        self.corner_margin_ratio = corner_margin_ratio

    def detect_watermark(self, 
                         image_bgr: np.ndarray, 
                         ocr_text: Optional[str] = None) -> Tuple[bool, float, str]:
        """
        Detects watermarks using computer vision cues and OCR text.
        Returns:
            watermark_detected (bool)
            confidence (float 0.0 - 1.0)
            reason (str)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, 0.0, "Empty image"

        # 1. OCR Keyword Check (Primary high-precision indicator)
        if ocr_text:
            normalized_text = ocr_text.lower()
            for kw in STRICT_WATERMARK_KEYWORDS:
                if kw in normalized_text:
                    # Verify it's not part of safe celebration text
                    is_safe = False
                    for safe in SAFE_DECOR_KEYWORDS:
                        if safe in normalized_text and kw == "stock":
                            is_safe = True
                            break
                    if not is_safe:
                        return True, 0.95, f"Detected watermark keyword: '{kw}'"

        # 2. Semi-Transparent Repeating Diagonal Grid Check (e.g. Shutterstock grid)
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=130, minLineLength=w // 5, maxLineGap=12)
            if lines is not None and len(lines) > 30:
                diagonal_count = 0
                for line in lines:
                    pt = line.reshape(-1)
                    if len(pt) < 4:
                        continue
                    x1, y1, x2, y2 = pt[:4]
                    angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
                    # Standard repetitive diagonals between 28 and 62 degrees
                    if 28 <= angle <= 62:
                        diagonal_count += 1
                if diagonal_count >= 16:
                    return True, 0.85, f"Detected {diagonal_count} repetitive diagonal watermark overlay lines"
        except Exception:
            pass

        # 3. Corner Logo Stamp Check (Requires both high edge density AND distinct high gradient contrast)
        cw = int(w * self.corner_margin_ratio)
        ch = int(h * self.corner_margin_ratio)

        # Check only extreme bottom corners where copyright badges appear
        bottom_right = edges[h - ch:h, w - cw:w]
        center = edges[ch:h - ch, cw:w - cw]
        center_density = float(np.mean(center > 0)) if center.size > 0 else 0.01

        if bottom_right.size > 0:
            corner_density = float(np.mean(bottom_right > 0))
            # Must be significantly denser than the center, exceed threshold, and not simply be solid background
            if corner_density > self.edge_density_threshold and corner_density > (center_density * 4.2 + 0.08):
                # Send to review with moderate confidence rather than hard rejecting
                return True, 0.65, "Suspected photographer watermark/logo in bottom-right corner"

        return False, 0.0, "No watermark detected"
