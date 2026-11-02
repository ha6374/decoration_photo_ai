"""
Unit tests for Computer Vision & AI Filtering Modules.
Validates blur detection, watermark analysis, camera angle classification,
and OCR text & number detection for celebration themes.
"""

import os
import unittest
import tempfile
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.blur_detector import BlurDetector
from modules.watermark_detector import WatermarkDetector
from modules.angle_detector import AngleDetector
from modules.text_detector import TextAndNumberDetector


class TestFilterModules(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_blur_detector_sharp_vs_blurry(self):
        detector = BlurDetector(default_threshold=100.0)

        # 1. Generate sharp high-contrast texture (grid of lines and dots)
        sharp_img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(0, 400, 10):
            cv2.line(sharp_img, (i, 0), (i, 400), (255, 255, 255), 2)
            cv2.line(sharp_img, (0, i), (400, i), (255, 255, 255), 2)

        is_blurry_sharp, sharp_score, _ = detector.evaluate(sharp_img, threshold=100.0)
        self.assertFalse(is_blurry_sharp)
        self.assertGreater(sharp_score, 100.0)

        # 2. Heavily blurred version
        blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)
        is_blurry_blur, blur_score, _ = detector.evaluate(blurry_img, threshold=100.0)
        self.assertTrue(is_blurry_blur)
        self.assertLess(blur_score, 100.0)

    def test_watermark_detection_via_keyword_and_corner_density(self):
        detector = WatermarkDetector()

        # Keyword test
        has_wm, conf, reason = detector.detect_watermark(
            image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            ocr_text="Exclusive Photo (c) Shutterstock images preview"
        )
        self.assertTrue(has_wm)
        self.assertGreaterEqual(conf, 0.90)

        # Corner density test: clean plain backdrop vs corner dense stamp
        plain_canvas = np.full((500, 500, 3), 200, dtype=np.uint8)
        has_wm_plain, _, _ = detector.detect_watermark(plain_canvas, ocr_text="")
        self.assertFalse(has_wm_plain)

    def test_angle_detector_front_vs_side(self):
        detector = AngleDetector()

        # Front-facing backdrop: perfectly symmetrical stage frame
        front_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Left pillar and right pillar equidistant from center
        cv2.rectangle(front_canvas, (100, 80), (140, 400), (200, 200, 200), -1)
        cv2.rectangle(front_canvas, (500, 80), (540, 400), (200, 200, 200), -1)
        # Arch top connecting them horizontally
        cv2.rectangle(front_canvas, (100, 80), (540, 120), (200, 200, 200), -1)
        # Horizontal stage line
        cv2.line(front_canvas, (50, 400), (590, 400), (255, 255, 255), 3)

        angle_class, conf, details = detector.evaluate_angle(front_canvas)
        self.assertEqual(angle_class, "FRONT")
        self.assertGreaterEqual(conf, 0.60)

    def test_text_and_number_detector_first_birthday(self):
        detector = TextAndNumberDetector()

        # Test text matching
        sample_ocr_output = "Welcome To Our Party Happy Birthday Little One"
        matched, missing = detector.matches_required_text(sample_ocr_output, ["Happy Birthday"])
        self.assertTrue(matched)
        self.assertEqual(len(missing), 0)

        # Test number matching for digit '1' and word 'ONE'
        nums = detector.parse_numbers(sample_ocr_output)
        self.assertIn("1", nums)  # 'one' parsed to canonical '1'

        matched_num, missing_num = detector.matches_required_numbers(
            detected_numbers=nums,
            required_numbers=["1", "ONE"],
            full_text=sample_ocr_output
        )
        self.assertTrue(matched_num)
        self.assertEqual(len(missing_num), 0)

        # Negative test: missing 'Happy Birthday'
        unrelated_text = "Wedding Reception of John and Jane"
        matched_neg, missing_neg = detector.matches_required_text(unrelated_text, ["Happy Birthday"])
        self.assertFalse(matched_neg)
        self.assertIn("Happy Birthday", missing_neg)


if __name__ == "__main__":
    unittest.main()
