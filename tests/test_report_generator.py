"""
Unit tests for ReportGenerator.
Verifies CSV and Excel exports contain all required columns and metadata fields.
"""

import os
import unittest
import tempfile
import shutil
import pandas as pd
from modules.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.report_gen = ReportGenerator(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_csv_and_excel(self):
        sample_records = [
            {
                "id": 1,
                "original_filename": "stage_front.jpg",
                "category": "Wedding Decoration",
                "status": "APPROVED",
                "rejection_reasons": ["All mandatory filters passed successfully."],
                "file_format": "JPG",
                "width": 1920,
                "height": 1080,
                "file_size_bytes": 204800,
                "sha256": "abcdef123456",
                "phash": "ffff0000ffff0000",
                "blur_score": 142.5,
                "human_detected": False,
                "human_count": 0,
                "watermark_detected": False,
                "watermark_confidence": 0.0,
                "detected_angle": "FRONT",
                "angle_confidence": 0.92,
                "detected_text": "Wedding Reception",
                "detected_numbers": [],
                "duplicate_of_id": None,
                "duplicate_similarity": 0.0,
                "upload_batch_id": "batch_test_01",
                "source_path": "/tmp/stage_front.jpg",
                "output_path": "/output/Wedding/Approved/stage_front_abc.jpg",
                "first_seen_at": "2026-11-01T20:00:00"
            }
        ]

        # 1. Export CSV
        csv_path, csv_bytes = self.report_gen.export_csv(sample_records, "test_report")
        self.assertTrue(os.path.exists(csv_path))
        self.assertGreater(len(csv_bytes), 0)

        df_csv = pd.read_csv(csv_path)
        self.assertEqual(len(df_csv), 1)
        self.assertIn("Original Filename", df_csv.columns)
        self.assertIn("Final Status", df_csv.columns)
        self.assertIn("SHA-256 Hash", df_csv.columns)
        self.assertIn("Perceptual Hash (pHash)", df_csv.columns)
        self.assertIn("Blur Score (Sharpness)", df_csv.columns)
        self.assertIn("Human Detected", df_csv.columns)
        self.assertIn("Watermark Detected", df_csv.columns)
        self.assertIn("OCR Detected Text", df_csv.columns)
        self.assertIn("Detected Angle", df_csv.columns)

        # 2. Export Excel
        excel_path, excel_bytes = self.report_gen.export_excel(sample_records, "test_report")
        self.assertTrue(os.path.exists(excel_path))
        self.assertGreater(len(excel_bytes), 0)

        df_excel = pd.read_excel(excel_path)
        self.assertEqual(len(df_excel), 1)
        self.assertEqual(df_excel["Original Filename"].iloc[0], "stage_front.jpg")
        self.assertEqual(df_excel["Final Status"].iloc[0], "APPROVED")


if __name__ == "__main__":
    unittest.main()
