"""
Unit tests for DuplicateDetector.
Verifies Level 1 (SHA-256), Level 2 (pHash), and cross-batch deduplication against SQLite.
"""

import os
import unittest
import tempfile
import shutil
from PIL import Image, ImageDraw

from database.db import DatabaseManager
from database.models import ImageMetadata
from modules.duplicate_detector import DuplicateDetector
from modules.image_loader import ImageLoader


class TestDuplicateDetector(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_dups.db")
        self.db = DatabaseManager(self.db_path)
        self.detector = DuplicateDetector(db=self.db, phash_threshold=6)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_image(self, path: str, color=(200, 100, 50), size=(300, 300)):
        img = Image.new("RGB", size, color=color)
        img.save(path)
        return path

    def test_level1_exact_sha256_duplicate(self):
        # Create image
        img_path = os.path.join(self.test_dir, "original.jpg")
        self._create_image(img_path)

        info = ImageLoader.inspect_image_file(img_path)
        # Seed record in DB
        meta = ImageMetadata(
            original_filename="original.jpg",
            file_size_bytes=info["file_size_bytes"],
            width=info["width"],
            height=info["height"],
            file_format="JPG",
            sha256=info["sha256"],
            phash=info["phash"],
            source_path=img_path,
            category="First Birthday Decoration",
            upload_batch_id="batch_1",
            status="APPROVED"
        )
        self.db.insert_image(meta)

        # Query identical file
        is_dup, dup_type, match_rec, score = self.detector.check_duplicate(
            filepath=img_path,
            sha256=info["sha256"],
            phash_str=info["phash"],
            category="First Birthday Decoration"
        )

        self.assertTrue(is_dup)
        self.assertEqual(dup_type, "EXACT_SHA256")
        self.assertEqual(score, 1.0)
        self.assertEqual(match_rec["original_filename"], "original.jpg")

    def test_level2_phash_resized_duplicate(self):
        # Create base image
        base_path = os.path.join(self.test_dir, "base_photo.jpg")
        self._create_image(base_path, color=(30, 150, 220), size=(800, 600))
        base_info = ImageLoader.inspect_image_file(base_path)

        # Store in DB
        meta = ImageMetadata(
            original_filename="base_photo.jpg",
            file_size_bytes=base_info["file_size_bytes"],
            width=base_info["width"],
            height=base_info["height"],
            file_format="JPG",
            sha256=base_info["sha256"],
            phash=base_info["phash"],
            source_path=base_path,
            category="Wedding Decoration",
            upload_batch_id="batch_wedding_1",
            status="APPROVED"
        )
        self.db.insert_image(meta)

        # Create resized copy (e.g. 400x300, JPEG compression 75)
        # Different SHA-256, but near-identical pHash!
        resized_path = os.path.join(self.test_dir, "resized_photo.jpg")
        with Image.open(base_path) as im:
            re_im = im.resize((400, 300))
            re_im.save(resized_path, quality=75)

        resized_info = ImageLoader.inspect_image_file(resized_path)
        self.assertNotEqual(base_info["sha256"], resized_info["sha256"])  # SHA-256 differs

        # Test deduplication
        is_dup, dup_type, match_rec, score = self.detector.check_duplicate(
            filepath=resized_path,
            sha256=resized_info["sha256"],
            phash_str=resized_info["phash"],
            category="Wedding Decoration"
        )

        self.assertTrue(is_dup)
        self.assertEqual(dup_type, "PERCEPTUAL_PHASH")
        self.assertGreaterEqual(score, 0.85)

    def test_non_duplicate_distinct_images(self):
        # Create image A with red circles and diagonal lines
        path_a = os.path.join(self.test_dir, "photo_a.jpg")
        img_a = Image.new("RGB", (300, 300), color=(255, 255, 255))
        draw_a = ImageDraw.Draw(img_a)
        draw_a.rectangle([20, 20, 150, 150], fill=(255, 0, 0))
        draw_a.line([(0, 0), (300, 300)], fill=(0, 0, 255), width=5)
        img_a.save(path_a)
        info_a = ImageLoader.inspect_image_file(path_a)

        meta = ImageMetadata(
            original_filename="photo_a.jpg",
            file_size_bytes=info_a["file_size_bytes"],
            width=info_a["width"],
            height=info_a["height"],
            file_format="JPG",
            sha256=info_a["sha256"],
            phash=info_a["phash"],
            source_path=path_a,
            category="Haldi Decoration",
            upload_batch_id="batch_haldi",
            status="APPROVED"
        )
        self.db.insert_image(meta)

        # Create totally different image B with green horizontal stripes and yellow ellipse
        path_b = os.path.join(self.test_dir, "photo_b.jpg")
        img_b = Image.new("RGB", (300, 300), color=(0, 0, 0))
        draw_b = ImageDraw.Draw(img_b)
        draw_b.ellipse([50, 50, 250, 250], fill=(255, 255, 0))
        for y in range(0, 300, 20):
            draw_b.line([(0, y), (300, y)], fill=(0, 255, 0), width=3)
        img_b.save(path_b)
        info_b = ImageLoader.inspect_image_file(path_b)

        is_dup, dup_type, match_rec, score = self.detector.check_duplicate(
            filepath=path_b,
            sha256=info_b["sha256"],
            phash_str=info_b["phash"],
            category="Haldi Decoration"
        )

        self.assertFalse(is_dup)
        self.assertIsNone(dup_type)


if __name__ == "__main__":
    unittest.main()
