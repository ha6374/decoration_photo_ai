"""
Unit tests for DatabaseManager and SQLite persistence.
Verifies category management, duplicate lookups, and state persistence across DB restarts.
"""

import os
import unittest
import tempfile
import shutil
from database.db import DatabaseManager
from database.models import CategoryRule, ImageMetadata, UploadBatch


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_decoration.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_categories_seeded(self):
        cats = self.db.list_categories()
        cat_names = [c.name for c in cats]
        self.assertIn("First Birthday Decoration", cat_names)
        self.assertIn("Second Birthday Decoration", cat_names)
        self.assertIn("Wedding Decoration", cat_names)

    def test_category_crud(self):
        custom = CategoryRule(
            name="Graduation Ceremony",
            description="Decorations for graduation parties",
            required_text=["Class of 2026", "Congrats"],
            required_numbers=["2026"],
            blur_threshold=120.0
        )
        self.assertTrue(self.db.save_category(custom))

        fetched = self.db.get_category("Graduation Ceremony")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.required_text, ["Class of 2026", "Congrats"])
        self.assertEqual(fetched.blur_threshold, 120.0)

        # Delete category
        self.assertTrue(self.db.delete_category("Graduation Ceremony"))
        self.assertIsNone(self.db.get_category("Graduation Ceremony"))

    def test_image_insertion_and_duplicate_queries(self):
        meta = ImageMetadata(
            original_filename="photo_stage.jpg",
            file_size_bytes=102400,
            width=1920,
            height=1080,
            file_format="JPG",
            sha256="abc123456789deadbeef",
            phash="ffff0000ffff0000",
            source_path="/tmp/photo_stage.jpg",
            category="Wedding Decoration",
            upload_batch_id="batch_001",
            status="APPROVED",
            blur_score=150.0
        )
        img_id = self.db.insert_image(meta)
        self.assertIsNotNone(img_id)

        # Exact SHA256 query
        dup = self.db.find_exact_duplicate_by_sha256("abc123456789deadbeef")
        self.assertIsNotNone(dup)
        self.assertEqual(dup["original_filename"], "photo_stage.jpg")

        # Perceptual hash query with identical phash
        p_dups = self.db.find_perceptual_duplicates("ffff0000ffff0000", max_hamming_distance=4)
        self.assertEqual(len(p_dups), 1)
        self.assertEqual(p_dups[0][1], 0)  # Distance 0

    def test_persistence_across_app_restart(self):
        # Insert image
        meta = ImageMetadata(
            original_filename="arch.jpg",
            file_size_bytes=50000,
            width=1200,
            height=800,
            file_format="JPG",
            sha256="restart_test_hash_999",
            phash="1111222233334444",
            source_path="/tmp/arch.jpg",
            category="First Birthday Decoration",
            upload_batch_id="batch_restart",
            status="APPROVED"
        )
        self.db.insert_image(meta)

        # Simulate application restart: create a new DatabaseManager instance pointing to same file
        restarted_db = DatabaseManager(self.db_path)
        dup = restarted_db.find_exact_duplicate_by_sha256("restart_test_hash_999")
        self.assertIsNotNone(dup)
        self.assertEqual(dup["original_filename"], "arch.jpg")
        self.assertEqual(dup["status"], "APPROVED")


if __name__ == "__main__":
    unittest.main()
