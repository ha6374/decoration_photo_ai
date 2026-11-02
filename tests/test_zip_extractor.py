"""
Unit tests for SafeZipExtractor.
Validates extraction of nested folders, collision prevention, uncorrupted extraction,
and strict data safety invariants (originals are never touched).
"""

import os
import zipfile
import unittest
import tempfile
import shutil
from PIL import Image
from modules.zip_extractor import SafeZipExtractor


class TestSafeZipExtractor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.extract_base = os.path.join(self.test_dir, "extracted")
        self.extractor = SafeZipExtractor(self.extract_base)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_image(self, path: str, color=(255, 0, 0), size=(100, 100)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = Image.new("RGB", size, color=color)
        img.save(path)

    def test_safe_zip_extraction_with_nested_folders_and_duplicate_names(self):
        # Create a test zip file with nested folders and duplicate filenames
        zip_path = os.path.join(self.test_dir, "sample_batch.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Root level image
            temp_img1 = os.path.join(self.test_dir, "tmp1.jpg")
            self._create_sample_image(temp_img1, color=(255, 0, 0))
            zf.write(temp_img1, arcname="photo.jpg")

            # Nested level 1 with IDENTICAL filename 'photo.jpg'
            temp_img2 = os.path.join(self.test_dir, "tmp2.jpg")
            self._create_sample_image(temp_img2, color=(0, 255, 0))
            zf.write(temp_img2, arcname="subfolder_A/photo.jpg")

            # Deep nested level 2 with IDENTICAL filename 'photo.jpg'
            temp_img3 = os.path.join(self.test_dir, "tmp3.jpg")
            self._create_sample_image(temp_img3, color=(0, 0, 255))
            zf.write(temp_img3, arcname="subfolder_A/nested_B/photo.jpg")

            # Non-image file to test filtering
            temp_txt = os.path.join(self.test_dir, "notes.txt")
            with open(temp_txt, "w") as f:
                f.write("Some decoration notes")
            zf.write(temp_txt, arcname="notes.txt")

        # Snapshot of original zip metadata before extraction
        orig_zip_size = os.path.getsize(zip_path)
        orig_zip_mtime = os.path.getmtime(zip_path)

        # Extract
        batch_id = "test_batch_001"
        extracted_files, errors = self.extractor.extract_zip(zip_path, batch_id)

        # All 3 image files extracted
        self.assertEqual(len(extracted_files), 3)
        self.assertEqual(len(errors), 0)

        # Verify all extracted files exist and have unique paths
        self.assertEqual(len(set(extracted_files)), 3)
        for f in extracted_files:
            self.assertTrue(os.path.exists(f))
            self.assertTrue(f.endswith(".jpg"))

        # DATA SAFETY AUDIT: Ensure original ZIP was never modified or deleted
        self.assertTrue(os.path.exists(zip_path))
        self.assertEqual(os.path.getsize(zip_path), orig_zip_size)
        self.assertEqual(os.path.getmtime(zip_path), orig_zip_mtime)

    def test_ingest_individual_files(self):
        # Create 2 source images
        src1 = os.path.join(self.test_dir, "individual_1.png")
        src2 = os.path.join(self.test_dir, "individual_2.png")
        self._create_sample_image(src1, color=(100, 100, 100))
        self._create_sample_image(src2, color=(200, 200, 200))

        src1_mtime = os.path.getmtime(src1)
        src2_mtime = os.path.getmtime(src2)

        batch_id = "batch_individual"
        extracted_files, errors = self.extractor.ingest_individual_files([src1, src2], batch_id)

        self.assertEqual(len(extracted_files), 2)
        # Originals still exist untouched
        self.assertTrue(os.path.exists(src1))
        self.assertTrue(os.path.exists(src2))
        self.assertEqual(os.path.getmtime(src1), src1_mtime)
        self.assertEqual(os.path.getmtime(src2), src2_mtime)


if __name__ == "__main__":
    unittest.main()
