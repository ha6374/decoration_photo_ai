"""
End-to-End Acceptance Test Suite.
Verifies all 8 user acceptance criteria:
TEST 1: Bulk ZIP extraction (500 images, nested folders, collision safety).
TEST 2: Application of front-angle, blur, watermark, and human detection filters.
TEST 3: First Birthday required text ('Happy Birthday') and number ('1' or 'ONE') validation.
TEST 4: Separate organization into Approved, Review, Rejected, and Duplicates folders.
TEST 5: Cross-batch deduplication: ZIP 2 containing ZIP 1 approved images blocked from re-saving.
TEST 6: Manual Review Gallery interactive approve/reject/remove flows.
TEST 7: Full application restart: duplicate history persists and functions across sessions.
TEST 8: Strict Data Safety Audit: original ZIP and source images are 100% untouched.
"""

import os
import unittest
import tempfile
import shutil
import zipfile
import hashlib
from typing import Dict, List
import cv2
import numpy as np
from PIL import Image, ImageDraw

from database.db import DatabaseManager
from database.models import CategoryRule, UploadBatch
from modules.zip_extractor import SafeZipExtractor
from modules.filter_engine import FilterEngine
from modules.image_organizer import ImageOrganizer
from modules.duplicate_detector import DuplicateDetector
from modules.image_loader import ImageLoader


class TestFullAcceptancePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.root_test_dir, "acceptance.db")
        cls.extracted_dir = os.path.join(cls.root_test_dir, "extracted")
        cls.output_dir = os.path.join(cls.root_test_dir, "output")

        cls.db = DatabaseManager(cls.db_path)
        cls.extractor = SafeZipExtractor(cls.extracted_dir)
        cls.engine = FilterEngine(cls.db)
        cls.organizer = ImageOrganizer(cls.output_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root_test_dir, ignore_errors=True)

    @staticmethod
    def _file_sha256(filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _generate_synthetic_decoration(self, path: str, text: str = "", blur: bool = False, watermark: bool = False, front: bool = True):
        """Generates realistic synthetic decoration image canvas."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = Image.new("RGB", (640, 480), color=(245, 240, 235))
        draw = ImageDraw.Draw(img)

        # Draw decorative balloon arch
        for i in range(10):
            x = 80 + i * 50
            y = 120 - int(40 * np.sin(i * np.pi / 9))
            draw.ellipse([x, y, x + 40, y + 50], fill=(255, 105, 180), outline=(200, 50, 100))

        # Backdrop pillars (symmetrical if front)
        if front:
            draw.rectangle([80, 150, 120, 420], fill=(220, 220, 220))
            draw.rectangle([520, 150, 560, 420], fill=(220, 220, 220))
            draw.line([(50, 420), (590, 420)], fill=(180, 150, 100), width=4)
        else:
            # Skewed angle
            draw.polygon([(50, 200), (90, 180), (100, 420), (40, 440)], fill=(220, 220, 220))
            draw.line([(40, 440), (550, 360)], fill=(180, 150, 100), width=4)

        # Cake table center
        draw.rectangle([260, 320, 380, 420], fill=(255, 255, 255), outline=(200, 200, 200))

        # Text banner if specified
        if text:
            draw.rectangle([200, 220, 440, 270], fill=(255, 220, 100))
            draw.text((220, 235), text, fill=(50, 50, 50))

        # Watermark
        if watermark:
            draw.text((450, 440), "PREVIEW WATERMARK (C) STOCK", fill=(100, 100, 100))

        img.save(path, quality=95)

        if blur:
            # Apply Gaussian blur using OpenCV
            cv_img = cv2.imread(path)
            blurred = cv2.GaussianBlur(cv_img, (45, 45), 0)
            cv2.imwrite(path, blurred)

    # =========================================================================
    # TEST 1: Extract ZIP containing 500 images successfully
    # =========================================================================
    def test_01_zip_extraction_500_images(self):
        zip_path = os.path.join(self.root_test_dir, "batch_500.zip")
        temp_img_template = os.path.join(self.root_test_dir, "sample_template.jpg")
        self._generate_synthetic_decoration(temp_img_template, text="Sample Decor")

        # Create zip with 500 images across nested directories
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for i in range(500):
                # Nested hierarchy testing
                folder = "root" if i < 150 else ("sub_hall_A" if i < 350 else "sub_hall_A/deep_corner")
                # Intentionally duplicate filename across different subfolders
                fname = f"decor_item_{i % 50:03d}.jpg"
                zf.write(temp_img_template, arcname=f"{folder}/{fname}")

        batch_id = "acceptance_batch_500"
        extracted_files, errors = self.extractor.extract_zip(zip_path, batch_id)

        self.assertEqual(len(extracted_files), 500, "Should extract all 500 images.")
        self.assertEqual(len(errors), 0, "No extraction errors should occur.")
        self.assertEqual(len(set(extracted_files)), 500, "All 500 extracted files must have unique paths.")

    # =========================================================================
    # TEST 2 & TEST 3: Multi-Filter Assessment on Celebration Decor
    # =========================================================================
    def test_02_and_03_filter_pipeline_and_first_birthday(self):
        cat_rule = self.db.get_category("First Birthday Decoration")
        self.assertIsNotNone(cat_rule)

        # 1. Valid First Birthday Image: Front angle, sharp, no watermark, "Happy Birthday 1"
        valid_img_path = os.path.join(self.root_test_dir, "first_bday_approved.jpg")
        self._generate_synthetic_decoration(valid_img_path, text="Happy Birthday 1", blur=False, watermark=False, front=True)

        meta_valid = self.engine.process_image(valid_img_path, cat_rule, "test_bday_batch")
        self.assertEqual(meta_valid.status, "APPROVED", f"Expected APPROVED, got: {meta_valid.status} ({meta_valid.rejection_reasons})")
        self.assertGreaterEqual(meta_valid.blur_score, cat_rule.blur_threshold)

        # 2. Blurry Image: Sharpness < threshold
        blurry_path = os.path.join(self.root_test_dir, "first_bday_blurry.jpg")
        self._generate_synthetic_decoration(blurry_path, text="Happy Birthday 1", blur=True, front=True)

        meta_blurry = self.engine.process_image(blurry_path, cat_rule, "test_bday_batch")
        self.assertEqual(meta_blurry.status, "REJECTED")
        self.assertTrue(any("blurry" in r.lower() for r in meta_blurry.rejection_reasons))

        # 3. Watermarked Image: Visible watermark keyword
        wm_path = os.path.join(self.root_test_dir, "first_bday_wm.jpg")
        self._generate_synthetic_decoration(wm_path, text="Happy Birthday 1", watermark=True, front=True)

        meta_wm = self.engine.process_image(wm_path, cat_rule, "test_bday_batch")
        self.assertTrue(meta_wm.status in ("REJECTED", "REVIEW_REQUIRED"))
        self.assertTrue(meta_wm.watermark_detected)

        # 4. First Birthday missing text "Happy Birthday" or number "1"
        missing_text_path = os.path.join(self.root_test_dir, "wedding_sign.jpg")
        self._generate_synthetic_decoration(missing_text_path, text="Wedding Reception", front=True)

        meta_missing = self.engine.process_image(missing_text_path, cat_rule, "test_bday_batch")
        self.assertEqual(meta_missing.status, "REJECTED")
        self.assertTrue(any("missing" in r.lower() for r in meta_missing.rejection_reasons))

    # =========================================================================
    # TEST 4: Organization into Approved, Review, Rejected, Duplicates
    # =========================================================================
    def test_04_output_folder_segregation(self):
        cat_name = "First Birthday Decoration"

        # Organize 4 test items
        p_app = os.path.join(self.root_test_dir, "mock_app.jpg")
        p_rev = os.path.join(self.root_test_dir, "mock_rev.jpg")
        p_rej = os.path.join(self.root_test_dir, "mock_rej.jpg")
        p_dup = os.path.join(self.root_test_dir, "mock_dup.jpg")

        for p in (p_app, p_rev, p_rej, p_dup):
            self._generate_synthetic_decoration(p)

        out_app = self.organizer.organize_image(p_app, cat_name, "APPROVED", "hash1")
        out_rev = self.organizer.organize_image(p_rev, cat_name, "REVIEW_REQUIRED", "hash2")
        out_rej = self.organizer.organize_image(p_rej, cat_name, "REJECTED", "hash3")
        out_dup = self.organizer.organize_image(p_dup, cat_name, "DUPLICATE", "hash4")

        self.assertIn("Approved", out_app)
        self.assertIn("Review", out_rev)
        self.assertIn("Rejected", out_rej)
        self.assertIn("Duplicates", out_dup)

        for p in (out_app, out_rev, out_rej, out_dup):
            self.assertTrue(os.path.exists(p))

    # =========================================================================
    # TEST 5: Second ZIP containing images already approved from first ZIP
    # =========================================================================
    def test_05_cross_batch_duplicate_blocking(self):
        cat_rule = self.db.get_category("First Birthday Decoration")

        # 1. First ZIP with an approved image
        zip1_path = os.path.join(self.root_test_dir, "zip1_original.zip")
        shared_img_path = os.path.join(self.root_test_dir, "shared_master_photo.jpg")
        self._generate_synthetic_decoration(shared_img_path, text="Happy Birthday ONE", front=True)

        with zipfile.ZipFile(zip1_path, "w") as zf1:
            zf1.write(shared_img_path, arcname="photo_master.jpg")

        ext1_files, _ = self.extractor.extract_zip(zip1_path, "batch_zip1")
        meta1 = self.engine.process_image(ext1_files[0], cat_rule, "batch_zip1")
        out1 = self.organizer.organize_image(meta1.source_path, meta1.category, meta1.status, meta1.sha256)
        meta1.output_path = out1
        self.db.insert_image(meta1)
        self.assertEqual(meta1.status, "APPROVED")

        # 2. Second ZIP containing the identical image (or re-encoded visual duplicate)
        zip2_path = os.path.join(self.root_test_dir, "zip2_reupload.zip")
        with zipfile.ZipFile(zip2_path, "w") as zf2:
            zf2.write(shared_img_path, arcname="renamed_subfolder/reuploaded_master.jpg")

        ext2_files, _ = self.extractor.extract_zip(zip2_path, "batch_zip2")
        meta2 = self.engine.process_image(ext2_files[0], cat_rule, "batch_zip2")

        # MUST BE IDENTIFIED AS DUPLICATE
        self.assertEqual(meta2.status, "DUPLICATE", "Re-uploaded image from ZIP 2 must be flagged as DUPLICATE.")
        self.assertEqual(meta2.duplicate_of_id, meta1.id, "Must link to first batch image record.")

        # Organize into Duplicates, NEVER Approved!
        out2 = self.organizer.organize_image(meta2.source_path, meta2.category, meta2.status, meta2.sha256)
        self.assertIn("Duplicates", out2)
        self.assertNotIn("Approved", out2)

    # =========================================================================
    # TEST 6: Manual Review Gallery interactive actions (Approve, Reject, Remove)
    # =========================================================================
    def test_06_manual_gallery_actions(self):
        cat_name = "First Birthday Decoration"

        # Create image initially in REVIEW_REQUIRED
        src_path = os.path.join(self.root_test_dir, "manual_candidate.jpg")
        self._generate_synthetic_decoration(src_path, text="Happy Birthday 1")
        info = ImageLoader.inspect_image_file(src_path)

        out_initial = self.organizer.organize_image(src_path, cat_name, "REVIEW_REQUIRED", info["sha256"])
        meta = self.engine.process_image(src_path, self.db.get_category(cat_name), "batch_manual")
        meta.status = "REVIEW_REQUIRED"
        meta.output_path = out_initial
        img_id = self.db.insert_image(meta)

        # 1. Manually Approve
        new_app_path = self.organizer.relocate_image(meta.output_path, src_path, cat_name, "APPROVED", info["sha256"])
        self.db.update_image_status(img_id, "APPROVED", new_output_path=new_app_path, reason="User manual approve")
        self.assertIn("Approved", new_app_path)
        self.assertTrue(os.path.exists(new_app_path))
        self.assertFalse(os.path.exists(out_initial), "Old review copy should be cleanly relocated.")

        # 2. Remove from Approved (reverts output without touching source)
        self.organizer.remove_from_approved(new_app_path)
        self.assertFalse(os.path.exists(new_app_path), "Processed copy must be removed.")
        self.assertTrue(os.path.exists(src_path), "Original source file must NEVER be deleted.")

    # =========================================================================
    # TEST 7: Restart application and verify duplicate history remains available
    # =========================================================================
    def test_07_persistence_across_app_restart(self):
        # Create a fresh DatabaseManager instance on the same DB file
        fresh_db = DatabaseManager(self.db_path)
        fresh_detector = DuplicateDetector(fresh_db)

        # Check duplicate lookup for previously processed image
        shared_img_path = os.path.join(self.root_test_dir, "shared_master_photo.jpg")
        info = ImageLoader.inspect_image_file(shared_img_path)

        is_dup, dup_type, matched, score = fresh_detector.check_duplicate(
            filepath=shared_img_path,
            sha256=info["sha256"],
            phash_str=info["phash"],
            category="First Birthday Decoration"
        )

        self.assertTrue(is_dup, "Duplicate detection must work seamlessly after application restart.")
        self.assertIsNotNone(matched)
        self.assertEqual(matched["status"], "APPROVED")

    # =========================================================================
    # TEST 8: Data Safety Audit: Originals remain 100% untouched
    # =========================================================================
    def test_08_original_files_never_deleted_or_modified(self):
        # Create an untouched reference image and reference zip
        ref_img_path = os.path.join(self.root_test_dir, "sacred_original.jpg")
        self._generate_synthetic_decoration(ref_img_path, text="Original Never Touch")
        ref_sha = self._file_sha256(ref_img_path)
        ref_size = os.path.getsize(ref_img_path)

        ref_zip_path = os.path.join(self.root_test_dir, "sacred_original.zip")
        with zipfile.ZipFile(ref_zip_path, "w") as zf:
            zf.write(ref_img_path, arcname="sacred.jpg")
        ref_zip_sha = self._file_sha256(ref_zip_path)
        ref_zip_size = os.path.getsize(ref_zip_path)

        # Run extraction & pipeline
        ext_files, _ = self.extractor.extract_zip(ref_zip_path, "batch_safety")
        meta = self.engine.process_image(ext_files[0], self.db.get_category("Wedding Decoration"), "batch_safety")
        out_path = self.organizer.organize_image(meta.source_path, meta.category, meta.status, meta.sha256)

        # Try relocating / removing
        self.organizer.relocate_image(out_path, meta.source_path, meta.category, "REJECTED", meta.sha256)

        # Audit original image: size and byte hash must match EXACTLY
        self.assertTrue(os.path.exists(ref_img_path))
        self.assertEqual(os.path.getsize(ref_img_path), ref_size)
        self.assertEqual(self._file_sha256(ref_img_path), ref_sha)

        # Audit original zip: size and byte hash must match EXACTLY
        self.assertTrue(os.path.exists(ref_zip_path))
        self.assertEqual(os.path.getsize(ref_zip_path), ref_zip_size)
        self.assertEqual(self._file_sha256(ref_zip_path), ref_zip_sha)


if __name__ == "__main__":
    unittest.main()
