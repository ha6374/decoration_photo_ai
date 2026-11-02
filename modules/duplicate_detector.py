"""
Multi-Level Global Duplicate Detection Engine.
Level 1: SHA-256 Exact Hash matching (bit-for-bit identical files).
Level 2: Perceptual Hash (pHash) Hamming distance (compressed, resized, or re-encoded files).
Level 3: Feature / Visual Similarity scoring (crops, color grading, minor modifications).
Checks across all historical batches via persistent SQLite database.
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import imagehash
from PIL import Image

from database.db import DatabaseManager
from modules.image_loader import ImageLoader


class DuplicateDetector:
    def __init__(self, db: DatabaseManager, phash_threshold: int = 6, similarity_threshold: float = 0.88, global_scope: bool = True):
        self.db = db
        self.phash_threshold = phash_threshold
        self.similarity_threshold = similarity_threshold
        self.global_scope = global_scope

    def check_duplicate(self, 
                        filepath: str, 
                        sha256: str, 
                        phash_str: str, 
                        category: str,
                        pil_image: Optional[Image.Image] = None) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], float]:
        """
        Runs multi-level duplicate verification against the global database.
        Returns:
            is_duplicate (bool)
            duplicate_type (str: 'EXACT_SHA256', 'PERCEPTUAL_PHASH', 'SIMILARITY', or None)
            matched_record (Dict or None)
            similarity_score (float 0.0 - 1.0)
        """
        # --- Level 1: SHA-256 Exact Hash ---
        exact_match = self.db.find_exact_duplicate_by_sha256(
            sha256_hash=sha256, 
            global_scope=self.global_scope, 
            category=category
        )
        if exact_match:
            return True, "EXACT_SHA256", exact_match, 1.0

        # --- Level 2: Perceptual Hash (pHash) ---
        if phash_str:
            p_matches = self.db.find_perceptual_duplicates(
                target_phash=phash_str, 
                max_hamming_distance=self.phash_threshold,
                global_scope=self.global_scope,
                category=category
            )
            if p_matches:
                best_match, distance = p_matches[0]
                # Normalized similarity: 64-bit phash -> 1 - (dist / 64)
                similarity = max(0.0, 1.0 - (distance / 64.0))
                return True, "PERCEPTUAL_PHASH", best_match, round(similarity, 4)

        # --- Level 3: Visual & Color Histogram Similarity (Uncertain matches) ---
        # When pHash distance is slightly beyond threshold (e.g. 7-10), check histogram similarity
        if phash_str:
            loose_matches = self.db.find_perceptual_duplicates(
                target_phash=phash_str,
                max_hamming_distance=12,
                global_scope=self.global_scope,
                category=category
            )
            if loose_matches and filepath:
                target_cv = ImageLoader.load_cv2_image(filepath)
                if target_cv is not None:
                    target_hist = self._compute_color_hist(target_cv)
                    for candidate_rec, dist in loose_matches:
                        cand_path = candidate_rec.get("output_path") or candidate_rec.get("source_path")
                        if cand_path:
                            cand_cv = ImageLoader.load_cv2_image(cand_path)
                            if cand_cv is not None:
                                cand_hist = self._compute_color_hist(cand_cv)
                                sim = cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_CORREL)
                                if sim >= self.similarity_threshold:
                                    return True, "SIMILARITY", candidate_rec, round(float(sim), 4)

        return False, None, None, 0.0

    @staticmethod
    def _compute_color_hist(image_bgr: np.ndarray) -> np.ndarray:
        """Computes normalized HSV color histogram for visual similarity."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
