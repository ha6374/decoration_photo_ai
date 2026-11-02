"""
Safe Image Loader & Metadata Extraction Module.
Loads images with Pillow and OpenCV, respects EXIF orientation, computes SHA-256 and pHash,
and provides thumbnail generators to avoid out-of-memory errors on large datasets.
"""

import os
import hashlib
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageOps
import imagehash
import cv2
import numpy as np


class ImageLoader:
    @staticmethod
    def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
        """Computes exact SHA-256 cryptographic hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_phash(pil_image: Image.Image) -> str:
        """Computes perceptual hash (pHash) for visual similarity matching."""
        try:
            h = imagehash.phash(pil_image)
            return str(h)
        except Exception:
            return ""

    @staticmethod
    def load_pil_image(filepath: str) -> Optional[Image.Image]:
        """
        Safely loads an image as a Pillow Image object.
        Auto-orients EXIF metadata and converts to RGB.
        Returns None if corrupted or unreadable.
        """
        try:
            with Image.open(filepath) as img:
                # Transpose according to EXIF orientation tag
                oriented_img = ImageOps.exif_transpose(img)
                # Convert to RGB to ensure uniform 3-channel layout
                rgb_img = oriented_img.convert("RGB")
                return rgb_img
        except Exception:
            return None

    @staticmethod
    def load_cv2_image(filepath: str) -> Optional[np.ndarray]:
        """
        Safely loads an image as an OpenCV BGR numpy array.
        Handles unicode Windows paths properly using numpy fromfile.
        """
        try:
            with open(filepath, "rb") as f:
                bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                return img
        except Exception:
            return None

    @classmethod
    def inspect_image_file(cls, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Extracts structural image attributes without keeping full image in memory.
        """
        if not os.path.exists(filepath):
            return None

        file_size = os.path.getsize(filepath)
        sha256 = cls.compute_sha256(filepath)

        pil_img = cls.load_pil_image(filepath)
        if pil_img is None:
            return None

        width, height = pil_img.size
        ext = os.path.splitext(filepath)[1].lower().replace(".", "").upper()
        phash = cls.compute_phash(pil_img)

        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "file_format": ext or "UNKNOWN",
            "sha256": sha256,
            "phash": phash,
        }

    @staticmethod
    def get_thumbnail(filepath: str, max_size: Tuple[int, int] = (300, 300)) -> Optional[Image.Image]:
        """
        Generates a lightweight thumbnail for UI galleries without loading full resolution.
        """
        try:
            with Image.open(filepath) as img:
                oriented_img = ImageOps.exif_transpose(img)
                rgb_img = oriented_img.convert("RGB")
                rgb_img.thumbnail(max_size, Image.Resampling.LANCZOS)
                return rgb_img
        except Exception:
            return None
