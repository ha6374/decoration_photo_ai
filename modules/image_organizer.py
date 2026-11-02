"""
File & Folder Organization Engine.
Structures outputs into category folders:
output/<Category>/Approved, Review, Rejected, Duplicates.
Prevents file overwriting via collision-proof hashes, never touches original source files,
and supports dynamic relocation during manual reviews.
"""

import os
import shutil
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ImageOrganizer:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = os.path.abspath(output_base_dir)
        os.makedirs(self.output_base_dir, exist_ok=True)

    @staticmethod
    def sanitize_category_name(category_name: str) -> str:
        """Converts human category name into safe folder name."""
        clean = re.sub(r"[^\w\s-]", "", category_name).strip()
        return re.sub(r"[-\s]+", "_", clean)

    def get_destination_folder(self, category_name: str, status: str) -> str:
        """
        Returns destination directory based on category and status.
        status: APPROVED -> Approved, REVIEW_REQUIRED -> Review, REJECTED -> Rejected, DUPLICATE -> Duplicates
        """
        cat_folder = self.sanitize_category_name(category_name)
        status_subfolders = {
            "APPROVED": "Approved",
            "REVIEW_REQUIRED": "Review",
            "REJECTED": "Rejected",
            "DUPLICATE": "Duplicates",
        }
        subfolder = status_subfolders.get(status, "Review")
        target_dir = os.path.join(self.output_base_dir, cat_folder, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def organize_image(self, source_path: str, category_name: str, status: str, sha256_hash: str) -> Optional[str]:
        """
        Safely copies processed image into appropriate output subfolder.
        Returns the output destination file path.
        """
        if not os.path.exists(source_path):
            logger.error(f"Cannot organize missing file: {source_path}")
            return None

        target_dir = self.get_destination_folder(category_name, status)
        basename = os.path.basename(source_path)
        name, ext = os.path.splitext(basename)

        # Unique, collision-proof filename
        short_hash = sha256_hash[:8] if sha256_hash else "copy"
        unique_name = f"{name}_{short_hash}{ext}"
        target_path = os.path.join(target_dir, unique_name)

        # In case target path already exists, append counter
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(target_dir, f"{name}_{short_hash}_{counter}{ext}")
            counter += 1

        try:
            # COPY, never move or delete source!
            shutil.copy2(source_path, target_path)
            return target_path
        except Exception as e:
            logger.error(f"Failed to copy file to {target_path}: {e}")
            return None

    def relocate_image(self, current_output_path: Optional[str], source_path: str, category_name: str, new_status: str, sha256_hash: str) -> Optional[str]:
        """
        Relocates an image when user updates status manually in Review Gallery.
        Deletes old output copy if it exists (NEVER touches source_path!).
        Copies source_path to new output location.
        """
        if current_output_path and os.path.exists(current_output_path):
            try:
                os.remove(current_output_path)
            except Exception as e:
                logger.warning(f"Failed removing old output copy {current_output_path}: {e}")

        if not os.path.exists(source_path):
            logger.error(f"Original source file not found at {source_path}")
            return None

        return self.organize_image(
            source_path=source_path,
            category_name=category_name,
            status=new_status,
            sha256_hash=sha256_hash
        )

    def remove_from_approved(self, current_output_path: Optional[str]) -> bool:
        """
        Safely removes processed copy from Approved folder.
        NEVER deletes original uploaded file!
        """
        if current_output_path and os.path.exists(current_output_path):
            try:
                os.remove(current_output_path)
                return True
            except Exception as e:
                logger.error(f"Error removing file {current_output_path}: {e}")
                return False
        return True
