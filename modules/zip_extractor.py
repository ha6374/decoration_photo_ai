"""
Safe ZIP Archive Extractor & File Ingestion Module.
Guarantees data safety: NEVER modifies or deletes original ZIP or image files.
Handles nested directories, filename collisions, corrupted files, and format validation.
"""

import os
import zipfile
import shutil
import uuid
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class SafeZipExtractor:
    def __init__(self, extraction_base_dir: str):
        self.extraction_base_dir = os.path.abspath(extraction_base_dir)
        os.makedirs(self.extraction_base_dir, exist_ok=True)

    def extract_zip(self, zip_file_path: str, batch_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Safely extracts an uploaded ZIP file into an isolated batch directory.
        Returns:
            extracted_files: List of valid image file paths
            errors: List of error dictionaries for any unreadable/corrupted files
        """
        extracted_files: List[str] = []
        errors: List[Dict[str, str]] = []

        batch_dir = os.path.join(self.extraction_base_dir, batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        if not os.path.exists(zip_file_path):
            errors.append({"file": zip_file_path, "error": "Source ZIP file not found."})
            return extracted_files, errors

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                # Test archive integrity
                bad_file = zf.testzip()
                if bad_file:
                    logger.warning(f"Corrupted file detected inside zip: {bad_file}")
                    errors.append({"file": bad_file, "error": "Corrupted CRC check inside zip archive."})

                file_counter = 0
                for member in zf.infolist():
                    # Skip directories and macOS metadata
                    if member.is_dir() or member.filename.startswith("__MACOSX/") or os.path.basename(member.filename).startswith("._"):
                        continue

                    ext = os.path.splitext(member.filename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue

                    # Generate collision-safe target filename
                    raw_basename = os.path.basename(member.filename)
                    clean_name = "".join(c for c in raw_basename if c.isalnum() or c in "._- ")
                    if not clean_name:
                        clean_name = f"image_{uuid.uuid4().hex[:8]}"

                    name_part, ext_part = os.path.splitext(clean_name)
                    file_counter += 1
                    target_filename = f"{file_counter:04d}_{name_part}{ext_part}"
                    target_filepath = os.path.join(batch_dir, target_filename)

                    # Extract safely by reading stream to avoid path traversal
                    try:
                        with zf.open(member) as source, open(target_filepath, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_files.append(target_filepath)
                    except Exception as ex:
                        logger.error(f"Failed extracting {member.filename}: {ex}")
                        errors.append({"file": member.filename, "error": str(ex)})

        except zipfile.BadZipFile:
            errors.append({"file": zip_file_path, "error": "File is not a valid ZIP file or is damaged."})
        except Exception as e:
            errors.append({"file": zip_file_path, "error": f"Unexpected error reading ZIP: {str(e)}"})

        return extracted_files, errors

    def ingest_individual_files(self, file_paths: List[str], batch_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Safely copies individual uploaded images to the isolated batch folder.
        Preserves original files completely.
        """
        extracted_files: List[str] = []
        errors: List[Dict[str, str]] = []

        batch_dir = os.path.join(self.extraction_base_dir, batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        counter = 0
        for src_path in file_paths:
            if not os.path.exists(src_path):
                errors.append({"file": src_path, "error": "File not found."})
                continue

            ext = os.path.splitext(src_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append({"file": src_path, "error": f"Unsupported format '{ext}'."})
                continue

            counter += 1
            raw_basename = os.path.basename(src_path)
            clean_name = "".join(c for c in raw_basename if c.isalnum() or c in "._- ")
            name_part, ext_part = os.path.splitext(clean_name)
            target_filename = f"{counter:04d}_{name_part}{ext_part}"
            target_filepath = os.path.join(batch_dir, target_filename)

            try:
                shutil.copy2(src_path, target_filepath)
                extracted_files.append(target_filepath)
            except Exception as e:
                errors.append({"file": src_path, "error": str(e)})

        return extracted_files, errors
