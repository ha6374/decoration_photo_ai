"""
OCR Text and Number Detection Module.
Extracts visible text and numbers from decoration banners, signs, and backdrops.
Supports EasyOCR, PyTesseract, and text normalizers with number parsing (digits and words: 1, ONE, 2, TWO).
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)

# Map textual representations of numbers to canonical digits
NUMBER_WORD_MAP = {
    "zero": "0", "one": "1", "first": "1", "1st": "1",
    "two": "2", "second": "2", "2nd": "2",
    "three": "3", "third": "3", "3rd": "3",
    "four": "4", "fourth": "4", "4th": "4",
    "five": "5", "fifth": "5", "5th": "5",
    "six": "6", "sixth": "6", "6th": "6",
    "seven": "7", "seventh": "7", "7th": "7",
    "eight": "8", "eighth": "8", "8th": "8",
    "nine": "9", "ninth": "9", "9th": "9",
    "ten": "10", "tenth": "10", "10th": "10"
}


class TextAndNumberDetector:
    _easyocr_reader = None

    def __init__(self, backend: str = "easyocr"):
        self.backend = backend
        self._ensure_backend()

    @classmethod
    def get_easyocr_reader(cls):
        if cls._easyocr_reader is None:
            try:
                import easyocr
                # Initialize English reader; gpu=False by default for universal compatibility
                cls._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("EasyOCR Reader successfully loaded.")
            except Exception as e:
                logger.warning(f"Could not load EasyOCR: {e}")
                cls._easyocr_reader = None
        return cls._easyocr_reader

    def _ensure_backend(self):
        if self.backend == "easyocr":
            self.get_easyocr_reader()

    def extract_text(self, image_bgr: np.ndarray) -> Tuple[str, List[str], float]:
        """
        Runs OCR on image.
        Returns:
            full_text (str): Concatenated text
            detected_numbers (List[str]): List of detected numbers/digits
            confidence (float): Average detection confidence
        """
        if image_bgr is None or image_bgr.size == 0:
            return "", [], 0.0

        raw_strings: List[str] = []
        confidences: List[float] = []

        # 1. EasyOCR
        reader = self.get_easyocr_reader()
        if reader is not None:
            try:
                # Downsample large images for faster OCR inference
                h, w = image_bgr.shape[:2]
                if max(h, w) > 1600:
                    scale = 1600.0 / max(h, w)
                    proc_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
                else:
                    proc_img = image_bgr

                results = reader.readtext(proc_img)
                for bbox, text, conf in results:
                    cleaned = text.strip()
                    if cleaned and conf >= 0.2:
                        raw_strings.append(cleaned)
                        confidences.append(float(conf))
            except Exception as e:
                logger.warning(f"EasyOCR extraction failed: {e}")

        # 2. PyTesseract fallback if EasyOCR produced nothing
        if not raw_strings:
            try:
                import pytesseract
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                txt = pytesseract.image_to_string(rgb)
                if txt.strip():
                    raw_strings.append(txt.strip())
                    confidences.append(0.6)
            except Exception:
                pass

        full_text = " ".join(raw_strings)
        avg_conf = float(np.mean(confidences)) if confidences else (0.5 if full_text else 0.0)

        # Extract numbers (both numeric digits and written words)
        detected_numbers = self.parse_numbers(full_text)

        return full_text, detected_numbers, round(avg_conf, 2)

    @classmethod
    def parse_numbers(cls, text: str) -> List[str]:
        """
        Finds all numbers in the text, converting written words like 'ONE' or '1st' to canonical '1'.
        """
        tokens = re.findall(r"\b[\w'-]+\b", text.lower())
        detected = set()

        for token in tokens:
            # Check pure digits
            if token.isdigit():
                detected.add(token)
            # Check word map (e.g. 'one', 'first', '1st')
            elif token in NUMBER_WORD_MAP:
                detected.add(NUMBER_WORD_MAP[token])
            # Check combined patterns like '1st', '2nd'
            m = re.match(r"^(\d+)(st|nd|rd|th)$", token)
            if m:
                detected.add(m.group(1))

        return sorted(list(detected))

    @classmethod
    def matches_required_text(cls, extracted_text: str, required_keywords: List[str]) -> Tuple[bool, List[str]]:
        """
        Checks if required keywords exist inside the extracted text (case-insensitive).
        Returns:
            passed (bool)
            missing_keywords (List[str])
        """
        if not required_keywords:
            return True, []

        normalized_text = re.sub(r"[^\w\s]", " ", extracted_text.lower())
        missing = []

        for req in required_keywords:
            clean_req = re.sub(r"[^\w\s]", " ", req.lower()).strip()
            # Substring match or token match
            if clean_req not in normalized_text:
                missing.append(req)

        return (len(missing) == 0), missing

    @classmethod
    def matches_required_numbers(cls, detected_numbers: List[str], required_numbers: List[str], full_text: str = "") -> Tuple[bool, List[str]]:
        """
        Checks if any required number matches.
        Allows either numeric (e.g. '1') or word (e.g. 'ONE').
        """
        if not required_numbers:
            return True, []

        # Convert required numbers to their canonical digit representations
        canonical_required = set()
        for r in required_numbers:
            r_str = str(r).strip().lower()
            if r_str in NUMBER_WORD_MAP:
                canonical_required.add(NUMBER_WORD_MAP[r_str])
            elif r_str.isdigit():
                canonical_required.add(r_str)
            else:
                canonical_required.add(r_str)

        detected_set = set(detected_numbers)
        # Also check directly in raw text for exact word tokens
        words_in_text = set(re.findall(r"\b[\w'-]+\b", full_text.lower()))

        matched = False
        for req in required_numbers:
            clean_r = str(req).strip().lower()
            canon = NUMBER_WORD_MAP.get(clean_r, clean_r)
            if canon in detected_set or clean_r in words_in_text:
                matched = True
                break

        if matched:
            return True, []
        else:
            return False, required_numbers
