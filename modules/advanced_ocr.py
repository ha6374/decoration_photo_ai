"""
Advanced Multi-Stage OCR Engine for Event Decoration Photography.
Features multi-variant image preprocessing (CLAHE, unsharp masking, inverted binarization for neon signs),
multi-region cropping (central backdrop, arch, table, flanks), and fuzzy token matching for curved & decorative fonts.
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Common festive aliases and synonyms
KEYWORD_SYNONYMS = {
    "happy birthday": [
        "happy birthday", "happy bday", "happy b'day", "happy b-day",
        "happy birthday to you", "hbd", "first birthday", "1st birthday",
        "2nd birthday", "birthday celebration"
    ],
    "baby shower": ["baby shower", "welcome baby", "baby boy", "baby girl", "mom to be"],
    "haldi": ["haldi", "haldi ceremony", "peethi", "haldi rasam"],
    "anniversary": ["anniversary", "happy anniversary", "wedding anniversary"],
    "engagement": ["engagement", "ring ceremony", "engaged", "she said yes"],
    "wedding": ["wedding", "wedding reception", "marriage", "just married", "vivah"]
}

# Number words to canonical digits
NUMBER_WORDS = {
    "0": "0", "zero": "0",
    "1": "1", "one": "1", "first": "1", "1st": "1", "one!": "1", "|": "1", "l": "1", "i": "1",
    "2": "2", "two": "2", "second": "2", "2nd": "2",
    "3": "3", "three": "3", "third": "3", "3rd": "3",
    "4": "4", "four": "4", "fourth": "4", "4th": "4",
    "5": "5", "five": "5", "fifth": "5", "5th": "5",
    "6": "6", "six": "6", "sixth": "6", "6th": "6",
    "7": "7", "seven": "7", "seventh": "7", "7th": "7",
    "8": "8", "eight": "8", "eighth": "8", "8th": "8",
    "9": "9", "nine": "9", "ninth": "9", "9th": "9",
    "10": "10", "ten": "10", "tenth": "10", "10th": "10"
}


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculates Levenshtein similarity ratio between 0.0 and 1.0."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    max_len = max(len1, len2)
    return 1.0 - (dp[len1][len2] / max_len)


class AdvancedOCREngine:
    _easyocr_reader = None

    def __init__(self, min_confidence: float = 0.20, use_gpu: bool = False):
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        self._init_reader()

    @classmethod
    def get_reader(cls, use_gpu: bool = False):
        if cls._easyocr_reader is None:
            try:
                import easyocr
                cls._easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                logger.info("EasyOCR Reader successfully initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize EasyOCR: {e}")
                cls._easyocr_reader = None
        return cls._easyocr_reader

    def _init_reader(self):
        self.get_reader(self.use_gpu)

    # ---------------- Image Preprocessing Variants ----------------

    @staticmethod
    def generate_preprocessing_variants(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Creates specialized vision variants to enhance contrast, neon glow, and small text.
        """
        variants: Dict[str, np.ndarray] = {"original": image_bgr}

        if image_bgr is None or image_bgr.size == 0:
            return variants

        # 1. CLAHE Contrast enhancement on Lightness channel
        try:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            enhanced = cv2.merge((l_clahe, a, b))
            variants["clahe"] = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

        # 2. Unsharp Masking (Sharpening)
        try:
            gaussian = cv2.GaussianBlur(image_bgr, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image_bgr, 1.5, gaussian, -0.5, 0)
            variants["sharpened"] = sharpened
        except Exception:
            pass

        # 3. Inverted Binary (Neon Sign Enhancement)
        # Neon signs are glowing white/bright on dark grass or backdrops
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # Threshold high-luminance elements
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Invert: text becomes black on white for OCR
            neon_enhanced = cv2.bitwise_not(bright_mask)
            variants["neon_inv"] = cv2.cvtColor(neon_enhanced, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

        # 4. Otsu Binarization on Grayscale
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants["otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

        return variants

    # ---------------- Regional Cropping ----------------

    @staticmethod
    def extract_regional_crops(image_bgr: np.ndarray) -> Dict[str, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extracts strategic regions where decoration signs and numbers commonly appear.
        Returns:
            dict of {region_name: (cropped_img, (x_offset, y_offset, crop_w, crop_h))}
        """
        h, w = image_bgr.shape[:2]
        regions = {
            "full": (image_bgr, (0, 0, w, h))
        }

        # Avoid cropping if image is extremely small
        if w < 200 or h < 200:
            return regions

        # Center Backdrop Region (Center 60% x 60% where banners, neon signs, and numbers hang)
        cx1, cy1 = int(w * 0.18), int(h * 0.15)
        cx2, cy2 = int(w * 0.82), int(h * 0.80)
        regions["center_backdrop"] = (image_bgr[cy1:cy2, cx1:cx2], (cx1, cy1, cx2 - cx1, cy2 - cy1))

        # Upper Arch Region (top 50% across width where arch banners hang)
        ux1, uy1 = int(w * 0.10), int(h * 0.05)
        ux2, uy2 = int(w * 0.90), int(h * 0.55)
        regions["upper_arch"] = (image_bgr[uy1:uy2, ux1:ux2], (ux1, uy1, ux2 - ux1, uy2 - uy1))

        # Lower Table / Cake Stage Region (bottom 50% where freestanding LED 1 or table letters sit)
        lx1, ly1 = int(w * 0.15), int(h * 0.45)
        lx2, ly2 = int(w * 0.85), int(h * 0.95)
        regions["lower_stage"] = (image_bgr[ly1:ly2, lx1:lx2], (lx1, ly1, lx2 - lx1, ly2 - ly1))

        # Left Flank (common position for freestanding large marquee number 1)
        flx1, fly1 = int(w * 0.05), int(h * 0.20)
        flx2, fly2 = int(w * 0.45), int(h * 0.88)
        regions["left_flank"] = (image_bgr[fly1:fly2, flx1:flx2], (flx1, fly1, flx2 - flx1, fly2 - fly1))

        # Right Flank
        frx1, fry1 = int(w * 0.55), int(h * 0.20)
        frx2, fry2 = int(w * 0.95), int(h * 0.88)
        regions["right_flank"] = (image_bgr[fry1:fry2, frx1:frx2], (frx1, fry1, frx2 - frx1, fry2 - fry1))

        return regions

    # ---------------- Multi-Stage Text Extraction ----------------

    def extract_all_text(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Executes multi-variant, multi-region OCR extraction.
        Returns aggregated text tokens with bounding boxes, confidence, and detected candidates.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "full_text": "",
                "raw_tokens": [],
                "detected_words": [],
                "detected_numbers": [],
                "highest_confidence": 0.0
            }

        reader = self.get_reader(self.use_gpu)
        all_tokens: List[Dict[str, Any]] = []
        seen_spans: Set[str] = set()

        # Step 1: Preprocess full image variants
        variants = self.generate_preprocessing_variants(image_bgr)
        regional_crops = self.extract_regional_crops(image_bgr)

        # Plan inspection passes:
        # Pass A: Full image on 'original' and 'clahe'
        # Pass B: Center backdrop on 'original', 'clahe', 'neon_inv'
        # Pass C: Upper arch and lower stage on 'clahe'
        passes = [
            ("full", "original", image_bgr, (0, 0, image_bgr.shape[1], image_bgr.shape[0])),
            ("full", "clahe", variants.get("clahe", image_bgr), (0, 0, image_bgr.shape[1], image_bgr.shape[0])),
            ("center_backdrop", "clahe", regional_crops["center_backdrop"][0], regional_crops["center_backdrop"][1]),
            ("center_backdrop", "neon_inv", cv2.cvtColor(
                cv2.bitwise_not(cv2.threshold(cv2.cvtColor(regional_crops["center_backdrop"][0], cv2.COLOR_BGR2GRAY), 195, 255, cv2.THRESH_BINARY)[1]),
                cv2.COLOR_GRAY2BGR
            ), regional_crops["center_backdrop"][1]),
            ("upper_arch", "clahe", regional_crops["upper_arch"][0], regional_crops["upper_arch"][1]),
            ("lower_stage", "clahe", regional_crops["lower_stage"][0], regional_crops["lower_stage"][1]),
            ("left_flank", "clahe", regional_crops["left_flank"][0], regional_crops["left_flank"][1]),
            ("right_flank", "clahe", regional_crops["right_flank"][0], regional_crops["right_flank"][1]),
        ]

        if reader is not None:
            for region_name, var_name, crop_img, (ox, oy, cw, ch) in passes:
                if crop_img is None or crop_img.size == 0 or cw < 40 or ch < 40:
                    continue

                try:
                    # Optional scale up for small regions
                    proc_crop = crop_img
                    scale_factor = 1.0
                    if max(cw, ch) < 500:
                        scale_factor = 2.0
                        proc_crop = cv2.resize(crop_img, (int(cw * scale_factor), int(ch * scale_factor)), interpolation=cv2.INTER_CUBIC)

                    results = reader.readtext(proc_crop)
                    for bbox, text, conf in results:
                        cleaned = text.strip()
                        if not cleaned or conf < self.min_confidence:
                            continue

                        # Rescale bbox coordinates back to full image space
                        global_bbox = []
                        for pt in bbox:
                            gx = int(ox + (pt[0] / scale_factor))
                            gy = int(oy + (pt[1] / scale_factor))
                            global_bbox.append([gx, gy])

                        token_key = f"{cleaned.lower()}_{global_bbox[0][0]//20}_{global_bbox[0][1]//20}"
                        if token_key not in seen_spans:
                            seen_spans.add(token_key)
                            all_tokens.append({
                                "text": cleaned,
                                "confidence": round(float(conf), 3),
                                "bbox": global_bbox,
                                "region": region_name,
                                "variant": var_name
                            })
                except Exception as e:
                    logger.debug(f"OCR pass {region_name}-{var_name} error: {e}")

        # Construct unified text and extract candidate words/numbers
        unique_texts = [t["text"] for t in all_tokens]
        full_text = " ".join(unique_texts)
        all_words = re.findall(r"\b[\w'-]+\b", full_text.lower())

        # Parse numbers
        detected_numbers = self.extract_numbers_from_tokens(all_tokens, full_text)

        highest_conf = max([t["confidence"] for t in all_tokens]) if all_tokens else 0.0

        return {
            "full_text": full_text,
            "raw_tokens": all_tokens,
            "detected_words": list(set(all_words)),
            "detected_numbers": detected_numbers,
            "highest_confidence": round(highest_conf, 3)
        }

    # ---------------- Number Recognition from Tokens ----------------

    @classmethod
    def extract_numbers_from_tokens(cls, tokens: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """
        Extracts numbers with bounding boxes and origin confidence.
        """
        results: List[Dict[str, Any]] = []
        seen = set()

        # 1. Search in individual tokens
        for t in tokens:
            token_text = t["text"].strip().lower()

            # Direct word map (e.g. 'one', '1', 'first', '1st')
            if token_text in NUMBER_WORDS:
                canon = NUMBER_WORDS[token_text]
                if canon not in seen:
                    seen.add(canon)
                    results.append({
                        "canonical": canon,
                        "raw_text": t["text"],
                        "confidence": t["confidence"],
                        "bbox": t["bbox"],
                        "region": t.get("region", "unknown"),
                        "method": "OCR_EXACT_TOKEN"
                    })

            # Check inside token via regex (e.g. 'Party 1st' or 'One!')
            for word in re.findall(r"\b[\w'-]+\b", token_text):
                if word in NUMBER_WORDS:
                    canon = NUMBER_WORDS[word]
                    if canon not in seen:
                        seen.add(canon)
                        results.append({
                            "canonical": canon,
                            "raw_text": word,
                            "confidence": t["confidence"],
                            "bbox": t["bbox"],
                            "region": t.get("region", "unknown"),
                            "method": "OCR_SUBWORD_MATCH"
                        })

        # 2. Additional regex sweep across full concatenated text
        for word in re.findall(r"\b[\w'-]+\b", full_text.lower()):
            if word in NUMBER_WORDS:
                canon = NUMBER_WORDS[word]
                if canon not in seen:
                    seen.add(canon)
                    results.append({
                        "canonical": canon,
                        "raw_text": word,
                        "confidence": 0.75,
                        "bbox": None,
                        "region": "full_text",
                        "method": "OCR_GLOBAL_REGEX"
                    })

        return results

    # ---------------- Fuzzy & Contextual Keyword Matching ----------------

    @classmethod
    def match_required_keywords(cls, 
                                ocr_tokens: List[Dict[str, Any]], 
                                full_text: str, 
                                required_keywords: List[str], 
                                fuzzy_threshold: float = 0.80) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """
        Context-aware keyword matching using exact, token-combination, and Levenshtein fuzzy matching.
        Returns:
            all_passed (bool)
            found_evidence (List[Dict])
            missing_keywords (List[str])
        """
        if not required_keywords:
            return True, [], []

        found_evidence: List[Dict[str, Any]] = []
        missing: List[str] = []
        normalized_full = re.sub(r"[^\w\s]", " ", full_text.lower())

        for req in required_keywords:
            req_clean = req.lower().strip()
            # Retrieve synonyms / aliases if available
            synonyms = KEYWORD_SYNONYMS.get(req_clean, [req_clean])
            req_found = False

            # Check 1: Direct substring in full text or synonyms
            for syn in synonyms:
                if syn in normalized_full:
                    req_found = True
                    found_evidence.append({
                        "required_target": req,
                        "matched_term": syn,
                        "match_type": "EXACT_SUBSTRING",
                        "confidence": 0.95
                    })
                    break

            if req_found:
                continue

            # Check 2: Token-pair proximity (e.g. token 'HAPPY' followed by token 'BIRTHDAY')
            req_words = [w for w in re.findall(r"\b\w+\b", req_clean) if len(w) > 1]
            if len(req_words) >= 2:
                # Count how many words of the phrase are found anywhere in tokens
                token_words = set(re.findall(r"\b\w+\b", normalized_full))
                matched_subwords = [w for w in req_words if w in token_words]
                if len(matched_subwords) == len(req_words):
                    req_found = True
                    found_evidence.append({
                        "required_target": req,
                        "matched_term": " + ".join(matched_subwords),
                        "match_type": "MULTI_TOKEN_PROXIMITY",
                        "confidence": 0.88
                    })
                    continue

            # Check 3: Fuzzy Levenshtein token matching
            for token in ocr_tokens:
                txt = token["text"].lower().strip()
                # Compare against all synonyms
                for syn in synonyms:
                    score = levenshtein_similarity(txt, syn)
                    if score >= fuzzy_threshold:
                        req_found = True
                        found_evidence.append({
                            "required_target": req,
                            "matched_term": token["text"],
                            "match_type": "FUZZY_LEVENSHTEIN",
                            "similarity": round(score, 3),
                            "confidence": token["confidence"],
                            "bbox": token["bbox"]
                        })
                        break
                if req_found:
                    break

            if not req_found:
                missing.append(req)

        return (len(missing) == 0), found_evidence, missing
