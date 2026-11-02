"""
Report Generation Module.
Exports comprehensive CSV and Excel reports containing complete verification records,
hashes, blur scores, OCR extractions, and classification decisions.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


class ReportGenerator:
    def __init__(self, reports_base_dir: str):
        self.reports_base_dir = os.path.abspath(reports_base_dir)
        self.csv_dir = os.path.join(self.reports_base_dir, "CSV")
        self.excel_dir = os.path.join(self.reports_base_dir, "Excel")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.excel_dir, exist_ok=True)

    def prepare_dataframe(self, image_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transforms database image records into formatted reporting DataFrame."""
        rows = []
        for r in image_records:
            reasons = r.get("rejection_reasons", [])
            reasons_str = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)

            detected_nums = r.get("detected_numbers", [])
            nums_str = ", ".join(detected_nums) if isinstance(detected_nums, list) else str(detected_nums)

            file_size_kb = round(r.get("file_size_bytes", 0) / 1024.0, 2)

            rows.append({
                "Image ID": r.get("id"),
                "Original Filename": r.get("original_filename"),
                "Category": r.get("category"),
                "Final Status": r.get("status"),
                "Rejection Reasons": reasons_str,
                "File Format": r.get("file_format"),
                "Width (px)": r.get("width"),
                "Height (px)": r.get("height"),
                "File Size (KB)": file_size_kb,
                "SHA-256 Hash": r.get("sha256"),
                "Perceptual Hash (pHash)": r.get("phash"),
                "Blur Score (Sharpness)": r.get("blur_score"),
                "Human Detected": "Yes" if r.get("human_detected") else "No",
                "Human Count": r.get("human_count", 0),
                "Watermark Detected": "Yes" if r.get("watermark_detected") else "No",
                "Watermark Confidence": r.get("watermark_confidence", 0.0),
                "Detected Angle": r.get("detected_angle"),
                "Angle Confidence": r.get("angle_confidence", 0.0),
                "OCR Detected Text": r.get("detected_text", ""),
                "Detected Numbers": nums_str,
                "Duplicate of Image ID": r.get("duplicate_of_id"),
                "Duplicate Similarity": r.get("duplicate_similarity", 0.0),
                "Batch ID": r.get("upload_batch_id"),
                "Source Path": r.get("source_path"),
                "Output Path": r.get("output_path"),
                "Processed Timestamp": r.get("first_seen_at")
            })

        return pd.DataFrame(rows)

    def export_csv(self, records: List[Dict[str, Any]], filename_prefix: str = "decoration_report") -> Tuple[str, bytes]:
        """Generates CSV report file and returns (filepath, raw_bytes)."""
        df = self.prepare_dataframe(records)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(self.csv_dir, filename)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        with open(filepath, "wb") as f:
            f.write(csv_bytes)

        return filepath, csv_bytes

    def export_excel(self, records: List[Dict[str, Any]], filename_prefix: str = "decoration_report") -> Tuple[str, bytes]:
        """Generates Excel report file and returns (filepath, raw_bytes)."""
        df = self.prepare_dataframe(records)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        filepath = os.path.join(self.excel_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Decorations", index=False)

        with open(filepath, "rb") as f:
            excel_bytes = f.read()

        return filepath, excel_bytes
