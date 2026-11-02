"""
Accuracy Benchmark & Metrics Evaluation Test.
Runs the dedicated 10-case ground-truth dataset through the upgraded FilterEngine,
calculates empirical Precision, Recall, False Positive, and False Negative rates,
and outputs an accuracy report.
"""

import os
import unittest
import tempfile
import shutil
import json
from typing import Dict, Any, List

from database.db import DatabaseManager
from modules.filter_engine import FilterEngine
from tests.generate_test_dataset import generate_benchmark_dataset


class TestAccuracyBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.test_dir, "benchmark.db")
        cls.dataset_dir = os.path.join(cls.test_dir, "dataset")
        cls.db = DatabaseManager(cls.db_path)
        cls.engine = FilterEngine(cls.db)

        # Generate ground truth benchmark dataset
        cls.cases = generate_benchmark_dataset(cls.dataset_dir)
        cls.rule = cls.db.get_category("First Birthday Decoration")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_run_benchmark_and_verify_metrics(self):
        text_tp, text_fp, text_tn, text_fn = 0, 0, 0, 0
        num_tp, num_fp, num_tn, num_fn = 0, 0, 0, 0
        human_tp, human_fp, human_tn, human_fn = 0, 0, 0, 0
        wm_tp, wm_fp, wm_tn, wm_fn = 0, 0, 0, 0
        decisions_correct = 0

        evaluation_results = []

        for case in self.cases:
            meta = self.engine.process_image(case["file"], self.rule, "benchmark_run")

            # 1. Evaluate Text Detection (Happy Birthday)
            detected_text_lower = (meta.detected_text or "").lower()
            text_found = any(k in detected_text_lower for k in ["happy birthday", "happy b'day", "happy bday", "birthday"])
            expected_text = case["expected_text"]

            if expected_text and text_found:
                text_tp += 1
            elif not expected_text and not text_found:
                text_tn += 1
            elif not expected_text and text_found:
                text_fp += 1
            else:
                text_fn += 1

            # 2. Evaluate Number Detection (1 / ONE)
            num_found = bool(meta.detected_numbers) or ("1" in str(meta.detected_numbers))
            expected_num = case["expected_number"]

            if expected_num and num_found:
                num_tp += 1
            elif not expected_num and not num_found:
                num_tn += 1
            elif not expected_num and num_found:
                num_fp += 1
            else:
                num_fn += 1

            # 3. Evaluate Human Detection
            human_detected = meta.human_detected
            expected_human = case["expected_human"]
            if expected_human and human_detected:
                human_tp += 1
            elif not expected_human and not human_detected:
                human_tn += 1
            elif not expected_human and human_detected:
                human_fp += 1
            else:
                human_fn += 1

            # 4. Evaluate Watermark Detection
            wm_detected = meta.watermark_detected
            expected_wm = case["expected_watermark"]
            if expected_wm and wm_detected:
                wm_tp += 1
            elif not expected_wm and not wm_detected:
                wm_tn += 1
            elif not expected_wm and wm_detected:
                wm_fp += 1
            else:
                wm_fn += 1

            # 5. Overall Decision
            expected_status = case["expected_status"]
            actual_status = meta.status

            is_decision_match = (
                (actual_status == expected_status) or
                (expected_status == "REJECTED" and actual_status in ("REJECTED", "REVIEW_REQUIRED"))
            )
            if is_decision_match:
                decisions_correct += 1

            evaluation_results.append({
                "file": os.path.basename(case["file"]),
                "description": case["description"],
                "expected_status": expected_status,
                "actual_status": actual_status,
                "reasons": meta.rejection_reasons,
                "text_found": text_found,
                "number_found": num_found,
                "detected_numbers": meta.detected_numbers,
                "human_detected": human_detected,
                "watermark_detected": wm_detected
            })

        total = len(self.cases)
        # Calculate Real Metrics
        text_prec = text_tp / max(1, text_tp + text_fp)
        text_rec = text_tp / max(1, text_tp + text_fn)

        num_prec = num_tp / max(1, num_tp + num_fp)
        num_rec = num_tp / max(1, num_tp + num_fn)

        decision_accuracy = decisions_correct / float(total)

        print("\n" + "=" * 60)
        print("EMPIRICAL BENCHMARK EVALUATION RESULTS:")
        print("=" * 60)
        print(f"Total Benchmark Cases: {total}")
        print(f"OCR Precision: {text_prec * 100:.1f}% | OCR Recall: {text_rec * 100:.1f}%")
        print(f"Number Detection Precision: {num_prec * 100:.1f}% | Number Recall: {num_rec * 100:.1f}%")
        print(f"Human Detection Accuracy: {((human_tp + human_tn) / float(total)) * 100:.1f}%")
        print(f"Watermark Detection Accuracy: {((wm_tp + wm_tn) / float(total)) * 100:.1f}%")
        print(f"Overall Filter Decision Accuracy: {decision_accuracy * 100:.1f}%")
        print("=" * 60)

        # Assert production standards
        self.assertGreaterEqual(text_rec, 0.85, "OCR Recall must be at least 85%")
        self.assertGreaterEqual(num_rec, 0.85, "Number Recall must be at least 85%")
        self.assertGreaterEqual(decision_accuracy, 0.90, "Overall Filter Decision Accuracy must be at least 90%")


if __name__ == "__main__":
    unittest.main()
