"""
Human & Person Detection Module.
Detects people inside event decoration photographs using YOLOv8 or fallback OpenCV HOG detector.
Configurable, modular, and replaceable.
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HumanDetector:
    def __init__(self, model_type: str = "auto", confidence_threshold: float = 0.45):
        """
        model_type: 'yolo', 'hog', or 'auto' (tries YOLOv8 first, falls back to OpenCV HOG).
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.yolo_model = None
        self.hog_detector = None
        self._init_models()

    def _init_models(self):
        # Initialize YOLOv8 if requested or in auto mode
        if self.model_type in ("auto", "yolo"):
            try:
                from ultralytics import YOLO
                # Load lightweight yolo8n (nano)
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 nano human detector initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO model: {e}. Falling back to HOG.")
                self.yolo_model = None

        # Initialize OpenCV HOG as fallback only if available
        if hasattr(cv2, "HOGDescriptor"):
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
                self.hog_detector = hog
            except Exception as e:
                logger.debug(f"HOG detector unavailable: {e}")
                self.hog_detector = None
        else:
            self.hog_detector = None

    def detect_humans(self, image_bgr: np.ndarray) -> Tuple[bool, int, float, str]:
        """
        Runs human detection on image array.
        Returns:
            human_detected (bool)
            human_count (int)
            max_confidence (float 0.0 - 1.0)
            model_used (str)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, 0, 0.0, "NONE"

        # 1. Try YOLO first if available
        if self.yolo_model is not None:
            try:
                # Class 0 in COCO dataset is 'person'
                results = self.yolo_model.predict(
                    source=image_bgr,
                    classes=[0],
                    conf=self.confidence_threshold,
                    verbose=False
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    count = len(boxes)
                    confs = boxes.conf.cpu().numpy()
                    max_conf = float(np.max(confs)) if len(confs) > 0 else 0.5
                    return True, count, round(max_conf, 2), "YOLOv8"
                return False, 0, 0.0, "YOLOv8"
            except Exception as e:
                logger.warning(f"YOLO inference failed: {e}. Trying HOG fallback.")

        # 2. Fallback to OpenCV HOG People Detector
        if self.hog_detector is not None:
            try:
                # Resize if image is huge for performance
                h, w = image_bgr.shape[:2]
                scale = 1.0
                if max(h, w) > 1280:
                    scale = 1280.0 / max(h, w)
                    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
                else:
                    resized = image_bgr

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                boxes, weights = self.hog_detector.detectMultiScale(
                    gray, winStride=(8, 8), padding=(8, 8), scale=1.05
                )

                if len(boxes) > 0:
                    valid_counts = 0
                    max_w = 0.0
                    for w_val in weights:
                        if float(w_val) >= 0.2:
                            valid_counts += 1
                            if float(w_val) > max_w:
                                max_w = float(w_val)
                    if valid_counts > 0:
                        conf = min(1.0, 0.5 + 0.3 * max_w)
                        return True, valid_counts, round(conf, 2), "OpenCV-HOG"
                return False, 0, 0.0, "OpenCV-HOG"
            except Exception as e:
                logger.error(f"HOG detector inference failed: {e}")

        # If no models could be evaluated
        return False, 0, 0.0, "UNAVAILABLE"
