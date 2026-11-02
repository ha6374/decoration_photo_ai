
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
from functools import lru_cache


# COCO person class ID
PERSON_CLASS_ID = 0

# Lower confidence to improve partial-person detection
CONFIDENCE_THRESHOLD = 0.15


@lru_cache(maxsize=1)
def load_model():
    """Load YOLO segmentation model only once."""

    return YOLO("yolo11n-seg.pt")


def detect_human(image_bytes):
    """
    Detect visible people using YOLO segmentation.
    Returns detection status and number of detected persons.
    """

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image_array = np.array(image)

    model = load_model()

    results = model.predict(
        source=image_array,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=1024,
        verbose=False
    )

    result = results[0]

    person_count = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )

    return {
        "human_detected": person_count > 0,
        "person_count": person_count
    }
