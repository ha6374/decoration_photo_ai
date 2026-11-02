
import cv2
import numpy as np


# Initial blur threshold
BLUR_THRESHOLD = 100


def detect_blur(image_bytes):
    """
    Detect whether an image is blurry using
    Variance of Laplacian technique.
    """

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_blurry = blur_score < BLUR_THRESHOLD

    return {
        "is_blurry": is_blurry,
        "blur_score": round(blur_score, 2),
        "threshold": BLUR_THRESHOLD
    }
