# image_process
import zipfile
import shutil
from pathlib import Path
from io import BytesIO

from PIL import Image

from modules.blur_detector import detect_blur
from modules.human_detector import detect_human


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

ACCEPTED_DIR = BASE_DIR / "output" / "accepted"
REJECTED_DIR = BASE_DIR / "output" / "rejected"

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


def clear_output_folders():
    """Clear previous processing results."""

    for folder in [ACCEPTED_DIR, REJECTED_DIR]:

        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(parents=True, exist_ok=True)


def process_zip(uploaded_file):
    """Extract and filter images based on blur and human detection."""

    clear_output_folders()

    results = []

    stats = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "invalid": 0,
        "human_detected": 0
    }

    zip_bytes = uploaded_file.getvalue()

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as archive:

            for index, item in enumerate(archive.infolist(), start=1):

                if item.is_dir():
                    continue

                original_name = Path(item.filename).name
                extension = Path(original_name).suffix.lower()

                if extension not in ALLOWED_EXTENSIONS:
                    continue

                stats["total"] += 1

                if item.file_size > 25 * 1024 * 1024:

                    stats["invalid"] += 1

                    results.append({
                        "name": original_name,
                        "status": "Invalid",
                        "reason": "File exceeds 25 MB"
                    })

                    continue

                try:
                    image_bytes = archive.read(item)

                    # Validate image
                    with Image.open(BytesIO(image_bytes)) as image:
                        image.verify()

                    # Blur detection
                    blur_result = detect_blur(image_bytes)

                    # Human detection
                    human_result = detect_human(image_bytes)

                    is_blurry = blur_result["is_blurry"]
                    has_human = human_result["human_detected"]

                    reasons = []

                    if is_blurry:
                        reasons.append("Blurry image")

                    if has_human:
                        reasons.append("Human detected")

                        stats["human_detected"] += 1

                    filename = f"{index}_{original_name}"

                    if is_blurry or has_human:

                        destination = REJECTED_DIR / filename
                        status = "Rejected"

                        reason = " + ".join(reasons)

                        stats["rejected"] += 1

                    else:

                        destination = ACCEPTED_DIR / filename
                        status = "Accepted"

                        reason = "Clear image, no human detected"

                        stats["accepted"] += 1

                    destination.write_bytes(image_bytes)

                    results.append({
                        "name": original_name,
                        "status": status,
                        "reason": reason,
                        "blur_score": blur_result["blur_score"],
                        "person_count": human_result["person_count"]
                    })

                except Exception as error:

                    stats["invalid"] += 1

                    results.append({
                        "name": original_name,
                        "status": "Invalid",
                        "reason": f"Unable to process image: {str(error)[:100]}"
                    })

    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    return stats, results
