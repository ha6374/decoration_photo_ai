"""
Synthetic Realistic Decoration Dataset Generator for Accuracy Benchmark.
Generates labeled benchmark decoration photographs with diverse font styles,
curved banners, neon signs, balloon numbers, floral numbers, people, and watermarks.
"""

import os
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_decorated_scene(
    filepath: str,
    text_content: str = "",
    number_content: str = "",
    text_style: str = "standard",  # 'standard', 'curved', 'neon', 'rotated', 'small'
    number_style: str = "marquee",  # 'marquee', 'balloon', 'word_one', 'none'
    blur: bool = False,
    watermark: bool = False,
    add_person: bool = False,
    is_front: bool = True
) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    w, h = 1000, 750

    # Base background (neutral indoor backdrop or organic green/pastel wall)
    if text_style == "neon":
        # Dark green or midnight blue wall for neon sign contrast
        base_color = (25, 45, 30)
    else:
        base_color = (248, 244, 240)

    img = Image.new("RGB", (w, h), color=base_color)
    draw = ImageDraw.Draw(img)

    # 1. Background Structure & Arch Frame
    if is_front:
        # Symmetrical circular/arch frame
        draw.ellipse([150, 60, 850, 700], outline=(220, 200, 180), width=6)
        # Floor stage
        draw.line([(50, 650), (950, 650)], fill=(200, 180, 160), width=5)
    else:
        # Skewed side angle
        draw.polygon([(100, 150), (350, 100), (450, 680), (120, 720)], outline=(220, 200, 180), width=6)
        draw.line([(80, 720), (900, 580)], fill=(200, 180, 160), width=5)

    # 2. Balloon Garland clusters
    colors = [(255, 182, 193), (255, 105, 180), (255, 215, 0), (255, 255, 255), (173, 216, 230)]
    for i in range(25):
        bx = int(180 + 640 * (i / 25.0) + np.sin(i) * 30)
        by = int(100 - np.sin(i * np.pi / 25.0) * 60 + np.cos(i) * 20)
        rad = np.random.randint(28, 45)
        c = colors[i % len(colors)]
        draw.ellipse([bx - rad, by - rad, bx + rad, by + rad], fill=c, outline=(200, 150, 150))

    # 3. Cake Table Center
    draw.rectangle([400, 480, 600, 650], fill=(255, 255, 255), outline=(210, 210, 210), width=2)
    # Cake
    draw.rectangle([460, 420, 540, 480], fill=(255, 230, 240), outline=(230, 180, 200))

    # 4. Text Banner / Sign Rendering
    if text_content:
        if text_style == "neon":
            # Bright luminous neon flex sign
            # Glowing outline
            draw.rectangle([250, 220, 750, 310], fill=(35, 55, 40), outline=(100, 255, 200), width=3)
            draw.text((280, 250), text_content.upper(), fill=(255, 255, 255))
        elif text_style == "curved":
            # Simulated curved arch banner
            words = text_content.split()
            for idx, word in enumerate(words):
                wx = 320 + idx * 180
                wy = 200 + int(25 * np.sin(idx))
                draw.text((wx, wy), word, fill=(40, 40, 40))
        elif text_style == "rotated":
            # Angled banner
            draw.rectangle([300, 230, 700, 290], fill=(255, 225, 120), outline=(200, 180, 80))
            draw.text((330, 245), text_content, fill=(30, 30, 30))
        else:
            # Standard clean backdrop text
            draw.rectangle([280, 220, 720, 290], fill=(255, 230, 130), outline=(210, 180, 90))
            draw.text((320, 245), text_content, fill=(40, 40, 40))

    # 5. Number 1 / ONE Decoration
    if number_content:
        if number_style == "marquee":
            # Tall vertical freestanding marquee 1 on stage floor (left flank)
            # Aspect ratio ~ 3.5:1
            draw.rectangle([200, 360, 280, 650], fill=(255, 255, 240), outline=(180, 160, 120), width=4)
            # Bulb dots
            for by in range(380, 640, 30):
                draw.ellipse([230, by, 250, by + 15], fill=(255, 235, 100))
            draw.text((225, 470), "1", fill=(60, 50, 30))
        elif number_style == "balloon":
            # Giant gold foil number 1 balloon (right flank)
            draw.rectangle([720, 350, 800, 640], fill=(230, 190, 50), outline=(180, 140, 30), width=4)
            draw.text((745, 470), "1", fill=(80, 60, 10))
        elif number_style == "word_one":
            # Freestanding wooden letters 'ONE' on table front
            draw.rectangle([420, 520, 580, 580], fill=(255, 240, 200), outline=(150, 120, 80))
            draw.text((450, 535), "ONE", fill=(60, 40, 20))

    # 6. Human addition if requested
    if add_person:
        # Draw stylized human standing beside table
        # Head
        draw.ellipse([640, 320, 700, 380], fill=(240, 190, 160))
        # Body torso
        draw.rectangle([630, 380, 710, 540], fill=(40, 60, 120))
        # Legs
        draw.line([(650, 540), (650, 650)], fill=(30, 30, 50), width=10)
        draw.line([(690, 540), (690, 650)], fill=(30, 30, 50), width=10)

    # 7. Watermark addition if requested
    if watermark:
        draw.text((620, 710), "SHUTTERSTOCK PREVIEW (C) ALL RIGHTS RESERVED", fill=(120, 120, 120))

    img.save(filepath, quality=95)

    # Apply Gaussian blur if requested
    if blur:
        cv_img = cv2.imread(filepath)
        blurred = cv2.GaussianBlur(cv_img, (55, 55), 0)
        cv2.imwrite(filepath, blurred)

    return filepath


def generate_benchmark_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Builds the complete ground-truth benchmark suite of 10 representative decoration images.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    benchmarks = []

    # Case 1: Ideal First Birthday (Clear banner + Marquee 1) -> APPROVED
    p1 = os.path.join(dataset_dir, "case01_clear_happy_birthday_1.jpg")
    create_decorated_scene(p1, text_content="Happy Birthday", number_content="1", number_style="marquee")
    benchmarks.append({
        "file": p1, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "APPROVED",
        "description": "Standard clean front-facing First Birthday backdrop with Happy Birthday text and marquee 1"
    })

    # Case 2: Decorative Cursive 'Happy B'day' + Floral 'ONE' -> APPROVED
    p2 = os.path.join(dataset_dir, "case02_decorative_happy_bday_one.jpg")
    create_decorated_scene(p2, text_content="Happy B'day", number_content="ONE", number_style="word_one")
    benchmarks.append({
        "file": p2, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "APPROVED",
        "description": "Decorative cursive text Happy B'day and table letters ONE"
    })

    # Case 3: Glowing Neon Sign 'Happy Birthday' on dark wall + Giant Foil Balloon 1 -> APPROVED
    p3 = os.path.join(dataset_dir, "case03_neon_birthday_balloon1.jpg")
    create_decorated_scene(p3, text_content="Happy Birthday", number_content="1", text_style="neon", number_style="balloon")
    benchmarks.append({
        "file": p3, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "APPROVED",
        "description": "High-contrast glowing neon sign with freestanding foil balloon 1"
    })

    # Case 4: Curved arch lettering 'Happy Birthday' + Marquee 1 -> APPROVED
    p4 = os.path.join(dataset_dir, "case04_curved_birthday_1.jpg")
    create_decorated_scene(p4, text_content="Happy Birthday", number_content="1", text_style="curved", number_style="marquee")
    benchmarks.append({
        "file": p4, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "APPROVED",
        "description": "Curved arch lettering banner Happy Birthday with marquee 1"
    })

    # Case 5: Heavy Blur -> REJECTED (Blur filter)
    p5 = os.path.join(dataset_dir, "case05_blurry_birthday.jpg")
    create_decorated_scene(p5, text_content="Happy Birthday", number_content="1", blur=True)
    benchmarks.append({
        "file": p5, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "REJECTED",
        "description": "Severely blurred photograph failing sharpness threshold"
    })

    # Case 6: Shutterstock Watermark Overlay -> REJECTED (Watermark filter)
    p6 = os.path.join(dataset_dir, "case06_watermarked_stock.jpg")
    create_decorated_scene(p6, text_content="Happy Birthday", number_content="1", watermark=True)
    benchmarks.append({
        "file": p6, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": True, "expected_status": "REJECTED",
        "description": "Stock photograph containing visible Shutterstock copyright watermark"
    })

    # Case 7: Person Present in Scene -> REJECTED (Human detection)
    p7 = os.path.join(dataset_dir, "case07_person_in_frame.jpg")
    create_decorated_scene(p7, text_content="Happy Birthday", number_content="1", add_person=True)
    benchmarks.append({
        "file": p7, "expected_text": True, "expected_number": True,
        "expected_human": True, "expected_watermark": False, "expected_status": "REJECTED",
        "description": "Guest / person standing in front of decoration setup"
    })

    # Case 8: Wedding Stage Decor (No birthday text or number) -> REJECTED (Category mismatch)
    p8 = os.path.join(dataset_dir, "case08_wedding_stage_no_birthday.jpg")
    create_decorated_scene(p8, text_content="Wedding Reception", number_content="")
    benchmarks.append({
        "file": p8, "expected_text": False, "expected_number": False,
        "expected_human": False, "expected_watermark": False, "expected_status": "REJECTED",
        "description": "Wedding setup missing Happy Birthday and number one"
    })

    # Case 9: Severe Side Perspective -> REJECTED or REVIEW_REQUIRED
    p9 = os.path.join(dataset_dir, "case09_side_angle_perspective.jpg")
    create_decorated_scene(p9, text_content="Happy Birthday", number_content="1", is_front=False)
    benchmarks.append({
        "file": p9, "expected_text": True, "expected_number": True,
        "expected_human": False, "expected_watermark": False, "expected_status": "REJECTED",
        "description": "Oblique side view of the decoration"
    })

    # Case 10: Haldi Ceremony Decor -> REJECTED (Category mismatch)
    p10 = os.path.join(dataset_dir, "case10_haldi_decor.jpg")
    create_decorated_scene(p10, text_content="Haldi Ceremony", number_content="")
    benchmarks.append({
        "file": p10, "expected_text": False, "expected_number": False,
        "expected_human": False, "expected_watermark": False, "expected_status": "REJECTED",
        "description": "Traditional yellow floral Haldi setup missing birthday criteria"
    })

    return benchmarks
