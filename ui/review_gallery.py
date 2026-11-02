"""
Manual Review Gallery UI View.
Allows visual inspection of processed decoration photos, bulk status updates,
full-resolution previews, and safe relocation of output copies while preserving original files.
"""

import os
import streamlit as st
from typing import List, Dict, Any

from database.db import DatabaseManager
from modules.image_organizer import ImageOrganizer
from modules.duplicate_detector import DuplicateDetector
from modules.image_loader import ImageLoader


def render_review_gallery(db: DatabaseManager, organizer: ImageOrganizer):
    st.title("Manual Review Gallery")
    st.markdown("Inspect classification decisions, verify borderline photos, and manually adjust approval states.")

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    categories = ["All"] + [c.name for c in db.list_categories()]
    with f_col1:
        cat_filter = st.selectbox("Filter Category:", categories, index=0)
    with f_col2:
        status_filter = st.selectbox(
            "Filter Status:",
            ["All", "REVIEW_REQUIRED", "APPROVED", "REJECTED", "DUPLICATE"],
            index=1  # Default to Review Required
        )
    with f_col3:
        batches = ["All"] + [b["batch_id"] for b in db.list_batches()]
        batch_filter = st.selectbox("Filter Batch:", batches, index=0)

    # Fetch matching images
    images = db.query_images(category=cat_filter, status=status_filter, batch_id=batch_filter, limit=100)

    if not images:
        st.info("No images match the current filter selection.")
        return

    st.write(f"Showing **{len(images)}** images:")

    # Bulk Action Toolbar
    st.markdown("---")
    bulk_col1, bulk_col2, bulk_col3 = st.columns([1, 1, 2])
    with bulk_col1:
        if st.button("Bulk Approve Displayed", type="primary"):
            _handle_bulk_action(images, "APPROVED", db, organizer)
            st.rerun()
    with bulk_col2:
        if st.button("Bulk Reject Displayed", type="secondary"):
            _handle_bulk_action(images, "REJECTED", db, organizer)
            st.rerun()

    st.markdown("---")

    # Grid Display
    # Render in 3 columns
    cols_per_row = 3
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = i + col_idx
            if img_idx >= len(images):
                break

            img_rec = images[img_idx]
            with cols[col_idx]:
                _render_image_card(img_rec, db, organizer)


def _render_image_card(img_rec: Dict[str, Any], db: DatabaseManager, organizer: ImageOrganizer):
    # Determine which file to display: prefer thumbnail, fallback to output_path or source_path
    display_path = img_rec.get("output_path") or img_rec.get("source_path")

    # Status color badge
    status = img_rec.get("status", "UNKNOWN")
    color_map = {
        "APPROVED": "green",
        "REVIEW_REQUIRED": "orange",
        "REJECTED": "red",
        "DUPLICATE": "blue"
    }
    badge_color = color_map.get(status, "grey")

    st.markdown(
        f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 15px;'>"
        f"<h4 style='margin:0; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{img_rec.get('original_filename')}</h4>"
        f"<p style='color:{badge_color}; font-weight:bold; margin:2px 0;'>[{status}]</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    if display_path and os.path.exists(display_path):
        thumb = ImageLoader.get_thumbnail(display_path, max_size=(320, 240))
        if thumb:
            st.image(thumb, use_container_width=True)
        else:
            st.warning("Thumbnail unavailable")
    else:
        st.warning("File preview unavailable")

    with st.expander("Card Details & Diagnostics"):
        st.write(f"**ID:** #{img_rec.get('id')} | **Category:** {img_rec.get('category')}")
        st.write(f"**Angle:** {img_rec.get('detected_angle')} (conf: {img_rec.get('angle_confidence', 0.0)})")
        st.write(f"**Blur Score:** {img_rec.get('blur_score')}")
        st.write(f"**Human Detected:** {'Yes' if img_rec.get('human_detected') else 'No'} (count: {img_rec.get('human_count', 0)})")
        st.write(f"**Watermark:** {'Yes' if img_rec.get('watermark_detected') else 'No'}")
        st.write(f"**OCR Text:** `{img_rec.get('detected_text') or 'None'}`")
        if img_rec.get("detected_numbers"):
            st.write(f"**Numbers:** {', '.join(img_rec.get('detected_numbers'))}")
        if img_rec.get("rejection_reasons"):
            st.write(f"**Reasons:**")
            for r in img_rec.get("rejection_reasons"):
                st.caption(f"• {r}")

        # Check for forensic debug JSON
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        short_hash = img_rec.get("sha256", "")[:12]
        debug_json_path = os.path.join(base_dir, "logs", "debug", f"debug_{short_hash}.json")
        if os.path.exists(debug_json_path):
            if st.checkbox("View Detailed AI & OCR JSON Evidence", key=f"dbg_{img_rec.get('id')}"):
                try:
                    import json
                    with open(debug_json_path, "r", encoding="utf-8") as f:
                        dbg = json.load(f)
                    st.json(dbg)
                except Exception:
                    pass

        # View Full Image
        if display_path and os.path.exists(display_path):
            if st.checkbox("Show Full Image", key=f"full_{img_rec.get('id')}"):
                st.image(display_path, use_container_width=True)

    # Action Buttons
    img_id = img_rec.get("id")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Approve", key=f"app_{img_id}", use_container_width=True):
            _manual_approve(img_rec, db, organizer)
            st.rerun()
    with b2:
        if st.button("Reject", key=f"rej_{img_id}", use_container_width=True):
            _manual_reject(img_rec, db, organizer)
            st.rerun()
    with b3:
        if status == "APPROVED":
            if st.button("Remove", key=f"rem_{img_id}", use_container_width=True):
                _manual_remove(img_rec, db, organizer)
                st.rerun()


def _manual_approve(img_rec: Dict[str, Any], db: DatabaseManager, organizer: ImageOrganizer):
    dup_detector = DuplicateDetector(db=db)
    # Check duplicate before approving
    is_dup, dup_type, dup_match, dup_sim = dup_detector.check_duplicate(
        filepath=img_rec.get("source_path"),
        sha256=img_rec.get("sha256"),
        phash_str=img_rec.get("phash"),
        category=img_rec.get("category")
    )

    if is_dup and dup_match and dup_match.get("id") != img_rec.get("id") and dup_match.get("status") == "APPROVED":
        st.error(f"Cannot approve! Matches existing approved image #{dup_match.get('id')} ({dup_type}).")
        return

    new_path = organizer.relocate_image(
        current_output_path=img_rec.get("output_path"),
        source_path=img_rec.get("source_path"),
        category_name=img_rec.get("category"),
        new_status="APPROVED",
        sha256_hash=img_rec.get("sha256")
    )
    db.update_image_status(img_rec["id"], new_status="APPROVED", new_output_path=new_path, reason="Manual user approval")


def _manual_reject(img_rec: Dict[str, Any], db: DatabaseManager, organizer: ImageOrganizer):
    new_path = organizer.relocate_image(
        current_output_path=img_rec.get("output_path"),
        source_path=img_rec.get("source_path"),
        category_name=img_rec.get("category"),
        new_status="REJECTED",
        sha256_hash=img_rec.get("sha256")
    )
    db.update_image_status(img_rec["id"], new_status="REJECTED", new_output_path=new_path, reason="Manual user rejection")


def _manual_remove(img_rec: Dict[str, Any], db: DatabaseManager, organizer: ImageOrganizer):
    # Remove from Approved folder (NEVER touch original source)
    organizer.remove_from_approved(img_rec.get("output_path"))
    db.update_image_status(img_rec["id"], new_status="REVIEW_REQUIRED", new_output_path=None, reason="Removed from approved folder")


def _handle_bulk_action(images: List[Dict[str, Any]], target_status: str, db: DatabaseManager, organizer: ImageOrganizer):
    for img_rec in images:
        if target_status == "APPROVED":
            _manual_approve(img_rec, db, organizer)
        elif target_status == "REJECTED":
            _manual_reject(img_rec, db, organizer)
