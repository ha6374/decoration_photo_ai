"""
Duplicate Management UI View.
Displays duplicate records across all historical upload batches,
shows side-by-side comparisons with canonical original images,
and provides an on-demand duplicate check tool.
"""

import os
import streamlit as st
import tempfile
from typing import Dict, Any

from database.db import DatabaseManager
from modules.duplicate_detector import DuplicateDetector
from modules.image_loader import ImageLoader
from modules.image_organizer import ImageOrganizer


def render_duplicate_manager(db: DatabaseManager, organizer: ImageOrganizer):
    st.title("Global Duplicate Management")
    st.markdown("Inspect and audit duplicate detections across all historical batches.")

    tab_overview, tab_lookup = st.tabs(["Duplicate Records", "On-Demand Duplicate Lookup"])

    # --- TAB 1: Duplicate Records ---
    with tab_overview:
        dup_images = db.query_images(status="DUPLICATE", limit=100)

        if not dup_images:
            st.info("No duplicate images detected yet in the global database.")
        else:
            st.write(f"Found **{len(dup_images)}** duplicate images:")

            for dup_rec in dup_images:
                with st.container():
                    st.markdown("---")
                    c_dup, c_orig, c_meta = st.columns([1, 1, 1])

                    # 1. Duplicate Image preview
                    with c_dup:
                        st.caption(f"**Duplicate Item (#{dup_rec.get('id')})**")
                        st.write(f"*{dup_rec.get('original_filename')}*")
                        dup_path = dup_rec.get("output_path") or dup_rec.get("source_path")
                        if dup_path and os.path.exists(dup_path):
                            thumb = ImageLoader.get_thumbnail(dup_path, max_size=(240, 200))
                            if thumb:
                                st.image(thumb, use_container_width=True)

                    # 2. Original / Canonical Image preview
                    with c_orig:
                        canonical_id = dup_rec.get("duplicate_of_id")
                        canonical_rec = db.get_image_by_id(canonical_id) if canonical_id else None

                        if canonical_rec:
                            st.caption(f"**Canonical Original (#{canonical_id})**")
                            st.write(f"*{canonical_rec.get('original_filename')}*")
                            canon_path = canonical_rec.get("output_path") or canonical_rec.get("source_path")
                            if canon_path and os.path.exists(canon_path):
                                thumb = ImageLoader.get_thumbnail(canon_path, max_size=(240, 200))
                                if thumb:
                                    st.image(thumb, use_container_width=True)
                        else:
                            st.caption("**Canonical Match**")
                            st.info("Primary record or path not available.")

                    # 3. Match Details & Action
                    with c_meta:
                        st.caption("**Detection Diagnostics**")
                        st.write(f"**Category:** {dup_rec.get('category')}")
                        st.write(f"**Batch:** {dup_rec.get('upload_batch_id')}")
                        st.write(f"**Similarity Score:** {dup_rec.get('duplicate_similarity', 0.0):.2f}")
                        if dup_rec.get("rejection_reasons"):
                            st.caption(f"Reason: {dup_rec.get('rejection_reasons')[0]}")

                        # Action to override if user asserts this is distinct
                        if st.button("Mark as Distinct (Send to Review)", key=f"dist_{dup_rec.get('id')}"):
                            new_path = organizer.relocate_image(
                                current_output_path=dup_rec.get("output_path"),
                                source_path=dup_rec.get("source_path"),
                                category_name=dup_rec.get("category"),
                                new_status="REVIEW_REQUIRED",
                                sha256_hash=dup_rec.get("sha256")
                            )
                            db.update_image_status(dup_rec["id"], "REVIEW_REQUIRED", new_output_path=new_path, reason="User override from Duplicate to Review")
                            st.success("Record moved to Review Required.")
                            st.rerun()

    # --- TAB 2: On-Demand Duplicate Lookup ---
    with tab_lookup:
        st.subheader("Lookup Candidate Image Against Database")
        st.markdown("Upload any single photo to check if it already exists in past batches.")

        lookup_file = st.file_uploader("Upload image to query:", type=["jpg", "jpeg", "png", "webp", "bmp"], key="lookup_upload")
        if lookup_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{lookup_file.name}") as tmp:
                tmp.write(lookup_file.getbuffer())
                tmp_path = tmp.name

            try:
                st.image(tmp_path, width=280)
                if st.button("Run Global Duplicate Check", type="primary"):
                    info = ImageLoader.inspect_image_file(tmp_path)
                    if info:
                        detector = DuplicateDetector(db=db)
                        is_dup, dup_type, matched, sim_score = detector.check_duplicate(
                            filepath=tmp_path,
                            sha256=info["sha256"],
                            phash_str=info["phash"],
                            category="All"
                        )

                        if is_dup:
                            st.error(f"DUPLICATE DETECTED! Match Type: {dup_type} (Similarity: {sim_score:.2f})")
                            if matched:
                                st.write(f"- **Matches Image ID:** #{matched.get('id')}")
                                st.write(f"- **Original Filename:** {matched.get('original_filename')}")
                                st.write(f"- **Category:** {matched.get('category')}")
                                st.write(f"- **Original Status:** {matched.get('status')}")
                                st.write(f"- **Batch:** {matched.get('upload_batch_id')}")
                                match_path = matched.get("output_path") or matched.get("source_path")
                                if match_path and os.path.exists(match_path):
                                    st.image(match_path, caption="Matched Original in DB", width=300)
                        else:
                            st.success("No duplicate detected! This image is unique across all database records.")
                            st.write(f"- **SHA-256:** `{info['sha256']}`")
                            st.write(f"- **pHash:** `{info['phash']}`")
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
