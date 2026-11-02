"""
Processing Results & Evidence Debugging UI View.
Displays detailed breakdown of batch processing outcomes, rejection reasons analysis,
and an interactive OCR & AI Evidence Debugger showing bounding boxes, confidence scores, and structured JSON logs.
"""

import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any
from PIL import Image, ImageDraw

from database.db import DatabaseManager
from modules.image_loader import ImageLoader


def render_processing_results(db: DatabaseManager):
    st.title("Batch Processing Results & AI Diagnostics")
    st.markdown("Detailed breakdown of automated filtering decisions, rejection causes, and forensic AI evidence logs.")

    tab_summary, tab_debugger = st.tabs(["Batch Analytics & Table", "OCR & AI Evidence Inspector"])

    # Select batch or view all
    batches = db.list_batches()
    batch_options = ["All Batches"] + [f"{b['batch_id']} - {b['source_name']}" for b in batches]

    with tab_summary:
        selected_batch_str = st.selectbox("Select Batch to Inspect:", batch_options, index=0)

        target_batch_id = None
        if selected_batch_str != "All Batches":
            target_batch_id = selected_batch_str.split(" - ")[0]

        # Category and status filters
        c1, c2 = st.columns(2)
        with c1:
            categories = ["All"] + [c.name for c in db.list_categories()]
            selected_cat = st.selectbox("Category Filter:", categories, key="res_cat")
        with c2:
            selected_status = st.selectbox("Status Filter:", ["All", "APPROVED", "REVIEW_REQUIRED", "REJECTED", "DUPLICATE"], key="res_status")

        images = db.query_images(category=selected_cat, status=selected_status, batch_id=target_batch_id, limit=500)

        if not images:
            st.info("No records found matching the selected batch and filters.")
        else:
            st.write(f"Displaying **{len(images)}** processed image records:")

            # Summary Rejection Reasons Chart
            rejection_causes = {}
            for img in images:
                reasons = img.get("rejection_reasons", [])
                for r in reasons:
                    label = r.split(":")[0] if ":" in r else r[:40]
                    rejection_causes[label] = rejection_causes.get(label, 0) + 1

            if rejection_causes:
                st.subheader("Top Decision & Rejection Causes")
                df_causes = pd.DataFrame(list(rejection_causes.items()), columns=["Reason", "Count"]).sort_values("Count", ascending=False)
                st.bar_chart(df_causes.set_index("Reason"))

            st.markdown("---")

            # Detailed Table
            st.subheader("Detailed Records Table")
            table_rows = []
            for r in images:
                table_rows.append({
                    "ID": r["id"],
                    "Filename": r["original_filename"],
                    "Category": r["category"],
                    "Status": r["status"],
                    "Blur Score": r["blur_score"],
                    "Angle": r["detected_angle"],
                    "Human": "Yes" if r["human_detected"] else "No",
                    "Watermark": "Yes" if r["watermark_detected"] else "No",
                    "Text": r["detected_text"][:35] if r["detected_text"] else "",
                    "Numbers": ", ".join(r["detected_numbers"]) if r["detected_numbers"] else "",
                    "Decision Reasons": "; ".join(r["rejection_reasons"]) if r["rejection_reasons"] else "",
                    "Processed At": r["first_seen_at"]
                })

            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True)

    with tab_debugger:
        st.subheader("Interactive OCR & Evidence Debugger")
        st.markdown("Select an image to inspect its OCR tokens, bounding boxes, semantic scene probabilities, and full decision JSON.")

        debug_images = db.query_images(limit=100)
        if not debug_images:
            st.info("No processed images available for debugging.")
            return

        img_choices = {f"#{im['id']} - {im['original_filename']} ({im['status']})": im for im in debug_images}
        chosen_label = st.selectbox("Select Image to Audit:", list(img_choices.keys()))
        selected_img_rec = img_choices[chosen_label]

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        short_hash = selected_img_rec["sha256"][:12]
        debug_json_path = os.path.join(base_dir, "logs", "debug", f"debug_{short_hash}.json")

        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("#### Image Preview")
            img_path = selected_img_rec.get("output_path") or selected_img_rec.get("source_path")
            if img_path and os.path.exists(img_path):
                # Load PIL image
                pil_disp = Image.open(img_path).convert("RGB")

                # If debug JSON has bounding boxes, allow user to overlay them
                show_boxes = st.checkbox("Overlay OCR & Number Bounding Boxes", value=True)
                if show_boxes and os.path.exists(debug_json_path):
                    try:
                        with open(debug_json_path, "r", encoding="utf-8") as f:
                            dbg_data = json.load(f)

                        draw = ImageDraw.Draw(pil_disp)
                        # Draw OCR boxes (green)
                        raw_tokens = dbg_data.get("ocr", {}).get("raw_tokens", [])
                        for tok in raw_tokens:
                            box = tok.get("bbox")
                            if box and len(box) == 4:
                                pts = [(p[0], p[1]) for p in box]
                                draw.polygon(pts, outline=(0, 255, 0), width=3)
                                draw.text((pts[0][0], max(0, pts[0][1] - 12)), tok.get("text", "")[:15], fill=(0, 255, 0))

                        # Draw Number boxes (cyan)
                        num_ev = dbg_data.get("required_numbers", {}).get("evidence", [])
                        for ne in num_ev:
                            bbox = ne.get("bbox")
                            if bbox and len(bbox) == 4:
                                if isinstance(bbox[0], list):
                                    pts = [(p[0], p[1]) for p in bbox]
                                    draw.polygon(pts, outline=(0, 255, 255), width=4)
                                else:
                                    draw.rectangle(bbox, outline=(0, 255, 255), width=4)
                                draw.text((bbox[0], max(0, bbox[1] - 14)), f"NUM {ne.get('canonical')}", fill=(0, 255, 255))
                    except Exception as e:
                        st.caption(f"Could not render bounding boxes: {e}")

                st.image(pil_disp, caption=f"ID #{selected_img_rec['id']} - {selected_img_rec['status']}", use_container_width=True)
            else:
                st.warning("Original image file not found on disk.")

        with d_col2:
            st.markdown("#### Diagnostic Evidence Findings")
            st.write(f"- **Final Decision:** `{selected_img_rec['status']}`")
            st.write(f"- **Sharpness Blur Score:** `{selected_img_rec['blur_score']}`")
            st.write(f"- **Detected Camera Angle:** `{selected_img_rec['detected_angle']}`")
            st.write(f"- **Detected Numbers:** `{selected_img_rec.get('detected_numbers')}`")
            st.write(f"- **OCR Detected Text:**")
            st.code(selected_img_rec.get("detected_text") or "None")

            st.write("**Decision Explanations:**")
            for r in selected_img_rec.get("rejection_reasons", []):
                st.info(f"• {r}")

            st.markdown("#### Structured Forensic JSON")
            if os.path.exists(debug_json_path):
                try:
                    with open(debug_json_path, "r", encoding="utf-8") as f:
                        dbg_data = json.load(f)
                    st.json(dbg_data)
                except Exception as ex:
                    st.error(f"Error parsing debug JSON: {ex}")
            else:
                # Synthesize on-the-fly debug summary if file wasn't created prior
                st.json({
                    "image": selected_img_rec["original_filename"],
                    "status": selected_img_rec["status"],
                    "blur_score": selected_img_rec["blur_score"],
                    "detected_text": selected_img_rec["detected_text"],
                    "detected_numbers": selected_img_rec["detected_numbers"],
                    "reasons": selected_img_rec["rejection_reasons"]
                })
