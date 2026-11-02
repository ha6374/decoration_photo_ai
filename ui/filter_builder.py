"""
Dynamic Filter Builder UI View.
Allows fine-tuning and testing all 10 filter modules interactively.
Supports instant single-image sandbox testing to observe detection confidence before running big batches.
"""

import os
import streamlit as st
import tempfile
from PIL import Image

from database.db import DatabaseManager
from database.models import CategoryRule
from modules.filter_engine import FilterEngine
from modules.image_loader import ImageLoader


def render_filter_builder(db: DatabaseManager, engine: FilterEngine):
    st.title("Dynamic Filter Builder & Sandbox")
    st.markdown("Configure and simulate filter rules on sample decoration photographs before running production batches.")

    categories = db.list_categories()
    if not categories:
        st.error("No categories found.")
        return

    cat_names = [c.name for c in categories]
    selected_name = st.selectbox("Select Category Ruleset to Tune:", cat_names)
    rule = db.get_category(selected_name)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Filter Pipeline Parameters")
        with st.form("tune_filter_form"):
            st.markdown("##### 1. Human Detection")
            h_on = st.checkbox("Enable Human Detection", value=rule.human_detection_enabled)
            h_act = st.selectbox("Action on Human Detected", ["REJECT", "REVIEW"], index=0 if rule.human_action == "REJECT" else 1)

            st.markdown("##### 2. Blur / Sharpness")
            b_on = st.checkbox("Enable Blur Check", value=rule.blur_detection_enabled)
            b_thresh = st.slider("Min Sharpness (Laplacian Var)", 10.0, 500.0, float(rule.blur_threshold), step=10.0)

            st.markdown("##### 3. Watermarks & Logos")
            wm_on = st.checkbox("Enable Watermark Check", value=rule.watermark_detection_enabled)

            st.markdown("##### 4. Camera Angle")
            ang_on = st.checkbox("Enable Angle Check", value=rule.angle_filter_enabled)
            pref_ang = st.multiselect("Preferred Angles", ["FRONT", "SIDE", "OTHER"], default=rule.preferred_angles)

            st.markdown("##### 5. OCR Text & Numbers")
            req_txt = st.text_input("Required Text (comma-separated):", value=", ".join(rule.required_text))
            req_num = st.text_input("Required Numbers/Words:", value=", ".join(rule.required_numbers))
            c_kws = st.text_input("Custom Keywords:", value=", ".join(rule.custom_keywords))

            st.markdown("##### 6. Resolution & File Size")
            c1, c2 = st.columns(2)
            with c1:
                mw = st.number_input("Min Width (px)", value=rule.min_width, step=50)
                mh = st.number_input("Min Height (px)", value=rule.min_height, step=50)
            with c2:
                mkb = st.number_input("Min Size (KB)", value=rule.min_file_size_kb, step=5.0)
                maxkb = st.number_input("Max Size (KB)", value=rule.max_file_size_kb, step=1000.0)

            st.markdown("##### 7. Custom AI Rule")
            ai_rule = st.text_input("Natural Language Requirement:", value=rule.custom_ai_prompt)

            save_btn = st.form_submit_button("Save Ruleset to Category", type="primary")
            if save_btn:
                updated = CategoryRule(
                    name=rule.name,
                    description=rule.description,
                    required_text=[t.strip() for t in req_txt.split(",") if t.strip()],
                    required_numbers=[n.strip() for n in req_num.split(",") if n.strip()],
                    custom_keywords=[k.strip() for k in c_kws.split(",") if k.strip()],
                    preferred_angles=pref_ang or ["FRONT"],
                    human_detection_enabled=h_on,
                    human_action=h_act,
                    blur_detection_enabled=b_on,
                    blur_threshold=b_thresh,
                    watermark_detection_enabled=wm_on,
                    angle_filter_enabled=ang_on,
                    min_width=int(mw),
                    min_height=int(mh),
                    min_file_size_kb=float(mkb),
                    max_file_size_kb=float(maxkb),
                    custom_ai_prompt=ai_rule.strip()
                )
                db.save_category(updated, overwrite_if_exists=True)
                st.success(f"Rules saved to '{rule.name}'!")
                st.rerun()

    with col_right:
        st.subheader("Test Sandbox")
        st.markdown("Upload a single image to test the current filtering rules.")
        test_file = st.file_uploader("Upload test image:", type=["jpg", "jpeg", "png", "webp", "bmp"], key="sandbox_upload")

        if test_file:
            # Temporary file write
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{test_file.name}") as tmp:
                tmp.write(test_file.getbuffer())
                tmp_path = tmp.name

            try:
                st.image(tmp_path, caption="Uploaded Test Photo", use_container_width=True)

                if st.button("Run Simulation Filter", type="primary"):
                    with st.spinner("Analyzing photo through filter pipeline..."):
                        sim_metadata = engine.process_image(
                            filepath=tmp_path,
                            category_rule=rule,
                            batch_id="sandbox_test"
                        )

                    st.markdown("### Simulation Result")
                    if sim_metadata.status == "APPROVED":
                        st.success("STATUS: APPROVED")
                    elif sim_metadata.status == "REVIEW_REQUIRED":
                        st.warning("STATUS: REVIEW REQUIRED")
                    elif sim_metadata.status == "REJECTED":
                        st.error("STATUS: REJECTED")
                    elif sim_metadata.status == "DUPLICATE":
                        st.info("STATUS: DUPLICATE")

                    st.write("**Evaluation Details:**")
                    st.write(f"- **Sharpness Blur Score:** {sim_metadata.blur_score} (Threshold: {rule.blur_threshold})")
                    st.write(f"- **Human Detected:** {sim_metadata.human_detected} (Count: {sim_metadata.human_count})")
                    st.write(f"- **Watermark Detected:** {sim_metadata.watermark_detected} (Conf: {sim_metadata.watermark_confidence})")
                    st.write(f"- **Camera Angle:** {sim_metadata.detected_angle} (Conf: {sim_metadata.angle_confidence})")
                    st.write(f"- **OCR Text:** `{sim_metadata.detected_text or 'No text detected'}`")
                    st.write(f"- **Detected Numbers:** {sim_metadata.detected_numbers}")
                    st.write(f"- **Resolution:** {sim_metadata.width} x {sim_metadata.height}")
                    st.write(f"- **Decision Reasons:**")
                    for r in sim_metadata.rejection_reasons:
                        st.write(f"  • {r}")

                    # Read generated debug json
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    short_hash = sim_metadata.sha256[:12] if sim_metadata.sha256 else ""
                    debug_json_path = os.path.join(base_dir, "logs", "debug", f"debug_{short_hash}.json")
                    if os.path.exists(debug_json_path):
                        with st.expander("View Full Structured AI Diagnostic JSON"):
                            import json
                            with open(debug_json_path, "r", encoding="utf-8") as jf:
                                st.json(json.load(jf))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
