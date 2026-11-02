"""
Category Management UI View.
Allows creating, editing, customizing, and deleting event decoration categories
with category-specific filter thresholds and keyword rules.
"""

import streamlit as st
from database.db import DatabaseManager
from database.models import CategoryRule


def render_category_manager(db: DatabaseManager):
    st.title("Category Management")
    st.markdown("Create and configure custom event categories with distinct filtering criteria.")

    categories = db.list_categories()
    tab_list, tab_create, tab_edit = st.tabs(["Active Categories", "Add New Category", "Edit / Delete"])

    # --- TAB 1: Active Categories Overview ---
    with tab_list:
        st.subheader("Configured Event Categories")
        if not categories:
            st.info("No categories configured yet.")
        else:
            for cat in categories:
                with st.expander(f"**{cat.name}** - {cat.description or 'No description'}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Required Text:** {', '.join(cat.required_text) if cat.required_text else 'None'}")
                        st.write(f"**Required Numbers:** {', '.join(cat.required_numbers) if cat.required_numbers else 'None'}")
                        st.write(f"**Preferred Angles:** {', '.join(cat.preferred_angles) if cat.preferred_angles else 'None'}")
                        st.write(f"**Human Detection:** {'Enabled (' + cat.human_action + ')' if cat.human_detection_enabled else 'Disabled'}")
                    with c2:
                        st.write(f"**Blur Threshold:** {cat.blur_threshold} (Variance)")
                        st.write(f"**Watermark Detection:** {'Enabled' if cat.watermark_detection_enabled else 'Disabled'}")
                        st.write(f"**Min Resolution:** {cat.min_width} x {cat.min_height} px")
                        st.write(f"**File Size Limits:** {cat.min_file_size_kb} KB to {cat.max_file_size_kb / 1024:.1f} MB")
                        if cat.custom_ai_prompt:
                            st.write(f"**AI Prompt:** '{cat.custom_ai_prompt}'")

    # --- TAB 2: Add New Category ---
    with tab_create:
        st.subheader("Create New Event Category")
        with st.form("create_category_form"):
            new_name = st.text_input("Category Name (e.g., '18th Birthday Decoration'):")
            new_desc = st.text_area("Description:", placeholder="Describe the type of event and typical decorations.")

            c1, c2 = st.columns(2)
            with c1:
                req_text = st.text_input("Required Text (comma-separated, e.g., 'Happy Birthday, Welcome'):")
                req_nums = st.text_input("Required Numbers/Words (comma-separated, e.g., '18, EIGHTEEN'):")
                custom_kws = st.text_input("Custom Keywords (comma-separated):")
                pref_angles = st.multiselect("Preferred Camera Angles:", ["FRONT", "SIDE", "OTHER"], default=["FRONT"])

            with c2:
                human_enabled = st.checkbox("Enable Human Detection", value=True)
                human_action = st.selectbox("Action on Human Detected:", ["REJECT", "REVIEW"], index=0)
                blur_enabled = st.checkbox("Enable Blur Detection", value=True)
                blur_thresh = st.slider("Blur Sharpness Threshold:", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
                watermark_enabled = st.checkbox("Enable Watermark Detection", value=True)

            c3, c4 = st.columns(2)
            with c3:
                min_w = st.number_input("Minimum Width (px):", value=400, step=50)
                min_h = st.number_input("Minimum Height (px):", value=400, step=50)
            with c4:
                min_kb = st.number_input("Min File Size (KB):", value=10.0, step=5.0)
                max_kb = st.number_input("Max File Size (KB):", value=50000.0, step=1000.0)

            ai_prompt = st.text_input("Custom AI Rule (optional natural language prompt):", placeholder="e.g. Balloon arch must be intact.")

            submitted = st.form_submit_button("Save Category", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Category name cannot be empty.")
                else:
                    new_rule = CategoryRule(
                        name=new_name.strip(),
                        description=new_desc.strip(),
                        required_text=[t.strip() for t in req_text.split(",") if t.strip()],
                        required_numbers=[n.strip() for n in req_nums.split(",") if n.strip()],
                        custom_keywords=[k.strip() for k in custom_kws.split(",") if k.strip()],
                        preferred_angles=pref_angles or ["FRONT"],
                        human_detection_enabled=human_enabled,
                        human_action=human_action,
                        blur_detection_enabled=blur_enabled,
                        blur_threshold=blur_thresh,
                        watermark_detection_enabled=watermark_enabled,
                        min_width=int(min_w),
                        min_height=int(min_h),
                        min_file_size_kb=float(min_kb),
                        max_file_size_kb=float(max_kb),
                        custom_ai_prompt=ai_prompt.strip()
                    )
                    success = db.save_category(new_rule, overwrite_if_exists=False)
                    if success:
                        st.success(f"Category '{new_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Category '{new_name}' already exists.")

    # --- TAB 3: Edit / Delete Category ---
    with tab_edit:
        st.subheader("Edit or Remove Category")
        if not categories:
            st.info("No categories to edit.")
            return

        cat_names = [c.name for c in categories]
        edit_target = st.selectbox("Select Category to Modify:", cat_names, key="edit_selector")
        target_rule = db.get_category(edit_target)

        if target_rule:
            with st.form("edit_category_form"):
                e_desc = st.text_area("Description:", value=target_rule.description)
                c1, c2 = st.columns(2)
                with c1:
                    e_text = st.text_input("Required Text (comma-separated):", value=", ".join(target_rule.required_text))
                    e_nums = st.text_input("Required Numbers/Words:", value=", ".join(target_rule.required_numbers))
                    e_kws = st.text_input("Custom Keywords:", value=", ".join(target_rule.custom_keywords))
                    e_angles = st.multiselect("Preferred Angles:", ["FRONT", "SIDE", "OTHER"], default=target_rule.preferred_angles)
                with c2:
                    e_human = st.checkbox("Enable Human Detection", value=target_rule.human_detection_enabled)
                    e_action = st.selectbox("Action on Human:", ["REJECT", "REVIEW"], index=0 if target_rule.human_action == "REJECT" else 1)
                    e_blur = st.checkbox("Enable Blur Detection", value=target_rule.blur_detection_enabled)
                    e_thresh = st.slider("Blur Threshold:", 10.0, 500.0, float(target_rule.blur_threshold), step=10.0)
                    e_wm = st.checkbox("Enable Watermark Detection", value=target_rule.watermark_detection_enabled)

                c3, c4 = st.columns(2)
                with c3:
                    e_min_w = st.number_input("Minimum Width (px):", value=target_rule.min_width)
                    e_min_h = st.number_input("Minimum Height (px):", value=target_rule.min_height)
                with c4:
                    e_min_kb = st.number_input("Min File Size (KB):", value=target_rule.min_file_size_kb)
                    e_max_kb = st.number_input("Max File Size (KB):", value=target_rule.max_file_size_kb)

                e_ai_prompt = st.text_input("Custom AI Rule:", value=target_rule.custom_ai_prompt)

                update_btn = st.form_submit_button("Update Category", type="primary")
                if update_btn:
                    updated_rule = CategoryRule(
                        name=target_rule.name,
                        description=e_desc.strip(),
                        required_text=[t.strip() for t in e_text.split(",") if t.strip()],
                        required_numbers=[n.strip() for n in e_nums.split(",") if n.strip()],
                        custom_keywords=[k.strip() for k in e_kws.split(",") if k.strip()],
                        preferred_angles=e_angles or ["FRONT"],
                        human_detection_enabled=e_human,
                        human_action=e_action,
                        blur_detection_enabled=e_blur,
                        blur_threshold=e_thresh,
                        watermark_detection_enabled=e_wm,
                        min_width=int(e_min_w),
                        min_height=int(e_min_h),
                        min_file_size_kb=float(e_min_kb),
                        max_file_size_kb=float(e_max_kb),
                        custom_ai_prompt=e_ai_prompt.strip()
                    )
                    db.save_category(updated_rule, overwrite_if_exists=True)
                    st.success(f"Category '{target_rule.name}' updated successfully!")
                    st.rerun()

            st.markdown("---")
            st.write("#### Danger Zone")
            if st.button(f"Delete Category '{target_rule.name}'", type="secondary"):
                deleted = db.delete_category(target_rule.name)
                if deleted:
                    st.warning(f"Category '{target_rule.name}' removed.")
                    st.rerun()
