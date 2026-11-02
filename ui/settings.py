"""
Settings & Integrations UI View.
Manages global duplicate detection sensitivities, AI model selections,
Google Drive cloud backups, and storage paths.
"""

import os
import streamlit as st
from database.db import DatabaseManager
from modules.gdrive_integration import GoogleDriveIntegration


def render_settings(db: DatabaseManager, output_dir: str):
    st.title("System Settings & Cloud Integration")
    st.markdown("Configure global parameters, AI detection backends, and Google Drive destinations.")

    tab_models, tab_dups, tab_gdrive, tab_system = st.tabs([
        "AI Models & Backends",
        "Duplicate Engine Rules",
        "Google Drive Cloud",
        "System & Paths"
    ])

    # --- TAB 1: AI Models & Backends ---
    with tab_models:
        st.subheader("Configurable Computer Vision Models")

        st.markdown("##### 1. Human Detection Backend")
        st.write("Current Default: **YOLOv8 Nano (Ultralytics)** with automatic fallback to **OpenCV HOG**.")
        model_choice = st.radio("Primary Human Detector:", ["Auto (YOLOv8 with HOG fallback)", "YOLOv8 Nano Only", "OpenCV HOG Only"])
        st.caption("YOLOv8 offers deep neural detection for occluded bodies, while OpenCV HOG is CPU-lightweight.")

        st.markdown("---")
        st.markdown("##### 2. OCR Text Recognition Engine")
        st.write("Current Backend: **EasyOCR (CRAFT + ResNet)** with PyTesseract fallback.")
        st.info("EasyOCR operates locally without cloud dependencies and extracts English words & digits accurately.")

        st.markdown("---")
        st.markdown("##### 3. Camera Angle Analyzer")
        st.write("Uses bilateral symmetry index and Hough perspective line convergence to isolate front-facing setups.")

    # --- TAB 2: Duplicate Engine Rules ---
    with tab_dups:
        st.subheader("Global Duplicate Matching Sensitivity")
        st.write("Controls how strictly images are flagged as duplicates across upload batches.")

        phash_tol = st.slider("Perceptual Hash (pHash) Hamming Distance Threshold:", min_value=0, max_value=16, value=6, step=1)
        st.caption("Distance 0 = identical visual pixels; Distance ≤ 6 detects re-compression, resizing, and watermarked copies.")

        sim_tol = st.slider("Visual / Color Histogram Similarity Threshold:", min_value=0.70, max_value=0.99, value=0.88, step=0.01)
        st.caption("Used for loose similarity checks on crops and edited variations.")

        global_scope = st.checkbox("Enforce Global Duplicate Checking across all categories", value=True)
        st.caption("When enabled, prevents an approved wedding image from being re-approved under anniversary.")

    # --- TAB 3: Google Drive Cloud Integration ---
    with tab_gdrive:
        st.subheader("Google Drive Synchronization")
        st.markdown(
            """
            Synchronize only **Approved** decoration photos to Google Drive.
            - **Data Safety**: Read-only local access, never deletes existing Drive files.
            - **Security**: No hardcoded secrets. Use service account JSON credentials.
            """
        )

        with st.form("gdrive_config_form"):
            gdrive_json = st.text_input(
                "Service Account JSON Filepath:",
                value=os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "credentials/gdrive_service_account.json"),
                placeholder="C:/path/to/credentials.json"
            )
            gdrive_folder = st.text_input(
                "Google Drive Destination Folder ID:",
                value=os.getenv("GDRIVE_FOLDER_ID", ""),
                placeholder="e.g. 1a2b3c4d5e6f7g8h9i"
            )
            save_gdrive = st.form_submit_button("Test & Validate Google Drive Connection", type="primary")

            if save_gdrive:
                os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"] = gdrive_json.strip()
                os.environ["GDRIVE_FOLDER_ID"] = gdrive_folder.strip()

                gdrive_client = GoogleDriveIntegration(
                    credentials_path=gdrive_json.strip(),
                    destination_folder_id=gdrive_folder.strip()
                )
                success, msg = gdrive_client.test_connection()
                if success:
                    st.success(msg)
                else:
                    st.warning(f"Connection test: {msg}")

        st.markdown("---")
        st.markdown("##### Manual Sync Approved Images to Google Drive")
        if st.button("Push Approved Images to Drive"):
            approved_records = db.query_images(status="APPROVED", limit=50)
            if not approved_records:
                st.info("No approved images to upload.")
            else:
                gdrive_client = GoogleDriveIntegration(
                    credentials_path=os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON"),
                    destination_folder_id=os.getenv("GDRIVE_FOLDER_ID")
                )
                if not gdrive_client.is_connected:
                    st.error("Google Drive is not authenticated. Please provide credentials above.")
                else:
                    upload_progress = st.progress(0.0)
                    for idx, rec in enumerate(approved_records):
                        out_path = rec.get("output_path")
                        if out_path and os.path.exists(out_path):
                            gdrive_client.upload_approved_image(out_path)
                        upload_progress.progress((idx + 1) / len(approved_records))
                    st.success("Sync completed.")

    # --- TAB 4: System & Storage Paths ---
    with tab_system:
        st.subheader("Application Directory Structure")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        st.write(f"**App Root:** `{base_dir}`")
        st.write(f"**Database File:** `{db.db_path}`")
        st.write(f"**Output Directory:** `{output_dir}`")

        # Database Health Check
        with st.expander("Database Status"):
            with db.get_connection() as conn:
                tbls = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                st.write(f"Active SQLite Tables: {', '.join([t['name'] for t in tbls])}")
