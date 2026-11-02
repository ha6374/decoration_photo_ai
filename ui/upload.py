"""
File Upload & Ingestion UI View.
Accepts multiple ZIP archives and individual image uploads.
Executes non-destructive batch extraction, invokes the filter pipeline,
and updates database records with real-time progress.
"""

import os
import uuid
import streamlit as st
from typing import List

from database.db import DatabaseManager
from database.models import UploadBatch
from modules.zip_extractor import SafeZipExtractor
from modules.filter_engine import FilterEngine
from modules.image_organizer import ImageOrganizer


def render_upload(db: DatabaseManager, extractor: SafeZipExtractor, engine: FilterEngine, organizer: ImageOrganizer, input_dir: str):
    st.title("Upload & Process Decoration Photos")
    st.markdown("Safely ingest ZIP archives or individual decoration photographs for automated filtering.")

    # 1. Select Category
    categories = db.list_categories()
    if not categories:
        st.error("No categories found. Please create at least one category in 'Category Management' first.")
        return

    cat_names = [c.name for c in categories]
    selected_cat_name = st.selectbox("Select Target Event Category:", cat_names)
    selected_rule = db.get_category(selected_cat_name)

    st.markdown("---")

    # 2. Upload Interface
    upload_mode = st.radio("Choose Upload Type:", ["ZIP Archive(s)", "Individual Images"], horizontal=True)

    if upload_mode == "ZIP Archive(s)":
        uploaded_zips = st.file_uploader(
            "Upload one or more ZIP archives containing decoration photos:",
            type=["zip"],
            accept_multiple_files=True
        )

        if uploaded_zips and st.button("Start Processing ZIP Archives", type="primary"):
            for zip_file in uploaded_zips:
                batch_id = f"batch_{uuid.uuid4().hex[:10]}"
                batch_input_dir = os.path.join(input_dir, batch_id)
                os.makedirs(batch_input_dir, exist_ok=True)

                # Safe copy of uploaded zip to input storage (read-only original preservation)
                saved_zip_path = os.path.join(batch_input_dir, zip_file.name)
                with open(saved_zip_path, "wb") as f:
                    f.write(zip_file.getbuffer())

                st.info(f"Extracting '{zip_file.name}' into isolated workspace...")
                extracted_files, extract_errors = extractor.extract_zip(saved_zip_path, batch_id)

                if extract_errors:
                    st.warning(f"Extracted with {len(extract_errors)} warnings/corrupted items.")
                    with st.expander("Extraction Warnings"):
                        st.json(extract_errors)

                if not extracted_files:
                    st.error(f"No supported image files found in '{zip_file.name}'.")
                    continue

                # Process extracted images
                _execute_batch_processing(
                    extracted_files=extracted_files,
                    batch_id=batch_id,
                    source_name=zip_file.name,
                    category_rule=selected_rule,
                    db=db,
                    engine=engine,
                    organizer=organizer
                )

    else:
        uploaded_imgs = st.file_uploader(
            "Upload individual decoration photographs (JPG, PNG, WEBP, BMP, TIFF):",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
            accept_multiple_files=True
        )

        if uploaded_imgs and st.button("Start Processing Uploaded Images", type="primary"):
            batch_id = f"batch_{uuid.uuid4().hex[:10]}"
            batch_input_dir = os.path.join(input_dir, batch_id)
            os.makedirs(batch_input_dir, exist_ok=True)

            temp_paths = []
            for img_file in uploaded_imgs:
                temp_path = os.path.join(batch_input_dir, img_file.name)
                with open(temp_path, "wb") as f:
                    f.write(img_file.getbuffer())
                temp_paths.append(temp_path)

            extracted_files, ingest_errors = extractor.ingest_individual_files(temp_paths, batch_id)

            _execute_batch_processing(
                extracted_files=extracted_files,
                batch_id=batch_id,
                source_name=f"{len(uploaded_imgs)} individual images",
                category_rule=selected_rule,
                db=db,
                engine=engine,
                organizer=organizer
            )


def _execute_batch_processing(extracted_files: List[str], 
                              batch_id: str, 
                              source_name: str, 
                              category_rule, 
                              db: DatabaseManager, 
                              engine: FilterEngine, 
                              organizer: ImageOrganizer):
    total = len(extracted_files)
    st.subheader(f"Processing Batch: {batch_id} ({total} images)")

    # Register initial batch record in DB
    batch_record = UploadBatch(
        batch_id=batch_id,
        source_name=source_name,
        category=category_rule.name,
        total_images=total
    )
    db.create_batch(batch_record)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    approved_count = 0
    review_count = 0
    rejected_count = 0
    duplicate_count = 0

    for idx, filepath in enumerate(extracted_files):
        filename = os.path.basename(filepath)
        status_text.text(f"Analyzing [{idx + 1}/{total}]: {filename}...")

        # 1. Run multi-filter pipeline
        metadata = engine.process_image(
            filepath=filepath,
            category_rule=category_rule,
            batch_id=batch_id
        )

        # 2. Organize output file safely
        output_path = organizer.organize_image(
            source_path=metadata.source_path,
            category_name=metadata.category,
            status=metadata.status,
            sha256_hash=metadata.sha256
        )
        metadata.output_path = output_path

        # 3. Save to database
        db.insert_image(metadata)

        # Update running tallies
        if metadata.status == "APPROVED":
            approved_count += 1
        elif metadata.status == "REVIEW_REQUIRED":
            review_count += 1
        elif metadata.status == "REJECTED":
            rejected_count += 1
        elif metadata.status == "DUPLICATE":
            duplicate_count += 1

        progress_bar.progress((idx + 1) / total)

    # Finalize batch counts in DB
    db.update_batch_counts(batch_id)
    status_text.text("Batch processing complete!")

    st.success(
        f"Processing finished successfully!\n"
        f"Approved: {approved_count} | Review Required: {review_count} | "
        f"Rejected: {rejected_count} | Duplicates: {duplicate_count}"
    )
