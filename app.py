"""
Main Application Entry Point for AI Decoration Photo Filtering and Management Application.
Event decoration photo curation, multi-level global duplicate detection,
and automated quality assessment pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so modules and database packages import cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st

from database.db import DatabaseManager
from modules.zip_extractor import SafeZipExtractor
from modules.filter_engine import FilterEngine
from modules.image_organizer import ImageOrganizer
from modules.report_generator import ReportGenerator

# UI Views
from ui.dashboard import render_dashboard
from ui.upload import render_upload
from ui.category_manager import render_category_manager
from ui.filter_builder import render_filter_builder
from ui.results_view import render_processing_results
from ui.review_gallery import render_review_gallery
from ui.duplicate_manager import render_duplicate_manager
from ui.reports_view import render_reports
from ui.settings import render_settings


# Configure Streamlit page
st.set_page_config(
    page_title="Decoration Photo AI Filter & Manager",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Establish Directory Paths
INPUT_DIR = os.path.join(BASE_DIR, "input")
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "Reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for directory in (INPUT_DIR, EXTRACTED_DIR, OUTPUT_DIR, REPORTS_DIR, LOGS_DIR):
    os.makedirs(directory, exist_ok=True)


@st.cache_resource
def get_database() -> DatabaseManager:
    db_path = os.path.join(BASE_DIR, "decoration_app.db")
    return DatabaseManager(db_path=db_path)


@st.cache_resource
def get_services():
    db = get_database()
    extractor = SafeZipExtractor(extraction_base_dir=EXTRACTED_DIR)
    engine = FilterEngine(db=db)
    organizer = ImageOrganizer(output_base_dir=OUTPUT_DIR)
    report_gen = ReportGenerator(reports_base_dir=REPORTS_DIR)
    return extractor, engine, organizer, report_gen


def main():
    db = get_database()
    extractor, engine, organizer, report_gen = get_services()

    # Sidebar Navigation
    st.sidebar.title("Decoration AI")
    st.sidebar.caption("Intelligent Image Filtering & Deduplication")

    sections = [
        "Dashboard",
        "Upload Files",
        "Category Management",
        "Filter Configuration",
        "Processing Results",
        "Manual Review Gallery",
        "Duplicate Management",
        "Reports",
        "Settings"
    ]

    selected_section = st.sidebar.radio("Navigation:", sections, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Status:**
        - **Storage:** Local Persistent SQLite
        - **Data Safety:** Read-Only Source Copies
        - **Deduplication:** Global Multi-Level
        """
    )

    # Route to Selected Section
    if selected_section == "Dashboard":
        render_dashboard(db)
    elif selected_section == "Upload Files":
        render_upload(db, extractor, engine, organizer, INPUT_DIR)
    elif selected_section == "Category Management":
        render_category_manager(db)
    elif selected_section == "Filter Configuration":
        render_filter_builder(db, engine)
    elif selected_section == "Processing Results":
        render_processing_results(db)
    elif selected_section == "Manual Review Gallery":
        render_review_gallery(db, organizer)
    elif selected_section == "Duplicate Management":
        render_duplicate_manager(db, organizer)
    elif selected_section == "Reports":
        render_reports(db, report_gen)
    elif selected_section == "Settings":
        render_settings(db, OUTPUT_DIR)


if __name__ == "__main__":
    main()
