"""
Reports UI View.
Allows generating, previewing, and downloading CSV and Excel audit reports
with complete hashes, blur scores, OCR extractions, and classification decisions.
"""

import streamlit as st
import pandas as pd
from database.db import DatabaseManager
from modules.report_generator import ReportGenerator


def render_reports(db: DatabaseManager, report_gen: ReportGenerator):
    st.title("Audit & Export Reports")
    st.markdown("Generate and download comprehensive spreadsheet reports for operations and accounting.")

    col1, col2 = st.columns(2)
    with col1:
        categories = ["All"] + [c.name for c in db.list_categories()]
        selected_cat = st.selectbox("Report Category:", categories)
    with col2:
        selected_status = st.selectbox("Report Status:", ["All", "APPROVED", "REVIEW_REQUIRED", "REJECTED", "DUPLICATE"])

    # Fetch matching records
    records = db.query_images(category=selected_cat, status=selected_status, limit=10000)

    st.write(f"Found **{len(records)}** records matching the criteria.")

    if not records:
        st.info("No records available to export.")
        return

    # Preview DataFrame
    df_preview = report_gen.prepare_dataframe(records)
    st.dataframe(df_preview.head(25), use_container_width=True)

    st.markdown("---")
    st.subheader("Download Reports")

    d_col1, d_col2 = st.columns(2)

    with d_col1:
        csv_path, csv_bytes = report_gen.export_csv(records, filename_prefix="decoration_report")
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"decoration_report_{selected_cat.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Saved locally to `{csv_path}`")

    with d_col2:
        excel_path, excel_bytes = report_gen.export_excel(records, filename_prefix="decoration_report")
        st.download_button(
            label="Download Excel Report (.xlsx)",
            data=excel_bytes,
            file_name=f"decoration_report_{selected_cat.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption(f"Saved locally to `{excel_path}`")
