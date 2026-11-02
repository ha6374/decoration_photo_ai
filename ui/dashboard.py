"""
Dashboard Overview UI View.
Displays aggregate statistics, classification metrics, category distributions,
and recent batch processing history.
"""

import streamlit as st
import pandas as pd
from database.db import DatabaseManager


def render_dashboard(db: DatabaseManager):
    st.title("Event Decoration AI Dashboard")
    st.markdown("Automated quality filtering, duplicate detection, and image classification system.")

    # Category filter for stats
    categories = ["All"] + [c.name for c in db.list_categories()]
    selected_cat = st.selectbox("Filter statistics by category:", categories, index=0)

    stats = db.get_statistics(category=selected_cat)

    # Key KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Uploaded", f"{stats['total_images']:,}")
    with col2:
        st.metric("Approved Images", f"{stats['approved']:,}", delta=f"{stats['success_rate']}% pass")
    with col3:
        st.metric("Review Required", f"{stats['review_required']:,}")
    with col4:
        st.metric("Rejected Images", f"{stats['rejected']:,}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Duplicate Images", f"{stats['duplicates']:,}")
    with col6:
        st.metric("Total Processed", f"{stats['total_processed']:,}")
    with col7:
        st.metric("Approval Rate", f"{stats['success_rate']}%")
    with col8:
        st.metric("Active Categories", len(categories) - 1)

    st.markdown("---")

    # Visual Status Breakdown
    st.subheader("Classification Distribution")
    dist_data = {
        "Status": ["Approved", "Review Required", "Rejected", "Duplicates"],
        "Count": [stats["approved"], stats["review_required"], stats["rejected"], stats["duplicates"]]
    }
    df_dist = pd.DataFrame(dist_data)

    chart_col, info_col = st.columns([2, 1])
    with chart_col:
        st.bar_chart(df_dist.set_index("Status"))
    with info_col:
        st.info(
            """
            **Status Definitions:**
            * **Approved**: Passed all mandatory sharpness, angle, human, watermark, and OCR tests.
            * **Review Required**: Low confidence detection or requires visual audit.
            * **Rejected**: Failed one or more mandatory category criteria.
            * **Duplicates**: Identified as duplicate across current or past batches.
            """
        )

    st.markdown("---")

    # Recent Upload Batches
    st.subheader("Recent Processing Batches")
    batches = db.list_batches()
    if batches:
        df_batches = pd.DataFrame(batches)
        rename_cols = {
            "batch_id": "Batch ID",
            "source_name": "Source / Filename",
            "category": "Category",
            "total_images": "Total",
            "approved_count": "Approved",
            "review_count": "Review Req",
            "rejected_count": "Rejected",
            "duplicate_count": "Duplicates",
            "created_at": "Timestamp"
        }
        df_batches = df_batches.rename(columns=rename_cols)
        st.dataframe(df_batches, use_container_width=True)
    else:
        st.caption("No upload batches recorded yet. Navigate to 'Upload Files' to begin processing.")
