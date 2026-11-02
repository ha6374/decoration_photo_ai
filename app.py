
import streamlit as st
import pandas as pd
from pathlib import Path

from modules.image_processor import process_zip

st.set_page_config(
    page_title="Image Quality Checker",
    page_icon="🖼️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

ACCEPTED_DIR = BASE_DIR / "output" / "accepted"
REJECTED_DIR = BASE_DIR / "output" / "rejected"


st.title("🖼️ Image Quality Checker")
st.caption("AI-powered image inspection and filtering system")

st.divider()

st.subheader("📁 Upload Images")

uploaded_file = st.file_uploader(
    "Upload a ZIP file containing images",
    type=["zip"]
)

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🚀 Start Image Inspection", type="primary"):

        try:
            with st.spinner("Analyzing images..."):

                stats, results = process_zip(uploaded_file)

                st.session_state["stats"] = stats
                st.session_state["results"] = results

            st.success("Image processing completed!")

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Processing failed: {error}")


if "stats" in st.session_state:

    stats = st.session_state["stats"]
    results = st.session_state["results"]

    st.divider()
    st.subheader("📊 Processing Results")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Images", stats["total"])
    col2.metric("Accepted", stats["accepted"])
    col3.metric("Rejected", stats["rejected"])
    col4.metric("Human Detected", stats["human_detected"])
    col5.metric("Invalid", stats["invalid"])

    st.divider()

    # Accepted and Rejected Tabs
    accepted_tab, rejected_tab, report_tab = st.tabs([
        f"✅ Accepted ({stats['accepted']})",
        f"❌ Rejected ({stats['rejected']})",
        "📋 Full Report"
    ])

    
    with accepted_tab:

        st.subheader("Accepted Images")
        st.caption("Images that are sharp and have no detected human presence.")

        accepted_images = list(ACCEPTED_DIR.glob("*"))

        if accepted_images:

            cols = st.columns(3)

            for index, image_path in enumerate(accepted_images):

                with cols[index % 3]:

                    st.image(
                        str(image_path),
                        caption=image_path.name,
                        use_container_width=True
                    )

                    if st.button(
                        "🗑️ Delete Image",
                        key=f"delete_{image_path.name}",
                        use_container_width=True
                    ):

                        image_path.unlink()

                        st.toast(
                            f"{image_path.name} deleted successfully!"
                        )

                        st.rerun()

        else:
            st.info("No accepted images found.")


    with rejected_tab:

        st.subheader("Rejected Images")
        st.caption("Images rejected due to blur or detected human presence.")

        rejected_images = list(REJECTED_DIR.glob("*"))

        if rejected_images:

            cols = st.columns(3)

            for index, image_path in enumerate(rejected_images):

                with cols[index % 3]:

                    st.image(
                        str(image_path),
                        caption=image_path.name,
                        use_container_width=True
                    )

        else:
            st.info("No rejected images found.")

    with report_tab:

        st.subheader("Complete Processing Report")

        if results:

            dataframe = pd.DataFrame(results)

            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No images available.")

else:

    st.info("Upload a ZIP file to start image inspection.")

st.divider()

st.caption(
    "Image Quality Checker | Python • Streamlit • OpenCV"
)
