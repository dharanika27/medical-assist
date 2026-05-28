import streamlit as st
from utils.api import upload_pdfs_api

def render_uploader():

    st.sidebar.header("Upload Medical documents (.PDFs)")

    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.sidebar.button("Upload DB"):

        if uploaded_files:

            response = upload_pdfs_api(uploaded_files)

            if response.status_code == 200:
                st.sidebar.success("Files uploaded successfully!")

            else:
                st.sidebar.error(f"Upload failed: {response.text}")

        else:
            st.sidebar.warning("Please select PDF files first.")