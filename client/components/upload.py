import streamlit as st
import requests

from utils.api import upload_pdfs_api


def render_uploader():
    st.sidebar.header("Upload Medical documents (.PDFs)")
    uploaded_files=st.sidebar.file_uploader("Upload multiple PDFs",type="pdf",accept_multiple_files=True)
    if st.sidebar.button("Upload DB") and uploaded_files:
        try:
            response=upload_pdfs_api(uploaded_files)
            if response.status_code==200:
                st.sidebar.success("Uploaded successfully")
            else:
                st.sidebar.error(f"Error:{response.text}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Could not connect to backend API: {e}")
