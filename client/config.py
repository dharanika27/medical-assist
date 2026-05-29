import os

import streamlit as st

DEFAULT_API_URL = "https://medical-assist-6m80.onrender.com"


def get_api_url():
    api_url = os.getenv("API_URL") or st.secrets.get("API_URL", DEFAULT_API_URL)
    return api_url.rstrip("/")


API_URL = get_api_url()
