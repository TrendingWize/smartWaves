# smart_waves/pages/04_SEC_Analysis.py

import streamlit as st
from utils import initialize_session_state, check_concurrent_login
from utils import logout_user
from components.sec_filing_analysis import sec_filing_analysis_tab_content

# Page configuration
st.set_page_config(page_title="SEC Filing Analysis", layout="wide")

# Initialize session and check login
initialize_session_state()

st.title("📑 SEC Filing Analysis")

if check_concurrent_login():
    st.stop()

if not st.session_state.get("logged_in"):
    st.warning("🔒 Please log in to access SEC Filing Analysis.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/00_Login.py")
    st.stop()

# Top-right logout
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🚪 Logout"):
        logout_user()

# Render the SEC Filing UI
sec_filing_analysis_tab_content()
