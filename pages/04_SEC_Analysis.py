"""
Streamlit page 03 – AI Analysis
Uses the shared tab renderer from components.ai_analysis.
"""

from __future__ import annotations

import streamlit as st
from styles import load_global_css
from utils.user_auth import (
    initialize_session_state,
    check_concurrent_login,
    logout_user,
)
from components.sec_filing_analysis import sec_filing_analysis_tab_content

# ——— page setup ————————————————————————————————————————————————
st.set_page_config(page_title="AI Analysis", layout="wide")
load_global_css()                       # inject once per page

initialize_session_state()

st.title("🤖 AI Analysis")

if check_concurrent_login():
    st.stop()

if not st.session_state.get("logged_in"):
    st.warning("🔒 Please log in to access AI Analysis.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/97_Login.py")
    st.stop()

# top-right logout
col_main, col_logout = st.columns([8, 1])
with col_logout:
    if st.button("🚪 Logout"):
        logout_user()

# optional back-to-dashboard
st.sidebar.button("⬅️ Back to Home", on_click=lambda: st.switch_page("Home.py"))

# ——— render the AI analysis tab content ————————————————
sec_filing_analysis_tab_content()
