# pages/05_Similarity_Analysis.py

import streamlit as st
from components.similar_companies_tab import similar_companies_tab_content
from utils import initialize_session_state, check_concurrent_login, get_neo4j_driver, fetch_sector_list, fetch_company_preview

# Initialize session state
initialize_session_state()
st.set_page_config(page_title="Fundamental Similarity Analysis", layout="wide")

# Authentication check
if not st.session_state.logged_in:
    st.warning("🔒 Please log in to access the Financial Dashboard.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/97_Login.py")
    st.stop()
    
# Concurrent login check
if check_concurrent_login():
    st.stop()

# --- Access control ---
if not st.session_state.logged_in:
    st.warning("🔒 Please log in to access this feature.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/97_Login.py")
    st.stop()

# --- Enhanced Similarity Tool UI ---
similar_companies_tab_content()
