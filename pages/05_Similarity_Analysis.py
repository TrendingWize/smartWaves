# pages/05_Similarity_Analysis.py
import streamlit as st

# It's good practice to set page config as the first Streamlit command
st.set_page_config(page_title="Fundamental Similarity Analysis", layout="wide")

from components.similar_companies_tab import similar_companies_tab_content
from utils.user_auth import initialize_session_state, check_concurrent_login, create_user_table # For auth

# Initialize session state (idempotent)
initialize_session_state()
create_user_table() # Ensure user table exists

# Authentication check
if not st.session_state.get('logged_in'): # Use .get for safety
    st.warning("🔒 Please log in to access the Similarity Analysis tool.")
    # Provide a way to navigate to login, e.g., if you have a multipage app structure
    # For example, if your login page is named "Login.py" or similar:
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/Login.py") # Adjust if your login page has a different name/path
    st.stop()

# Concurrent login check
if check_concurrent_login(): # This function handles logout and rerun
    st.stop()

# --- Main Content ---
# The user is logged in and no concurrent session detected.
similar_companies_tab_content()
