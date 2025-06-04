# smart_waves/pages/01_Home.py
import streamlit as st
from utils.user_auth  import check_concurrent_login, initialize_session_state

initialize_session_state() # Ensure session state is set up

st.set_page_config(page_title="Home - Smart Waves", layout="wide", initial_sidebar_state="auto")

# --- Concurrent Login Check ---
if check_concurrent_login(): # If true, user was logged out, stop rendering
    st.stop()

st.title("🏠 Home")
st.markdown("---")

st.header("Welcome to Smart Waves!")
st.markdown("""
Smart Waves is your cutting-edge platform for navigating the complex world of finance.
We provide powerful tools and insights to help you make informed decisions.

**Explore our features:**
- **Financial Dashboard:** Visualize key market data and your portfolio.
- **AI Analysis:** Leverage artificial intelligence for market predictions and sentiment analysis.
- **SEC Analysis:** Dive deep into SEC filings and company reports.
- **Similarity Analysis:** Compare financial instruments or documents.
- **News & Blog:** Stay updated with the latest financial news and expert opinions.

Please **Sign Up** or **Login** to access the full suite of tools.
""")

if st.session_state.logged_in:
    st.success(f"You are logged in as {st.session_state.username}.")
else:
    st.info("You are currently viewing as a guest. Login or Sign Up for more features.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px;'>
        <p style='font-size: 0.9em; color: #777;'>
            Smart Waves © 2024. All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
