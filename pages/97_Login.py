import streamlit as st
from utils import login_user, initialize_session_state, check_concurrent_login
from streamlit import switch_page

# Initialize session
initialize_session_state()

# Check for concurrent login
if check_concurrent_login():
    st.stop()

# Page config must be first
st.set_page_config(page_title="Login - Smart Waves", layout="wide")

st.title("🔑 Login to Smart Waves")
st.markdown("---")

# If already logged in, redirect
if st.session_state.logged_in:
    st.success(f"You are already logged in as {st.session_state.username}.")
    switch_page("pages/01_Home.py")
    st.stop()

# Login form
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.form_submit_button("Login")

    if login_btn:
        if login_user(username, password):  # must return True/False
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            switch_page("pages/01_Home.py")
        else:
            st.error("Invalid username or password.")
