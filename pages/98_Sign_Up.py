# smart_waves/pages/10_Sign_Up.py
import streamlit as st
from utils import add_user, initialize_session_state

initialize_session_state()

# st.set_page_config(page_title="Sign Up - Smart Waves", layout="wide", initial_sidebar_state="collapsed") # REMOVE THIS LINE

st.title("✍️ Sign Up for Smart Waves")
st.markdown("---")

if st.session_state.logged_in:
    st.info("You are already logged in. If you want to create a new account, please log out first.")
    if st.button("Logout to Sign Up"):
        from utils import logout_user # Local import to avoid circularity if utils imports st too early
        logout_user()
    st.stop()

with st.form("signup_form"):
    st.subheader("Create a New Account")
    new_username = st.text_input("Choose a Username", key="signup_username")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

    cols = st.columns([1,3]) # For better layout of checkbox
    with cols[0]:
        agree_terms = st.checkbox("I agree to the terms and conditions (placeholder)")

    submit_button = st.form_submit_button("Sign Up")

    if submit_button:
        if not new_username or not new_password or not confirm_password:
            st.error("Please fill in all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        elif len(new_password) < 4: # Basic password policy
            st.error("Password must be at least 6 characters long.")
        elif not agree_terms:
            st.error("You must agree to the terms and conditions.")
        else:
            if add_user(new_username, new_password):
                st.success("Account created successfully! Please wait for an administrator to approve your account.")
                st.balloons()
                st.info("You will be able to log in once your account is approved.")
            else:
                st.error("Username already exists. Please choose a different one.")