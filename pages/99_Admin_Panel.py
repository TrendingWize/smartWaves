# smart_waves/pages/99_Admin_Panel.py
import streamlit as st
from utils import (
    get_unapproved_users,
    approve_user,
    check_concurrent_login,
    initialize_session_state,
)

initialize_session_state()
from utils.blog_db import get_all_blog_posts, init_blog_db

init_blog_db()

# --- Concurrent Login Check & Auth Check ---
if check_concurrent_login():
    st.stop()

if not st.session_state.logged_in or not st.session_state.is_admin:
    st.error("🚫 Access Denied. You must be an administrator to view this page.")
    if not st.session_state.logged_in:
         # MODIFIED LINE: Added ?action=login
         st.page_link("app.py", label="Go to Login", icon="🔑")

    st.stop()

st.title("🛠️ Admin Panel")
st.markdown("---")
st.write(f"Welcome, Admin {st.session_state.username}!")

st.subheader("User Approval Management")

unapproved_users = get_unapproved_users()

if not unapproved_users:
    st.success("No users are currently pending approval.")
else:
    st.info(f"Found {len(unapproved_users)} user(s) pending approval:")
    
    for user in unapproved_users:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Username: **{user['username']}** (ID: {user['id']})")
        with col2:
            if st.button(f"Approve {user['username']}", key=f"approve_{user['id']}"):
                approve_user(user['id'])
                st.success(f"User '{user['username']}' has been approved!")
                st.rerun() # Refresh the list

st.markdown("---")
st.subheader("Other Admin Functions")
st.info("More administrative tools can be added here (e.g., view all users, site statistics).")