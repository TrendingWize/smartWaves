# smart_waves/app.py
import streamlit as st
from streamlit_navigation_bar import st_navbar 
from utils import initialize_session_state, create_user_table, login_user, logout_user, check_concurrent_login

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Waves",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Initialize Database and Session State ---
create_user_table() # Ensures table and admin user exist
initialize_session_state()

# --- Custom CSS (Optional, for finer control) ---
st.markdown("""
    <style>
        /* Add any global styles here */
        .stApp {
            /* background-color: #f0f2f6; */ /* Example light background */
        }
        .stDeployButton {
            visibility: hidden;
        }
        /* You can style the navigation bar further if needed */
    </style>
""", unsafe_allow_html=True)


# --- Check for Concurrent Logins on each run ---
# This needs to be done carefully to avoid infinite loops with rerun
# We'll call it before rendering content in pages that require login
# if check_concurrent_login():
#    st.stop() # If logged out due to concurrency, stop further execution for this run

# --- Main App Layout ---
st.markdown("<h1 style='text-align: center; color: #007bff;'>🌊 Smart Waves</h1>", unsafe_allow_html=True)
st.markdown("---")


# --- Navigation Bar ---
# Define pages available to everyone
public_pages = ["Home", "Support", "Contact Us"]
user_pages = ["Financial Dashboard", "AI Analysis", "SEC Analysis", "Ask AI", "Similarity Analysis", "News", "Blog"]

# Dynamically build navigation items
nav_items = list(public_pages) # Start with public pages

if st.session_state.logged_in:
    nav_items.extend(user_pages)
    if st.session_state.is_admin:
        nav_items.append("Admin Panel")
    nav_items.append("Logout")
else:
    nav_items.append("Login")
    nav_items.append("Sign Up")

# Sort pages to ensure specific order if needed, or use a predefined order
# For now, this order is fine.
# For a fixed order:
# ordered_nav_items = ["Home", "Financial Dashboard", ..., "Login/Logout", "Sign Up", "Admin Panel"]
# selected_page = st_navbar([item for item in ordered_nav_items if item in nav_items])

selected_page = st_navbar(nav_items)


# --- Page Content Display ---
# This is the core of the multi-page app logic.
# Streamlit's native multi-page app feature handles this by looking at the `pages/` directory.
# The `st_navbar` just helps in changing what's displayed *conceptually*
# We need to update `st.session_state.current_page` to then conditionally show content
# or use Streamlit's automatic page switching.

# The `st_navbar` returns the selected page name. We can use this to navigate.
# However, Streamlit's multi-page app feature means we don't *manually* switch pages here.
# The `pages/` directory structure does that.
# The `st_navbar` is more for the *visual* navigation.
# We will handle Login/Logout/Sign Up actions here directly.

if selected_page == "Login":
    st.session_state.current_page = "Login"
    st.subheader("Login")
    with st.form("login_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        if login_button:
            login_user(login_username, login_password)
            # No need to st.rerun() here, login_user does it on success

elif selected_page == "Logout":
    logout_user()
    # No need to st.rerun() here, logout_user does it

elif selected_page == "Sign Up":
    st.switch_page("pages/10_Sign_Up.py") # Direct to sign up page

elif selected_page == "Admin Panel" and st.session_state.is_admin:
    st.switch_page("pages/99_Admin_Panel.py")

elif selected_page == "Home":
    st.switch_page("pages/01_Home.py") # Default or explicit home
    # Or, if app.py is your home, display home content here. For clarity, use pages/01_Home.py

# Streamlit will automatically load the content of the selected page from the `pages` directory
# if the `selected_page` matches a filename (e.g., "Home" matches `01_Home.py` if `st.Page("pages/01_Home.py", title="Home")` was used with `st.navigation`)
# Since we are using `st_navbar`, we handle specific actions like Login/Logout here.
# For content pages, we rely on the user clicking them and Streamlit switching.
# If `app.py` is intended to be the Home page itself, you'd put Home content here if selected_page == "Home".

# Display a placeholder message if no specific action page is selected from navbar
# and it's not a page from the `pages` directory (which Streamlit handles automatically)
if selected_page not in ["Login", "Logout", "Sign Up", "Admin Panel"] and st.session_state.current_page == "Home" and selected_page != "Home":
    # This logic is a bit tricky with st_navbar and Streamlit's MPA.
    # Essentially, if the navbar changes to "Financial Dashboard", Streamlit's MPA should take over.
    # The `st.switch_page` calls help.
    pass

# This is the landing spot, effectively the Home page if no other page is explicitly loaded by st.switch_page
# or by Streamlit's own MPA mechanism if `app.py` is the entry point.
if selected_page == "Home" or st.session_state.current_page == "Home":
    # This will be shown if `app.py` is run and no other page is selected yet,
    # or if "Home" is explicitly clicked.
    # It's better to have a dedicated `pages/01_Home.py` for consistency.
    # For now, let's assume app.py doesn't show detailed home content itself.
    st.info(f"You are on the main application page. Select an option from the navigation bar. Current conceptual page: {selected_page}")

# To ensure users see content corresponding to the navbar selection when it's a page from the `pages/` directory:
# This logic might be redundant if Streamlit's MPA is handling it well based on URL.
# But st_navbar selection needs to align with what Streamlit thinks is the current page.
page_mapping = {
    "Home": "pages/01_Home.py",
    "Financial Dashboard": "pages/02_Financial_Dashboard.py",
    "AI Analysis": "pages/03_AI_Analysis.py",
    "SEC Analysis": "pages/04_SEC_Analysis.py",
    "Ask AI":"pages/11_ask_ai.py",
    "Similarity Analysis": "pages/05_Similarity_Analysis.py",
    "News": "pages/06_News.py",
    "Blog": "pages/07_Blog.py",
    "Support": "pages/08_Support.py",
    "Contact Us": "pages/09_Contact_Us.py",
}

if selected_page in page_mapping:
    st.switch_page(page_mapping[selected_page])


# Fallback for the initial run or if something goes wrong with page switching.
# Typically, Streamlit handles displaying the content of pages/01_Home.py (or the first page) by default.
# If `app.py` is the first page users see, then some default content should be here.
# Given the structure, Streamlit will likely try to load `pages/01_Home.py` if it exists.
# If `app.py` itself is the entry and should display Home content:
# if st.session_state.current_page == "Home": # Or based on actual Streamlit page context
#     st.header("Welcome to Smart Waves!")
#     st.write("Your intelligent platform for financial insights and analysis.")
#     st.write("Please login or sign up to access more features.")