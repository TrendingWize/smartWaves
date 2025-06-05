import streamlit as st
from components.similar_companies_tab import similar_companies_tab_content, similarity_over_time_tab_content
from utils.user_auth import initialize_session_state, check_concurrent_login, create_user_table

initialize_session_state()
create_user_table()

if not st.session_state.get('logged_in'):
    st.warning("🔒 Please log in to access the Similarity Analysis tool.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/97_Login.py")
    st.stop()

if check_concurrent_login():
    st.stop()

tab1, tab2 = st.tabs(["Top Similar Companies", "Similarity Over Time"])
with tab1:
    similar_companies_tab_content()
with tab2:
    similarity_over_time_tab_content()
