# smart_waves/pages/02_Financial_Dashboard.py
import streamlit as st
st.set_page_config(page_title="Financial Dashboard", layout="wide")

import sys
import os
from utils.user_auth import logout_user
from utils.user_auth import check_concurrent_login, initialize_session_state # This should now work
from components.company_profile import company_profile_tab_content
from components.income_statement import income_statement_tab_content
from components.balance_sheet import balance_sheet_tab_content
from components.cash_flow import cash_flow_tab_content
#from components.ai_analysis import ai_analysis_tab_content
#from components.sec_filing_analysis import sec_filing_analysis_tab_content
from utils.blog_db import insert_blog_post, init_blog_db


initialize_session_state()
st.title("📊 Financial Dashboard")

if check_concurrent_login():
    st.stop()

#if not st.session_state.logged_in:
    #st.warning("🔒 Please log in to access the Financial Dashboard.")
    #if st.button("🔑 Go to Login"):
     #   st.switch_page("pages/97_Login.py")
    #st.stop()



# Top-right logout
col1, col2 = st.columns([8, 1])
with col2:
    if st.session_state.get("logged_in"):
        if st.button("🚪 Logout"):
            logout_user()

# Company symbol input
symbol = st.text_input("Enter Company Symbol (e.g., AAPL, TSLA):", key="dashboard_symbol")
if not symbol:
    st.info("Please enter a symbol to begin.")
    st.stop()

# Tabs and function mapping
tab_titles = [
    "🏢 Company Profile",
    "📈 Income Statement",
    "📊 Balance Sheet",
    "💵 Cash Flow",
    #"🧠 AI Analysis",
   # "📑 SEC Filings"
]

tab_functions = [
    company_profile_tab_content,
    income_statement_tab_content,
    balance_sheet_tab_content,
    cash_flow_tab_content,
    #lambda symbol: ai_analysis_tab_content(symbol),
    #lambda symbol: sec_filing_analysis_tab_content()
]

tabs = st.tabs(tab_titles)

for i, tab in enumerate(tabs):
    with tab:
        if tab_titles[i] == "📑 SEC Filings":
            tab_functions[i](None)
        else:
            tab_functions[i](symbol.strip().upper())
