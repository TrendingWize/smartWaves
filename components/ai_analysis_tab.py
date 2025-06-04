import streamlit as st
from components.analysis_pipeline import generate_ai_report
from utils.user_auth import display_ai_analysis_dashboard
from styles import load_global_css

load_global_css()

if not st.session_state.get("logged_in", False):
    st.warning("🔒 Please log in to access the Financial Dashboard.")
    if st.button("🔑 Go to Login"):
        st.switch_page("pages/97_Login.py")
    st.stop()
    
def ai_analysis_tab_content(default_symbol="AAPL"):
    st.subheader("🧠 AI-Powered Financial Analysis")

    symbol = st.text_input("Enter a Company Symbol", value=default_symbol)
    report_period = st.selectbox("Select Report Period", ["annual", "quarter"])

    if st.button("Generate AI Report"):
        with st.spinner("Generating AI-powered analysis..."):
            report_data = generate_ai_report(symbol.strip().upper(), report_period)

            if not report_data or "analysis" not in report_data:
                st.error(f"No report available for {symbol}.")
            else:
                analysis = report_data["analysis"]
                metadata = report_data.get("metadata", {})
                display_ai_analysis_dashboard(analysis, metadata, symbol)
