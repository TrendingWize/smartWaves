# --- pages/03_AI_Analysis.py ---
import streamlit as st
from components.analysis_pipeline import generate_ai_report
from utils.user_auth import display_ai_analysis_dashboard
from styles import load_global_css
from components.financial_data_module import FinancialDataModule, Config

st.set_page_config(page_title="AI Analysis - Smart Waves", layout="wide")
load_global_css()

st.title("🧠 AI-Powered Financial Analysis")

#if not st.session_state.get("logged_in", False):
 #   st.warning("?? Please log in to access the Financial Dashboard.")
  #  if st.button("?? Go to Login"):
   #     st.switch_page("pages/97_Login.py")
    #st.stop()
    
symbol = st.text_input("Enter a Company Symbol", value="AAPL")
report_period = st.selectbox("Select Report Period", ["annual", "quarter"])

if st.button("Generate AI Report"):
    with st.spinner("Generating AI-powered analysis..."):
        report_data = generate_ai_report(symbol.strip().upper(), report_period)

        if not report_data or "analysis" not in report_data:
            st.error(f"No report available for {symbol.upper()} [{report_period}].")
        else:
            analysis = report_data["analysis"]
            metadata = report_data.get("metadata", {})
            display_ai_analysis_dashboard(analysis, metadata, symbol)
