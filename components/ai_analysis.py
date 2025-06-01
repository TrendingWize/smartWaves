import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from components.analysis_pipeline import generate_ai_report
from components.ai_analysis_tab import display_ai_analysis_dashboard

def ai_analysis_tab_content(default_symbol="AAPL"):
    st.subheader("?? AI-Powered Financial Analysis")

    symbol = st.text_input("Enter a Company Symbol", value=default_symbol)
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