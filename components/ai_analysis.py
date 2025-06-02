"""
AI-powered analysis tab.

Key improvements
────────────────
• Strict ticker validation (A–Z, 1–5 chars)
• Async + threaded pipeline so Streamlit UI stays responsive
• 24 h resource cache to avoid paying OpenAI twice for the same report
• Rich error / metadata display for easier debugging
"""

from __future__ import annotations

import asyncio
import re
import textwrap
from typing import Any, Dict

import streamlit as st

# ——— constants & helpers ————————————————————————————————————————
TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
DEFAULT_TICKER = "AAPL"
PERIOD_CHOICES = {"Annual": "annual", "Quarterly": "quarter"}

SPINNER_MSG = "🔎 Crunching numbers with AI … this can take 30-45 s"

# import here to avoid circulars and costly import until needed
from utils.ai_pipeline import generate_ai_report  # noqa: E402


def _validate_symbol(sym: str) -> bool:
    if not TICKER_RE.fullmatch(sym):
        st.warning("Ticker symbols are 1–5 capital letters (e.g. **AAPL**).")
        return False
    return True


@st.cache_resource(ttl=24 * 3600, show_spinner=False)  # one report per ticker/period per day
def _cached_generate(symbol: str, period: str) -> Dict[str, Any]:
    """Wrapper for expensive analysis (runs in worker thread)."""
    return asyncio.run(asyncio.to_thread(generate_ai_report, symbol, period))


# ——— main tab renderer ————————————————————————————————————————————
def ai_analysis_tab_content() -> None:
    st.subheader("🧠 AI-Powered Financial Analysis")  # fixed emoji

    # --- user inputs -----------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Ticker", value=DEFAULT_TICKER).upper().strip()
    with col2:
        period = st.radio("Period", list(PERIOD_CHOICES.keys()), horizontal=True)

    run_btn = st.button("🚀 Generate analysis", use_container_width=True)

    # --- trigger ---------------------------------------------------------
    if run_btn:
        if not _validate_symbol(symbol):
            return

        with st.spinner(SPINNER_MSG):
            try:
                report_data = _cached_generate(
                    symbol, PERIOD_CHOICES[period]
                )
            except Exception as exc:
                st.error(f"⚠️ Analysis failed:\n\n```{exc}```")
                return

        # --- error path from pipeline -----------------------------------
        if report_data.get("error"):
            st.error(report_data["error"])
            with st.expander("Show debug details"):
                st.write(report_data.get("traceback", "—"))
            return

        # --- happy path --------------------------------------------------
        md = report_data.get("markdown")
        if md:
            st.markdown(md, unsafe_allow_html=True)
            st.download_button(
                "💾 Download Markdown",
                md,
                file_name=report_data.get("md_filename", f"{symbol}_analysis.md"),
                mime="text/markdown",
            )
        else:
            st.warning("No report generated. Please try again later.")

        # debug / telemetry block
        meta = report_data.get("metadata", {})
        if meta:
            with st.expander("ℹ️ Debug metadata"):
                st.json(meta)
