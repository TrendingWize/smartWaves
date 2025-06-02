# smart_waves/pages/components/sec_filing_analysis.py
import subprocess, os, pathlib, streamlit as st
import sys # <--- Added import for sys

OUTPUT_DIR = "test_analysis"
K_SCRIPT   = "10-k.py"
Q_SCRIPT   = "10-q.py"

def _run_generator(script: str, ticker: str) -> tuple[pathlib.Path | None, subprocess.CompletedProcess | None]:
    env = os.environ.copy()
    env["TICKER_SYMBOL"] = ticker.upper()
    proc = subprocess.run([sys.executable, script], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        st.error(f"{script} failed.\n\n```{proc.stderr or proc.stdout}```")
        return None, proc

    output_path = pathlib.Path(OUTPUT_DIR)
    if not output_path.exists():
        st.error(f"Output directory '{OUTPUT_DIR}' does not exist.")
        return None, proc

    htmls = sorted(output_path.rglob("*.html"), key=os.path.getmtime)
    return (htmls[-1] if htmls else None), proc


def sec_filing_analysis_tab_content() -> None:
    st.subheader("SEC Filing Analysis")
    filing_type = st.toggle("10-Q (quarterly) / 10-K (annual)", value=False)
    script  = Q_SCRIPT if filing_type else K_SCRIPT
    ticker  = st.text_input("Ticker symbol", value="INTC").upper()

    if st.button("Generate analysis"):
        with st.spinner("Running generator …"):
            html_path, proc = _run_generator(script, ticker)


        if html_path and html_path.exists():
            st.success("Report generated!")
            with st.expander("► View report"):
                try:
                    st.components.v1.html(html_path.read_text(encoding="utf-8"),
                                          height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error reading or displaying HTML report: {e}")
            try:
                st.download_button("Download HTML", html_path.read_bytes(),
                                   file_name=html_path.name, mime="text/html")
            except Exception as e:
                st.error(f"Error preparing HTML report for download: {e}")

        elif html_path is None and proc.returncode != 0: # Error already displayed by _run_generator
            pass
        else:
            st.error("No HTML report was produced – check the script output or logs if available.")

st.markdown(report_md, unsafe_allow_html=True)

