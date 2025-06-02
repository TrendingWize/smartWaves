import subprocess, os, pathlib, streamlit as st

OUTPUT_DIR = "test_analysis"
K_SCRIPT   = "10-k.py"
Q_SCRIPT   = "10-q.py"

def _run_generator(script: str, ticker: str) -> pathlib.Path | None:
    env = os.environ.copy()
    env["TICKER_SYMBOL"] = ticker.upper()
    proc = subprocess.run(["python", script], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        st.error(f"{script} failed.\n\n```{proc.stderr or proc.stdout}```")
        return None

    htmls = sorted(pathlib.Path(OUTPUT_DIR).rglob("*.html"), key=os.path.getmtime)
    return htmls[-1] if htmls else None

def sec_filing_analysis_tab_content() -> None:
    st.subheader("SEC Filing Analysis")
    filing_type = st.toggle("10-Q (quarterly) / 10-K (annual)", value=False)
    script  = Q_SCRIPT if filing_type else K_SCRIPT
    ticker  = st.text_input("Ticker symbol", value="INTC").upper()

    if st.button("Generate analysis"):
        with st.spinner("Running generator …"):
            html_path = _run_generator(script, ticker)

        if html_path and html_path.exists():
            st.success("Report generated!")
            with st.expander("► View report"):
                st.components.v1.html(html_path.read_text(encoding="utf-8"),
                                      height=800, scrolling=True)
            st.download_button("Download HTML", html_path.read_bytes(),
                               file_name=html_path.name, mime="text/html")
        else:
            st.error("No HTML report was produced – check the script output.")
