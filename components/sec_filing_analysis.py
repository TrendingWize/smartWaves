import subprocess, os, pathlib, sys   # 🔹 use sys for correct interpreter
import streamlit as st

OUTPUT_DIR = "test_analysis"
K_SCRIPT   = "10-k.py"
Q_SCRIPT   = "10-q.py"

# ---------------------------------------------------------------------------
# 🔹 helper: run generator and return the newest {html, md} pair (if any)
# ---------------------------------------------------------------------------
def _run_generator(script: str, ticker: str) -> tuple[pathlib.Path | None,
                                                      pathlib.Path | None]:
    env = os.environ.copy()
    env["TICKER_SYMBOL"] = ticker.upper()

    proc = subprocess.run(
        [sys.executable, script],           # 🔹 guarantees same venv/python
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        st.error(f"{script} failed.\n\n```{proc.stderr or proc.stdout}```")
        return None, None

    # pick the most recent HTML & MD produced by the generator
    htmls = sorted(pathlib.Path(OUTPUT_DIR).rglob("*.html"), key=os.path.getmtime)
    mds   = sorted(pathlib.Path(OUTPUT_DIR).rglob("*.md"),   key=os.path.getmtime)

    return (htmls[-1] if htmls else None,
            mds[-1]   if mds   else None)

# ---------------------------------------------------------------------------
def sec_filing_analysis_tab_content() -> None:
    st.subheader("SEC Filing Analysis")
    filing_type = st.toggle("10-Q (quarterly) / 10-K (annual)", value=False)
    script  = Q_SCRIPT if filing_type else K_SCRIPT
    ticker  = st.text_input("Ticker symbol", value="INTC").upper()

    if st.button("Generate analysis"):
        with st.spinner("Running generator …"):
            html_path, md_path = _run_generator(script, ticker)

        if html_path and html_path.exists():
            st.success("Report generated!")

            # 🔹 Prefer the colored Markdown view if present
            if md_path and md_path.exists():
                md_text = md_path.read_text(encoding="utf-8")
                st.markdown(md_text, unsafe_allow_html=True)   # colored MD
                st.download_button("Download Markdown", md_text,
                                   file_name=md_path.name,
                                   mime="text/markdown")
                with st.expander("► View HTML version"):
                    st.components.v1.html(
                        html_path.read_text(encoding="utf-8"),
                        height=800,
                        scrolling=True,
                    )
            else:
                # fallback: show HTML directly
                with st.expander("► View report"):
                    st.components.v1.html(
                        html_path.read_text(encoding="utf-8"),
                        height=800,
                        scrolling=True,
                    )

            # 🔹 always offer the HTML download
            st.download_button("Download HTML", html_path.read_bytes(),
                               file_name=html_path.name, mime="text/html")

        else:
            st.error("No report was produced – check the script output.")
