import streamlit as st
import pandas as pd
import inspect
from datetime import datetime
from typing import Dict, Any

from utils import (
    get_neo4j_driver,
    get_nearest_aggregate_similarities,
    fetch_financial_details_for_companies,
    format_value,
)


# --------------------------------------------------------------------------------------
# Streamlit tab builder – "Similar Companies" with configurable aggregation weighting
# --------------------------------------------------------------------------------------

DEFAULT_DECAY = 0.7  # λ for recency‑weighted mean when no slider provided


def similar_companies_tab_content() -> None:
    """Render the *Find Similar Companies* tab.

    Users can choose the embedding family **and** how yearly vectors are aggregated:
    * Equal‑weighted mean (long‑term similarity)
    * Recency‑weighted mean (λ‑decay, emphasises recent fundamentals)
    * Latest year only (most current snapshot)
    """

    # ---------- Header & Intro ----------
    st.title("🔗 Find Similar Companies")
    st.markdown(
        "Discover companies with similar financial‑statement embeddings, "
        "with a choice of **how** yearly vectors are aggregated."
    )

    # ---------- User Inputs ----------
    col_sym, col_family = st.columns([2, 2])
    with col_sym:
        target_symbol: str = (
            st.text_input(
                "Target Company Symbol (e.g., NVDA, AAPL)",
                value=st.session_state.get("similar_target_sym", "NVDA"),
                key="similar_symbol_input",
            )
            .strip()
            .upper()
        )

    # Embedding family (which statement vectors)
    embedding_family_options: Dict[str, str] = {
        "Cash‑Flow (cf_vec_)": "cf_vec_",
        "Income‑Statement (is_vec_)": "is_vec_",
        "Balance‑Sheet (bs_vec_)": "bs_vec_",
    }
    with col_family:
        family_display = st.selectbox(
            "Similarity Type (Embedding Family)",
            options=list(embedding_family_options.keys()),
            index=0,
            key="similar_family_select",
        )
    family_value = embedding_family_options[family_display]

    # Aggregation weighting scheme -------------------------------
    weight_scheme_options: Dict[str, str] = {
        "Equal‑weighted mean": "mean",
        "Recency‑weighted mean (λ‑decay)": "decay",
        "Latest year only": "latest",
    }
    col_ws, col_lambda = st.columns([2, 1])
    with col_ws:
        weight_display = st.selectbox(
            "Aggregation Weighting Scheme",
            options=list(weight_scheme_options.keys()),
            index=0,
            key="similar_weight_scheme_select",
        )
    weight_value = weight_scheme_options[weight_display]

    # λ slider only when decay selected
    decay_lambda = DEFAULT_DECAY
    with col_lambda:
        if weight_value == "decay":
            decay_lambda = st.slider(
                "λ (decay factor)",
                min_value=0.5,
                max_value=0.95,
                value=DEFAULT_DECAY,
                step=0.05,
                key="similar_decay_slider",
            )

    # Year range – always up to LAST full financial year
    end_similarity_year = datetime.now().year - 1
    start_similarity_year = end_similarity_year - 5  # 5‑year window by default

    # Number of peers to surface
    k = st.slider(
        "Number of Similar Companies", 5, 25, value=10, step=1, key="similar_k_slider"
    )

    # ---------- Session‑state keys ----------
    for key, default in [
        ("similar_companies", []),
        ("similar_details", {}),
        ("last_symbol", None),
        ("last_family", None),
        ("last_weight", None),
    ]:
        st.session_state.setdefault(key, default)

    # ---------- Trigger search ----------
    trigger_col, _ = st.columns([1, 3])
    with trigger_col:
        if st.button("🚀 Find Similar Companies", use_container_width=True):
            if not target_symbol:
                st.warning("Please enter a company symbol first.")
                st.stop()

            # Reset previous results
            st.session_state.similar_companies = []
            st.session_state.similar_details = {}

            # Connect & query
            neo_driver = get_neo4j_driver()
            if not neo_driver:
                st.error("Database connection failed.")
                st.stop()

            with st.spinner(f"Computing peers for {target_symbol} …"):
                try:
                    # Build kwargs dynamically to match whatever signature utils provides
                    peer_kwargs: Dict[str, Any] = {
                        'target_sym': target_symbol,
                        'embedding_family': family_value,
                        'start_year': start_similarity_year,
                        'end_year': end_similarity_year,
                        'k': k,
                    }
                    # Choose driver arg name based on signature to avoid cache hashing errors
                    sig_peer = inspect.signature(get_nearest_aggregate_similarities)
                    if '_driver' in sig_peer.parameters:
                        peer_kwargs['_driver'] = neo_driver
                    else:
                        peer_kwargs['driver'] = neo_driver
                    if 'weight_scheme' in sig_peer.parameters:
                        peer_kwargs['weight_scheme'] = weight_value
                    if 'decay' in sig_peer.parameters:
                        peer_kwargs['decay'] = decay_lambda if weight_value == 'decay' else None

                    peers = get_nearest_aggregate_similarities(**peer_kwargs)
                finally:
                    if hasattr(neo_driver, "close"):
                        neo_driver.close()

            st.session_state.similar_companies = peers or []
            st.session_state.last_symbol = target_symbol
            st.session_state.last_family = family_display
            st.session_state.last_weight = weight_display

            # Fetch fundamentals for display (only if peers returned)
            if peers:
                symbols = [sym for sym, _ in peers]
                with st.spinner("Fetching financial details …"):
                    neo_driver = get_neo4j_driver()
                    try:
                        sig_det = inspect.signature(fetch_financial_details_for_companies)
                        # Decide driver argument position/name
                        if '_driver' in sig_det.parameters or 'driver' in sig_det.parameters:
                            # Pass both parameters positionally to side‑step keyword mismatch
                            details = fetch_financial_details_for_companies(neo_driver, symbols)
                        else:
                            # Fallback: try keyword 'symbols' only
                            details = fetch_financial_details_for_companies(symbols=symbols)
                    finally:
                        if hasattr(neo_driver, "close"):
                            neo_driver.close()
                st.session_state.similar_details = details or {}

            st.rerun()

    # ---------- Display results ----------
    if st.session_state.last_symbol:
        st.markdown("---")
        st.subheader(
            f"Top {k} companies similar to **{st.session_state.last_symbol}** "
            f"using **{st.session_state.last_family}** vectors "
            f"aggregated via **{st.session_state.last_weight}**"
        )

        peers = st.session_state.similar_companies
        if not peers:
            st.info("No similar companies found with the selected parameters.")
            return

        details = st.session_state.similar_details
        for idx, (sym, score) in enumerate(peers, start=1):
            meta = details.get(sym, {})
            company_name = meta.get("companyName", sym)
            sector = meta.get("sector", "N/A")
            industry = meta.get("industry", "N/A")

            with st.container():
                st.markdown(f"**{idx}. {company_name} ({sym})** — similarity **{score:.4f}**")
                st.caption(f"Sector: {sector} | Industry: {industry}")

                col_is, col_bs, col_cf = st.columns(3)

                with col_is:
                    st.markdown("###### Income Statement")
                    st.metric("Revenue", format_value(meta.get("revenue")))
                    st.metric("Net Income", format_value(meta.get("netIncome")))
                    st.metric("Operating Inc.", format_value(meta.get("operatingIncome")))
                    st.metric("Gross Profit", format_value(meta.get("grossProfit")))

                with col_bs:
                    st.markdown("###### Balance Sheet")
                    st.metric("Assets", format_value(meta.get("totalAssets")))
                    st.metric("Liabilities", format_value(meta.get("totalLiabilities")))
                    st.metric("Equity", format_value(meta.get("totalStockholdersEquity")))
                    st.metric("Cash", format_value(meta.get("cashAndCashEquivalents")))

                with col_cf:
                    st.markdown("###### Cash‑Flow")
                    st.metric("Op. CF", format_value(meta.get("operatingCashFlow")))
                    st.metric("Free CF", format_value(meta.get("freeCashFlow")))
                    st.metric("Δ Cash", format_value(meta.get("netChangeInCash")))
                    st.metric("CapEx", format_value(meta.get("capitalExpenditure")))
                st.markdown("---")


# --------------------------------------------------------------------------------------
# Stand‑alone demo mode – lightweight mocks so the file is runnable without the full app
# --------------------------------------------------------------------------------------
if __name__ == "__main__":

    class _MockDriver:
        """Very small Neo4j stub sufficient for demo."""

        def close(self):
            pass

    def get_neo4j_driver():  # type: ignore
        return _MockDriver()

    # Fake utils behaviours ---------------------------------------------------
    def get_nearest_aggregate_similarities(  # type: ignore
        *,
        driver,  # noqa: ANN001
        target_sym: str,
        embedding_family: str,
        start_year: int,
        end_year: int,
        k: int,
        weight_scheme: str,
        decay: float | None = None,
    ):
        rng = np.random.default_rng(hash(target_sym) & 0xFFFF)
        peers = [
            (sym, float(rng.uniform(0.8, 0.95)))
            for sym in ["AMD", "INTC", "QCOM", "TSM", "AVGO"][:k]
        ]
        return peers

    def fetch_financial_details_for_companies(  # type: ignore
        driver, symbols  # noqa: ANN001
    ):
        return {
            s: {
                "companyName": f"{s} Inc.",
                "sector": "Tech",
                "industry": "Semiconductors",
                "revenue": 1_000_000_000,
                "netIncome": 120_000_000,
                "operatingIncome": 200_000_000,
                "grossProfit": 500_000_000,
                "totalAssets": 2_000_000_000,
                "totalLiabilities": 800_000_000,
                "totalStockholdersEquity": 1_200_000_000,
                "cashAndCashEquivalents": 300_000_000,
                "operatingCashFlow": 260_000_000,
                "freeCashFlow": 150_000_000,
                "netChangeInCash": 60_000_000,
                "capitalExpenditure": -100_000_000,
            }
            for s in symbols
        }

    def format_value(value):  # type: ignore
        if value is None:
            return "N/A"
        abs_val = abs(value)
        if abs_val >= 1e9:
            return f"${value / 1e9:,.2f} B"
        if abs_val >= 1e6:
            return f"${value / 1e6:,.2f} M"
        return f"${value:,.0f}"

    # Monkey‑patch the names imported at the top so our demo uses these mocks
    import sys as _sys

    _sys.modules[__name__].get_neo4j_driver = get_neo4j_driver  # type: ignore
    _sys.modules[__name__].get_nearest_aggregate_similarities = (
        get_nearest_aggregate_similarities  # type: ignore
    )
    _sys.modules[__name__].fetch_financial_details_for_companies = (
        fetch_financial_details_for_companies  # type: ignore
    )
    _sys.modules[__name__].format_value = format_value  # type: ignore

    st.set_page_config(layout="wide")
    similar_companies_tab_content()
