# components/similar_companies_tab.py

import streamlit as st
import numpy as np
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Any


from utils import (
    get_neo4j_driver,
    fetch_financial_details_for_companies,
    fetch_sector_list,
    fetch_company_preview,
    format_value
)

DEFAULT_DECAY = 0.7  # λ for recency-weighted mean

# --- Market Cap Class Fetcher ---
def fetch_market_cap_classes(driver=None):
    if not driver:
        driver = get_neo4j_driver()
    query = """
    MATCH (c:Company)
    WHERE c.marketCapClass IS NOT NULL
    RETURN DISTINCT c.marketCapClass AS cap_class
    ORDER BY cap_class
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record["cap_class"] for record in result]
    except Exception as e:
        print(f"Error fetching market cap classes: {str(e)}")
        return []
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()

# --- Vector Loader with Filtering ---
@st.cache_data(ttl="1h")
def load_vectors_for_similarity(
    _driver, 
    year: int, 
    family: str = "cf_vec_", 
    sectors: List[str]=None, 
    cap_classes: List[str]=None, 
    target_sym: str=None
):
    prop = f"{family}{year}"
    where_clauses = [f"c.{prop} IS NOT NULL", "c.ipoDate IS NOT NULL"]
    if sectors:
        where_clauses.append("c.sector IN $sectors")
    if cap_classes:
        where_clauses.append("c.marketCapClass IN $cap_classes")
    where_str = " AND ".join(where_clauses)
    query = (
        f"MATCH (c:Company)\n"
        f"WHERE {where_str}\n"
        f"RETURN c.symbol AS sym, c.{prop} AS vec, c.sector AS sector, c.marketCapClass AS cap_class"
    )
    params = {}
    if sectors:
        params["sectors"] = sectors
    if cap_classes:
        params["cap_classes"] = cap_classes
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query, **params)
            all_vecs = {r["sym"]: (np.asarray(r["vec"], dtype=np.float32), r["sector"], r["cap_class"]) for r in results}
            target_vector = all_vecs.get(target_sym, (None, None, None))[0]
            candidate_vecs = {sym: vec for sym, (vec, _, _) in all_vecs.items() if sym != target_sym}
            return target_vector, candidate_vecs
    except Exception as e:
        st.error(f"Error loading vectors for {prop}: {e}")
        return None, {}

# --- Cosine Similarity ---
def calculate_similarity_scores(target_vector: np.ndarray, vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    if target_vector is None or not vectors:
        return {}
    matrix_of_vectors = np.vstack(list(vectors.values()))
    matrix_of_vectors /= (np.linalg.norm(matrix_of_vectors, axis=1, keepdims=True) + 1e-12)
    target_vector /= (np.linalg.norm(target_vector) + 1e-12)
    similarities = matrix_of_vectors @ target_vector
    return dict(zip(vectors.keys(), similarities.astype(float)))

# --- Main Similarity Function ---
@st.cache_data(ttl="1h", show_spinner="Calculating aggregated similarity scores...")
def get_nearest_aggregate_similarities(_driver, 
                                       target_sym: str, 
                                       embedding_family: str, 
                                       start_year: int = 2017, 
                                       end_year: int = 2023, 
                                       sectors=None,
                                       cap_classes=None,
                                       weight_scheme=None,
                                       normalize=True,
                                       k: int = 10) -> List[Tuple[str, float]]:
    if not _driver:
        return []
    from collections import defaultdict
    cumulative_scores = defaultdict(float)
    years_processed_count = 0

    for year in range(start_year, end_year + 1):
        target_vector, yearly_vectors = load_vectors_for_similarity(
            _driver, year, embedding_family, sectors=sectors, cap_classes=cap_classes, target_sym=target_sym
        )
        if target_vector is None or not yearly_vectors:
            continue
        yearly_similarity_scores = calculate_similarity_scores(target_vector, yearly_vectors)
        for sym, score in yearly_similarity_scores.items():
            cumulative_scores[sym] += score
        years_processed_count += 1

    if years_processed_count == 0:
        st.warning(f"No data found for {target_sym} or its comparables in the selected year range and embedding family.")
        return []

    average_scores = {sym: score / years_processed_count for sym, score in cumulative_scores.items()}
    best_k_similar = sorted(average_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return best_k_similar

# --- UI Function ---
def similar_companies_tab_content() -> None:
    st.title("🔍 Fundamental Similarity Analysis")
    st.markdown("""
    Discover fundamentally similar companies using financial statement embeddings.
    *Features:* Custom time ranges, sector and market cap class filters, metric selection & visual comparisons.
    """)
    
    # Pre-fetch sectors & market cap classes
    neo_driver = get_neo4j_driver()
    sectors, cap_classes = [], []
    try:
        sectors = fetch_sector_list(neo_driver)
        cap_classes = fetch_market_cap_classes(neo_driver)
    except Exception as e:
        st.error(f"Couldn't fetch sector/market cap: {str(e)}")
        sectors = ["Technology", "Healthcare", "Financial"]
        cap_classes = ["Large Cap", "Mid Cap", "Small Cap"]
    finally:
        if hasattr(neo_driver, "close"):
            neo_driver.close()

    with st.expander("🔧 Search Parameters", expanded=True):
        col_sym, col_family = st.columns([2, 2])

        with col_sym:
            target_symbol = (
                st.text_input(
                    "Target Company Symbol (e.g., NVDA, AAPL)",
                    value=st.session_state.get("similar_target_sym", "NVDA"),
                    key="similar_symbol_input",
                )
                .strip()
                .upper()
            )
            if target_symbol:
                preview_driver = get_neo4j_driver()
                try:
                    preview = fetch_company_preview(target_symbol, preview_driver)
                    if preview:
                        st.caption(f"**{preview['companyName']}** | {preview.get('sector', 'N/A')} | {preview.get('industry', 'N/A')}")
                    else:
                        st.warning("Symbol not found in database")
                except Exception:
                    st.warning("Couldn't fetch company details")
                finally:
                    if hasattr(preview_driver, "close"):
                        preview_driver.close()

        embedding_family_options: Dict[str, str] = {
            "Combined Financials (all_vec_)": "all_vec_",
            "Income Statement (is_vec_)": "is_vec_",
            "Balance Sheet (bs_vec_)": "bs_vec_",
            "Cash Flow (cf_vec_)": "cf_vec_",
        }
        with col_family:
            family_display = st.selectbox(
                "Embedding Type",
                options=list(embedding_family_options.keys()),
                index=0,
                key="similar_family_select",
            )
        family_value = embedding_family_options[family_display]

        current_year = datetime.now().year
        col_start, col_end = st.columns(2)
        with col_start:
            start_year = st.number_input(
                "Start Year", 
                min_value=1990, 
                max_value=current_year-1,
                value=current_year-5,
                key="start_year"
            )
        with col_end:
            end_year = st.number_input(
                "End Year", 
                min_value=1990, 
                max_value=current_year,
                value=current_year-1,
                key="end_year"
            )
        if start_year > end_year:
            st.error("⚠️ Start year must be before end year")
            st.stop()

        selected_sectors = st.multiselect(
            "Filter by Sector",
            options=sectors,
            default=sectors,
            key="sector_filter"
        )

        selected_cap_classes = st.multiselect(
            "Filter by Market Cap Class",
            options=cap_classes,
            default=cap_classes,
            key="cap_class_filter"
        )

        weight_scheme_options: Dict[str, str] = {
            "Equal-weighted mean": "mean",
            "Recency-weighted mean (λ-decay)": "decay",
            "Latest year only": "latest",
        }
        col_ws, col_lambda = st.columns([2, 1])
        with col_ws:
            weight_display = st.selectbox(
                "Time Weighting",
                options=list(weight_scheme_options.keys()),
                index=0,
                key="similar_weight_scheme_select",
            )
        weight_value = weight_scheme_options[weight_display]

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
                    help="Higher λ = more weight on recent years"
                )

        metric_options = [
            "Revenue", "Net Income", "Gross Profit", "Operating Income",
            "Total Assets", "Total Liabilities", "Equity", "Cash",
            "Operating Cash Flow", "Free Cash Flow", "CapEx", "ROE", "Current Ratio"
        ]
        selected_metrics = st.multiselect(
            "Key Metrics to Compare",
            options=metric_options,
            default=metric_options[:8],
            key="metric_selection"
        )

        k = st.slider(
            "Number of Results", 
            5, 50, value=10, step=5, 
            key="similar_k_slider"
        )

    state_keys = [
        ("similar_companies", []),
        ("similar_details", {}),
        ("last_symbol", None),
        ("last_family", None),
        ("last_weight", None),
        ("metric_year", None)
    ]
    for key, default in state_keys:
        st.session_state.setdefault(key, default)

    search_col, _ = st.columns([1, 3])
    with search_col:
        if st.button("🚀 Find Similar Companies", use_container_width=True, type="primary"):
            if not target_symbol:
                st.warning("⚠️ Please enter a valid company symbol")
                st.stop()
            if not selected_sectors:
                st.warning("⚠️ Please select at least one sector")
                st.stop()
            if not selected_cap_classes:
                st.warning("⚠️ Please select at least one market cap class")
                st.stop()

            for key in ["similar_companies", "similar_details"]:
                st.session_state[key] = {}

            neo_driver = get_neo4j_driver()
            if not neo_driver:
                st.error("❌ Database connection failed")
                st.stop()

            with st.spinner(f"🔍 Analyzing {target_symbol} fundamentals..."):
                try:
                    peer_kwargs: Dict[str, Any] = {
                        'target_sym': target_symbol,
                        'embedding_family': family_value,
                        'start_year': start_year,
                        'end_year': end_year,
                        'k': k + 5,
                        'sectors': selected_sectors,
                        'cap_classes': selected_cap_classes,
                        'weight_scheme': weight_value,
                        'normalize': True
                    }
                    sig_peer = inspect.signature(get_nearest_aggregate_similarities)
                    if 'driver' in sig_peer.parameters:
                        peer_kwargs['driver'] = neo_driver
                    elif '_driver' in sig_peer.parameters:
                        peer_kwargs['_driver'] = neo_driver
                    peers = get_nearest_aggregate_similarities(**peer_kwargs)
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    return
                finally:
                    if hasattr(neo_driver, "close"):
                        neo_driver.close()

            if not peers:
                st.info("ℹ️ No similar companies found with current filters")
                return

            st.session_state.similar_companies = peers[:k]
            st.session_state.last_symbol = target_symbol
            st.session_state.last_family = family_display
            st.session_state.last_weight = weight_display
            st.session_state.metric_year = end_year

            symbols = [sym for sym, _ in peers[:k]]
            with st.spinner("📊 Compiling financial profiles..."):
                neo_driver = get_neo4j_driver()
                try:
                    sig_det = inspect.signature(fetch_financial_details_for_companies)
                    if 'year' in sig_det.parameters:
                        details = fetch_financial_details_for_companies(
                            neo_driver,
                            company_symbols=symbols,
                            year=end_year,
                            sectors=selected_sectors
                        )
                    else:
                        details = fetch_financial_details_for_companies(_driver, company_symbols=symbols)
                    st.session_state.similar_details = details or {}
                except Exception as e:
                    st.error(f"❌ Failed to fetch details: {str(e)}")
                    st.session_state.similar_details = {}
                finally:
                    if hasattr(neo_driver, "close"):
                        neo_driver.close()

    if st.session_state.similar_companies:
        st.divider()
        st.subheader(
            f"🔎 Companies Similar to {st.session_state.last_symbol} "
            f"({start_year}-{end_year} {family_display.split('(')[0].strip()})"
        )
        st.caption(f"Aggregation: {weight_display} | Sectors: {', '.join(selected_sectors)} | Market Cap: {', '.join(selected_cap_classes)}")
        
        peers = st.session_state.similar_companies
        details = st.session_state.similar_details
        metric_year = st.session_state.metric_year or end_year

        metric_handlers = {
            "Revenue": lambda m: m.get("revenue"),
            "Net Income": lambda m: m.get("netIncome"),
            "Gross Profit": lambda m: m.get("grossProfit"),
            "Operating Income": lambda m: m.get("operatingIncome"),
            "Total Assets": lambda m: m.get("totalAssets"),
            "Total Liabilities": lambda m: m.get("totalLiabilities"),
            "Equity": lambda m: m.get("totalStockholdersEquity"),
            "Cash": lambda m: m.get("cashAndCashEquivalents"),
            "Operating Cash Flow": lambda m: m.get("operatingCashFlow"),
            "Free Cash Flow": lambda m: m.get("freeCashFlow"),
            "CapEx": lambda m: m.get("capitalExpenditure"),
            "ROE": lambda m: (
                (m.get("netIncome") / m.get("totalStockholdersEquity")) 
                if m.get("totalStockholdersEquity") else None
            ),
            "Current Ratio": lambda m: (
                (m.get("totalCurrentAssets") / m.get("totalCurrentLiabilities")) 
                if m.get("totalCurrentLiabilities") else None
            )
        }
        
        num_cols = min(4, len(selected_metrics))
        cols = st.columns(num_cols)
        for idx, (sym, score) in enumerate(peers, start=1):
            meta = details.get(sym, {})
            company_name = meta.get("companyName", sym)
            sector = meta.get("sector", "N/A")
            industry = meta.get("industry", "N/A")
            with st.container():
                header_col, score_col = st.columns([4, 1])
                with header_col:
                    st.markdown(f"**{idx}. [{sym}] {company_name}**")
                    st.caption(f"{sector} › {industry}")
                with score_col:
                    st.metric("Similarity", f"{score:.0%}")
                    st.progress(float(score), text=f"Cosine: {score:.4f}")
                col_index = 0
                for metric in selected_metrics:
                    if col_index >= num_cols:
                        col_index = 0
                        cols = st.columns(num_cols)
                    with cols[col_index]:
                        if metric in metric_handlers:
                            value = metric_handlers[metric](meta)
                            st.metric(
                                f"{metric} ({metric_year})", 
                                format_value(value),
                                help=f"Latest available data for {metric_year}"
                            )
                        else:
                            st.metric(metric, "N/A")
                    col_index += 1
                st.divider()
