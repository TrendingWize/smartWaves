# components/similar_companies_tab.py
import streamlit as st
import numpy as np
# import inspect # No longer strictly needed for param checking if signatures are consistent
from datetime import datetime
from typing import Dict, List, Tuple, Any

# --- Neo4j and Data Helpers (from utils.utils_helpers) ---
from .utils_helpers import ( # Correct: relative import from sibling module
    get_neo4j_driver,
    fetch_market_cap_classes,
    load_vectors_for_similarity,
    calculate_similarity_scores,
    get_nearest_aggregate_similarities,
    fetch_sector_list,
    fetch_company_preview,
    fetch_financial_details_for_companies,
    fetch_income_statement_data,
    format_value
)


__all__ = [
    # Auth
    "get_db_connection", "create_user_table", "hash_password", "verify_password",
    "add_user", "get_user", "get_unapproved_users", "approve_user",
    "update_session_token", "initialize_session_state", "login_user", "logout_user",
    "check_concurrent_login",

    # Neo4j & Data
    "get_neo4j_driver", "fetch_market_cap_classes", "load_vectors_for_similarity",
    "calculate_similarity_scores", "get_nearest_aggregate_similarities",
    "fetch_sector_list", "fetch_company_preview",
    "fetch_financial_details_for_companies", "fetch_income_statement_data",
    "format_value",
]

DEFAULT_DECAY = 0.7  # λ for recency-weighted mean

# --- UI Function ---
def similar_companies_tab_content() -> None:
    st.title("🔍 Fundamental Similarity Analysis")
    st.markdown("""
    Discover fundamentally similar companies using financial statement embeddings.
    *Features:* Custom time ranges, sector and market cap class filters, time weighting, metric selection & visual comparisons.
    """)
    
    neo_driver = get_neo4j_driver()
    if not neo_driver:
        st.error("❌ Database connection failed. Please check Neo4j credentials and connectivity.")
        st.stop()

    sectors, cap_classes_list = [], [] # Renamed to avoid conflict
    try:
        sectors = fetch_sector_list(neo_driver)
        cap_classes_list = fetch_market_cap_classes(neo_driver)
    except Exception as e:
        st.error(f"Couldn't pre-fetch sector/market cap classes: {str(e)}")
        # Provide some defaults if fetching fails, or handle more gracefully
        sectors = ["Technology", "Healthcare", "Financial Services"] 
        cap_classes_list = ["Large Cap", "Mid Cap", "Small Cap"]
    # No neo_driver.close() here, as it's managed by @st.cache_resource

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
                # Preview driver is the same main driver
                try:
                    preview = fetch_company_preview(neo_driver, target_symbol)
                    if preview:
                        st.caption(f"**{preview.get('companyName', 'N/A')}** | {preview.get('sector', 'N/A')} | {preview.get('industry', 'N/A')}")
                    else:
                        st.warning(f"Symbol '{target_symbol}' not found or details unavailable.")
                except Exception as e:
                    st.warning(f"Couldn't fetch company preview: {e}")


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
                index=0, # Default to "Combined Financials"
                key="similar_family_select",
            )
        family_value = embedding_family_options[family_display]

        current_year = datetime.now().year
        col_start, col_end = st.columns(2)
        with col_start:
            start_year = st.number_input(
                "Start Year", 
                min_value=1990, 
                max_value=current_year -1, # Max is previous year if current year data isn't final
                value=max(1990, current_year - 6), # Default to 5 years ago (e.g., 2023-5 = 2018, for 2017-2022)
                key="start_year_sim" # Unique key
            )
        with col_end:
            end_year = st.number_input(
                "End Year", 
                min_value=1990, 
                max_value=current_year -1, # Max is previous year
                value=current_year - 1, # Default to previous year
                key="end_year_sim" # Unique key
            )
        if start_year > end_year:
            st.error("⚠️ Start year must be less than or equal to end year.")
            st.stop()

        selected_sectors = st.multiselect(
            "Filter by Sector (for candidates)",
            options=sectors,
            default=[], # Default to no sector filter or a common one
            key="sector_filter_sim" # Unique key
        )
        if not selected_sectors: # If no sectors selected, imply all. Pass None to function.
            db_selected_sectors = None
        else:
            db_selected_sectors = selected_sectors


        selected_cap_classes = st.multiselect(
            "Filter by Market Cap Class (for candidates)",
            options=cap_classes_list,
            default=[], # Default to no cap class filter
            key="cap_class_filter_sim" # Unique key
        )
        if not selected_cap_classes:
            db_selected_cap_classes = None
        else:
            db_selected_cap_classes = selected_cap_classes

        weight_scheme_options: Dict[str, str] = {
            "Equal-weighted mean": "mean",
            "Recency-weighted mean (λ-decay)": "decay",
            "Latest year only": "latest",
        }
        col_ws, col_lambda = st.columns([2, 1])
        with col_ws:
            weight_display = st.selectbox(
                "Time Weighting for Scores",
                options=list(weight_scheme_options.keys()),
                index=0,
                key="similar_weight_scheme_select",
            )
        weight_value = weight_scheme_options[weight_display]

        decay_lambda_val = DEFAULT_DECAY # Renamed to avoid conflict
        with col_lambda:
            if weight_value == "decay":
                decay_lambda_val = st.slider(
                    "λ (decay factor)",
                    min_value=0.5,
                    max_value=0.99, # Max < 1
                    value=DEFAULT_DECAY,
                    step=0.01,
                    key="similar_decay_slider",
                    help="Higher λ gives more weight to recent years (e.g., 0.9 weights recent more than 0.5)"
                )
            else:
                st.write("") # Placeholder to keep layout consistent

        metric_options = [
            "Revenue", "Net Income", "Gross Profit", "Operating Income",
            "Total Assets", "Total Liabilities", "Total Stockholders Equity", "Cash And Cash Equivalents", # Matched to DB fields
            "Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Market Cap",
            "ROE", "Current Ratio"
        ]
        selected_metrics = st.multiselect(
            "Key Metrics to Compare (from latest available data)",
            options=metric_options,
            default=metric_options[:8], # Default to first 8
            key="metric_selection_sim" # Unique key
        )

        k_results = st.slider( # Renamed variable
            "Number of Similar Companies to Display",
            min_value=5, max_value=30, value=10, step=1,
            key="similar_k_slider"
        )

    # Initialize session state keys for this tab
    state_keys_defaults = {
        "similar_companies_found": [],
        "similar_companies_details": {},
        "last_search_target_sym": None,
        "last_search_family_display": None,
        "last_search_weight_display": None,
        "last_search_metric_year": None,
    }
    for key, default_val in state_keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    search_col, _ = st.columns([1, 3]) # Or adjust ratio
    with search_col:
        if st.button("🚀 Find Similar Companies", use_container_width=True, type="primary"):
            if not target_symbol:
                st.warning("⚠️ Please enter a target company symbol.")
                st.stop()
            # Sector and Cap Class filters are optional; if empty, means all.

            # Clear previous results
            st.session_state.similar_companies_found = []
            st.session_state.similar_companies_details = {}
            
            # Neo driver should already be available and checked
            with st.spinner(f"🔍 Analyzing fundamental similarity for {target_symbol}..."):
                try:
                    peers = get_nearest_aggregate_similarities(
                        driver=neo_driver, # Pass the driver consistently
                        target_sym=target_symbol,
                        embedding_family=family_value,
                        start_year=start_year,
                        end_year=end_year,
                        sectors=db_selected_sectors, # Use the processed list (None if all)
                        cap_classes=db_selected_cap_classes, # Use the processed list (None if all)
                        weight_scheme=weight_value,
                        decay_lambda=decay_lambda_val,
                        # normalize=True, # Normalization is internal to similarity calculation
                        k=k_results + 5 # Fetch a bit more in case some don't have details
                    )
                except Exception as e:
                    st.error(f"❌ Similarity search failed: {str(e)}")
                    peers = [] # Ensure peers is empty on failure
                
            if not peers:
                st.info("ℹ️ No similar companies found with the current parameters and filters.")
            else:
                st.session_state.similar_companies_found = peers[:k_results] # Trim to desired k
                st.session_state.last_search_target_sym = target_symbol
                st.session_state.last_search_family_display = family_display
                st.session_state.last_search_weight_display = weight_display
                st.session_state.last_search_metric_year = end_year # Year for which details are primarily fetched

                symbols_to_fetch = [sym for sym, _ in st.session_state.similar_companies_found]
                if target_symbol not in symbols_to_fetch: # Add target company for comparison display
                    symbols_to_fetch.insert(0, target_symbol)
                
                if symbols_to_fetch:
                    with st.spinner("📊 Compiling financial profiles..."):
                        try:
                            details = fetch_financial_details_for_companies(
                                driver=neo_driver,
                                company_symbols=symbols_to_fetch,
                                year=end_year # Fetch for the end_year primarily
                            )
                            st.session_state.similar_companies_details = details or {}
                        except Exception as e:
                            st.error(f"❌ Failed to fetch financial details for peers: {str(e)}")
                            st.session_state.similar_companies_details = {}
                else:
                    st.session_state.similar_companies_details = {}


    if st.session_state.similar_companies_found:
        st.divider()
        st.subheader(
            f"🔎 Top {len(st.session_state.similar_companies_found)} Companies Similar to {st.session_state.last_search_target_sym} "
            f"({start_year}-{end_year}, {st.session_state.last_search_family_display.split('(')[0].strip()})"
        )
        caption_parts = [f"Time Weighting: {st.session_state.last_search_weight_display}"]
        if selected_sectors: caption_parts.append(f"Sectors: {', '.join(selected_sectors)}")
        if selected_cap_classes: caption_parts.append(f"Market Cap: {', '.join(selected_cap_classes)}")
        st.caption(" | ".join(caption_parts))
        
        # Target company details for comparison header (if available)
        target_details_for_header = st.session_state.similar_companies_details.get(st.session_state.last_search_target_sym, {})
        if target_details_for_header:
            st.markdown(f"**Target: [{st.session_state.last_search_target_sym}] {target_details_for_header.get('companyName', '')}**")
            cols_header = st.columns(len(selected_metrics) if selected_metrics else 1)
            metric_year_display = st.session_state.last_search_metric_year or end_year
            
            # Define metric handlers (simplified, direct access or calculated)
            # Ensure keys match those returned by fetch_financial_details_for_companies
            metric_access_keys = {
                "Revenue": "revenue", "Net Income": "netIncome", "Gross Profit": "grossProfit",
                "Operating Income": "operatingIncome", "Total Assets": "totalAssets",
                "Total Liabilities": "totalLiabilities", "Total Stockholders Equity": "totalStockholdersEquity",
                "Cash And Cash Equivalents": "cashAndCashEquivalents", "Operating Cash Flow": "operatingCashFlow",
                "Free Cash Flow": "freeCashFlow", "Capital Expenditure": "capitalExpenditure",
                "Market Cap": "marketCap",
            }
            
            for i, metric_name in enumerate(selected_metrics):
                with cols_header[i % len(cols_header)]: # Cycle through columns
                    value_to_display = "N/A"
                    if metric_name == "ROE":
                        ni = target_details_for_header.get("netIncome")
                        eq = target_details_for_header.get("totalStockholdersEquity")
                        if ni is not None and eq is not None and eq != 0:
                            value_to_display = format_value((ni / eq) * 100, is_percent=True)
                    elif metric_name == "Current Ratio":
                        ca = target_details_for_header.get("totalCurrentAssets")
                        cl = target_details_for_header.get("totalCurrentLiabilities")
                        if ca is not None and cl is not None and cl != 0:
                            value_to_display = format_value(ca / cl, is_ratio=True)
                    elif metric_name in metric_access_keys:
                        raw_value = target_details_for_header.get(metric_access_keys[metric_name])
                        value_to_display = format_value(raw_value)
                    
                    st.metric(label=f"{metric_name} ({metric_year_display})", value=value_to_display)
            st.divider()


        # Display Peer Companies
        for idx, (peer_sym, score) in enumerate(st.session_state.similar_companies_found, start=1):
            peer_details = st.session_state.similar_companies_details.get(peer_sym, {})
            company_name = peer_details.get("companyName", peer_sym)
            sector = peer_details.get("sector", "N/A")
            industry = peer_details.get("industry", "N/A")

            with st.container():
                header_col, score_col = st.columns([4, 1])
                with header_col:
                    st.markdown(f"**{idx}. [{peer_sym}] {company_name}**")
                    st.caption(f"{sector} › {industry}")
                with score_col:
                    st.metric("Similarity Score", f"{score:.0%}", help=f"Raw Cosine Similarity: {score:.4f}")
                    st.progress(float(score))

                # Display metrics for the peer
                if selected_metrics:
                    num_metric_cols = min(4, len(selected_metrics)) # Max 4 metrics per row for peers
                    metric_cols = st.columns(num_metric_cols)
                    col_idx = 0
                    for metric_name in selected_metrics:
                        with metric_cols[col_idx % num_metric_cols]:
                            value_to_display = "N/A"
                            if metric_name == "ROE":
                                ni = peer_details.get("netIncome")
                                eq = peer_details.get("totalStockholdersEquity")
                                if ni is not None and eq is not None and eq != 0:
                                    value_to_display = format_value((ni / eq) * 100, is_percent=True)
                            elif metric_name == "Current Ratio":
                                ca = peer_details.get("totalCurrentAssets")
                                cl = peer_details.get("totalCurrentLiabilities")
                                if ca is not None and cl is not None and cl != 0:
                                    value_to_display = format_value(ca / cl, is_ratio=True)
                            elif metric_name in metric_access_keys:
                                raw_value = peer_details.get(metric_access_keys[metric_name])
                                value_to_display = format_value(raw_value)
                            
                            st.metric(
                                label=f"{metric_name}", # Label is shorter for peers
                                value=value_to_display,
                                # help=f"{metric_name} for {peer_sym} ({metric_year_display})"
                            )
                        col_idx += 1
                st.divider()
