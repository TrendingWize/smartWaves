# components/similar_companies_tab.py
import streamlit as st
import numpy as np
# import inspect # Not needed if using centralized utils
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Import everything from the utils package (which uses __init__.py)
from utils import (
    get_neo4j_driver,
    fetch_financial_details_for_companies,
    fetch_sector_list,
    fetch_company_preview,       # Will use the version from utils_helpers via __init__
    fetch_market_cap_classes,    # Will use the version from utils_helpers via __init__
    load_vectors_for_similarity, # Will use the version from utils_helpers via __init__
    calculate_similarity_scores, # Will use the version from utils_helpers via __init__
    get_nearest_aggregate_similarities, # Will use the version from utils_helpers via __init__
    format_value
)

DEFAULT_DECAY = 0.7  # λ for recency-weighted mean

# --- UI Function ---
def similar_companies_tab_content() -> None:
    st.title("🔍 Fundamental Similarity Analysis")
    st.markdown("""
    Discover fundamentally similar companies using financial statement embeddings.
    *Features:* Custom time ranges, sector and market cap class filters, metric selection & visual comparisons.
    """)

    # Get the cached Neo4j driver once
    neo_driver = get_neo4j_driver()
    if not neo_driver:
        st.error("❌ Neo4j Database connection failed. Cannot load this page.")
        st.stop()

    # Pre-fetch sectors & market cap classes
    sectors_list, cap_classes_list = [], [] # Use different names to avoid conflict
    try:
        sectors_list = fetch_sector_list(neo_driver)
        cap_classes_list = fetch_market_cap_classes(neo_driver) # Uses the imported version
    except Exception as e:
        st.error(f"Couldn't fetch initial sector/market cap data: {str(e)}")
        # Provide defaults or handle error more gracefully
        sectors_list = ["Technology", "Healthcare", "Financial"]
        cap_classes_list = ["Large Cap", "Mid Cap", "Small Cap"]

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
                try:
                    # Call with standardized signature: driver first, then symbol
                    preview = fetch_company_preview(neo_driver, target_symbol)
                    if preview:
                        st.caption(f"**{preview.get('companyName', 'N/A')}** | {preview.get('sector', 'N/A')} | {preview.get('industry', 'N/A')}")
                    else:
                        st.warning(f"Symbol '{target_symbol}' not found in database or details unavailable.")
                except Exception as e: # Catch specific exceptions if possible
                    st.warning(f"Could not fetch company details for {target_symbol}: {e}")

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
                max_value=current_year - 1,
                value=max(1990, current_year - 6), # Default e.g. 2018 for 2024 current
                key="start_year_sim_tab" # Make keys unique
            )
        with col_end:
            end_year = st.number_input(
                "End Year",
                min_value=1990,
                max_value=current_year -1, # Usually data for current year is not final for embeddings
                value=current_year - 1,
                key="end_year_sim_tab"
            )
        if start_year > end_year:
            st.error("⚠️ Start year must be before or same as end year")
            st.stop()

        st.write("Candidate peers:", candidate_peers)


        # Use the fetched lists for options
        selected_sectors = st.multiselect(
            "Filter by Sector (for candidates)",
            options=sectors_list,
            default=[], # Start with no filter or select all: sectors_list
            key="sector_filter_sim_tab"
        )
        db_selected_sectors = selected_sectors if selected_sectors else None


        selected_cap_classes = st.multiselect(
            "Filter by Market Cap Class (for candidates)",
            options=cap_classes_list,
            default=[], # Start with no filter or select all: cap_classes_list
            key="cap_class_filter_sim_tab"
        )
        db_selected_cap_classes = selected_cap_classes if selected_cap_classes else None


        weight_scheme_options: Dict[str, str] = {
            "Equal-weighted mean": "mean",
            "Recency-weighted mean (λ-decay)": "decay",
            "Latest year only": "latest",
        }
        col_ws, col_lambda_ui = st.columns([2, 1]) # Renamed col_lambda to col_lambda_ui
        with col_ws:
            weight_display = st.selectbox(
                "Time Weighting",
                options=list(weight_scheme_options.keys()),
                index=0,
                key="similar_weight_scheme_select",
            )
        weight_value = weight_scheme_options[weight_display]

        decay_lambda_val = DEFAULT_DECAY # Use a different variable name
        with col_lambda_ui:
            if weight_value == "decay":
                decay_lambda_val = st.slider(
                    "λ (decay factor)",
                    min_value=0.5,
                    max_value=0.99, # Max < 1 for decay
                    value=DEFAULT_DECAY,
                    step=0.01,
                    key="similar_decay_slider",
                    help="Higher λ = more weight on recent years"
                )
            else:
                st.empty() # Keep layout consistent

        metric_options = [
            "Revenue", "Net Income", "Gross Profit", "Operating Income",
            "Total Assets", "Total Liabilities", "Equity", "Cash", # Match 'Equity' to 'totalStockholdersEquity'
            "Operating Cash Flow", "Free Cash Flow", "CapEx", "ROE", "Current Ratio"
        ]
        selected_metrics = st.multiselect(
            "Key Metrics to Compare",
            options=metric_options,
            default=metric_options[:8],
            key="metric_selection_sim_tab"
        )

        k_results = st.slider( # Renamed
            "Number of Results",
            5, 30, value=10, step=1, # Reduced max for typical screen
            key="similar_k_slider_tab"
        )

    # Session state keys specific to this tab's results
    if "sim_tab_companies" not in st.session_state: st.session_state.sim_tab_companies = []
    if "sim_tab_details" not in st.session_state: st.session_state.sim_tab_details = {}
    if "sim_tab_last_symbol" not in st.session_state: st.session_state.sim_tab_last_symbol = None
    if "sim_tab_last_family" not in st.session_state: st.session_state.sim_tab_last_family = None
    # ... add other state persistence as needed

    search_col, _ = st.columns([1, 3])
    with search_col:
        if st.button("🚀 Find Similar Companies", use_container_width=True, type="primary"):
            if not target_symbol:
                st.warning("⚠️ Please enter a valid company symbol")
                st.stop()
            # Removed mandatory selection for sectors/cap_classes, empty means no filter
            # if not selected_sectors:
            #     st.warning("⚠️ Please select at least one sector")
            #     st.stop()
            # if not selected_cap_classes:
            #     st.warning("⚠️ Please select at least one market cap class")
            #     st.stop()

            st.session_state.sim_tab_companies = [] # Clear previous results
            st.session_state.sim_tab_details = {}

            with st.spinner(f"🔍 Analyzing {target_symbol} fundamentals..."):
                try:
                    # Call the imported get_nearest_aggregate_similarities
                    peers = get_nearest_aggregate_similarities(
                        _driver=neo_driver, # Pass the main driver
                        target_sym=target_symbol,
                        embedding_family=family_value,
                        start_year=start_year,
                        end_year=end_year,
                        k=k_results + 5, # Fetch a bit more for processing
                        sectors=db_selected_sectors,
                        cap_classes=db_selected_cap_classes,
                        weight_scheme=weight_value,
                        decay_lambda=decay_lambda_val, # Pass the decay lambda
                        normalize=True # This param is in utils_helpers version, though normalization is internal
                    )
                except Exception as e:
                    st.error(f"❌ Similarity Search failed: {str(e)}")
                    peers = [] # Ensure peers is an empty list on error
                # No neo_driver.close() here

            if not peers:
                st.info("ℹ️ No similar companies found with current filters.")
            else:
                st.session_state.sim_tab_companies = peers[:k_results] # Trim to k
                st.session_state.sim_tab_last_symbol = target_symbol
                st.session_state.sim_tab_last_family = family_display
                # ... update other state vars for display

                symbols_for_details = [sym for sym, _ in st.session_state.sim_tab_companies]
                # Optionally add target symbol to fetch its details for comparison header
                if target_symbol not in symbols_for_details:
                    symbols_for_details.insert(0, target_symbol)

                if symbols_for_details:
                    with st.spinner("📊 Compiling financial profiles..."):
                        try:
                            # Call imported fetch_financial_details_for_companies
                            # Removed 'sectors' from this call, to get details for any found peer
                            details = fetch_financial_details_for_companies(
                                _driver=neo_driver, # Pass main driver
                                company_symbols=symbols_for_details,
                                year=end_year # Fetch details for the most recent year in range
                            )
                            st.session_state.sim_tab_details = details or {}
                        except Exception as e:
                            st.error(f"❌ Failed to fetch financial details: {str(e)}")
                            st.session_state.sim_tab_details = {}
                else:
                    st.session_state.sim_tab_details = {}


    if st.session_state.sim_tab_companies:
        st.divider()
        st.subheader(
            f"🔎 Companies Similar to {st.session_state.sim_tab_last_symbol} "
            f"({start_year}-{end_year}, {st.session_state.sim_tab_last_family.split('(')[0].strip()})"
        )
        # Construct caption string carefully
        caption_parts = [f"Aggregation: {weight_display}"]
        if selected_sectors: caption_parts.append(f"Sectors: {', '.join(selected_sectors)}")
        if selected_cap_classes: caption_parts.append(f"Market Cap: {', '.join(selected_cap_classes)}")
        st.caption(" | ".join(caption_parts))

        peers_to_display = st.session_state.sim_tab_companies
        details_map = st.session_state.sim_tab_details
        metric_year_display = end_year # The year for which metrics are displayed

        # Metric handlers mapping display names to how to get them from 'details_map' items
        # Keys in details_map come from fetch_financial_details_for_companies query
        metric_handlers = {
            "Revenue": lambda m: m.get("revenue"),
            "Net Income": lambda m: m.get("netIncome"),
            "Gross Profit": lambda m: m.get("grossProfit"),
            "Operating Income": lambda m: m.get("operatingIncome"),
            "Total Assets": lambda m: m.get("totalAssets"),
            "Total Liabilities": lambda m: m.get("totalLiabilities"),
            "Equity": lambda m: m.get("totalStockholdersEquity"), # Corrected key
            "Cash": lambda m: m.get("cashAndCashEquivalents"),    # Corrected key
            "Operating Cash Flow": lambda m: m.get("operatingCashFlow"),
            "Free Cash Flow": lambda m: m.get("freeCashFlow"),
            "CapEx": lambda m: m.get("capitalExpenditure"),
            "ROE": lambda m: (
                (m.get("netIncome") / m.get("totalStockholdersEquity")) * 100
                if m.get("netIncome") is not None and m.get("totalStockholdersEquity") and m.get("totalStockholdersEquity") != 0
                else None
            ),
            "Current Ratio": lambda m: (
                (m.get("totalCurrentAssets") / m.get("totalCurrentLiabilities"))
                if m.get("totalCurrentAssets") is not None and m.get("totalCurrentLiabilities") and m.get("totalCurrentLiabilities") != 0
                else None
            )
        }

        num_metric_display_cols = min(4, len(selected_metrics) if selected_metrics else 1)

        for idx, (sym, score) in enumerate(peers_to_display, start=1):
            meta = details_map.get(sym, {})
            company_name = meta.get("companyName", sym)
            peer_sector = meta.get("sector", "N/A") # Use different var name
            peer_industry = meta.get("industry", "N/A") # Use different var name

            with st.container(): # Use st.container for each peer's block
                header_col, score_col = st.columns([4, 1])
                with header_col:
                    st.markdown(f"**{idx}. [{sym}] {company_name}**")
                    st.caption(f"{peer_sector} › {peer_industry}")
                with score_col:
                    st.metric("Similarity", f"{score:.0%}", help=f"Cosine Score: {score:.4f}")
                    st.progress(float(score)) # Removed text from progress, metric is enough

                if selected_metrics:
                    metric_display_cols = st.columns(num_metric_display_cols)
                    col_idx = 0
                    for metric_name in selected_metrics:
                        with metric_display_cols[col_idx % num_metric_display_cols]:
                            value_to_format = None
                            is_percent_metric = metric_name == "ROE"
                            is_ratio_metric = metric_name == "Current Ratio"

                            if metric_name in metric_handlers:
                                value_to_format = metric_handlers[metric_name](meta)

                            formatted_val_str = format_value(value_to_format, is_percent=is_percent_metric, is_ratio=is_ratio_metric)
                            st.metric(
                                f"{metric_name} ({metric_year_display})",
                                formatted_val_str
                                # help=f"Data for {metric_year_display}" # Optional help text
                            )
                        col_idx += 1
                st.divider()


def similarity_over_time_tab_content():
    st.title("📈 Similarity Over Time")
    st.markdown(
        "Find companies whose **fundamental similarity** to the target has increased the most over your selected time window."
    )

    neo_driver = get_neo4j_driver()
    if not neo_driver:
        st.error("❌ Neo4j Database connection failed.")
        st.stop()

    try:
        sectors_list = fetch_sector_list(neo_driver)
        cap_classes_list = fetch_market_cap_classes(neo_driver)
    except Exception:
        sectors_list = []
        cap_classes_list = []

    # --- UI FILTERS (all variables defined before button) ---
    with st.expander("🔧 Search Parameters", expanded=True):
        col_sym, col_family = st.columns([2, 2])
        with col_sym:
            target_symbol = (
                st.text_input(
                    "Target Company Symbol (e.g., NVDA, AAPL)",
                    value=st.session_state.get("so_time_target_sym", "NVDA"),
                    key="so_time_symbol_input",
                )
                .strip()
                .upper()
            )
            if target_symbol:
                try:
                    preview = fetch_company_preview(neo_driver, target_symbol)
                    if preview:
                        st.caption(f"**{preview.get('companyName', 'N/A')}** | {preview.get('sector', 'N/A')} | {preview.get('industry', 'N/A')}")
                    else:
                        st.warning(f"Symbol '{target_symbol}' not found in database.")
                except Exception as e:
                    st.warning(f"Could not fetch company details for {target_symbol}: {e}")

        embedding_family_options = {
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
                key="so_time_family_select",
            )
        family_value = embedding_family_options[family_display]

        current_year = datetime.now().year
        col_start, col_end = st.columns(2)
        with col_start:
            start_year = st.number_input(
                "Start Year",
                min_value=1990,
                max_value=current_year - 1,
                value=max(1990, current_year - 6),
                key="so_time_start_year"
            )
        with col_end:
            end_year = st.number_input(
                "End Year",
                min_value=1990,
                max_value=current_year - 1,
                value=current_year - 1,
                key="so_time_end_year"
            )
        if start_year >= end_year:
            st.error("⚠️ Start year must be before end year")
            st.stop()

        selected_sectors = st.multiselect(
            "Filter by Sector",
            options=sectors_list,
            default=[],
            key="so_time_sector_filter"
        )
        db_selected_sectors = selected_sectors if selected_sectors else None

        selected_cap_classes = st.multiselect(
            "Filter by Market Cap Class",
            options=cap_classes_list,
            default=[],
            key="so_time_cap_class_filter"
        )
        db_selected_cap_classes = selected_cap_classes if selected_cap_classes else None

        st.caption("Results show the top 20 companies whose similarity to the target **increased the most** over the window.")

    # --- ACTION BUTTON ---
    if st.button("🚀 Find Movers Towards Target", key="so_time_search_button"):
        with st.spinner("Searching..."):
            try:
                # Candidate universe (like 'similar_companies', but k=150 for bigger pool)
                candidate_peers = get_nearest_aggregate_similarities(
                    _driver=neo_driver,
                    target_sym=target_symbol,
                    embedding_family=family_value,
                    start_year=start_year,
                    end_year=end_year,
                    k=150,
                    sectors=db_selected_sectors,
                    cap_classes=db_selected_cap_classes,
                    weight_scheme="mean",
                    decay_lambda=DEFAULT_DECAY,
                    normalize=True
                )
            except Exception as e:
                st.error(f"❌ Similarity search failed: {str(e)}")
                return

            movers = []
            for sym, _ in candidate_peers:
                try:
                    # Per-year similarity
                    sim_start = calculate_similarity_scores(
                        _driver=neo_driver,
                        target_sym=target_symbol,
                        candidate_sym=sym,
                        embedding_family=family_value,
                        year=start_year,
                        normalize=True
                    )
                    sim_end = calculate_similarity_scores(
                        _driver=neo_driver,
                        target_sym=target_symbol,
                        candidate_sym=sym,
                        embedding_family=family_value,
                        year=end_year,
                        normalize=True
                    )
                    delta = sim_end - sim_start
                    if delta > 0:
                        movers.append((sym, sim_start, sim_end, delta))
                except Exception:
                    continue

            # Sort by delta descending, take top 20
            movers.sort(key=lambda x: -x[3])
            movers = movers[:20]

            if not movers:
                st.info("ℹ️ No movers found.")
            else:
                st.subheader(f"🏃‍♂️ Top 20 Companies Moving Towards {target_symbol} ({start_year}→{end_year})")
                for idx, (sym, s_start, s_end, delta) in enumerate(movers, 1):
                    st.markdown(
                        f"**{idx}. {sym}**: Δ Similarity = `{delta:.2%}` "
                        f"({s_start:.2%} → {s_end:.2%})"
                    )
                st.success("Analysis complete!")
