# components/similar_companies_tab.py
import streamlit as st
import numpy as np
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Corrected import path
from utils import (
    get_neo4j_driver,
    get_nearest_aggregate_similarities,
    fetch_financial_details_for_companies,
    format_value,
    fetch_sector_list,
    fetch_company_preview
)


DEFAULT_DECAY = 0.7  # λ for recency-weighted mean

def similar_companies_tab_content() -> None:
    """Render enhanced 'Find Similar Companies' tab"""
    st.title("🔍 Fundamental Similarity Analysis")
    st.markdown("""
    Discover fundamentally similar companies using financial statement embeddings.
    *Features:* Custom time ranges, sector filters, metric selection & visual comparisons.
    """)
    
    # Initialize driver early for sector list
    neo_driver = get_neo4j_driver()
    sectors = []
    try:
        sectors = fetch_sector_list(neo_driver)
    except Exception as e:
        st.error(f"Couldn't fetch sector list: {str(e)}")
        sectors = ["Technology", "Healthcare", "Financial"]
    finally:
        if hasattr(neo_driver, "close"):
            neo_driver.close()

    # ---------- User Inputs Section ----------
    with st.expander("🔧 Search Parameters", expanded=True):
        col_sym, col_family = st.columns([2, 2])
        
        # Symbol input with preview
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
            # Company preview
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

        # Embedding family selection
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

        # Year range selection
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
        
        # Validate year range
        if start_year > end_year:
            st.error("⚠️ Start year must be before end year")
            st.stop()

        # Sector filtering
        selected_sectors = st.multiselect(
            "Filter by Sector",
            options=sectors,
            default=sectors,
            key="sector_filter"
        )

        # Aggregation weighting scheme
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
                    help="Higher λ = more weight on recent years"
                )

        # Metric selection
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

        # Number of results
        k = st.slider(
            "Number of Results", 
            5, 50, value=10, step=5, 
            key="similar_k_slider"
        )

    # ---------- Session State Management ----------
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

    # ---------- Search Execution ----------
    search_col, _ = st.columns([1, 3])
    with search_col:
        if st.button("🚀 Find Similar Companies", use_container_width=True, type="primary"):
            if not target_symbol:
                st.warning("⚠️ Please enter a valid company symbol")
                st.stop()
                
            if not selected_sectors:
                st.warning("⚠️ Please select at least one sector")
                st.stop()

            # Reset previous results
            for key in ["similar_companies", "similar_details"]:
                st.session_state[key] = {}

            # Connect & query
            neo_driver = get_neo4j_driver()
            if not neo_driver:
                st.error("❌ Database connection failed")
                st.stop()

            with st.spinner(f"🔍 Analyzing {target_symbol} fundamentals..."):
                try:
                    # Build query parameters
                    peer_kwargs: Dict[str, Any] = {
                        'target_sym': target_symbol,
                        'embedding_family': family_value,
                        'start_year': start_year,
                        'end_year': end_year,
                        'k': k + 5,  # Over-fetch for filtering
                        'sectors': selected_sectors,
                        'weight_scheme': weight_value,
                        #'decay': decay_lambda if weight_value == 'decay' else None,
                        'normalize': True  # Ensure vector normalization
                    }
                    
                    # Handle driver parameter
                    sig_peer = inspect.signature(get_nearest_aggregate_similarities)
                    if 'driver' in sig_peer.parameters:
                        peer_kwargs['driver'] = neo_driver
                    elif '_driver' in sig_peer.parameters:
                        peer_kwargs['_driver'] = neo_driver
                    
                    # Execute similarity search
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
                
            # Store results
            st.session_state.similar_companies = peers[:k]  # Trim to requested size
            st.session_state.last_symbol = target_symbol
            st.session_state.last_family = family_display
            st.session_state.last_weight = weight_display
            st.session_state.metric_year = end_year  # Use last analysis year

            # Fetch financial details
            symbols = [sym for sym, _ in peers[:k]]
            with st.spinner("📊 Compiling financial profiles..."):
                neo_driver = get_neo4j_driver()
                try:
                    sig_det = inspect.signature(fetch_financial_details_for_companies)
                    if 'year' in sig_det.parameters:
                        details = fetch_financial_details_for_companies(
                            neo_driver,
                            company_symbols=symbols,
                            year=end_year  # Get latest available data
                        )
                    else:
                        # Fallback to default implementation
                        details = fetch_financial_details_for_companies(_driver, company_symbols=symbols)
                    
                    st.session_state.similar_details = details or {}
                except Exception as e:
                    st.error(f"❌ Failed to fetch details: {str(e)}")
                    st.session_state.similar_details = {}
                finally:
                    if hasattr(neo_driver, "close"):
                        neo_driver.close()

    # ---------- Results Display ----------
    if st.session_state.similar_companies:
        st.divider()
        st.subheader(
            f"🔎 Companies Similar to {st.session_state.last_symbol} "
            f"({start_year}-{end_year} {family_display.split('(')[0].strip()})"
        )
        st.caption(f"Aggregation: {weight_display} | Sectors: {', '.join(selected_sectors)}")
        
        peers = st.session_state.similar_companies
        details = st.session_state.similar_details
        metric_year = st.session_state.metric_year or end_year

        # Generate metric display functions
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
        
        # Column layout - dynamic based on metric selection
        num_cols = min(4, len(selected_metrics))
        cols = st.columns(num_cols)
        
        for idx, (sym, score) in enumerate(peers, start=1):
            meta = details.get(sym, {})
            company_name = meta.get("companyName", sym)
            sector = meta.get("sector", "N/A")
            industry = meta.get("industry", "N/A")
            
            with st.container():
                # Header with similarity visual
                header_col, score_col = st.columns([4, 1])
                with header_col:
                    st.markdown(f"**{idx}. [{sym}] {company_name}**")
                    st.caption(f"{sector} › {industry}")
                with score_col:
                    st.metric("Similarity", f"{score:.0%}")
                    st.progress(float(score), text=f"Cosine: {score:.4f}")
                
                # Metrics display
                col_index = 0
                for metric in selected_metrics:
                    if col_index >= num_cols:
                        col_index = 0
                        cols = st.columns(num_cols)  # New row
                    
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
