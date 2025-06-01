# components/balance_sheet.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (                   # ← pull everything from utils.py instead
    get_neo4j_driver,
    format_value,
    calculate_delta,
    _arrow,
    R_display_metric_card,
)
# --- Data Fetching for Balance Sheet ---
@st.cache_data(ttl="1h")
def fetch_balance_sheet_data(_driver, symbol: str, start_year: int) -> pd.DataFrame:
    if not _driver:
        return pd.DataFrame()
    # Query to fetch balance sheet data
    query = """
    MATCH (c:Company)-[:HAS_BALANCE_SHEET]->(bs:BalanceSheet) // Assuming this relationship
    WHERE c.symbol = $sym AND bs.fillingDate.year >= $start_yr
    RETURN 
        bs.fillingDate.year AS year,
        // Assets
        bs.cashAndCashEquivalents AS cashAndCashEquivalents,
        bs.shortTermInvestments AS shortTermInvestments,
        bs.cashAndShortTermInvestments AS cashAndShortTermInvestments,
        bs.netReceivables AS netReceivables,
        bs.inventory AS inventory,
        bs.otherCurrentAssets AS otherCurrentAssets,
        bs.totalCurrentAssets AS totalCurrentAssets,
        bs.propertyPlantEquipmentNet AS propertyPlantEquipmentNet,
        bs.goodwill AS goodwill,
        bs.intangibleAssets AS intangibleAssets,
        bs.longTermInvestments AS longTermInvestments,
        bs.taxAssets AS taxAssets, 
        bs.otherNonCurrentAssets AS otherNonCurrentAssets,
        bs.totalNonCurrentAssets AS totalNonCurrentAssets,
        bs.totalAssets AS totalAssets,
        // Liabilities
        bs.accountPayables AS accountPayables, 
        bs.shortTermDebt AS shortTermDebt,
        bs.deferredRevenue AS deferredRevenue, 
        bs.taxPayables AS taxPayables,
        bs.otherCurrentLiabilities AS otherCurrentLiabilities,
        bs.totalCurrentLiabilities AS totalCurrentLiabilities,
        bs.longTermDebt AS longTermDebt,
        bs.deferredRevenueNonCurrent AS deferredRevenueNonCurrent, // Corrected from deferredRevenue.1 if it was a typo
        bs.deferredTaxLiabilitiesNonCurrent AS deferredTaxLiabilitiesNonCurrent,
        bs.otherNonCurrentLiabilities AS otherNonCurrentLiabilities,
        bs.totalNonCurrentLiabilities AS totalNonCurrentLiabilities,
        bs.totalLiabilities AS totalLiabilities,
        // Equity
        bs.commonStock AS commonStock,
        bs.retainedEarnings AS retainedEarnings,
        bs.accumulatedOtherComprehensiveIncomeLoss AS accumulatedOtherComprehensiveIncomeLoss,
        bs.totalStockholdersEquity AS totalStockholdersEquity, 
        bs.totalLiabilitiesAndStockholdersEquity AS totalLiabilitiesAndStockholdersEquity, // bs.totalLiabilitiesAndTotalEquity if that's the field name
        // Other items that might be directly on BalanceSheet node or Company node
        // For simplicity, assuming they are on BalanceSheet node for this query
        bs.totalInvestments AS totalInvestments, // if available
        bs.totalDebt AS totalDebt,             // if available, or calculate
        bs.netDebt AS netDebt                 // if available, or calculate
    ORDER BY year ASC
    """
    try:
        with _driver.session(database="neo4j") as session:
            result = session.run(query, sym=symbol, start_yr=start_year)
            data = [record.data() for record in result]
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['year'] = df['year'].astype(int)

        # Ensure all expected columns are present, fill with NA if not
        # This list should match the fields in your Cypher query and derived metrics
        expected_bs_cols = [ 
            'year', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 
            'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 
            'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'longTermInvestments', 
            'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'totalAssets', 
            'accountPayables', 'shortTermDebt', 'deferredRevenue', 'taxPayables', 
            'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 
            'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 
            'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'totalLiabilities', 
            'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 
            'totalStockholdersEquity', 'totalLiabilitiesAndStockholdersEquity', 
            'totalInvestments', 'totalDebt', 'netDebt'
        ]
        for col in expected_bs_cols:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Calculate derived metrics if base data exists
        if 'totalDebt' not in df.columns or df['totalDebt'].isnull().all():
            if 'shortTermDebt' in df.columns and 'longTermDebt' in df.columns:
                 df['totalDebt'] = df['shortTermDebt'].fillna(0) + df['longTermDebt'].fillna(0)

        if 'netDebt' not in df.columns or df['netDebt'].isnull().all():
            if 'totalDebt' in df.columns and 'cashAndShortTermInvestments' in df.columns: # More common to use Cash & ST Inv.
                 df['netDebt'] = df['totalDebt'] - df['cashAndShortTermInvestments'].fillna(0)
            elif 'totalDebt' in df.columns and 'cashAndCashEquivalents' in df.columns:
                 df['netDebt'] = df['totalDebt'] - df['cashAndCashEquivalents'].fillna(0)


        df['currentRatio'] = df.apply(lambda row: (row['totalCurrentAssets'] / row['totalCurrentLiabilities']) \
                                      if pd.notna(row['totalCurrentAssets']) and pd.notna(row['totalCurrentLiabilities']) and row['totalCurrentLiabilities'] != 0 \
                                      else pd.NA, axis=1)
        df['debtToEquityRatio'] = df.apply(lambda row: (row['totalDebt'] / row['totalStockholdersEquity']) \
                                           if pd.notna(row['totalDebt']) and pd.notna(row['totalStockholdersEquity']) and row['totalStockholdersEquity'] != 0 \
                                           else pd.NA, axis=1)
        df['quickRatio'] = df.apply(lambda row: ((row.get('cashAndCashEquivalents',0) + row.get('shortTermInvestments',0) + row.get('netReceivables',0)) / row['totalCurrentLiabilities']) \
                                    if pd.notna(row.get('totalCurrentLiabilities')) and row['totalCurrentLiabilities'] != 0 \
                                    else pd.NA, axis=1)
        
        # Add derived ratios to expected_cols if they are always calculated
        derived_ratios = ['currentRatio', 'debtToEquityRatio', 'quickRatio']
        for col in derived_ratios:
            if col not in expected_bs_cols: expected_bs_cols.append(col)
        # Re-ensure all (including newly derived) are present for safety, though calculation above should handle it
        for col in expected_bs_cols: 
            if col not in df.columns: df[col] = pd.NA

        return df
    except Exception as e:
        st.error(f"Error fetching or processing balance sheet data for {symbol}: {e}")
        return pd.DataFrame()

# --- Main Tab Function for Balance Sheet ---
def balance_sheet_tab_content(selected_symbol_from_app):
    # The main title for the tab (e.g., "Balance Sheet Analysis") is now in 1_Financial_Dashboard.py

    symbol = selected_symbol_from_app
    start_year_default = 2017
    start_year = st.number_input(
        "View Data From Year:", 
        min_value=2000, 
        max_value=pd.Timestamp.now().year, 
        value=start_year_default, 
        step=1, 
        key=f"bs_start_year_{symbol}"
    )

    if not symbol:
        st.info("Symbol not provided to Balance Sheet tab.")
        return

    neo_driver = get_neo4j_driver()
    if not neo_driver: return
    df_bs = fetch_balance_sheet_data(neo_driver, symbol, start_year)
    
    if df_bs.empty:
        st.warning(f"No balance sheet data found for {symbol} from {start_year}.")
        return

    # Header for this specific content, now less prominent as tab itself has a header
    # st.markdown(f"### Financial Position for {symbol} ({df_bs['year'].min()} - {df_bs['year'].max()})")
    latest_data = df_bs.iloc[-1] if not df_bs.empty else pd.Series(dtype='float64')
    prev_data = df_bs.iloc[-2] if len(df_bs) > 1 else pd.Series(dtype='float64')
    
    bs_metric_groups_config = [
        {"section_title": "📊 Key Position & Liquidity", "separator": True},
        {"card_metric": "totalAssets", "card_title": "Total Assets", "help": "Total value of company assets"},
        {"card_metric": "totalLiabilities", "card_title": "Total Liabilities", "help": "Total company obligations"},
        {"card_metric": "totalStockholdersEquity", "card_title": "Total Equity", "help": "Net worth of the company (Assets - Liabilities)"},
        {"card_metric": "currentRatio", "card_title": "Current Ratio", "help": "Liquidity measure (Current Assets / Current Liabilities)", "is_ratio": True},

        {"section_title": "💳 Debt & Solvency", "separator": True},
        {"card_metric": "totalDebt", "card_title": "Total Debt", "help": "Sum of short-term and long-term debt obligations"},
        {"card_metric": "netDebt", "card_title": "Net Debt", "help": "Total Debt minus cash and highly liquid investments"},
        {"card_metric": "debtToEquityRatio", "card_title": "Debt/Equity Ratio", "help": "Leverage measure (Total Debt / Total Equity)", "is_ratio": True},
        {"card_metric": "quickRatio", "card_title": "Quick Ratio (Acid Test)", "help": "Stringent liquidity ( (Cash + ST Inv + Receivables) / Current Liab.)", "is_ratio": True},

        {"section_title": "📄 Asset Composition (Selected)", "separator": True},
        {"card_metric": "cashAndShortTermInvestments", "card_title": "Cash & ST Inv.", "help": "Cash, equivalents, and short-term marketable securities"},
        {"card_metric": "netReceivables", "card_title": "Net Receivables", "help": "Money owed by customers for goods/services delivered"},
        {"card_metric": "inventory", "card_title": "Inventory", "help": "Value of goods available for sale or use in production"},
        {"card_metric": "propertyPlantEquipmentNet", "card_title": "PP&E (Net)", "help": "Property, Plant, and Equipment, net of depreciation"},
    ]

    active_bs_metric_groups = []
    current_bs_section_groups = []
    for group_config in bs_metric_groups_config:
        if group_config.get("section_title"):
            if current_bs_section_groups:
                active_bs_metric_groups.append({"is_section_data": True, "groups": current_bs_section_groups})
                current_bs_section_groups = []
            active_bs_metric_groups.append(group_config)
        elif group_config.get("card_metric") and pd.notna(latest_data.get(group_config["card_metric"])):
            current_bs_section_groups.append(group_config)
    if current_bs_section_groups:
        active_bs_metric_groups.append({"is_section_data": True, "groups": current_bs_section_groups})

    for section_item in active_bs_metric_groups:
        if section_item.get("section_title"):
            if section_item.get("separator") and section_item != active_bs_metric_groups[0]:
                 st.markdown("---") 
            st.subheader(section_item["section_title"])
            continue
        
        if section_item.get("is_section_data"):
            groups_to_display = section_item["groups"]
            if not groups_to_display: continue
            
            num_active_groups = len(groups_to_display)
            card_cols = st.columns(num_active_groups if num_active_groups > 0 else 1)
            
            for i, group in enumerate(groups_to_display):
                col_to_use = card_cols[i % num_active_groups] if num_active_groups > 0 else card_cols[0]
                with col_to_use:
                    R_display_metric_card(st, 
                                         group["card_metric"], 
                                         latest_data, 
                                         prev_data, 
                                         is_ratio=group.get("is_ratio", False), # Pass is_ratio correctly
                                         help_text=group.get("help", ""),
                                         currency_symbol="$" if not group.get("is_ratio") else "") # No currency for ratios

            st.markdown("##### Trends")
            chart_cols = st.columns(num_active_groups if num_active_groups > 0 else 1)
            for i, group in enumerate(groups_to_display):
                col_to_use_chart = chart_cols[i % num_active_groups] if num_active_groups > 0 else chart_cols[0]
                with col_to_use_chart:
                    chart_metric = group.get("chart_metric", group["card_metric"])
                    chart_title = group.get("chart_title", group["card_title"])
                    is_ratio_chart = group.get("is_ratio", False) # Check if it's a ratio for y-axis label
                    
                    if chart_metric in df_bs.columns and df_bs[chart_metric].notna().any():
                        fig = px.line(df_bs, x="year", y=chart_metric, title=chart_title, markers=True,
                                      labels={chart_metric: ("Ratio" if is_ratio_chart else "USD")}) # Y-axis label
                        common_layout_updates = dict(title_x=0.5, title_font_size=14, margin=dict(t=35, b=5, l=5, r=5), height=250)
                        if is_ratio_chart: 
                            fig.update_layout(**common_layout_updates) # Ratios don't need specific tickformat usually
                        else: 
                            fig.update_layout(yaxis_tickformat="$,.0f", **common_layout_updates) # Monetary values
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.caption(f"{chart_title} data N/A")

    # --- Detailed Balance Sheet Table (Original Structure - Not yet refactored like Income Statement) ---
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📄 Detailed Balance Sheet")
    st.markdown("##### Interactive View: Click 📊 to Plot Trend")

    bs_table_metric_categories = {
        "Assets": {
            "Current Assets": ['cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets'],
            "Non-Current Assets": ['propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets'],
            "Total Assets": ['totalAssets']
        },
        "Liabilities": {
            "Current Liabilities": ['accountPayables', 'shortTermDebt', 'deferredRevenue', 'taxPayables', 'otherCurrentLiabilities', 'totalCurrentLiabilities'],
            "Non-Current Liabilities": ['longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities'],
            "Total Liabilities": ['totalLiabilities']
        },
        "Equity": {
            "Stockholders' Equity": ['commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'totalStockholdersEquity'],
            "Total Liabilities & Equity": ['totalLiabilitiesAndStockholdersEquity'] # Changed from TotalLiabilitiesAndTotalEquity
        },
        "Key Ratios & Items": ['currentRatio', 'debtToEquityRatio', 'quickRatio', 'totalDebt', 'netDebt', 'totalInvestments'] 
    }

    all_ordered_metrics_for_bs_table = []
    for main_cat, sub_cats_or_metrics in bs_table_metric_categories.items():
        if isinstance(sub_cats_or_metrics, dict):
            for sub_cat_name, metric_list in sub_cats_or_metrics.items():
                for metric in metric_list:
                    if metric in df_bs.columns: all_ordered_metrics_for_bs_table.append(metric)
        else: 
            for metric in sub_cats_or_metrics:
                if metric in df_bs.columns: all_ordered_metrics_for_bs_table.append(metric)
    
    # Session state for interactive chart in BS tab
    if 'active_chart_metric_bs' not in st.session_state: st.session_state.active_chart_metric_bs = None
    if 'current_symbol_bs' not in st.session_state or st.session_state.current_symbol_bs != symbol:
        st.session_state.active_chart_metric_bs = None
        st.session_state.scroll_to_chart_bs = False 
        st.session_state.current_symbol_bs = symbol
    
    st.markdown("<a id='chart_anchor_bs'></a>", unsafe_allow_html=True) 
    chart_slot_bs = st.empty()

    if st.session_state.get('scroll_to_chart_bs', False): # scroll_to_chart_bs might not be set
        js_scroll_script_bs = """<script> setTimeout(function() { const anchor = document.getElementById('chart_anchor_bs'); if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, 100); </script>"""
        st.components.v1.html(js_scroll_script_bs, height=0, scrolling=False)
        st.session_state.scroll_to_chart_bs = False

    if st.session_state.active_chart_metric_bs:
        metric_to_plot_bs = st.session_state.active_chart_metric_bs
        is_ratio_plot_bs = "Ratio" in metric_to_plot_bs 
        
        # Get a display name for the chart title (using a simplified version of R_display_metric_card's label logic)
        metric_display_name_chart_bs = metric_to_plot_bs.replace("Ratio"," Ratio").replace("Equivalents", "Equiv.").replace("Receivables","Recv.").title().replace("And", "&")
        
        with chart_slot_bs.container():
            st.markdown(f"#### Trend for {metric_display_name_chart_bs}")
            fig = px.bar(df_bs, x="year", y=metric_to_plot_bs, title=f"{metric_display_name_chart_bs} Over Time for {symbol}", 
                         labels={"year": "Year", metric_to_plot_bs: metric_display_name_chart_bs}, 
                         text_auto=(".2f" if is_ratio_plot_bs else ".2s"))
            if not is_ratio_plot_bs: fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            if st.button("Clear Chart", key=f"clear_chart_btn_bs_{metric_to_plot_bs}_{symbol}"):
                st.session_state.active_chart_metric_bs = None; st.session_state.scroll_to_chart_bs = False; chart_slot_bs.empty(); st.rerun()
    
    st.markdown("---")
    if df_bs.empty or not all_ordered_metrics_for_bs_table:
        st.info("No data for detailed interactive balance sheet table.")
    else:
        df_display_interactive_bs = df_bs.set_index("year")[all_ordered_metrics_for_bs_table].transpose()
        num_year_cols = len(df_display_interactive_bs.columns)
        
        # Styles for table (can be moved to CSS block in styles.py if preferred)
        table_cell_common_style = "padding: 6px 8px; text-align: right; border-bottom: 1px solid #e9ecef;"
        metric_cell_style_base = "padding: 6px 8px; text-align: left; font-weight: normal; border-bottom: 1px solid #e9ecef;" # Base style
        plot_cell_style = "padding: 6px 0px; text-align: center; border-bottom: 1px solid #e9ecef;"
        # These category styles are better defined globally if this table structure is kept
        main_cat_style_html = "font-weight:bold; font-size:1.1em; padding: 12px 5px 8px 5px; background-color:#e9ecef; border-bottom: 2px solid #ced4da;"
        sub_cat_style_html = "font-weight:bold; font-size:1.0em; padding: 10px 5px 5px 10px; background-color:#f8f9fa; border-bottom: 1px solid #e9ecef;"

        # Table Header
        header_cols = st.columns([2.5] + [1]*num_year_cols + [0.5]) 
        header_cols[0].markdown(f"<div style='{metric_cell_style_base} font-weight:bold; background-color: #f0f2f6;'>Metric</div>", unsafe_allow_html=True)
        for i, year_val in enumerate(df_display_interactive_bs.columns):
            header_cols[i+1].markdown(f"<div style='{table_cell_common_style} font-weight:bold; background-color: #f0f2f6;'>{year_val}</div>", unsafe_allow_html=True)
        header_cols[-1].markdown(f"<div style='{plot_cell_style} font-weight:bold; background-color: #f0f2f6;'>Plot</div>", unsafe_allow_html=True)

        # Table Body
        first_main_category_processed = False
        for main_cat_name, sub_categories_or_metrics_dict in bs_table_metric_categories.items():
            if first_main_category_processed:
                 st.markdown("<hr style='border:none; border-top: 3px solid #adb5bd; margin-top:15px; margin-bottom:15px;'>", unsafe_allow_html=True)
            first_main_category_processed = True
            st.markdown(f"<div style='{main_cat_style_html}'>{main_cat_name}</div>", unsafe_allow_html=True)

            if isinstance(sub_categories_or_metrics_dict, dict): # It has sub-categories
                for sub_cat_name, metrics_in_sub_category_list in sub_categories_or_metrics_dict.items():
                    available_metrics_in_sub_cat = [m for m in metrics_in_sub_category_list if m in df_display_interactive_bs.index]
                    if not available_metrics_in_sub_cat: continue
                    
                    st.markdown(f"<div style='{sub_cat_style_html}'>{sub_cat_name}</div>", unsafe_allow_html=True)
                    for metric_key_name in available_metrics_in_sub_cat:
                        row_cols = st.columns([2.5] + [1]*num_year_cols + [0.5])
                        is_ratio_metric_row = "Ratio" in metric_key_name
                        # Simplified display name logic for row
                        metric_display_name_row = metric_key_name.replace("Ratio"," Ratio").replace("Equivalents", "Equiv.").replace("Receivables","Recv.").title().replace("And", "&")
                        
                        row_cols[0].markdown(f"<div style='{metric_cell_style_base} padding-left: 25px;'>{metric_display_name_row}</div>", unsafe_allow_html=True)
                        for i, year_val in enumerate(df_display_interactive_bs.columns):
                            current_val = df_display_interactive_bs.loc[metric_key_name, year_val]
                            arrow_html_val = "" 
                            if i > 0: 
                                prev_val = df_display_interactive_bs.loc[metric_key_name, df_display_interactive_bs.columns[i-1]]
                                arrow_html_val = _arrow(prev_val, current_val, is_percent=is_ratio_metric_row) # is_percent used for arrow direction logic
                            formatted_val = format_value(current_val, is_ratio=is_ratio_metric_row, currency_symbol="" if is_ratio_metric_row else "$")
                            cell_content = "N/A" if pd.isna(current_val) else f"{formatted_val}{arrow_html_val}"
                            row_cols[i+1].markdown(f"<div style='{table_cell_common_style}'>{cell_content}</div>", unsafe_allow_html=True)
                        with row_cols[-1]:
                             st.markdown(f"<div style='{plot_cell_style} display:flex; align-items:center; justify-content:center; height:100%;'>", unsafe_allow_html=True)
                             if st.button("📊", key=f"plot_btn_bs_{metric_key_name}_{symbol}", help=f"Plot trend for {metric_display_name_row}"):
                                 st.session_state.active_chart_metric_bs = metric_key_name; st.session_state.scroll_to_chart_bs = True; st.rerun()
                             st.markdown("</div>", unsafe_allow_html=True)
            else: # It's a flat list of metrics (e.g., Key Ratios & Items)
                for metric_key_name in sub_categories_or_metrics_dict: 
                    if metric_key_name not in df_display_interactive_bs.index: continue
                    row_cols = st.columns([2.5] + [1]*num_year_cols + [0.5])
                    is_ratio_metric_row = "Ratio" in metric_key_name
                    metric_display_name_row = metric_key_name.replace("Ratio"," Ratio").title().replace("And", "&")
                    row_cols[0].markdown(f"<div style='{metric_cell_style_base} padding-left: 15px;'>{metric_display_name_row}</div>", unsafe_allow_html=True) 
                    for i, year_val in enumerate(df_display_interactive_bs.columns):
                        current_val = df_display_interactive_bs.loc[metric_key_name, year_val]
                        arrow_html_val = ""
                        if i > 0: 
                            prev_val = df_display_interactive_bs.loc[metric_key_name, df_display_interactive_bs.columns[i-1]]
                            arrow_html_val = _arrow(prev_val, current_val, is_percent=is_ratio_metric_row)
                        formatted_val = format_value(current_val, is_ratio=is_ratio_metric_row, currency_symbol="" if is_ratio_metric_row else "$")
                        cell_content = "N/A" if pd.isna(current_val) else f"{formatted_val}{arrow_html_val}"
                        row_cols[i+1].markdown(f"<div style='{table_cell_common_style}'>{cell_content}</div>", unsafe_allow_html=True)
                    with row_cols[-1]:
                         st.markdown(f"<div style='{plot_cell_style} display:flex; align-items:center; justify-content:center; height:100%;'>", unsafe_allow_html=True)
                         if st.button("📊", key=f"plot_btn_bs_{metric_key_name}_{symbol}", help=f"Plot trend for {metric_display_name_row}"):
                             st.session_state.active_chart_metric_bs = metric_key_name; st.session_state.scroll_to_chart_bs = True; st.rerun()
                         st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none; border-top: 2px solid #dee2e6; margin-top:10px;'>", unsafe_allow_html=True)


# Standalone testing mock (ensure it aligns with any data changes)
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Balance Sheet Test")
    class MockNeo4jDriver:
        def session(self, database=None): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def run(self, query, **kwargs): return [] # Basic mock, needs data for real test
        def verify_connectivity(self): pass

    _original_get_driver = get_neo4j_driver
    def mock_get_neo4j_driver(): return MockNeo4jDriver()
    get_neo4j_driver = mock_get_neo4j_driver
    
    if 'global_selected_symbol' not in st.session_state:
        st.session_state.global_selected_symbol = "AAPL" # Mock symbol
    balance_sheet_tab_content(st.session_state.global_selected_symbol)
    get_neo4j_driver = _original_get_driver