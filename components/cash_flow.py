# components/cash_flow.py

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

# --- Helper to display metric cards (can be a shared utility) ---
def display_cf_metric_card(column_or_st, label, latest_data, prev_data, help_text=None, currency_symbol="$"):
    # This function is nearly identical to display_bs_metric_card / display_metric_card
    # Ideally, this would be a single function in utils.py
    current_val = latest_data.get(label)
    prev_val = prev_data.get(label)
    
    delta_val = calculate_delta(current_val, prev_val)
    
    delta_display = None
    if delta_val is not None:
        delta_prefix = ""
        if delta_val > 0: delta_prefix = "+"
        
        # Format delta value without currency, prefix handles sign
        val_for_format = abs(delta_val)
        if abs(val_for_format) >= 1_000_000_000: formatted_abs_delta = f"{val_for_format / 1_000_000_000:.2f}B"
        elif abs(val_for_format) >= 1_000_000: formatted_abs_delta = f"{val_for_format / 1_000_000:.2f}M"
        elif abs(val_for_format) >= 1_000: formatted_abs_delta = f"{val_for_format / 1_000:.2f}K"
        else: formatted_abs_delta = f"{val_for_format:,.0f}"
        delta_display = f"{delta_prefix}{formatted_abs_delta}"

    help_str = f"Latest: {format_value(current_val, currency_symbol=currency_symbol) if pd.notna(current_val) else 'N/A'}"
    if pd.notna(prev_val):
        help_str += f" | Previous: {format_value(prev_val, currency_symbol=currency_symbol)}"
    if help_text:
        help_str = f"{help_text}\n{help_str}"

    metric_container = column_or_st
    metric_container.metric(
        label=label.replace("Activities", "Act.").replace("ProvidedBy", "/").replace("UsedFor","Used For").replace("Expenditure","Exp.").title().replace("And","&"),
        value=format_value(current_val, currency_symbol=currency_symbol) if pd.notna(current_val) else "N/A",
        delta=delta_display,
        help=help_str
    )

# --- Data Fetching for Cash Flow Statement ---
@st.cache_data(ttl="1h")
def fetch_cash_flow_data(_driver, symbol: str, start_year: int) -> pd.DataFrame:
    if not _driver:
        return pd.DataFrame()

    # Query assumes year can be extracted from 'fillingDate'
    # and the node 'n' has all properties.
    query = """
    MATCH (n:CashFlowStatement)
    WHERE n.symbol = $sym AND n.fillingDate.year >= $start_yr
    RETURN 
        n.fillingDate.year AS year, // Or n.fiscalYear if that's more consistent
        // Operating Activities
        n.netIncome AS netIncome,
        n.depreciationAndAmortization AS depreciationAndAmortization,
        n.stockBasedCompensation AS stockBasedCompensation,
        n.changeInWorkingCapital AS changeInWorkingCapital,
        // n.accountsReceivables, // Component of changeInWorkingCapital - usually not shown as standalone in CF summary
        // n.inventory,             // Component of changeInWorkingCapital
        // n.accountsPayables,      // Component of changeInWorkingCapital
        n.otherNonCashItems AS otherNonCashItems,
        n.netCashProvidedByOperatingActivities AS netCashProvidedByOperatingActivities, // aka operatingCashFlow
        // Investing Activities
        n.investmentsInPropertyPlantAndEquipment AS investmentsInPropertyPlantAndEquipment, // aka capitalExpenditure (negative)
        n.acquisitionsNet AS acquisitionsNet,
        n.purchasesOfInvestments AS purchasesOfInvestments,
        n.salesMaturitiesOfInvestments AS salesMaturitiesOfInvestments,
        n.otherInvestingActivities AS otherInvestingActivities,
        n.netCashProvidedByInvestingActivities AS netCashProvidedByInvestingActivities, // Often n.netCashUsedForInvestingActivities
        // Financing Activities
        n.netDebtIssuance AS netDebtIssuance, // Often debtRepayment or proceedsFromDebt
        n.netCommonStockIssuance AS netCommonStockIssuance, // Often commonStockIssued or commonStockRepurchased
        // n.commonStockIssuance AS commonStockIssuance, // Component
        // n.commonStockRepurchased AS commonStockRepurchased, // Component (negative)
        n.netDividendsPaid AS netDividendsPaid, // Often dividendsPaid (negative)
        n.otherFinancingActivities AS otherFinancingActivities,
        n.netCashProvidedByFinancingActivities AS netCashProvidedByFinancingActivities, // Often n.netCashUsedProvidedByFinancingActivities
        // Summary
        n.netChangeInCash AS netChangeInCash,
        n.cashAtEndOfPeriod AS cashAtEndOfPeriod,
        n.cashAtBeginningOfPeriod AS cashAtBeginningOfPeriod,
        // Commonly derived/reported
        n.operatingCashFlow AS operatingCashFlow, // Often same as netCashProvidedByOperatingActivities
        n.capitalExpenditure AS capitalExpenditure, // Often same as investmentsInPropertyPlantAndEquipment (negative)
        n.freeCashFlow AS freeCashFlow, // OCF - CapEx
        n.incomeTaxesPaid AS incomeTaxesPaid, // For info
        n.interestPaid AS interestPaid // For info
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

        # Ensure all expected columns are present
        expected_cf_cols = [
            'netIncome', 'depreciationAndAmortization', 'stockBasedCompensation', 'changeInWorkingCapital', 
            'otherNonCashItems', 'netCashProvidedByOperatingActivities', 
            'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 
            'salesMaturitiesOfInvestments', 'otherInvestingActivities', 'netCashProvidedByInvestingActivities',
            'netDebtIssuance', 'netCommonStockIssuance', 'netDividendsPaid', 'otherFinancingActivities', 
            'netCashProvidedByFinancingActivities', 'netChangeInCash', 'cashAtEndOfPeriod', 
            'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow',
            'incomeTaxesPaid', 'interestPaid'
        ]
        for col in expected_cf_cols:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Recalculate FCF if components are present and FCF might be missing
        if 'freeCashFlow' not in df.columns or df['freeCashFlow'].isnull().all():
            if 'operatingCashFlow' in df.columns and 'capitalExpenditure' in df.columns:
                 df['freeCashFlow'] = df['operatingCashFlow'] + df['capitalExpenditure'] # CapEx is usually negative
            elif 'netCashProvidedByOperatingActivities' in df.columns and 'investmentsInPropertyPlantAndEquipment' in df.columns:
                 df['freeCashFlow'] = df['netCashProvidedByOperatingActivities'] + df['investmentsInPropertyPlantAndEquipment']


        return df
    except Exception as e:
        st.error(f"Error fetching or processing cash flow data for {symbol}: {e}")
        return pd.DataFrame()

# --- Main Tab Function for Cash Flow Statement ---
def cash_flow_tab_content(selected_symbol_from_app):
    # local_css_cash_flow() # Define if needed
    st.title("🌊 Cash Flow Statement Analysis")

    symbol = selected_symbol_from_app
    start_year_default = 2017
    start_year = st.number_input(
        "View Data From Year:", 
        min_value=2000, 
        max_value=pd.Timestamp.now().year, 
        value=start_year_default, 
        step=1, 
        key=f"cf_start_year_{symbol}" # Unique key
    )

    if not symbol:
        st.info("Symbol not provided to Cash Flow tab.")
        return

    neo_driver = get_neo4j_driver()
    if not neo_driver: return
    df_cf = fetch_cash_flow_data(neo_driver, symbol, start_year)
    
    if df_cf.empty:
        st.warning(f"No cash flow data found for {symbol} from {start_year}.")
        return

    st.markdown(f"### Cash Flow Dynamics for {symbol} ({df_cf['year'].min()} - {df_cf['year'].max()})")
    latest_data = df_cf.iloc[-1] if not df_cf.empty else pd.Series(dtype='float64')
    prev_data = df_cf.iloc[-2] if len(df_cf) > 1 else pd.Series(dtype='float64')
    
    # --- Define Metric Groups for Cash Flow Statement ---
    cf_metric_groups_config = [
        {"section_title": "💨 Key Cash Flows", "separator": True},
        {"card_metric": "netCashProvidedByOperatingActivities", "card_title": "Operating Cash Flow", "help": "Cash from core business operations (CFO/OCF)"},
        {"card_metric": "netCashProvidedByInvestingActivities", "card_title": "Investing Cash Flow", "help": "Cash from investments & divestitures (CFI)"}, # Title can be "Cash Flow from Investing"
        {"card_metric": "netCashProvidedByFinancingActivities", "card_title": "Financing Cash Flow", "help": "Cash from debt, equity, and dividends (CFF)"},
        {"card_metric": "freeCashFlow", "card_title": "Free Cash Flow (FCF)", "help": "OCF - Capital Expenditures"},

        {"section_title": "🔧 Components of Operating Cash Flow", "separator": True},
        {"card_metric": "netIncome", "card_title": "Net Income (Start)", "help": "Starting point for indirect CFO"},
        {"card_metric": "depreciationAndAmortization", "card_title": "Depr. & Amort.", "help": "Non-cash expense added back"},
        {"card_metric": "stockBasedCompensation", "card_title": "Stock Comp.", "help": "Non-cash stock compensation"},
        {"card_metric": "changeInWorkingCapital", "card_title": "Δ in Working Capital", "help": "Changes in current assets/liabilities"},

        {"section_title": "💸 Investing & Financing Details (Selected)", "separator": True},
        {"card_metric": "capitalExpenditure", "card_title": "Capital Expenditure", "help": "Investment in PP&E (usually negative)"}, # Often investmentsInPropertyPlantAndEquipment
        {"card_metric": "netDebtIssuance", "card_title": "Net Debt Issuance", "help": "Net cash from issuing/repaying debt"},
        {"card_metric": "netCommonStockIssuance", "card_title": "Net Stock Issuance", "help": "Net cash from issuing/repurchasing stock"},
        {"card_metric": "netDividendsPaid", "card_title": "Dividends Paid", "help": "Cash paid as dividends (usually negative)"},
        
        {"section_title": "💰 Cash Position Summary", "separator": True},
        {"card_metric": "netChangeInCash", "card_title": "Net Change in Cash", "help": "Overall change in cash balance"},
        {"card_metric": "cashAtBeginningOfPeriod", "card_title": "Cash (Beginning)", "help": "Cash at start of period"},
        {"card_metric": "cashAtEndOfPeriod", "card_title": "Cash (End)", "help": "Cash at end of period"},
        {"card_metric": "incomeTaxesPaid", "card_title": "Income Taxes Paid", "help": "Actual cash paid for income taxes"}, 
    ]

    active_cf_metric_groups = []
    current_cf_section_groups = []
    for group_config in cf_metric_groups_config:
        if group_config.get("section_title"):
            if current_cf_section_groups:
                active_cf_metric_groups.append({"is_section_data": True, "groups": current_cf_section_groups})
                current_cf_section_groups = []
            active_cf_metric_groups.append(group_config)
        elif pd.notna(latest_data.get(group_config["card_metric"])):
            current_cf_section_groups.append(group_config)
    if current_cf_section_groups:
        active_cf_metric_groups.append({"is_section_data": True, "groups": current_cf_section_groups})

    # --- Loop through sections and display cards & charts ---
    for section_item in active_cf_metric_groups:
        if section_item.get("section_title"):
            if section_item.get("separator"):
                st.markdown("---") if section_item != active_cf_metric_groups[0] else None
            st.subheader(section_item["section_title"])
            continue
        
        if section_item.get("is_section_data"):
            groups_to_display = section_item["groups"]
            if not groups_to_display: continue
            num_active_groups = len(groups_to_display)

            card_cols = st.columns(num_active_groups)
            for i, group in enumerate(groups_to_display):
                with card_cols[i]:
                    display_cf_metric_card(st, 
                                         group["card_metric"], 
                                         latest_data, 
                                         prev_data, 
                                         help_text=group.get("help", ""))

            st.markdown("##### Trends")
            chart_cols = st.columns(num_active_groups)
            for i, group in enumerate(groups_to_display):
                with chart_cols[i]:
                    chart_metric = group.get("chart_metric", group["card_metric"])
                    chart_title = group.get("chart_title", group["card_title"])
                    
                    if chart_metric in df_cf.columns and df_cf[chart_metric].notna().any():
                        # Cash flow charts are often better as bar charts to show inflow/outflow
                        fig = px.bar(df_cf, x="year", y=chart_metric, title=chart_title, 
                                      labels={chart_metric: "USD"}, text_auto=".2s") # Show values on bars
                        
                        common_layout_updates = dict(title_x=0.5, title_font_size=14, margin=dict(t=40, b=10, l=5, r=5), height=300)
                        fig.update_layout(yaxis_tickformat="$,.0f", **common_layout_updates)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(f"{chart_title} data N/A")

    # --- Detailed Cash Flow Table (Interactive) ---
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📄 Detailed Cash Flow Statement")
    st.markdown("##### Interactive View: Click 📊 to Plot Trend")

    cf_table_metric_categories = {
        "Cash Flow from Operating Activities": [
            'netIncome', 'depreciationAndAmortization', 'stockBasedCompensation', 
            'changeInWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities'
        ],
        "Cash Flow from Investing Activities": [
            'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments',
            'salesMaturitiesOfInvestments', 'otherInvestingActivities', 'netCashProvidedByInvestingActivities'
        ],
        "Cash Flow from Financing Activities": [
            'netDebtIssuance', 'netCommonStockIssuance', 'netDividendsPaid', 
            'otherFinancingActivities', 'netCashProvidedByFinancingActivities'
        ],
        "Cash Reconciliation & Key Metrics": [
            'netChangeInCash', 'cashAtBeginningOfPeriod', 'cashAtEndOfPeriod',
            'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow'
        ],
        "Supplemental": ['incomeTaxesPaid', 'interestPaid']
    }

    all_ordered_metrics_for_cf_table = []
    for category_name, metric_list in cf_table_metric_categories.items():
        for metric in metric_list:
            if metric in df_cf.columns:
                all_ordered_metrics_for_cf_table.append(metric)
    
    # Session state for interactive chart in CF tab
    if 'active_chart_metric_cf' not in st.session_state: st.session_state.active_chart_metric_cf = None
    if 'current_symbol_cf' not in st.session_state: st.session_state.current_symbol_cf = None
    if 'scroll_to_chart_cf' not in st.session_state: st.session_state.scroll_to_chart_cf = False

    if st.session_state.current_symbol_cf != symbol:
        st.session_state.active_chart_metric_cf = None
        st.session_state.scroll_to_chart_cf = False
        st.session_state.current_symbol_cf = symbol
    
    st.markdown("<a id='chart_anchor_cf'></a>", unsafe_allow_html=True)
    chart_slot_cf = st.empty()

    if st.session_state.get('scroll_to_chart_cf', False):
        js_scroll_script_cf = """<script> setTimeout(function() { const anchor = document.getElementById('chart_anchor_cf'); if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, 100); </script>"""
        st.components.v1.html(js_scroll_script_cf, height=0, scrolling=False)
        st.session_state.scroll_to_chart_cf = False

    if st.session_state.active_chart_metric_cf:
        metric_to_plot = st.session_state.active_chart_metric_cf
        metric_display_name_chart = metric_to_plot.replace("Activities", "Act.").replace("ProvidedBy","/").title().replace("And","&")
        
        with chart_slot_cf.container():
            st.markdown(f"#### Trend for {metric_display_name_chart}")
            # Bar chart is generally good for cash flow items
            fig = px.bar(df_cf, x="year", y=metric_to_plot, title=f"{metric_display_name_chart} Over Time for {symbol}", labels={"year": "Year", metric_to_plot: metric_display_name_chart}, text_auto=".2s")
            fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)
            if st.button("Clear Chart", key=f"clear_chart_btn_cf_{metric_to_plot}_{symbol}"):
                st.session_state.active_chart_metric_cf = None; st.session_state.scroll_to_chart_cf = False; chart_slot_cf.empty(); st.rerun()
    
    st.markdown("---")
    if df_cf.empty or not all_ordered_metrics_for_cf_table:
        st.info("No data for detailed interactive cash flow table.")
    else:
        df_display_interactive_cf = df_cf.set_index("year")[all_ordered_metrics_for_cf_table].transpose()
        num_year_cols = len(df_display_interactive_cf.columns)
        
        table_cell_common_style = "padding: 6px 8px; text-align: right; border-bottom: 1px solid #e9ecef;"
        metric_cell_style = "padding: 6px 8px; text-align: left; font-weight: 500; border-bottom: 1px solid #e9ecef; padding-left: 15px;"
        plot_cell_style = "padding: 6px 0px; text-align: center; border-bottom: 1px solid #e9ecef;"
        category_header_style = "font-weight:bold; font-size:1.05em; padding: 10px 5px 5px 5px; background-color:#f0f2f6; border-bottom: 1px solid #e9ecef;"

        header_cols = st.columns([2.5] + [1]*num_year_cols + [0.5])
        header_cols[0].markdown(f"<div style='{metric_cell_style} font-weight:bold; background-color: #f8f9fa;'>Metric</div>", unsafe_allow_html=True)
        for i, year_val in enumerate(df_display_interactive_cf.columns):
            header_cols[i+1].markdown(f"<div style='{table_cell_common_style} font-weight:bold; background-color: #f8f9fa;'>{year_val}</div>", unsafe_allow_html=True)
        header_cols[-1].markdown(f"<div style='{plot_cell_style} font-weight:bold; background-color: #f8f9fa;'>Plot</div>", unsafe_allow_html=True)

        first_category_processed = False
        for category_name, metrics_in_category in cf_table_metric_categories.items():
            available_metrics_in_category = [m for m in metrics_in_category if m in df_display_interactive_cf.index]
            if not available_metrics_in_category: continue
            
            if first_category_processed:
                 st.markdown("<hr style='border:none; border-top: 2px solid #dee2e6; margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
            first_category_processed = True
            st.markdown(f"<div style='{category_header_style}'>{category_name}</div>", unsafe_allow_html=True)
            
            for metric_key_name in available_metrics_in_category:
                row_cols = st.columns([2.5] + [1]*num_year_cols + [0.5])
                metric_display_name_row = metric_key_name.replace("Activities", "Act.").replace("ProvidedBy","/").replace("Expenditure","Exp.").title().replace("And","&")
                row_cols[0].markdown(f"<div style='{metric_cell_style}'>{metric_display_name_row}</div>", unsafe_allow_html=True)
                
                for i, year_val in enumerate(df_display_interactive_cf.columns):
                    current_val = df_display_interactive_cf.loc[metric_key_name, year_val]
                    arrow_html_val = ""; 
                    if i > 0: prev_val = df_display_interactive_cf.loc[metric_key_name, df_display_interactive_cf.columns[i-1]]; arrow_html_val = _arrow(prev_val, current_val)
                    formatted_val = format_value(current_val) # Default currency formatting
                    cell_content = "N/A" if pd.isna(current_val) else f"{formatted_val}{arrow_html_val}"
                    row_cols[i+1].markdown(f"<div style='{table_cell_common_style}'>{cell_content}</div>", unsafe_allow_html=True)
                
                button_key = f"plot_btn_cf_{metric_key_name}_{symbol}"
                with row_cols[-1]:
                     st.markdown(f"<div style='{plot_cell_style} display:flex; align-items:center; justify-content:center; height:100%;'>", unsafe_allow_html=True)
                     if st.button("📊", key=button_key, help=f"Plot trend for {metric_key_name.title()}"):
                         st.session_state.active_chart_metric_cf = metric_key_name; st.session_state.scroll_to_chart_cf = True; st.rerun()
                     st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none; border-top: 2px solid #dee2e6; margin-top:10px;'>", unsafe_allow_html=True)

# --- Mock data for standalone testing ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cash Flow Test")
    
    class MockNeo4jDriver:
        def session(self, database=None): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def run(self, query, **kwargs): return []
        def verify_connectivity(self): pass

    def get_neo4j_driver(): return MockNeo4jDriver()
    
    def format_value(value, is_percent=False, currency_symbol="$"): # Ensure full mock
        if pd.isna(value): return "N/A"
        if is_percent : return f"{value:.2f}%"
        if isinstance(value, (int, float)):
            num_str = ""
            if abs(value) >= 1_000_000_000_000: num_str = f"{value / 1_000_000_000_000:.2f}T"
            elif abs(value) >= 1_000_000_000: num_str = f"{value / 1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000: num_str = f"{value / 1_000_000:.2f}M"
            elif abs(value) >= 1_000: num_str = f"{value / 1_000:.2f}K"
            else: num_str = f"{value:,.0f}"
            return f"{currency_symbol}{num_str}" if currency_symbol else num_str
        return str(value)

    def calculate_delta(curr, prev): return curr - prev if pd.notna(curr) and pd.notna(prev) and prev != 0 else None
    def _arrow(prev, curr, is_percent=False):
        if pd.isna(prev) or pd.isna(curr): return " →"
        if curr > prev: return " <span style='color:green; font-weight:bold;'>↑</span>"
        if curr < prev: return " <span style='color:red; font-weight:bold;'>↓</span>"
        return " →"

    def mock_fetch_cash_flow_data(driver, symbol, start_year):
        if symbol == "AAPL": # Using AAPL as in your data example for CF
            data_list = []
            for i, yr_offset in enumerate(range(2, -1, -1)): # 2022, 2023, 2024
                year = 2022 + i
                factor = 1 - (yr_offset * 0.05) # slight variation factor
                data = {
                    "year": year, "symbol": "AAPL", "fillingDate": pd.Timestamp(f"{year}-11-01"), # Mock fillingDate
                    "netIncome": int(93736000000 * factor), "depreciationAndAmortization": int(11445000000 * factor),
                    "stockBasedCompensation": int(11688000000 * factor), "changeInWorkingCapital": int(3651000000 * factor),
                    "otherNonCashItems": int(-2266000000 * factor), 
                    "netCashProvidedByOperatingActivities": int(118254000000 * factor),
                    "investmentsInPropertyPlantAndEquipment": int(-9447000000 * factor), "acquisitionsNet": 0,
                    "purchasesOfInvestments": int(-48656000000 * factor), "salesMaturitiesOfInvestments": int(62346000000 * factor),
                    "otherInvestingActivities": int(-1308000000 * factor), 
                    "netCashProvidedByInvestingActivities": int(2935000000 * factor),
                    "netDebtIssuance": int(-5998000000 * factor), "netCommonStockIssuance": int(-94949000000 * factor),
                    "netDividendsPaid": int(-15234000000 * factor), "otherFinancingActivities": int(-5802000000 * factor),
                    "netCashProvidedByFinancingActivities": int(-121983000000 * factor),
                    "netChangeInCash": int(-794000000 * factor), "cashAtEndOfPeriod": int(29943000000 * factor),
                    "cashAtBeginningOfPeriod": int(30737000000 * factor), 
                    "operatingCashFlow": int(118254000000 * factor), # Often same as netCashProvidedByOperatingActivities
                    "capitalExpenditure": int(-9447000000 * factor), # Often same as investmentsInPropertyPlantAndEquipment
                    "freeCashFlow": int(108807000000 * factor),
                    "incomeTaxesPaid": int(26102000000 * factor),
                    "interestPaid": int(0 * factor) # Or some small value
                }
                data_list.append(data)
            
            df = pd.DataFrame(data_list)
            df = df[df['year'] >= start_year] # Filter by start_year for mock
             # Ensure all expected cf cols are present in mock
            expected_cf_cols_mock = [
                'netIncome', 'depreciationAndAmortization', 'stockBasedCompensation', 'changeInWorkingCapital', 
                'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 
                'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivities', 
                'netCashProvidedByInvestingActivities', 'netDebtIssuance', 'netCommonStockIssuance', 'netDividendsPaid', 
                'otherFinancingActivities', 'netCashProvidedByFinancingActivities', 'netChangeInCash', 
                'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 
                'freeCashFlow', 'incomeTaxesPaid', 'interestPaid'
            ]
            for col in expected_cf_cols_mock:
                if col not in df.columns: df[col] = pd.NA
            return df
        return pd.DataFrame()
    
    original_fetch_cf = fetch_cash_flow_data 
    fetch_cash_flow_data = mock_fetch_cash_flow_data

    cash_flow_tab_content("AAPL")

    fetch_cash_flow_data = original_fetch_cf