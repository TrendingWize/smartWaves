# components/income_statement.py
import streamlit as st
import pandas as pd
import numpy as np # Import numpy for np.nan if you prefer that over None
import plotly.express as px
from utils.utils_helpers import (                   # ← pull everything from utils.py instead
    calculate_delta,
    _arrow,
    R_display_metric_card,
    fetch_income_statement_data,
    get_neo4j_driver,
    format_value
)
# --- Main Tab Function ---
def income_statement_tab_content(selected_symbol_from_app):
    symbol = selected_symbol_from_app
    start_year_default = 2017

    start_year = st.number_input(
        "View Data From Year:",
        min_value=2000,
        max_value=pd.Timestamp.now().year,
        value=start_year_default,
        step=1,
        key=f"income_start_year_{symbol}"
    )

    if not symbol:
        st.info("Symbol not provided to Income Statement tab.")
        return

    neo_driver = get_neo4j_driver()
    if not neo_driver:
        return

    df_income_raw = fetch_income_statement_data(neo_driver, symbol, start_year) # Fetch raw data

    if df_income_raw.empty:
        st.warning(f"No income statement data found for {symbol} from {start_year}.")
        return

    # Convert pd.NA to None for Plotly compatibility
    # Create a copy to avoid modifying the cached DataFrame directly if it's important
    df_income = df_income_raw.copy()
    for col in df_income.columns:
        if df_income[col].dtype == "object" or pd.api.types.is_string_dtype(df_income[col]):
            # For object columns, explicitly replace pd.NA if they might exist
             # (less common for numeric data from DB, but good for safety)
            df_income[col] = df_income[col].replace({pd.NA: None})
        elif pd.api.types.is_numeric_dtype(df_income[col]):
            # For numeric columns, convert to float to allow None/np.nan, then replace pd.NA
            # This is crucial if columns were integers and pd.NA was used.
            try:
                # Attempt to convert to a float type that supports NaN (like float64)
                # if not pd.api.types.is_float_dtype(df_income[col]):
                #     df_income[col] = df_income[col].astype(float) # This might fail if original type cannot cast directly
                
                # More robust: fillna(np.nan) is usually sufficient if plotly handles np.nan
                # Or replace pd.NA with None specifically if that's preferred for JSON.
                # If a column is numeric and contains pd.NA, replace with np.nan
                # This will make the column float if it wasn't already.
                if df_income[col].hasnans: # Check if there are any NA-like values
                    # Replace pd.NA specifically with np.nan, which Plotly handles well.
                    # Using .astype(object).where(df_income[col].notna(), None) also works to replace with None.
                    df_income[col] = df_income[col].apply(lambda x: np.nan if pd.isna(x) else x).astype(float)

            except Exception as e:
                st.warning(f"Could not convert column {col} to float for NA handling: {e}")


    latest_data = df_income.iloc[-1] if not df_income.empty else pd.Series(dtype='float64')
    prev_data = df_income.iloc[-2] if len(df_income) > 1 else pd.Series(dtype='float64')

    metric_groups_config = [
        {"section_title": "💵 Revenue & Profitability", "separator": True},
        {"card_metric": "revenue", "card_title": "Revenue", "chart_metric": "revenue", "chart_title": "Revenue", "help": "Total Revenue", "is_percent_card": False},
        {"card_metric": "grossProfit", "card_title": "Gross Profit", "chart_metric": "grossProfit", "chart_title": "Gross Profit", "help": "Revenue - Cost of Revenue", "is_percent_card": False},
        {"card_metric": "grossProfitMargin", "card_title": "Gross Profit Mgn", "chart_metric": "grossProfitMargin", "chart_title": "Gross Profit Mgn", "help": "Gross Profit / Revenue", "is_percent_card": True},
        {"card_metric": "operatingIncome", "card_title": "Operating Income", "chart_metric": "operatingIncome", "chart_title": "Operating Income", "help": "Income from Core Business Operations", "is_percent_card": False},

        {"section_title": "⚙️ Operating Expenses", "separator": True},
        {"card_metric": "operatingExpenses", "card_title": "Total OpEx", "chart_metric": "operatingExpenses", "chart_title": "Total OpEx", "help": "Total Operating Expenses", "is_percent_card": False},
        {"card_metric": "researchAndDevelopmentExpenses", "card_title": "R&D Exp.", "chart_metric": "researchAndDevelopmentExpenses", "chart_title": "R&D Exp.", "help": "Research & Development Expenses", "is_percent_card": False},
        {"card_metric": "sellingGeneralAndAdministrativeExpenses", "card_title": "SG&A Exp.", "chart_metric": "sellingGeneralAndAdministrativeExpenses", "chart_title": "SG&A Exp.", "help": "Selling, General & Administrative Expenses", "is_percent_card": False},
        {"card_metric": "generalAndAdministrativeExpenses", "card_title": "G&A Exp.", "chart_metric": "generalAndAdministrativeExpenses", "chart_title": "G&A Exp.", "help": "General & Administrative Expenses (if separate from total SG&A)", "is_percent_card": False},

        {"section_title": "🏆 Net Income & Final Metrics", "separator": True},
        {"card_metric": "incomeBeforeTax", "card_title": "Income Before Tax", "chart_metric": "incomeBeforeTax", "chart_title": "Income Before Tax", "help": "Pre-Tax Income", "is_percent_card": False},
        {"card_metric": "netIncome", "card_title": "Net Income", "chart_metric": "netIncome", "chart_title": "Net Income", "help": "Net Income After Taxes", "is_percent_card": False},
        {"card_metric": "netIncomeMargin", "card_title": "Net Income Mgn", "chart_metric": "netIncomeMargin", "chart_title": "Net Income Mgn", "help": "Net Income / Revenue", "is_percent_card": True},
    ]

    active_metric_groups = []
    current_section_groups = []
    for group_config in metric_groups_config:
        if group_config.get("section_title"):
            if current_section_groups:
                active_metric_groups.append({"is_section_data": True, "groups": current_section_groups})
                current_section_groups = []
            active_metric_groups.append(group_config)
        # Use latest_data (which comes from the NA-handled df_income)
        elif group_config.get("card_metric") and pd.notna(latest_data.get(group_config["card_metric"])):
            current_section_groups.append(group_config)
    if current_section_groups:
        active_metric_groups.append({"is_section_data": True, "groups": current_section_groups})

    for section_item in active_metric_groups:
        if section_item.get("section_title"):
            if section_item.get("separator") and section_item != active_metric_groups[0]:
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
                    R_display_metric_card(st, group["card_metric"], latest_data, prev_data,
                                        is_percent=group.get("is_percent_card", False),
                                        help_text=group.get("help", ""),
                                        currency_symbol="$")

            st.markdown("##### Trends")
            chart_cols = st.columns(num_active_groups if num_active_groups > 0 else 1)
            for i, group in enumerate(groups_to_display):
                col_to_use_chart = chart_cols[i % num_active_groups] if num_active_groups > 0 else chart_cols[0]
                with col_to_use_chart:
                    chart_metric = group.get("chart_metric", group["card_metric"])
                    chart_title = group.get("chart_title", group["card_title"])
                    is_percent_chart = "Margin" in chart_metric

                    # Ensure the data for the chart from df_income is used (which has NA handled)
                    if chart_metric in df_income.columns and df_income[chart_metric].notna().any():
                        fig = px.line(df_income, x="year", y=chart_metric, title=chart_title, markers=True,
                                      labels={chart_metric: ("%" if is_percent_chart else "USD")})
                        common_layout_updates = dict(title_x=0.5, title_font_size=14, margin=dict(t=35, b=5, l=5, r=5), height=250)
                        if is_percent_chart: fig.update_layout(yaxis_ticksuffix="%", **common_layout_updates)
                        else: fig.update_layout(yaxis_tickformat="$,.0f", **common_layout_updates)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.caption(f"{chart_title} data N/A")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📄 Detailed Income Statement")

    metric_categories_for_table = {
        "Revenue & Gross Profit": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitMargin'],
        "Operating Expenses": ['researchAndDevelopmentExpenses', 'sellingGeneralAndAdministrativeExpenses', 'generalAndAdministrativeExpenses', 'operatingExpenses'],
        "Operating Income": ['operatingIncome', 'operatingIncomeMargin'],
        "Non-Operating & Pre-Tax Income": ['interestIncome', 'interestExpense', 'incomeBeforeTax'],
        "Taxes & Net Income": ['incomeTaxExpense', 'netIncome', 'netIncomeMargin']
    }

    all_metrics_in_categories = [metric for sublist in metric_categories_for_table.values() for metric in sublist]
    available_metrics_for_table = [m for m in all_metrics_in_categories if m in df_income.columns]

    if 'active_chart_metric_is' not in st.session_state:
        st.session_state.active_chart_metric_is = "None"
    if 'current_symbol_is' not in st.session_state or st.session_state.current_symbol_is != symbol:
        st.session_state.active_chart_metric_is = "None"
        st.session_state.current_symbol_is = symbol

    selectbox_options = [("None (Hide Chart)", "None")]
    for cat_name, metric_list_for_cat in metric_categories_for_table.items():
        for metric_key_name in metric_list_for_cat:
            if metric_key_name in available_metrics_for_table:
                disp_name = metric_key_name.replace("Margin", " Mgn").replace("Expenses", " Exp.").replace("Income", "Inc.").title().replace("And", "&")
                if "Margin" in metric_key_name : disp_name += " (%)"
                selectbox_options.append((f"{cat_name} - {disp_name}", metric_key_name))

    try:
        current_selection_index = [opt[1] for opt in selectbox_options].index(st.session_state.active_chart_metric_is)
    except ValueError:
        current_selection_index = 0

    selected_idx = st.selectbox(
        "Select metric to plot from table:",
        options=range(len(selectbox_options)),
        format_func=lambda x: selectbox_options[x][0],
        index=current_selection_index,
        key=f"plot_select_is_{symbol}"
    )
    st.session_state.active_chart_metric_is = selectbox_options[selected_idx][1]

    chart_slot_is = st.empty()
    if st.session_state.active_chart_metric_is != "None":
        metric_to_plot = st.session_state.active_chart_metric_is
        is_percent_metric_chart = "Margin" in metric_to_plot

        metric_display_name_chart = "Metric"
        for disp_name_tuple in selectbox_options:
            if disp_name_tuple[1] == metric_to_plot:
                metric_display_name_chart = disp_name_tuple[0].split(" - ")[-1]
                break

        with chart_slot_is.container():
            st.markdown(f"#### Trend for {metric_display_name_chart}")
            # Use df_income (NA handled) for plotting
            fig = px.bar(df_income, x="year", y=metric_to_plot,
                         title=f"{metric_display_name_chart} Over Time for {symbol}",
                         labels={"year": "Year", metric_to_plot: metric_display_name_chart},
                         text_auto=(".2f" if is_percent_metric_chart else ".2s"))
            if is_percent_metric_chart:
                fig.update_layout(yaxis_ticksuffix="%")
            else:
                fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    st.markdown("##### Historical Data Table")

    if df_income.empty or not available_metrics_for_table:
        st.info("No data available for the detailed table.")
    else:
        # Use df_income (NA handled) for table display as well
        df_display_interactive = df_income.set_index("year")[available_metrics_for_table].transpose()

        header_cols_html = f"<th class='table-metric-name' style='background-color: #f8f9fa; border-bottom: 1px solid #ddd;'>Metric</th>"
        for year_val in df_display_interactive.columns:
            header_cols_html += f"<th class='table-value-cell' style='background-color: #f8f9fa; border-bottom: 1px solid #ddd;'>{year_val}</th>"

        table_html = f"<table style='width:100%; border-collapse: collapse; font-size: 0.9em;'><thead><tr>{header_cols_html}</tr></thead><tbody>"

        first_category_processed = False
        for category_name, metrics_in_category_list in metric_categories_for_table.items():
            current_cat_metrics = [m for m in metrics_in_category_list if m in df_display_interactive.index]
            if not current_cat_metrics:
                continue

            if first_category_processed:
                table_html += "<tr><td colspan='100%' style='border-top: 2px solid #dee2e6; height:10px;'></td></tr>"
            first_category_processed = True

            table_html += f"<tr><td colspan='100%' class='table-category-header'>{category_name}</td></tr>"

            for metric_key_name in current_cat_metrics:
                metric_display_name_row = metric_key_name.replace("Margin", " Mgn").replace("Expenses", " Exp.").replace("Income", "Inc.").title().replace("And", "&")
                if "Margin" in metric_key_name: metric_display_name_row += " (%)"

                table_html += f"<tr><td class='table-metric-name' style='font-weight:normal; border-bottom: 1px solid #eee;'>{metric_display_name_row}</td>"

                is_percent_metric_row = "Margin" in metric_key_name

                for i, year_val in enumerate(df_display_interactive.columns):
                    current_val = df_display_interactive.loc[metric_key_name, year_val]
                    if isinstance(current_val, pd.Series):
                        current_val = current_val.iloc[0]

                    arrow_html_val = ""
                    if i > 0:
                        prev_val = df_display_interactive.loc[metric_key_name, df_display_interactive.columns[i-1]]
                        try:
                            scalar_prev = prev_val.item() if hasattr(prev_val, "item") else prev_val
                            scalar_curr = current_val.item() if hasattr(current_val, "item") else current_val
                        except Exception:
                            scalar_prev = prev_val
                            scalar_curr = current_val

                        arrow_html_val = _arrow(scalar_prev, scalar_curr, is_percent=is_percent_metric_row)


                    # format_value will handle np.nan or None by returning "N/A"
                    formatted_val = format_value(current_val,
                                                 is_percent=is_percent_metric_row,
                                                 currency_symbol="$" if not is_percent_metric_row else "",
                                                 is_ratio=False,
                                                 decimals=2)

                    cell_content = f"{formatted_val}{arrow_html_val}" if pd.notna(current_val) else "N/A" # Ensure N/A if val is NA before formatting
                    table_html += f"<td class='table-value-cell' style='padding:4px 8px; border-bottom: 1px solid #eee;'>{cell_content}</td>"
                table_html += "</tr>"

        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("<hr style='border:none; border-top: 2px solid #dee2e6; margin-top:10px;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Income Statement Test")
    # ... (Mock functions as before, they should work with the NA handling) ...
    # Ensure mock data does not produce pd.NA if your mock driver doesn't handle it; use np.nan instead for mocks.
    # Or, if your mock data DOES produce pd.NA, this new code should handle it.

    class MockNeo4jDriver:
        def session(self, database=None): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def run(self, query, **kwargs):
            if kwargs.get("sym") == "NVDA": # Example symbol
                sample_data_nvda = {
                    'year': [2020, 2021, 2022, 2023],
                    'revenue': [100e6, 120e6, pd.NA, 180e6], # Test with pd.NA
                    'costOfRevenue': [40e6, 50e6, 60e6, 70e6],
                    'grossProfit': [60e6, 70e6, 90e6, 110e6],
                    'researchAndDevelopmentExpenses': [10e6, 12e6, 15e6, 18e6],
                    'sellingGeneralAndAdministrativeExpenses': [8e6, 9.5e6, 11e6, 12.5e6],
                    'generalAndAdministrativeExpenses': [3e6, 3.5e6, 4e6, 4.5e6],
                    'operatingExpenses': [18e6, 21.5e6, 26e6, 30.5e6],
                    'operatingIncome': [42e6, 48.5e6, pd.NA, 79.5e6], # Test with pd.NA
                    'interestIncome': [0.5e6, 0.6e6, 0.7e6, 0.8e6],
                    'interestExpense': [1e6, 1.1e6, 1.2e6, 1.3e6],
                    'incomeBeforeTax': [41.5e6, 48e6, 63.5e6, 79e6],
                    'incomeTaxExpense': [8.3e6, 9.6e6, 12.7e6, 15.8e6],
                    'netIncome': [33.2e6, 38.4e6, 50.8e6, 63.2e6],
                }
                records = []
                for i in range(len(sample_data_nvda['year'])):
                    record_data = {key: sample_data_nvda[key][i] for key in sample_data_nvda}
                    class MockRecord:
                        def __init__(self, data): self._data = data
                        def data(self): return self._data
                    records.append(MockRecord(record_data))
                return records
            return []
        def verify_connectivity(self): pass

    _original_get_driver = get_neo4j_driver
    def mock_get_neo4j_driver(): return MockNeo4jDriver()
    get_neo4j_driver = mock_get_neo4j_driver
    
    if 'global_selected_symbol' not in st.session_state:
        st.session_state.global_selected_symbol = "NVDA"
    income_statement_tab_content(st.session_state.global_selected_symbol)
    get_neo4j_driver = _original_get_driver
