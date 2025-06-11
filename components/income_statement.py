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

    df_income_raw = fetch_income_statement_data(neo_driver, symbol, start_year)

    if df_income_raw.empty:
        st.warning(f"No income statement data found for {symbol} from {start_year}.")
        return

    df_income = df_income_raw.copy()
    for col in df_income.columns:
        if df_income[col].dtype == "object" or pd.api.types.is_string_dtype(df_income[col]):
            df_income[col] = df_income[col].replace({pd.NA: None})
        elif pd.api.types.is_numeric_dtype(df_income[col]):
            try:
                if df_income[col].hasnans:
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

    df_display_interactive = df_income.set_index("year")
    df_transposed = df_display_interactive.transpose()

    metric_categories_for_table = {
        "Revenue & Gross Profit": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitMargin'],
        "Operating Expenses": ['researchAndDevelopmentExpenses', 'sellingGeneralAndAdministrativeExpenses', 'generalAndAdministrativeExpenses', 'operatingExpenses'],
        "Operating Income": ['operatingIncome', 'operatingIncomeMargin'],
        "Non-Operating & Pre-Tax Income": ['interestIncome', 'interestExpense', 'incomeBeforeTax'],
        "Taxes & Net Income": ['incomeTaxExpense', 'netIncome', 'netIncomeMargin']
    }

    available_metrics = [m for m in df_transposed.index if m in sum(metric_categories_for_table.values(), [])]

    st.markdown("##### Historical Data Table")
    table_html = "<table><thead><tr><th>Metric</th>" + "".join(f"<th>{col}</th>" for col in df_transposed.columns) + "</tr></thead><tbody>"

    for cat, metrics in metric_categories_for_table.items():
        cat_metrics = [m for m in metrics if m in available_metrics]
        if not cat_metrics:
            continue

        table_html += f"<tr><td colspan='{len(df_transposed.columns)+1}'><strong>{cat}</strong></td></tr>"
        for metric in cat_metrics:
            table_html += f"<tr><td>{metric}</td>"
            for i, year in enumerate(df_transposed.columns):
                val = df_transposed.loc[metric, year]
                arrow = ""
                if i > 0:
                    prev = df_transposed.loc[metric, df_transposed.columns[i - 1]]
                    try:
                        prev_scalar = prev.item() if hasattr(prev, "item") else prev
                        val_scalar = val.item() if hasattr(val, "item") else val
                    except Exception:
                        prev_scalar = prev
                        val_scalar = val
                    arrow = _arrow(prev_scalar, val_scalar, is_percent="Margin" in metric)

                formatted = format_value(val, is_percent="Margin" in metric, currency_symbol="$" if "Margin" not in metric else "")
                table_html += f"<td>{formatted}{arrow}</td>"
            table_html += "</tr>"

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)



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
