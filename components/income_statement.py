# components/income_statement.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.utils_helpers import (
    calculate_delta,
    _arrow,
    R_display_metric_card,
    fetch_income_statement_data,
    get_neo4j_driver,
    format_value
)

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
