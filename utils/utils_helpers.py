import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from pandas.io.formats.style import Styler # pandas ≥1.0
import numpy as np # Added
from collections import defaultdict # Added
from typing import Dict, List, Tuple # Added
from typing import Any, Dict, List, Optional

cumulative_scores = defaultdict(float)
# ── Neo4j Config ───────────────────────────────────────────────────────
# Store sensitive credentials securely. In a deployed Streamlit app,
# use st.secrets (secrets.toml file). For local development, you might
# use environment variables or a local config file (not committed).
NEO4J_URI = st.secrets.get("NEO4J_URI", "neo4j+s://f9f444b7.databases.neo4j.io")
NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0")

# utils.py
@st.cache_data(ttl="1h", show_spinner="Calculating aggregated similarity scores...")
def get_nearest_aggregate_similarities(_driver, 
                                       target_sym: str, 
                                       embedding_family: str, 
                                       start_year: int = 2017, 
                                       end_year: int = 2023, 
                                       sectors=None,
                                       weight_scheme=None,
                                       normalize=True,
                                       cap_classes=cap_classes,
                                       k: int = 10) -> List[Tuple[str, float]]:
    if not _driver:
        return []

    cumulative_scores = defaultdict(float)
    years_processed_count = 0

    for year in range(start_year, end_year + 1):
        target_vector, candidate_vectors = load_vectors_for_similarity(
            _driver, year, embedding_family, sectors=sectors, target_sym=target_sym
        )
        if target_vector is None or not candidate_vectors:
            continue
        yearly_similarity_scores = calculate_similarity_scores(target_vector, candidate_vectors)
        for sym, score in yearly_similarity_scores.items():
            cumulative_scores[sym] += score
        years_processed_count += 1

    if years_processed_count == 0:
        st.warning(f"No data found for {target_sym} or its comparables in the selected year range and embedding family.")
        return []

    average_scores = {sym: score / years_processed_count for sym, score in cumulative_scores.items()}
    best_k_similar = sorted(average_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return best_k_similar

    
    cumulative_scores = defaultdict(float)  # <-- REQUIRED
    years_processed_count = 0

    for year in range(start_year, end_year + 1):
        yearly_vectors = load_vectors_for_similarity(_driver, year, embedding_family, sectors=sectors)
        target_vector = yearly_vectors.pop(target_sym, None)
        if target_vector is None:
            continue
        if not yearly_vectors:
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
                                           
def fetch_sector_list(driver=None) -> List[str]:
    """Fetch distinct sectors from database"""
    if not driver:
        driver = get_neo4j_driver()
    
    query = """
    MATCH (s:Sector)
    RETURN DISTINCT s.name AS sector
    ORDER BY sector
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record["sector"] for record in result]
    except Exception as e:
        print(f"Error fetching sectors: {str(e)}")
        return []
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()

def fetch_company_preview(symbol: str, driver=None) -> Dict[str, Any]:
    """Get basic company info"""
    if not driver:
        driver = get_neo4j_driver()
    
    query = """
    MATCH (c:Company {symbol: $symbol})
    RETURN 
        c.companyName AS companyName,
        c.sector AS sector,
        c.industry AS industry,
        c.description AS description
    LIMIT 1
    """
    try:
        with driver.session() as session:
            result = session.run(query, symbol=symbol)
            record = result.single()
            return record.data() if record else {}
    except Exception as e:
        print(f"Error fetching company preview: {str(e)}")
        return {}
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()
            
def fetch_sector_list(driver=None) -> List[str]:
    """Fetch distinct sectors from database"""
    if not driver:
        driver = get_neo4j_driver()
    
    query = """
    MATCH (s:Sector)
    RETURN DISTINCT s.name AS sector
    ORDER BY sector
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record["sector"] for record in result]
    except Exception as e:
        print(f"Error fetching sectors: {str(e)}")
        return []
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()

def fetch_company_preview(symbol: str, driver=None) -> Dict[str, Any]:
    """Get basic company info"""
    if not driver:
        driver = get_neo4j_driver()
    
    query = """
    MATCH (c:Company {symbol: $symbol})
    RETURN 
        c.companyName AS companyName,
        c.sector AS sector,
        c.industry AS industry,
        c.description AS description
    LIMIT 1
    """
    try:
        with driver.session() as session:
            result = session.run(query, symbol=symbol)
            record = result.single()
            return record.data() if record else {}
    except Exception as e:
        print(f"Error fetching company preview: {str(e)}")
        return {}
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()
# ── Neo4j Driver Cache ─────────────────────────────────────────────────
@st.cache_resource
def get_neo4j_driver():
    """
    Establishes and caches a connection to the Neo4j database.
    Returns the Neo4j driver object or None if connection fails.
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity() # Check if the connection is valid
        return driver
    except Exception as e:
        st.error(f"Neo4j connection error: {e}")
        return None

# ── Data Fetching ──────────────────────────────────────────────────────
@st.cache_data(ttl="1h") # Cache data for 1 hour
def fetch_income_statement_data(_driver, symbol: str, start_year: int = 2017) -> pd.DataFrame:
    """
    Fetches income statement data for a given company symbol and start year from Neo4j.
    """
    if not _driver:
        return pd.DataFrame()

    query = """
    MATCH (c:Company)-[:HAS_INCOME_STATEMENT]->(i:IncomeStatement)
    WHERE c.symbol IN $symbols
      AND ($year IS NULL OR i.calendarYear = $year)
      AND ($sectors IS NULL OR c.sector IN $sectors)
      AND i.fillingDate.year >= $start_yr
    RETURN
      i.fillingDate.year                     AS year,
      // Revenue & Profitability
      i.revenue                              AS revenue,
      i.costOfRevenue                        AS costOfRevenue,
      i.grossProfit                          AS grossProfit,
      // Operating Expenses
      i.researchAndDevelopmentExpenses       AS researchAndDevelopmentExpenses,
      i.generalAndAdministrativeExpenses     AS generalAndAdministrativeExpenses,
      i.sellingGeneralAndAdministrativeExpenses AS sellingGeneralAndAdministrativeExpenses,    
      i.operatingExpenses                    AS operatingExpenses, 
      i.operatingIncome                      AS operatingIncome, 
      // Other Income / (Expense)
      i.interestIncome                       AS interestIncome,
      i.interestExpense                      AS interestExpense,
      i.incomeBeforeTax                      AS incomeBeforeTax,
      // Taxes & Net Income
      i.incomeTaxExpense                     AS incomeTaxExpense,
      i.netIncome                            AS netIncome
    ORDER BY year ASC
    """
    try:
        with _driver.session(database="neo4j") as session: # Specify database if not default
            result = session.run(query, symbols=company_symbols, year=year, sectors=sectors)
            data = [record.data() for record in result]
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Ensure all expected columns are present, fill with NA if not
        expected_cols = [
            'year', 'revenue', 'costOfRevenue', 'grossProfit',
            'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
            'sellingGeneralAndAdministrativeExpenses', # Corrected name
            'operatingExpenses', 'operatingIncome',
            'interestIncome', 'interestExpense', 'incomeBeforeTax',
            'incomeTaxExpense', 'netIncome'
            # Derived metrics will be added after this block
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA 

        # Calculate derived metrics/ratios after ensuring base columns exist
        # Handle potential division by zero or NA values gracefully
        df['grossProfitMargin'] = df.apply(lambda row: (row['grossProfit'] / row['revenue']) * 100 \
                                           if pd.notna(row['grossProfit']) and pd.notna(row['revenue']) and row['revenue'] != 0 \
                                           else pd.NA, axis=1)
        df['operatingIncomeMargin'] = df.apply(lambda row: (row['operatingIncome'] / row['revenue']) * 100 \
                                               if pd.notna(row['operatingIncome']) and pd.notna(row['revenue']) and row['revenue'] != 0 \
                                               else pd.NA, axis=1)
        df['netIncomeMargin'] = df.apply(lambda row: (row['netIncome'] / row['revenue']) * 100 \
                                         if pd.notna(row['netIncome']) and pd.notna(row['revenue']) and row['revenue'] != 0 \
                                         else pd.NA, axis=1)
        
        if 'year' in df.columns : df['year'] = df['year'].astype(int) # Ensure year is integer
        return df
    except Exception as e:
        st.error(f"Error fetching or processing income data for {symbol}: {e}")
        return pd.DataFrame()


# ── Formatting and Styling Helpers ─────────────────────────────────────
def format_value(value, is_percent=False, currency_symbol="$", is_ratio=False, decimals=2):
    """
    Formats a numerical value into a string with appropriate suffixes (B, M, K),
    currency symbols, or percentage signs.
    Handles NA values by returning "N/A".
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    if is_percent: # For percentages
        return f"{value:.{decimals}f}%"
    
    if is_ratio: # For raw ratios (e.g., Current Ratio)
        return f"{value:.{decimals}f}" # Display with specified decimal places
        
    # For numerical monetary values
    if isinstance(value, (int, float)):
        num_str = ""
        abs_value = abs(value) # Use absolute value for threshold checks
        
        # Determine suffix based on magnitude
        if abs_value >= 1_000_000_000_000: num_str = f"{value / 1_000_000_000_000:.{decimals}f}T" # Trillions
        elif abs_value >= 1_000_000_000: num_str = f"{value / 1_000_000_000:.{decimals}f}B"    # Billions
        elif abs_value >= 1_000_000: num_str = f"{value / 1_000_000:.{decimals}f}M"       # Millions
        elif abs_value >= 1_000: num_str = f"{value / 1_000:.{decimals}f}K"          # Thousands
        else: # Smaller numbers or those to show fully (e.g. stock price)
            num_str = f"{value:,.{decimals}f}" # Default to specified decimal places with comma for thousands
        
        # Prepend currency symbol if provided and value is not just a suffix
        return f"{currency_symbol}{num_str}" if currency_symbol and num_str else num_str
        
    return str(value) # Fallback for non-numeric, non-NA types (should be rare for metrics)

def calculate_delta(current_value, previous_value):
    """
    Calculates the difference between current and previous values.
    Returns None if data is insufficient or previous value is zero (to avoid division by zero issues downstream).
    """
    if pd.isna(current_value) or pd.isna(previous_value) or previous_value == 0:
        return None 
    return current_value - previous_value

def _arrow(prev_val, curr_val, is_percent=False): # is_percent can influence how "better" is judged for some metrics
    """
    Generates an HTML span with an arrow indicating change direction.
    Green up arrow for increase, red down arrow for decrease, neutral for no change or NA.
    """
    if pd.isna(prev_val) or pd.isna(curr_val):
        return " →" # Neutral arrow for missing data
    
    # Simple comparison: higher is up, lower is down.
    # For specific metrics (e.g., expenses), a decrease might be positive.
    # This function provides a generic visual cue; context is handled by interpretation.
    if curr_val > prev_val:
        return " <span style='color:green; font-weight:bold;'>↑</span>"
    if curr_val < prev_val:
        return " <span style='color:red; font-weight:bold;'>↓</span>"
    return " →" # Neutral for no change


def R_display_metric_card(st_container, label: str, latest_data: pd.Series, prev_data: pd.Series, 
                         is_percent: bool = False, is_ratio: bool = False, 
                         help_text: str = None, currency_symbol: str = "$"):
    """
    Displays a styled metric card using st.metric, including value and delta.
    st_container: The Streamlit container (e.g., st or a column object) to place the metric in.
    label: The raw metric key (e.g., 'grossProfitMargin').
    latest_data: Pandas Series containing the latest period's data.
    prev_data: Pandas Series containing the previous period's data.
    is_percent: True if the value is a percentage.
    is_ratio: True if the value is a raw ratio (no currency, specific decimal formatting).
    help_text: Custom help text for the metric card.
    currency_symbol: Currency symbol to use for monetary values.
    """
    current_val = latest_data.get(label)
    prev_val = prev_data.get(label)
    
    delta_val = calculate_delta(current_val, prev_val)
    
    delta_display = None # String for st.metric's delta parameter
    if delta_val is not None:
        delta_prefix = ""
        # Monetary values
        if not is_percent and not is_ratio:
            if delta_val > 0: delta_prefix = "+"
            # Format the absolute delta value with B/M/K suffixes
            abs_delta_for_format = abs(delta_val)
            if abs_delta_for_format >= 1_000_000_000: formatted_abs_delta = f"{abs_delta_for_format / 1_000_000_000:.2f}B"
            elif abs_delta_for_format >= 1_000_000: formatted_abs_delta = f"{abs_delta_for_format / 1_000_000:.2f}M"
            elif abs_delta_for_format >= 1_000: formatted_abs_delta = f"{abs_delta_for_format / 1_000:.2f}K"
            else: formatted_abs_delta = f"{abs_delta_for_format:,.0f}" # Default to integer for smaller monetary deltas
            delta_display = f"{delta_prefix}{formatted_abs_delta}"
        # Percentage point changes
        elif is_percent: 
            delta_display = f"{delta_val:+.2f}pp" # "pp" for percentage points
        # Raw ratio changes
        elif is_ratio: 
            delta_display = f"{delta_val:+.2f}" # Show with sign and 2 decimal places

    # Construct help text string for the metric card
    current_formatted_val = format_value(current_val, is_percent, currency_symbol if not (is_percent or is_ratio) else "", is_ratio=is_ratio)
    prev_formatted_val = format_value(prev_val, is_percent, currency_symbol if not (is_percent or is_ratio) else "", is_ratio=is_ratio)

    help_str_parts = []
    if pd.notna(current_val): help_str_parts.append(f"Latest: {current_formatted_val}")
    else: help_str_parts.append("Latest: N/A")
    if pd.notna(prev_val): help_str_parts.append(f"Previous: {prev_formatted_val}")
    
    final_help_text_for_metric = " | ".join(help_str_parts)
    if help_text: # Prepend custom help text if provided
        final_help_text_for_metric = f"{help_text}\n{final_help_text_for_metric}"

    # Create a more readable display label for the metric card
    # Common abbreviations and capitalizations
    display_label = label.replace("Margin", " Mgn").replace("Expenses", " Exp.") \
                         .replace("Equivalents", "Equiv.").replace("Receivables","Recv.") \
                         .replace("Payables","Pay.").replace("Liabilities","Liab.") \
                         .replace("Assets","Ast.").replace("Activities", "Act.") \
                         .replace("ProvidedByOperating", "Op.").replace("ProvidedByInvesting", "Inv.").replace("ProvidedByFinancing", "Fin.") \
                         .replace("ProvidedBy", "/").replace("UsedFor","Used For") \
                         .replace("Expenditure","Exp.").replace("Income", "Inc.") \
                         .replace("Statement","Stmt.").replace("Interest","Int.") \
                         .replace("Development","Dev.").replace("Administrative","Admin.") \
                         .replace("General","Gen.").replace("ShortTerm","ST") \
                         .replace("LongTerm","LT").replace("Total","Tot.") \
                         .replace("StockholdersEquity", "Equity").replace("PropertyPlantEquipmentNet", "PP&E (Net)")
    # Capitalize words, handle "And" -> "&"
    display_label = ' '.join(word.capitalize() if not word.isupper() else word for word in display_label.replace("And", "&").split())


    st_container.metric(
        label=display_label,
        value=current_formatted_val if pd.notna(current_val) else "N/A",
        delta=delta_display,
        help=final_help_text_for_metric
    )

def build_styled_dataframe(df_long: pd.DataFrame, metrics_to_display: list) -> Styler:
    """
    Builds a Pandas Styler object for a table, with values formatted and arrows indicating change.
    This function is kept as a general utility but might not be directly used if manual HTML tables
    offer more control for specific categorized layouts.
    """
    if df_long.empty or not metrics_to_display:
        return pd.DataFrame().style # Return empty Styler if no data or metrics

    if 'year' not in df_long.columns:
        # This function expects df_long to have a 'year' column for pivoting.
        # If 'year' is already an index, the logic would need adjustment.
        st.warning("Dataframe for 'build_styled_dataframe' is expected to have a 'year' column.")
        return pd.DataFrame(index=metrics_to_display).style # Return empty styler with metric names as index
        
    # Filter metrics_to_display to include only those present in df_long columns
    valid_metrics = [m for m in metrics_to_display if m in df_long.columns]
    if not valid_metrics:
        st.warning(f"None of the metrics for the styled table are present in the data: {metrics_to_display}")
        return pd.DataFrame(index=metrics_to_display).style # Styler with metric names as index
            
    df_for_pivot = df_long[['year'] + valid_metrics]
    try:
        # Pivot the data: years as columns, metrics as rows
        df_display_transposed = df_for_pivot.set_index("year")[valid_metrics].transpose()
    except KeyError as e: 
        st.warning(f"Pivoting error in 'build_styled_dataframe': {e}. Check 'year' column and metric names.")
        return pd.DataFrame(index=metrics_to_display).style

    # Create a new DataFrame to hold HTML-formatted strings (value + arrow)
    out_df_with_html = pd.DataFrame(index=df_display_transposed.index, 
                                    columns=df_display_transposed.columns, 
                                    dtype=object) # Use object dtype for mixed content

    for r_idx, metric_name in enumerate(out_df_with_html.index):
        # Heuristic to determine if a metric is a percentage or ratio for formatting
        is_percent_or_ratio_metric = "Margin" in metric_name or "Ratio" in metric_name 

        for c_idx, year_val in enumerate(out_df_with_html.columns):
            current_val = df_display_transposed.loc[metric_name, year_val]
            
            if pd.isna(current_val):
                out_df_with_html.iat[r_idx, c_idx] = "N/A" # Display N/A for missing values
                continue

            # Format the current value
            formatted_current_val = format_value(current_val, 
                                                 is_percent=is_percent_or_ratio_metric and "Margin" in metric_name, # Only % for margins
                                                 currency_symbol="$" if not is_percent_or_ratio_metric else "", # No currency for %/ratios
                                                 is_ratio=is_percent_or_ratio_metric and "Ratio" in metric_name, # Explicitly for ratios
                                                 decimals=2) 
            
            arrow_symbol_html = "" # Default to no arrow (e.g., for the first year)
            if c_idx > 0: # Calculate arrow if not the first year column
                prev_year_val = out_df_with_html.columns[c_idx-1]
                prev_val = df_display_transposed.loc[metric_name, prev_year_val]
                arrow_symbol_html = _arrow(prev_val, current_val, is_percent=is_percent_or_ratio_metric) # Arrow logic
            
            # Combine formatted value and HTML arrow into a single string for the cell
            out_df_with_html.iat[r_idx, c_idx] = f"{formatted_current_val}{arrow_symbol_html}"

    styler = out_df_with_html.style
    # Use a formatter that just returns the value as is (identity function)
    # This is crucial because the cells already contain pre-rendered HTML.
    styler = styler.format(lambda x: x) 

    # Apply table-level and cell-level properties using Pandas Styler
    styler = styler.set_properties(**{
        "white-space": "nowrap",   # Prevent text wrapping in cells
        "text-align": "right",     # Align content to the right (common for numerical data)
        "padding": "6px 8px",      # Add some padding within cells
        "border": "1px solid #eee" # Light border for cells
    })

    styler = styler.set_table_styles([
        # Style for column headers (years)
        {'selector': 'th.col_heading', 'props': [('background-color', '#f7f7f7'), ('font-weight', 'bold'), ('text-align', 'center'), ('padding', '6px 8px')]},
        # Style for row headers (metric names)
        {'selector': 'th.row_heading', 'props': [('background-color', '#f7f7f7'), ('font-weight', 'bold'), ('text-align', 'left'), ('padding-left', '10px')]},
        # General style for data cells (td)
        {'selector': 'td', 'props': [('text-align', 'right'), ('padding', '5px 8px')]}, # Slightly less padding for data cells
        # Hover effect for table rows
        {'selector': 'tr:hover', 'props': [('background-color', '#f1f1f1')]} 
    ])
    # Note: To use this Styler object with st.markdown, you'd convert it to HTML:
    # html_table = styler.to_html(escape=False)
    # st.markdown(html_table, unsafe_allow_html=True)
    # Or, if using st.dataframe, Streamlit handles the Styler object directly:
    # st.dataframe(styler) - but st.dataframe doesn't render HTML content within cells by default.
            
    return styler
    
@st.cache_data(ttl="1h", show_spinner="Loading vector data for year...")
def load_vectors_for_similarity(_driver, year: int, family: str = "cf_vec_", sectors: List[str]=None, cap_classes: List[str]=None, target_sym: str=None):
    prop = f"{family}{year}"
    params = {}
    filter_clauses = ["c.{prop} IS NOT NULL", "c.ipoDate IS NOT NULL"]
    if sectors:
        filter_clauses.append("c.sector IN $sectors")
        params["sectors"] = sectors
    if cap_classes:
        filter_clauses.append("c.marketCapClass IN $cap_classes")
        params["cap_classes"] = cap_classes
    query = f"""
    MATCH (c:Company)
    WHERE {' AND '.join(filter_clauses)}
    RETURN c.symbol AS sym, c.{prop} AS vec, c.sector AS sector
    """
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query, **params)
            all_vecs = {r["sym"]: (np.asarray(r["vec"], dtype=np.float32), r["sector"]) for r in results}
            target_vector = all_vecs.get(target_sym, (None, None))[0]
            # Filter by sector/cap for candidates, but not for the anchor
            candidate_vecs = {sym: vec for sym, (vec, sector) in all_vecs.items() if sym != target_sym}
            return target_vector, candidate_vecs
    except Exception as e:
        st.error(f"Error loading vectors for {prop}: {e}")
        return None, {}


def calculate_similarity_scores(target_vector: np.ndarray, vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculates cosine similarity between a target vector and a dictionary of other vectors."""
    if target_vector is None or not vectors:
        return {}
    
    other_symbols = list(vectors.keys())
    matrix_of_vectors = np.vstack(list(vectors.values()))
    
    # Normalize matrix and target vector
    matrix_of_vectors /= (np.linalg.norm(matrix_of_vectors, axis=1, keepdims=True) + 1e-12)
    target_vector /= (np.linalg.norm(target_vector) + 1e-12)
    
    similarities = matrix_of_vectors @ target_vector
    return dict(zip(other_symbols, similarities.astype(float)))


@st.cache_data(ttl="1h", show_spinner="Fetching financial details for similar companies...")
def fetch_financial_details_for_companies(
    _driver, 
    company_symbols: List[str], 
    year: int = None,
    sectors: List[str] = None
) -> Dict[str, Dict]:
    query_option_b_no_apoc = """
    UNWIND $symbols AS sym_param
    MATCH (c:Company {symbol: sym_param})
    WHERE $sectors IS NULL OR c.sector IN $sectors

    OPTIONAL MATCH (c)-[:HAS_INCOME_STATEMENT]->(is_node:IncomeStatement)
    WHERE is_node.fillingDate IS NOT NULL AND ($year IS NULL OR is_node.calendarYear = $year)
    WITH c, sym_param, is_node ORDER BY is_node.fillingDate DESC
    WITH c, sym_param, COLLECT(is_node)[0] AS latest_is

    OPTIONAL MATCH (c)-[:HAS_BALANCE_SHEET]->(bs_node:BalanceSheet)
    WHERE bs_node.fillingDate IS NOT NULL AND ($year IS NULL OR bs_node.calendarYear = $year)
    WITH c, sym_param, latest_is, bs_node ORDER BY bs_node.fillingDate DESC
    WITH c, sym_param, latest_is, COLLECT(bs_node)[0] AS latest_bs

    OPTIONAL MATCH (c)-[:HAS_CASH_FLOW_STATEMENT]->(cf_node:CashFlowStatement)
    WHERE cf_node.fillingDate IS NOT NULL AND ($year IS NULL OR cf_node.calendarYear = $year)
    WITH c, sym_param, latest_is, latest_bs, cf_node ORDER BY cf_node.fillingDate DESC
    WITH c, sym_param, latest_is, latest_bs, COLLECT(cf_node)[0] AS latest_cf

    RETURN sym_param AS symbol,
           c.companyName AS companyName,
           c.sector AS sector,
           c.industry AS industry,
           latest_is.revenue AS revenue,
           latest_is.netIncome AS netIncome,
           latest_is.operatingIncome AS operatingIncome,
           latest_is.grossProfit AS grossProfit,
           latest_bs.totalAssets AS totalAssets,
           latest_bs.totalLiabilities AS totalLiabilities,
           latest_bs.totalStockholdersEquity AS totalStockholdersEquity,
           latest_bs.cashAndCashEquivalents AS cashAndCashEquivalents,
           latest_cf.operatingCashFlow AS operatingCashFlow,
           latest_cf.freeCashFlow AS freeCashFlow,
           latest_cf.netChangeInCash AS netChangeInCash,
           latest_cf.capitalExpenditure AS capitalExpenditure
    """
    details = {}
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query_option_b_no_apoc, symbols=company_symbols, sectors=sectors, year=year)
            for record in results:
                details[record["symbol"]] = record.data()
        return details
    except Exception as e:
        st.error(f"Error fetching financial details: {e}")
        return {}

