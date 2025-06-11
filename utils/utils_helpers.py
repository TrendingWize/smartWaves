import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from pandas.io.formats.style import Styler # pandas ≥1.0
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ── Neo4j Config ───────────────────────────────────────────────────────
NEO4J_URI = st.secrets.get("NEO4J_URI", "neo4j+s://f9f444b7.databases.neo4j.io")
NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0")


def calculate_delta(current_value, previous_value):
    """
    Calculates the difference between current and previous values.
    Returns None if data is insufficient or previous value is zero (to avoid division by zero issues downstream).
    """
    if pd.isna(current_value) or pd.isna(previous_value) or previous_value == 0:
        return None 
    return current_value - previous_value

def _arrow(prev_val, curr_val, is_percent=False):
    try:
        prev_val = prev_val.item() if hasattr(prev_val, "item") else prev_val
        curr_val = curr_val.item() if hasattr(curr_val, "item") else curr_val
    except Exception:
        return " →"

    if isinstance(prev_val, pd.Series) or isinstance(curr_val, pd.Series):
        return " →"

    if pd.isna(prev_val) or pd.isna(curr_val):
        return " →"

    if curr_val > prev_val:
        return ' <span style="color:green">↑</span>'
    elif curr_val < prev_val:
        return ' <span style="color:red">↓</span>'
    else:
        return ' →'



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
                             
# ── Neo4j Driver Cache ─────────────────────────────────────────────────
@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Neo4j connection error: {e}")
        return None

# --- Market Cap Class Fetcher ---
def fetch_market_cap_classes(driver) -> List[str]:
    if not driver:
        st.error("Neo4j driver not provided to fetch_market_cap_classes.")
        return []
    query = """
    MATCH (c:Company)
    WHERE c.marketCapClass IS NOT NULL
    RETURN DISTINCT c.marketCapClass AS cap_class
    ORDER BY cap_class
    """
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query)
            return [record["cap_class"] for record in result]
    except Exception as e:
        st.error(f"Error fetching market cap classes: {str(e)}")
        return []
    # No explicit driver.close() here as it's managed by @st.cache_resource or passed externally

# --- Vector Loader with Filtering ---
@st.cache_data(ttl="12h")
def load_vectors_for_similarity(
    _driver,
    year: int,
    family: str = "cf_vec_",
    sectors: Optional[List[str]] = None,      # Make Optional explicit
    cap_classes: Optional[List[str]] = None,  # Make Optional explicit
    target_sym: str = None
) -> Tuple[Optional[np.ndarray], Dict[str, Tuple[np.ndarray, str, str]]]:
    if not _driver:
        st.error("Neo4j driver not provided to load_vectors_for_similarity.")
        return None, {}
    if not target_sym:
        st.error("Target symbol not provided to load_vectors_for_similarity.")
        return None, {}

    prop = f"{family}{year}"
    target_vector: Optional[np.ndarray] = None
    candidate_vecs_data: Dict[str, Tuple[np.ndarray, str, str]] = {}

    # Step 1: Fetch the target company's vector unconditionally (for its specific embedding property)
    target_query = f"""
        MATCH (t:Company {{symbol: $target_sym}})
        WHERE t.`{prop}` IS NOT NULL
        RETURN t.`{prop}` AS vec
        LIMIT 1
    """
    try:
        with _driver.session(database="neo4j") as session:
            result = session.run(target_query, target_sym=target_sym)
            record = result.single()
            if record and record["vec"]:
                target_vector = np.asarray(record["vec"], dtype=np.float32)
            else:
                # st.info(f"No vector '{prop}' found for target {target_sym} in year {year}.") # Can be noisy
                return None, {} # If target vector for the year isn't found, can't proceed for this year

        # Step 2: Fetch candidate companies, applying filters
        candidate_where_clauses = [
            f"c.symbol <> $target_sym", # Exclude target from candidates
            f"c.`{prop}` IS NOT NULL",
            "c.ipoDate IS NOT NULL"
        ]
        params_candidate = {"target_sym": target_sym}

        if sectors:
            candidate_where_clauses.append("c.sector IN $sectors")
            params_candidate["sectors"] = sectors
        if cap_classes:
            candidate_where_clauses.append("c.marketCapClass IN $cap_classes")
            params_candidate["cap_classes"] = cap_classes

        candidate_where_str = " AND ".join(candidate_where_clauses)
        candidate_query = f"""
            MATCH (c:Company)
            WHERE {candidate_where_str}
            RETURN c.symbol AS sym, c.`{prop}` AS vec, c.sector AS sector, c.marketCapClass AS cap_class
        """
        with _driver.session(database="neo4j") as session:
            results_candidate = session.run(candidate_query, **params_candidate)
            for r_cand in results_candidate:
                if r_cand["vec"]: # Ensure candidate vector exists
                    candidate_vecs_data[r_cand["sym"]] = (
                        np.asarray(r_cand["vec"], dtype=np.float32),
                        r_cand["sector"],
                        r_cand["cap_class"]
                    )
        
        return target_vector, candidate_vecs_data

    except Exception as e:
        st.error(f"Error loading vectors for {prop} (target: {target_sym}, year: {year}): {e}")
        return None, {}


# --- Cosine Similarity ---
def calculate_similarity_scores(target_vector: np.ndarray,
                                candidate_vectors_data: Dict[str, Tuple[np.ndarray, str, str]]) -> Dict[str, float]:
    if target_vector is None or not candidate_vectors_data:
        return {}
    
    candidate_symbols = list(candidate_vectors_data.keys())
    # Extract just the vectors for calculation
    matrix_of_vectors = np.vstack([data[0] for data in candidate_vectors_data.values()])

    # Normalize matrix and target vector
    norm_matrix = np.linalg.norm(matrix_of_vectors, axis=1, keepdims=True)
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 1e-12: # Avoid division by zero for zero target vector
        return {sym: 0.0 for sym in candidate_symbols}

    # Avoid division by zero for candidate vectors
    # Create a mask for rows with zero norm
    zero_norm_mask = (norm_matrix < 1e-12).flatten()

    # Normalize non-zero norm vectors
    # Initialize similarities array
    similarities = np.zeros(len(candidate_symbols), dtype=float)
    
    # Normalize valid vectors
    valid_matrix_indices = ~zero_norm_mask
    if np.any(valid_matrix_indices):
        matrix_of_vectors[valid_matrix_indices] /= norm_matrix[valid_matrix_indices]
    
    target_vector_normalized = target_vector / norm_target
    
    # Calculate dot product for valid, normalized vectors
    if np.any(valid_matrix_indices):
       similarities[valid_matrix_indices] = matrix_of_vectors[valid_matrix_indices] @ target_vector_normalized

    return dict(zip(candidate_symbols, similarities.astype(float)))


# --- Main Similarity Function ---
@st.cache_data(ttl="1h", show_spinner="Calculating aggregated similarity scores...")
def get_nearest_aggregate_similarities(_driver, # <--- RENAMED driver to _driver
                                       target_sym: str,
                                       embedding_family: str,
                                       start_year: int = 2017,
                                       end_year: int = 2023,
                                       sectors: Optional[List[str]] = None,
                                       cap_classes: Optional[List[str]] = None,
                                       weight_scheme: str = "mean",
                                       decay_lambda: float = 0.7,
                                       normalize: bool = True, # This param is still here
                                       k: int = 10) -> List[Tuple[str, float]]:
    if not _driver: # Check the renamed argument
        st.error("Neo4j driver not provided to get_nearest_aggregate_similarities.")
        return []

    all_yearly_scores = defaultdict(list)
    # processed_years_for_sym = defaultdict(list) # Not strictly used in the provided logic

    for year in range(start_year, end_year + 1):
        # Pass the _driver argument to functions it calls
        target_vector, yearly_candidate_data = load_vectors_for_similarity(
            _driver, year, embedding_family, sectors=sectors, cap_classes=cap_classes, target_sym=target_sym
        )
        if target_vector is None:
            # st.info(f"No vector data for target {target_sym} in {year} with current filters.") # Can be noisy
            continue
        if not yearly_candidate_data:
            # st.info(f"No candidate vectors found for comparison in {year} with current filters.") # Can be noisy
            continue

        yearly_similarity_scores = calculate_similarity_scores(target_vector, yearly_candidate_data)

        for sym, score in yearly_similarity_scores.items():
            all_yearly_scores[sym].append({'year': year, 'score': score})
            # processed_years_for_sym[sym].append(year) # Not strictly used

    if not all_yearly_scores:
        st.warning(f"No similarity scores computed for {target_sym} or its comparables in the selected year range ({start_year}-{end_year}), embedding family ({embedding_family}), and filters.")
        return []

    aggregated_scores = defaultdict(float)

    if weight_scheme == "latest":
        for sym, scores_data in all_yearly_scores.items():
            if not scores_data: continue
            # Get data for the specified end_year if available, otherwise latest available within range
            relevant_scores = [sd for sd in scores_data if sd['year'] == end_year]
            if not relevant_scores: # If no data for exact end_year, take the latest available in range
                 relevant_scores = sorted(scores_data, key=lambda x: x['year'], reverse=True)

            if relevant_scores:
                 aggregated_scores[sym] = relevant_scores[0]['score']
            # else:
            #    st.info(f"No score found for {sym} to apply 'latest' weight scheme.")

    elif weight_scheme == "decay":
        for sym, scores_data in all_yearly_scores.items():
            if not scores_data: continue
            current_sum_score = 0.0
            current_sum_weight = 0.0
            # Sort scores by year to ensure consistent decay calculation if needed, though power handles it
            # sorted_scores_data = sorted(scores_data, key=lambda x: x['year'])
            for data_point in scores_data: # Use original scores_data
                year_val = data_point['year']
                score_val = data_point['score']
                # Weight is higher for years closer to end_year
                weight = decay_lambda ** (end_year - year_val)
                current_sum_score += score_val * weight
                current_sum_weight += weight
            if current_sum_weight > 1e-9: # Avoid division by zero or tiny weights
                aggregated_scores[sym] = current_sum_score / current_sum_weight
            # else:
            #    st.info(f"Could not apply 'decay' weighting for {sym} due to zero total weight.")


    else:  # Default is "mean"
        for sym, scores_data in all_yearly_scores.items():
            if not scores_data: continue
            sum_scores = sum(d['score'] for d in scores_data)
            count_scores = len(scores_data)
            if count_scores > 0:
                aggregated_scores[sym] = sum_scores / count_scores

    if not aggregated_scores:
         st.warning(f"No companies found after applying weighting scheme '{weight_scheme}'.")
         return []

    best_k_similar = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return best_k_similar

# You also need to apply this to other cached functions that take the driver:

def fetch_sector_list(driver) -> List[str]:
    if not driver:
        st.error("Neo4j driver not provided to fetch_sector_list.")
        return []
    query = """
    MATCH (c:Company) WHERE c.sector IS NOT NULL
    RETURN DISTINCT c.sector AS sector
    ORDER BY sector
    """ # Changed to directly query Company.sector as schema shows Sector node but not relation
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query)
            return [record["sector"] for record in result]
    except Exception as e:
        st.error(f"Error fetching sectors: {str(e)}")
        return []

def fetch_company_preview(driver, symbol: str) -> Dict[str, Any]:
    if not driver:
        st.error("Neo4j driver not provided to fetch_company_preview.")
        return {}
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
        with driver.session(database="neo4j") as session:
            result = session.run(query, symbol=symbol)
            record = result.single()
            return record.data() if record else {}
    except Exception as e:
        st.error(f"Error fetching company preview for {symbol}: {str(e)}")
        return {}

@st.cache_data(ttl="1h", show_spinner="Fetching financial details for similar companies...")
def fetch_financial_details_for_companies(
    _driver, # <--- RENAMED driver to _driver
    company_symbols: List[str],
    year: Optional[int] = None,
) -> Dict[str, Dict]:
    if not _driver: # Check the renamed argument
        st.error("Neo4j driver not provided to fetch_financial_details_for_companies.")
        return {}
    # ... rest of the function, ensuring you use _driver internally ...
    if not company_symbols:
        return {}
    query_option_b_no_apoc = """
    UNWIND $symbols AS sym_param
    MATCH (c:Company {symbol: sym_param})

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
           c.marketCap AS marketCap,
           latest_is.revenue AS revenue,
           latest_is.netIncome AS netIncome,
           latest_is.operatingIncome AS operatingIncome,
           latest_is.grossProfit AS grossProfit,
           latest_bs.totalAssets AS totalAssets,
           latest_bs.totalLiabilities AS totalLiabilities,
           latest_bs.totalStockholdersEquity AS totalStockholdersEquity,
           latest_bs.cashAndCashEquivalents AS cashAndCashEquivalents,
           latest_bs.totalCurrentAssets AS totalCurrentAssets, 
           latest_bs.totalCurrentLiabilities AS totalCurrentLiabilities, 
           latest_cf.operatingCashFlow AS operatingCashFlow,
           latest_cf.freeCashFlow AS freeCashFlow,
           latest_cf.netChangeInCash AS netChangeInCash,
           latest_cf.capitalExpenditure AS capitalExpenditure
    """
    details = {}
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query_option_b_no_apoc, symbols=company_symbols, year=year)
            for record in results:
                details[record["symbol"]] = record.data()
        return details
    except Exception as e:
        st.error(f"Error fetching financial details: {e}")
        return {}


# ── Data Fetching for specific statements (example) ───────────────────
@st.cache_data(ttl="1h")
def fetch_income_statement_data(_driver, symbol: str, start_yr: int = 2017, end_yr: Optional[int] = None) -> pd.DataFrame:
    if not _driver: # Check the renamed argument
        st.error("Neo4j driver not provided to fetch_income_statement_data.")
        return pd.DataFrame()
    # ... rest of the function, using _driver ...
    year_condition = "i.calendarYear >= $start_yr"
    params = {"symbol": symbol, "start_yr": start_yr}
    if end_yr:
        year_condition += " AND i.calendarYear <= $end_yr"
        params["end_yr"] = end_yr
        
    query = f"""
    MATCH (c:Company {{symbol: $symbol}})-[:HAS_INCOME_STATEMENT]->(i:IncomeStatement)
    WHERE {year_condition}
    RETURN
      i.calendarYear AS year, 
      i.revenue AS revenue,
      i.costOfRevenue AS costOfRevenue,
      i.grossProfit AS grossProfit,
      i.researchAndDevelopmentExpenses AS researchAndDevelopmentExpenses,
      i.generalAndAdministrativeExpenses AS generalAndAdministrativeExpenses,
      i.sellingGeneralAndAdministrativeExpenses AS sellingGeneralAndAdministrativeExpenses,
      i.operatingExpenses AS operatingExpenses,
      i.operatingIncome AS operatingIncome,
      i.interestIncome AS interestIncome,
      i.interestExpense AS interestExpense,
      i.incomeBeforeTax AS incomeBeforeTax,
      i.incomeTaxExpense AS incomeTaxExpense,
      i.netIncome AS netIncome
    ORDER BY year ASC
    """
    try:
        with _driver.session(database="neo4j") as session:
            result = session.run(query, **params)
            data = [record.data() for record in result]

        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        # ... (rest of df processing)
        return df
    except Exception as e:
        st.error(f"Error fetching or processing income data for {symbol}: {e}")
        return pd.DataFrame()

# ── Formatting and Styling Helpers ─────────────────────────────────────
def format_value(value, is_percent=False, currency_symbol="$", is_ratio=False, decimals=2):
    if pd.isna(value) or value is None:
        return "N/A"
    if is_percent:
        return f"{value:.{decimals}f}%"
    if is_ratio:
        return f"{value:.{decimals}f}"
    if isinstance(value, (int, float)):
        num_str = ""
        abs_value = abs(value)
        if abs_value >= 1_000_000_000_000: num_str = f"{value / 1_000_000_000_000:.{decimals}f}T"
        elif abs_value >= 1_000_000_000: num_str = f"{value / 1_000_000_000:.{decimals}f}B"
        elif abs_value >= 1_000_000: num_str = f"{value / 1_000_000:.{decimals}f}M"
        elif abs_value >= 1_000: num_str = f"{value / 1_000:.{decimals}f}K"
        else: num_str = f"{value:,.{decimals}f}"
        return f"{currency_symbol}{num_str}" if currency_symbol and num_str else num_str
    return str(value)
