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
@st.cache_data(ttl="1h")
@st.cache_data(ttl="1h")
def load_vectors_for_similarity(
    _driver, # <--- RENAMED driver to _driver
    year: int,
    family: str = "cf_vec_",
    sectors: List[str] = None,
    cap_classes: List[str] = None,
    target_sym: str = None
) -> Tuple[Optional[np.ndarray], Dict[str, Tuple[np.ndarray, str, str]]]: # Return type updated
    if not _driver: # Check the renamed argument
        st.error("Neo4j driver not provided to load_vectors_for_similarity.")
        return None, {}
    # ... rest of the function, ensuring you use _driver internally ...
    prop = f"{family}{year}"
    where_clauses = [f"c.`{prop}` IS NOT NULL", "c.ipoDate IS NOT NULL"]
    params = {} # Removed prop_name as it's not used in a parameterized way in query string

    if sectors:
        where_clauses.append("c.sector IN $sectors")
        params["sectors"] = sectors
    if cap_classes:
        where_clauses.append("c.marketCapClass IN $cap_classes")
        params["cap_classes"] = cap_classes

    where_str = " AND ".join(where_clauses)
    query = f"""
        MATCH (c:Company)
        WHERE {where_str}
        RETURN c.symbol AS sym, c.`{prop}` AS vec, c.sector AS sector, c.marketCapClass AS cap_class
    """
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query, **params) # Pass parameters correctly
            all_vecs_data = {
                r["sym"]: (np.asarray(r["vec"], dtype=np.float32), r["sector"], r["cap_class"])
                for r in results if r["vec"] is not None
            }
            
            target_data = all_vecs_data.get(target_sym)
            target_vector = target_data[0] if target_data and target_data[0] is not None else None # Ensure vec exists
            
            candidate_vecs_data = {
                sym: data for sym, data in all_vecs_data.items() if sym != target_sym and data[0] is not None # Ensure vec exists
            }
            return target_vector, candidate_vecs_data
    except Exception as e:
        st.error(f"Error loading vectors for {prop}: {e}")
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
