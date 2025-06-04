# components/similar_companies_tab.py

import streamlit as st
import numpy as np
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Any

from utils import (
    get_neo4j_driver,
    fetch_financial_details_for_companies,
    fetch_sector_list,
    fetch_company_preview,
    format_value
)

DEFAULT_DECAY = 0.7  # λ for recency-weighted mean

# --- Market Cap Class Fetcher ---
def fetch_market_cap_classes(driver=None):
    if not driver:
        driver = get_neo4j_driver()
    query = """
    MATCH (c:Company)
    WHERE c.marketCapClass IS NOT NULL
    RETURN DISTINCT c.marketCapClass AS cap_class
    ORDER BY cap_class
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record["cap_class"] for record in result]
    except Exception as e:
        print(f"Error fetching market cap classes: {str(e)}")
        return []
    finally:
        if driver and hasattr(driver, 'close'):
            driver.close()

# --- Vector Loader with Filtering ---
@st.cache_data(ttl="1h")
def load_vectors_for_similarity(
    _driver, 
    year: int, 
    family: str = "cf_vec_", 
    sectors: List[str] = None, 
    cap_classes: List[str] = None, 
    target_sym: str = None
):
    prop = f"{family}{year}"
    where_clauses = [f"c.`{prop}` IS NOT NULL", "c.ipoDate IS NOT NULL"]
    if sectors:
        where_clauses.append("c.sector IN $sectors")
    if cap_classes:
        where_clauses.append("c.marketCapClass IN $cap_classes")
    where_str = " AND ".join(where_clauses)
    query = f"""
        MATCH (c:Company)
        WHERE {where_str}
        RETURN c.symbol AS sym, c.`{prop}` AS vec, c.sector AS sector, c.marketCapClass AS cap_class
    """
    params = {}
    if sectors:
        params["sectors"] = sectors
    if cap_classes:
        params["cap_classes"] = cap_classes
    try:
        with _driver.session(database="neo4j") as session:
            results = session.run(query, **params)
            all_vecs = {r["sym"]: (np.asarray(r["vec"], dtype=np.float32), r["sector"], r["cap_class"]) for r in results}
            target_vector = all_vecs.get(target_sym, (None, None, None))[0]
            candidate_vecs = {sym: vec for sym, (vec, _, _) in all_vecs.items() if sym != target_sym}
            return target_vector, candidate_vecs
    except Exception as e:
        st.error(f"Error loading vectors for {prop}: {e}")
        return None, {}

# --- Cosine Similarity ---
def calculate_similarity_scores(target_vector: np.ndarray, vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    if target_vector is None or not vectors:
        return {}
    matrix_of_vectors = np.vstack(list(vectors.values()))
    matrix_of_vectors /= (np.linalg.norm(matrix_of_vectors, axis=1, keepdims=True) + 1e-12)
    target_vector /= (np.linalg.norm(target_vector) + 1e-12)
    similarities = matrix_of_vectors @ target_vector
    return dict(zip(vectors.keys(), similarities.astype(float)))

# --- Main Similarity Function ---
@st.cache_data(ttl="1h", show_spinner="Calculating aggregated similarity scores...")
def get_nearest_aggregate_similarities(_driver, 
                                       target_sym: str, 
                                       embedding_family: str, 
                                       start_year: int = 2017, 
                                       end_year: int = 2023, 
                                       sectors=None,
                                       cap_classes=None,
                                       weight_scheme=None,
                                       normalize=True,
                                       k: int = 10) -> List[Tuple[str, float]]:
    if not _driver:
        return []
    from collections import defaultdict
    cumulative_scores = defaultdict(float)
    years_processed_count = 0

    for year in range(start_year, end_year + 1):
        target_vector, yearly_vectors = load_vectors_for_similarity(
            _driver, year, embedding_family, sectors=sectors, cap_classes=cap_classes, target_sym=target_sym
        )
        if target_vector is None or not yearly_vectors:
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
