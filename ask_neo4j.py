# ask_neo4j.py (Revised for Streamlit Integration)

from __future__ import annotations
import os, sys, time, math, json, traceback, re, asyncio
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache # Not used in provided snippet, can remove if not needed elsewhere
import numpy as np
import pandas as pd
# from tqdm import tqdm # Not typically used in Streamlit apps
from tenacity import retry, wait_exponential, stop_after_attempt # Keep if LLM calls use it
from dotenv import load_dotenv # Keep for local .env, Streamlit secrets for deployment
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.prompts import PromptTemplate
import streamlit as st # ADDED for Streamlit caching and session state
# --- ENV + GLOBALS ---

# Use Streamlit secrets for deployed apps, fall back to .env for local
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER='openai'
#GOOGLE_API_KEY  = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
# NEO4J creds will be passed from utils.py's get_neo4j_driver or initialized here if standalone

NEO4J_URI_ASKAI = os.getenv("NEO4J_URI") or st.secrets.get("NEO4J_URI")
NEO4J_USERNAME_ASKAI = os.getenv("NEO4J_USER") or st.secrets.get("NEO4J_USER")
NEO4J_PASSWORD_ASKAI = os.getenv("NEO4J_PASSWORD") or st.secrets.get("NEO4J_PASSWORD")
st.write("Neo4j User:", NEO4J_USERNAME_ASKAI)


VECTOR_INDEX_NAME   = "company_embedding_ix" # Or make configurable
VECTOR_NODE_LABEL   = "Company"
VECTOR_PROPERTY_KEY = "metadata_embedding" # Or "all_embedding" or other from your schema

# Global placeholders for LLM and embeddings, initialized on demand
# Using Streamlit session state is better for managing these across reruns
# if 'ask_ai_llm' not in st.session_state:
#     st.session_state.ask_ai_llm = None
# if 'ask_ai_embeddings' not in st.session_state:
#     st.session_state.ask_ai_embeddings = None
# if 'ask_ai_llm_provider' not in st.session_state:
#     st.session_state.ask_ai_llm_provider = None

# --- LLM and Embeddings Initialization (Cached) ---
@st.cache_resource(show_spinner="Initializing AI Models...")
def initialize_llm_and_embeddings_askai(provider: str):
    # global llm, embeddings, LLM_PROVIDER # Avoid globals in functions
    llm_provider_internal = provider.lower()
    llm_internal = None
    embeddings_internal = None

    if llm_provider_internal == "openai":
        if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            st.error("OpenAI API Key is missing or invalid. Please set it in secrets.")
            raise ValueError("OPENAI_API_KEY missing or looks invalid.")
        embeddings_internal = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
        # embeddings_internal.embed_query("ping") # Smoke test
        llm_internal = ChatOpenAI(model="gpt-4o", 
        #temperature=0, 
        openai_api_key=OPENAI_API_KEY) # Using a common model
        # llm_internal.invoke("ping")
    elif llm_provider_internal == "gemini":
        if not GOOGLE_API_KEY:
            st.error("Google API Key is missing. Please set it in secrets.")
            raise ValueError("GOOGLE_API_KEY missing.")
        embeddings_internal = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=GOOGLE_API_KEY)
        # embeddings_internal.embed_query("ping")
        llm_internal = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
        # llm_internal.invoke("ping")
    else:
        st.error(f"Unsupported AI provider: {provider}. Choose 'openai' or 'gemini'.")
        raise ValueError("Unsupported provider (openai|gemini)")
    
    # st.success("AI Models Initialized!") # Can be noisy
    return llm_internal, embeddings_internal, llm_provider_internal

# --- Neo4j Graph Instance (Can use the one from utils.py or a specific one) ---
# For simplicity, let's assume we'll pass the driver from utils.py to ask_neo4j
# If you want this module to manage its own Neo4jGraph instance:
@st.cache_resource(show_spinner="Connecting to Neo4j for AI queries...")
def get_neo4j_graph_for_askai():
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI_ASKAI, 
            username="neo4j+s://f9f444b7.databases.neo4j.io",
            password="BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0",
            database="neo4j",
            refresh_schema=True # Important for the LLM to get current schema
        )
        # graph.query("RETURN 1") # Simple connectivity test
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j for AI queries: {e}")
        return None

def get_graph_schema_dynamic_askai(graph: Neo4jGraph) -> str: # Renamed to avoid conflict
    try:
        if hasattr(graph, "get_schema"):
            schema = graph.get_schema
            return schema() if callable(schema) else schema
        else:
            # Fallback or simpler schema representation if needed
            # This part is crucial for the LLM. If it's too verbose or incomplete, results suffer.
            # Your schema.txt implies a very detailed schema. The LLM might struggle with its full length.
            # Consider a summarized version or focus on key nodes/relationships for the prompt.
            # For now, using the default from Neo4jGraph.
            return "(Schema method not found on graph)"
    except Exception as e:
        return f"(Failed to fetch schema: {e})"

# --- Prompts (as before) ---
PROMPT_SYSTEM = r"""
Task: Generate a Cypher query for a Neo4j graph database based on the provided schema and user question.

Instructions:

1. **Intent Detection**
   - If the question asks for specific factual attributes (e.g., revenue of company X, CEO name, assets > Y), use standard Cypher with `MATCH`.
   - If the question implies semantic similarity (e.g., "companies similar to X", "companies in logistics"), use vector search with `CALL db.index.vector.queryNodes`.

2. **Company Identification**
   - Always identify companies by their `symbol` property.
   - If the user gives a company name (e.g., "Nvidia"), you MUST resolve and use its correct `symbol`.
   - If the company is Saudi-based, append `.SR` to the symbol (e.g., "1010.SR").

3. **Period Handling**
   - Use only **annual** data (`period = 'FY'`) unless the user explicitly requests **quarterly** (`period = 'Q1,Q2,Q3,or Q4'` or mentions "quarter", "Q1", etc.).
   - Always return `symbol`, `period`, and `calendarYear` if available.

4. **Schema Adherence**
   - Use only node labels, relationship types, and properties from the schema.
   - Pay close attention to property names (e.g., `companyName` vs `name`).

5. **Standard Cypher MATCH (for factual queries)**
   - Use these standard aliases:
     - `c` for Company
     - `is` for IncomeStatement
     - `bs` for BalanceSheet
     - `cf` for CashFlowStatement
     - `fr` for FinancialRatio
     - `i` for Industry
     - `s` for Sector
   - Use specific `MATCH` patterns (e.g., `MATCH (c:Company)-[:HAS_INCOME_STATEMENT]->(is:IncomeStatement)`).
   - Filter with `WHERE`, and for partial name matches use:
     `toLower(entity.property) CONTAINS toLower($param)`.
   - To get the latest records, use:
     `ORDER BY financial_node.calendarYear DESC, financial_node.period DESC LIMIT 1`.
   - Return only properties requested.
   -If you use UNION or UNION ALL, ensure that all subqueries return the same number and names of columns, in the same order. Use `AS` to rename columns and `NULL` as placeholder where necessary.
    - For quarterly, `period` will be one of `'Q1'`, `'Q2'`, `'Q3'`, `'Q4'`.
    - To match quarterly data, use: `WHERE is.period IN ['Q1','Q2','Q3','Q4']`

6. **Vector Search (for semantic similarity)**
   - Use **only** for concept-based questions (e.g., “similar companies”, “companies in healthcare”).
   - Use the vector index: `{VECTOR_INDEX_NAME}` # This IS a real placeholder, so single braces are correct
   - Query format:
     ```
     CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', k_value, $query_embedding) # This IS a real placeholder
     YIELD node AS c, score
     ```
   - Optionally expand with:
     `OPTIONAL MATCH (c)-[:IN_INDUSTRY]->(i:Industry)`
   - Return: `c.symbol`, `c.companyName`, and `score`
   - Do NOT include `$query_embedding` in the `"params"` object — it will be added externally.

7. **Math & Calculations**
   - Use `^` for exponentiation.
   - Example CAGR:
     ```
     RETURN ((latest.revenue / earliest.revenue) ^ (1.0 / years) - 1) * 100 AS cagr
     ```

8. **Critical Output Format**
   - Respond with a single **valid JSON object only** — no markdown, code fences, or extra text.
   - The object must have **two keys only**:
     - `"cypher"`: string
     - `"params"`: dictionary (empty if no params required)

- When filtering by industry or sector, prefer exact matches from schema if available. Otherwise, use `CONTAINS`.

9. **Valid Aliases:**
   - When using `AS` to create aliases for returned values (e.g., `RETURN some.value AS MyAlias`), ensure `MyAlias` is a valid Cypher identifier.
   - Valid identifiers start with a letter or underscore and contain only letters, numbers, or underscores.
   - **If an alias MUST contain special characters (like spaces, hyphens, ampersands), it MUST be enclosed in backticks.** For example:
     `RETURN item.value AS `R&D Expense Increase``
     `RETURN item.name AS `Company Name``
   - The `ORDER BY` clause must also use these exact (backticked, if necessary) aliases.
   
10. **List Slicing:**
   - To get a sub-section (slice) of a list, use bracket notation with a range.
   - `myList[startIndex .. endIndexExclusive]` (e.g., `myList[0..3]` gets the first 3 elements at indices 0, 1, 2).
   - `myList[startIndex ..]` (gets elements from startIndex to the end).
   - `myList[.. endIndexExclusive]` (gets elements from the beginning up to, but not including, endIndex).
   - Do NOT use a function like `sublist()`; it does not exist.
   
11. **Calculating Averages from Lists:**
   - When calculating an average from a list of numbers (e.g., collected historical values):
     `RETURN reduce(total = 0.0, x IN myList | total + x) / size(myList) AS average`
   - Always ensure `size(myList)` is greater than 0 before dividing to avoid division by zero errors, or use a CASE statement.
   
12. **Using Parameters in Cypher:**
   - If your query needs dynamic values (e.g., a specific year, a threshold), these values should be passed as parameters.
   - The JSON output should include these in the `"params": {{}}` object.
   - **Crucially, when referencing these parameters within the Cypher query string itself, they MUST be prefixed with a dollar sign (`$`).**
   - Example:
     If params is `{{"target_year": 2023, "min_revenue": 1000000}}`
     Cypher should be: `MATCH (c:Company)-[:HAS_SALES]->(s) WHERE s.year = $target_year AND s.revenue > $min_revenue RETURN c.name`
   - Do NOT use the parameter name directly in the Cypher without the `$` prefix.
   
13. **Handling "Most Recent" or "Last N" Time Periods (Preferred Dynamic Method):**
   - When the user asks for data for the "most recent year/quarter," "last N years/quarters," or a trend over recent periods:
     1. For each company, `MATCH` all relevant financial nodes (e.g., `IncomeStatement`, `BalanceSheet`, `FinancialRatio`, `KeyMetrics`) for the required `period` type (e.g., 'FY' for annual, or ['Q1','Q2','Q3','Q4'] for quarterly).
     2. **Crucially, `ORDER BY` the `calendarYear` DESCENDING, and then by `period` DESCENDING (if applicable for quarterly data, e.g. using a sortable period key or assuming Q4 > Q3 > Q2 > Q1 lexicographically if string).**
     3. `WITH` the company and other necessary fields, `collect()` the relevant data points (e.g., `collect(is.EPS) AS eps_values`, `collect(fr.calendarYear) AS years`). Due to the `ORDER BY`, the collected lists will have the most recent data at the beginning (index 0).
     4. To get the "most recent," take the head of the list (e.g., `eps_values[0]`).
     5. To get the "last N periods," take a slice of the list (e.g., `eps_values[0..N]`). Ensure `size(list) >= N`.
     6. If comparing two specific periods (e.g., latest vs. previous), you would use `list[0]` (latest) and `list[1]` (previous).
   - **Example for "latest 2 annual EPS values":**
     ```cypher
     MATCH (c:Company)-[:HAS_INCOME_STATEMENT]->(is:IncomeStatement)
     WHERE is.period = 'FY' AND is.EPS IS NOT NULL
     WITH c.symbol AS companySymbol, is.calendarYear AS year, is.EPS AS eps
     ORDER BY companySymbol, year DESC
     WITH companySymbol, collect(eps) AS all_annual_eps, collect(year) AS all_annual_years
     WHERE size(all_annual_eps) >= 2 // Ensure at least 2 years of data
     RETURN companySymbol, all_annual_eps[0] AS latest_eps, all_annual_eps[1] AS previous_eps, all_annual_years[0] AS latest_year
     ```
   - This dynamic approach is generally preferred over calculating years based on `{current_year}`, as it uses the actual latest data present in the database.
   - You can still use `{current_year}` (the current calendar year is: {current_year}) as general context if needed, but prioritize finding max/latest data from the graph.  
   
13. **When a user asks about GeographicRegion,ProductSegment or RevenueFact then do **not** apply filter or use **where clause** to filter ProductSegment, GeographicRegion or RevenueFact, instead return all data then answer the question**, apiPeriod period are always "annual" or "quarter".
    -Example: question: list revenue percentage from China for AAPL last 3 years.
    - Cypher query: WITH "AAPL" AS targetSymbolParameter 

MATCH (company:Company)
WHERE company.symbol = targetSymbolParameter
MATCH (company)-[:HAS_INCOME_STATEMENT]->(is:IncomeStatement)
WHERE is.period IN ["FY",'Q1','Q2','Q3','Q4'] 
WITH targetSymbolParameter, company, COLLECT(DISTINCT is.calendarYear) AS allFiscalYears
ORDER BY allFiscalYears DESC
WITH targetSymbolParameter, company, allFiscalYears[0..3] AS lastThreeFiscalYears

UNWIND lastThreeFiscalYears AS fiscalYear

MATCH (company)-[:HAS_INCOME_STATEMENT]->(totalRevenueIS:IncomeStatement)
WHERE totalRevenueIS.calendarYear = fiscalYear AND totalRevenueIS.period = "FY" 

OPTIONAL MATCH (company)-[:REPORTED]->(rf:RevenueFact)-[:FOR_REGION]->(gr:GeographicRegion)
WHERE rf.fiscalYear = fiscalYear
  AND rf.apiPeriod IN ["annual",'quarter'] 


WITH fiscalYear,
     totalRevenueIS.revenue AS totalCompanyRevenue,
     gr.name AS geographicRegionName,
     rf.revenue AS regionRevenue
WHERE geographicRegionName IS NOT NULL 

RETURN
    fiscalYear,
    geographicRegionName,
    regionRevenue,
    totalCompanyRevenue,
    CASE
        WHEN totalCompanyRevenue IS NULL OR totalCompanyRevenue = 0 THEN 0
        ELSE (toFloat(regionRevenue) / totalCompanyRevenue) * 100
    END AS regionRevenuePercentage
ORDER BY fiscalYear DESC, regionRevenuePercentage DESC



14. **marketCapClass values are [LargeCap,MidCap,SmallCap,NanoCap].
Context:
- Schema: {schema}  # Real placeholder for Python .format()
- Valid Sector Names (from schema or inferred): [{valid_sectors}] # Real placeholder for Python .format()
- Valid Industry Names (from schema or inferred): [{valid_industries}]  # Real placeholder for Python .format()
- The current calendar year is: {current_year} # Real placeholder for Python .format()
- User Question: {question}  

JSON Response:
""".strip()

answer_prompt_template = PromptTemplate.from_template(
    """You are an AI assistant that provides **clear and well-formatted answers** based on user question and answer from DB.
The company symbol will likely appear in the question. You can replace the symbol with the company name
from the data you will obtain from the database and answer the question. 
your answer must be comprehinsive and in details and probvide your insigt alongside with periods and dates details.
if no results from Database ask him to try write wright name or wright symbol again.
## **User Question:**
"{question}"

## **Answer from DB:**
"{result}"

## **Your Response:**
"""
)

def _ensure_embeddings(embeddings_instance): # Helper to ensure embeddings are ready
    if embeddings_instance is None:
        st.error("Embeddings client not initialized. Please select an AI provider.")
        raise RuntimeError("Embeddings client not initialized.")
    return embeddings_instance

# --- Core ask_neo4j Function (Modified to return results) ---
def ask_neo4j_logic(graph_instance: Neo4jGraph, 
                    question_text: str, 
                    llm_instance, 
                    embeddings_instance, 
                    llm_provider_name: str, 
                    explain_flag: bool = True) -> Tuple[str, str, str, List[Dict] | str | None]:
    """
    Processes a question, generates Cypher, executes it, and optionally generates an LLM explanation.
    Returns: (generated_cypher, query_params_str, final_llm_answer_str, raw_db_result)
    """
    # st.write(f"❓ Processing Question: {question_text}") # For debugging in Streamlit

    current_year_str = str(datetime.now().year)
    
    # For schema, valid_sectors, valid_industries - these would ideally be fetched once
    # or passed in if they don't change frequently for better performance.
    # For now, fetching schema each time for simplicity here.
    schema_for_prompt = get_graph_schema_dynamic_askai(graph_instance)
    # Simplified entity lists for now, as fetching them live can be slow / complex
    valid_sectors_list = [] 
    valid_industries_list = []
    sectors_for_prompt = ", ".join(f'"{s}"' for s in valid_sectors_list if s)
    industries_for_prompt = ", ".join(f'"{i}"' for i in valid_industries_list if i)

    final_llm_answer = "Could not generate an answer."
    generated_cypher_query = "No Cypher generated."
    query_parameters_str = "{}"
    raw_database_result = None

    try:
        current_prompt_obj = PromptTemplate.from_template(PROMPT_SYSTEM) # Use 'obj' to avoid conflict
        formatted_prompt_str = current_prompt_obj.format( # Use 'str' to avoid conflict
            schema=schema_for_prompt,
            valid_sectors=f"[{sectors_for_prompt}]",
            valid_industries=f"[{industries_for_prompt}]",
            VECTOR_INDEX_NAME=VECTOR_INDEX_NAME,
            current_year=current_year_str,
            question=question_text
        )
    except Exception as e_fmt:
        error_msg = f"Error formatting prompt: {e_fmt}"
        # st.error(error_msg)
        return "Prompt Formatting Error", "{}", error_msg, None

    # st.write("🧠 Generating Cypher via LLM…") # Debug
    llm_response_content = ""
    try:
        llm_response = llm_instance.invoke(formatted_prompt_str)
        llm_response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        # st.text_area("Raw LLM for Cypher:", llm_response_content, height=100) # Debug

        json_str_parsed = llm_response_content # Renamed for clarity
        match_found = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_response_content, re.DOTALL | re.IGNORECASE)
        if match_found:
            json_str_parsed = match_found.group(1).strip()
        else:
            first_brace_idx = json_str_parsed.find('{') # Renamed
            last_brace_idx = json_str_parsed.rfind('}') # Renamed
            if first_brace_idx != -1 and last_brace_idx != -1 and last_brace_idx > first_brace_idx:
                json_str_parsed = json_str_parsed[first_brace_idx : last_brace_idx+1]
        
        try:
            data_from_llm = json.loads(json_str_parsed) # Renamed
        except json.JSONDecodeError as e_json: # Renamed
            error_msg = f"Could not parse LLM Cypher response as JSON. Error: {e_json}. Response: {json_str_parsed}"
            # st.error(error_msg)
            return "LLM JSON Parsing Error", "{}", error_msg, None

        generated_cypher_query = data_from_llm.get("cypher")
        params_from_llm = data_from_llm.get("params", {}) # Renamed

        if not generated_cypher_query or not isinstance(generated_cypher_query, str):
            error_msg = "LLM did not return a valid Cypher query string."
            # st.warning(error_msg)
            return "Invalid Cypher from LLM", "{}", error_msg, None
        if not isinstance(params_from_llm, dict):
            # st.warning("LLM 'params' field was not a dictionary. Using empty parameters.")
            params_from_llm = {}
        
        query_parameters_str = json.dumps(params_from_llm, indent=2) # For display

        if "db.index.vector.queryNodes" in generated_cypher_query and VECTOR_INDEX_NAME in generated_cypher_query:
            # st.write("Vector search detected. Embedding question...") # Debug
            try:
                current_embeddings = _ensure_embeddings(embeddings_instance) # Renamed
                query_embedding_vector_val = current_embeddings.embed_query(question_text) # Renamed
                params_from_llm["query_embedding"] = query_embedding_vector_val
                query_parameters_str = json.dumps(params_from_llm, indent=2) # Update for display
            except RuntimeError as e_runtime: # Renamed
                 error_msg = f"Embeddings client not initialized: {e_runtime}. Cannot perform vector search."
                 # st.error(error_msg)
                 return generated_cypher_query, query_parameters_str, error_msg, None
            except Exception as e_embed: # Renamed
                error_msg = f"Failed to generate or add query embedding: {e_embed}."
                # st.warning(error_msg)
                # Vector search will likely fail if we proceed, so might be better to return
                return generated_cypher_query, query_parameters_str, error_msg, None
        
        # st.write("🚀 Executing query…") # Debug
        raw_database_result = graph_instance.query(generated_cypher_query, params=params_from_llm)

        if raw_database_result is None: # Should not happen if graph_instance is valid
            final_llm_answer = "Query execution returned None (check Neo4j connection or driver issue)."
            return generated_cypher_query, query_parameters_str, final_llm_answer, None
        
        # Prepare result for final answer prompt (can be stringified JSON or a summary)
        result_for_llm_str = json.dumps(raw_database_result, indent=2, default=str) # default=str for non-serializable
        if not raw_database_result:
            result_for_llm_str = "The database query returned no matching records."
            # final_llm_answer = "The database query returned no matching records for your question."

        if explain_flag:
            # st.write("🧠 Generating final answer from LLM…") # Debug
            try:
                answer_chain = answer_prompt_template | llm_instance # Use template obj
                final_llm_answer = answer_chain.invoke({
                    "question": question_text,
                    "result": result_for_llm_str # Pass stringified result
                }).content
            except Exception as e_explain:
                final_llm_answer = f"Failed to generate LLM explanation: {e_explain}. Raw DB result: {result_for_llm_str}"
                # st.warning(final_llm_answer)
        else:
            final_llm_answer = f"Query Executed. Raw results:\n```json\n{result_for_llm_str}\n```"

    except AttributeError as e_attr: # Renamed
        final_llm_answer = f"AttributeError during LLM interaction: {e_attr}. LLM response: {llm_response_content}"
        # st.error(final_llm_answer)
    except Exception as e_ask: # Renamed
        final_llm_answer = f"Unexpected error in ask_neo4j_logic: {e_ask}. LLM response: {llm_response_content}"
        # st.error(final_llm_answer)
        traceback.print_exc(file=sys.stdout) # Print to console for server logs

    return generated_cypher_query, query_parameters_str, final_llm_answer, raw_database_result


# --- Removed CLI part (if __name__ == "__main__": ...) ---
# The Streamlit app will call ask_neo4j_logic directly.