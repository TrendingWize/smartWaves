# analysis_pipeline.py (formerly load_annual_analysis.py - heavily condensed for this example)
from __future__ import annotations
import datetime
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime as dt_class, timedelta
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError, OpenAIError
import requests
import pandas as pd
import streamlit as st
from .financial_data_module import FinancialDataModule, Config

APP_CONFIG = None
FDM_MODULE_INSTANCE = None
OPENAI_CLIENT_INSTANCE = None
NEO4J_DRIVER_INSTANCE = None

FMP_API_KEY  = st.secrets.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", "")
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USER = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

APP_CONFIG = Config(
    fmp_key=FMP_API_KEY,
    openai_key=OPENAI_API_KEY,
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
)


try:
    from neo4j import GraphDatabase, Driver, unit_of_work
except ImportError:
    GraphDatabase = None; Driver = None; unit_of_work = None
    print("WARNING: neo4j driver not installed.")


# OpenAI is optional.  Install the package *and* set OPENAI_API_KEY to enable.
try:
    pass # Already imported above
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Logging & global constants
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("annual")

DEFAULT_MAX_WORKERS = os.cpu_count() or 4
SYMBOL_PROCESSING_WORKERS = 1 # Set as needed, start with 1 for easier debugging

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
symbol='AAL' # Can be a list of symbols later
period='annual' # 'annual' or 'quarter'
os.environ["FMP_API_KEY"] = os.environ.get("FMP_API_KEY", "Aw0rlddPHSnxmi3VmZ6jN4u3b2vvUvxn") # Use provided or existing
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
FMP_API_KEY  = st.secrets.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", "")

# --- Add Neo4j Config ---

NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USER = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

period_back = 0



config = Config(fmp_key="YOUR_FMP_KEY")
fdm = FinancialDataModule(config)


def load_prompt_instruction(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "annual_analysis_instructions.md")
PROMPT_INSTRUCTION = load_prompt_instruction(PROMPT_PATH)

def initialize_analysis_resources():
    global APP_CONFIG, FDM_MODULE_INSTANCE, OPENAI_CLIENT_INSTANCE, NEO4J_DRIVER_INSTANCE

    if APP_CONFIG is None:
        os.environ["FMP_API_KEY"] = FMP_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["NEO4J_URI"] = NEO4J_URI
        os.environ["NEO4J_USER"] = NEO4J_USER
        os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
        APP_CONFIG = Config()

    if FDM_MODULE_INSTANCE is None and APP_CONFIG:
        FDM_MODULE_INSTANCE = FinancialDataModule(config=APP_CONFIG)

    if OPENAI_CLIENT_INSTANCE is None and APP_CONFIG and APP_CONFIG.openai_key and OpenAI:
        try:
            OPENAI_CLIENT_INSTANCE = OpenAI(api_key=APP_CONFIG.openai_key)
            logger.info("OpenAI client initialized for analysis pipeline.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client in pipeline: {e}")
            OPENAI_CLIENT_INSTANCE = None

    # --- ADDED/MODIFIED SECTION FOR NEO4J DRIVER ---
    if NEO4J_DRIVER_INSTANCE is None and APP_CONFIG: # Check if already initialized
        if GraphDatabase and APP_CONFIG.neo4j_uri and APP_CONFIG.neo4j_user and APP_CONFIG.neo4j_password: # Ensure GraphDatabase is imported and config exists
            try:
                logger.info(f"Attempting to initialize Neo4j driver for URI: {APP_CONFIG.neo4j_uri}")
                NEO4J_DRIVER_INSTANCE = GraphDatabase.driver(
                    APP_CONFIG.neo4j_uri,
                    auth=(APP_CONFIG.neo4j_user, APP_CONFIG.neo4j_password)
                )
                NEO4J_DRIVER_INSTANCE.verify_connectivity()
                logger.info("Neo4j driver initialized successfully globally via initialize_analysis_resources.")
            except Exception as e:
                logger.error(f"Failed to initialize global Neo4j driver in pipeline: {e}", exc_info=True)
                NEO4J_DRIVER_INSTANCE = None # Ensure it's None if init fails
        else:
            missing_details = []
            if not GraphDatabase: missing_details.append("neo4j library not imported")
            if not APP_CONFIG.neo4j_uri: missing_details.append("NEO4J_URI missing")
            if not APP_CONFIG.neo4j_user: missing_details.append("NEO4J_USER missing")
            if not APP_CONFIG.neo4j_password: missing_details.append("NEO4J_PASSWORD missing") # Added check
            logger.warning(f"Neo4j driver not initialized globally. Missing: {', '.join(missing_details)}.")
            NEO4J_DRIVER_INSTANCE = None # Explicitly set to None
            
        pass
    # --- END OF ADDED/MODIFIED SECTION ---

# --- Main function to be called by Streamlit ---
def generate_ai_report(
    symbol_to_process: str,
    report_period: str,
    period_back_offset: int = 0
    # REMOVE st_neo4j_driver parameter if you exclusively use the global
) -> Optional[Dict[str, Any]]:
    initialize_analysis_resources() # Sets global NEO4J_DRIVER_INSTANCE

    report_data = process_symbol_logic(
        symbol_to_process=symbol_to_process,
        current_period_back_val=period_back_offset,
        fdm_module_instance=FDM_MODULE_INSTANCE,
        openai_client_instance=OPENAI_CLIENT_INSTANCE,
        neo4j_driver_instance=NEO4J_DRIVER_INSTANCE, # <--- USE THE GLOBAL INSTANCE
        app_config=APP_CONFIG,
        # Make sure to pass report_period to process_symbol_logic too
        # process_symbol_logic(..., report_period_param=report_period)
    )
    # ...
    return report_data
    """
    Generates an AI financial analysis report for a given symbol and period.
    Returns the full analysis report dictionary or None on failure.
    """
    initialize_analysis_resources() # Ensure global instances are ready

    if not APP_CONFIG or not FDM_MODULE_INSTANCE:
        logger.error("Analysis pipeline resources (Config/FDM) not initialized.")
        return {"status": "error_init", "message": "Core analysis modules not initialized."}

    logger.info(f"Calling process_symbol_logic for {symbol_to_process}, period: {report_period}, back: {period_back_offset}")

    try:
        # You'll need to adjust the call to your `process_symbol_logic`
        # to ensure it uses the `report_period` and `period_back_offset`
        # and the globally initialized or passed `FDM_MODULE_INSTANCE`,
        # `OPENAI_CLIENT_INSTANCE`, and `st_neo4j_driver`.

        # If process_symbol_logic is directly callable and adapted:
        report_data = process_symbol_logic(
            symbol_to_process=symbol_to_process,
            current_period_back_val=period_back_offset, # Ensure this matches param name in process_symbol_logic
            fdm_module_instance=FDM_MODULE_INSTANCE,    # Pass initialized FDM
            openai_client_instance=OPENAI_CLIENT_INSTANCE, # Pass initialized OpenAI
            neo4j_driver_instance=st_neo4j_driver,       # Pass Neo4j driver
            app_config=APP_CONFIG                       # Pass config
            # You might need to also explicitly pass `report_period` if process_symbol_logic
            # uses a global `period` variable internally.
        )
        # Ensure process_symbol_logic itself uses the `report_period` (annual/quarter)
        # when calling fdm_module.get_financial_statements(..., period_param=report_period, ...)

        if isinstance(report_data, dict) and "analysis" in report_data and isinstance(report_data["analysis"], dict):
            return report_data # Return the full report
        else:
            logger.error(f"Analysis generation failed or returned unexpected format for {symbol_to_process}. Report data: {str(report_data)[:500]}")
            if isinstance(report_data, dict): return report_data # Return error dict
            return {"status": "error_unexpected_report_format", "symbol": symbol_to_process, "message": "Unexpected report format from processing."}

    except Exception as e:
        logger.error(f"Exception during generate_ai_report for {symbol_to_process}: {e}", exc_info=True)
        return {"status": "error_exception", "symbol": symbol_to_process, "message": str(e)}
        
        
def _must_get(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable '{name}' is missing.")
    return val


@unit_of_work(timeout=10) # type: ignore
def get_analysis_from_neo4j(tx, symbol_param: str, filling_date_str: str) -> Optional[Dict[str, Any]]:
    query = (
        "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($filling_date_str)}) "
        "RETURN ar"
    )
    result = tx.run(query, symbol_param=symbol_param, filling_date_str=filling_date_str)
    record = result.single()
    if record and record["ar"]:
        # The properties of the node 'ar' form the dictionary
        # Need to handle Neo4j specific types like Date, DateTime if they are not auto-converted by driver
        report_data = dict(record["ar"])
        # Ensure metadata and analysis are dictionaries (they should be from Cypher load)
        if 'metadata' in report_data and isinstance(report_data['metadata'], str):
            try: report_data['metadata'] = json.loads(report_data['metadata'])
            except json.JSONDecodeError: pass # Keep as string if not valid JSON
        if 'analysis' in report_data and isinstance(report_data['analysis'], str):
            try: report_data['analysis'] = json.loads(report_data['analysis'])
            except json.JSONDecodeError: pass
        
        # Convert Neo4j DateTime/Date objects in metadata back to ISO strings if necessary
        if 'metadata' in report_data and isinstance(report_data['metadata'], dict):
            for key, val in report_data['metadata'].items():
                if hasattr(val, 'iso_format'): # Check if it's a Neo4j Date/DateTime object
                    report_data['metadata'][key] = val.iso_format()
        if hasattr(report_data.get('analysis_generated_at'), 'iso_format'):
             report_data['analysis_generated_at'] = report_data['analysis_generated_at'].iso_format()

        return report_data
    return None

@unit_of_work(timeout=30) # type: ignore
def save_analysis_to_neo4j(tx, symbol_param: str, analysis_report_data: Dict[str, Any]):
    # analysis_report_data is the JSON from OpenAI plus your added fields.
    
    metadata_block = analysis_report_data.get("metadata", {}) # This is a dict
    analysis_block = analysis_report_data.get("analysis", {})   # This is a dict
    fmp_data_snapshot_block = analysis_report_data.get("fmp_data_for_analysis", {}) # This is a dict

    filling_date_str = metadata_block.get("fillingDate") 

    if not filling_date_str:
        logger.error(f"Cannot save analysis for {symbol_param} to Neo4j: missing fillingDate in metadata.")
        # Optionally raise an error here to prevent proceeding
        raise ValueError(f"Missing fillingDate in metadata for symbol {symbol_param} during Neo4j save.")

    # Merge Company Node
    company_cypher = (
        "MERGE (c:Company {symbol: $symbol_param}) "
        "ON CREATE SET c.companyName = $companyName, c.exchange = $exchange, c.sector = $sector, c.industry = $industry, c.lastUpdated = datetime() "
        "ON MATCH SET c.companyName = COALESCE(c.companyName, $companyName), c.exchange = COALESCE(c.exchange, $exchange), "
        "c.sector = COALESCE(c.sector, $sector), c.industry = COALESCE(c.industry, $industry), c.lastUpdated = datetime()"
    )
    company_params = {
        "symbol_param": symbol_param,
        "companyName": metadata_block.get("company_name"),
        "exchange": metadata_block.get("exchange"),
        "sector": metadata_block.get("sector"),
        "industry": metadata_block.get("industry")
    }
    # logger.debug(f"Executing Company Cypher with params: {company_params}") # Optional debug
    tx.run(company_cypher, **company_params)

    # Prepare parameters for AnalysisReport, serializing dicts to JSON strings
    params_for_ar = {
        "symbol_param": symbol_param, # Used in MERGE (ar) and MATCH (c)
        "filling_date_str": filling_date_str, # Used in MERGE (ar)
        
        "metadata_json_str": json.dumps(metadata_block), 
        "analysis_json_str": json.dumps(analysis_block), 
        #"fmp_data_snapshot_json_str": json.dumps(fmp_data_snapshot_block, default=str), 

        "prompt_tokens": analysis_report_data.get("prompt_tokens"),
        "completion_tokens": analysis_report_data.get("completion_tokens"),
        "total_tokens": analysis_report_data.get("total_tokens"),
        "analysis_generated_at_str": analysis_report_data.get("analysis_generated_at"), 
        "model_used": analysis_report_data.get("model_used"),
        "symbol_processing_duration": analysis_report_data.get("symbol_processing_duration_total"),
        "calendarYear": analysis_report_data.get("metadata", {}).get("calendarYear"),
    }

    # ---- START DETAILED LOGGING ----
    logger.info("--- Preparing to save AnalysisReport. Parameter types: ---")
    for key, value in params_for_ar.items():
        logger.info(f"Param: {key}, Type: {type(value)}, Value (first 100 chars if str): {str(value)[:100] if isinstance(value, str) else value}")
    logger.info("--- End of parameter types ---")
    # ---- END DETAILED LOGGING ----

    query_ar = (
        "MERGE (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($filling_date_str)}) "
        "ON CREATE SET "
        "  ar.metadata_json = $metadata_json_str, "
        "  ar.analysis_json = $analysis_json_str, "
        "  ar.prompt_tokens = $prompt_tokens, "
        "  ar.completion_tokens = $completion_tokens, "
        "  ar.total_tokens = $total_tokens, "
        "  ar.analysis_generated_at = datetime($analysis_generated_at_str), "
        "  ar.model_used = $model_used, "
        "  ar.symbol_processing_duration = $symbol_processing_duration, "
        "  ar.calendarYear = $calendarYear, "
        "  ar.lastUpdated = datetime() "
        "ON MATCH SET "
        "  ar.metadata_json = $metadata_json_str, "
        "  ar.analysis_json = $analysis_json_str, "
        "  ar.prompt_tokens = $prompt_tokens, "
        "  ar.completion_tokens = $completion_tokens, "
        "  ar.total_tokens = $total_tokens, "
        "  ar.analysis_generated_at = datetime($analysis_generated_at_str), "
        "  ar.model_used = $model_used, "
        "  ar.symbol_processing_duration = $symbol_processing_duration, "
        "  ar.calendarYear = $calendarYear, "
        "  ar.lastUpdated = datetime() "
        "WITH ar "
        "MATCH (c:Company {symbol: $symbol_param}) " # $symbol_param is used here
        "MERGE (c)-[:HAS_ANALYSIS_REPORT]->(ar) "
        "RETURN ar"
    )
    tx.run(query_ar, **params_for_ar) # This is where the error occurs during commit
    logger.info(f"Analysis for {symbol_param} (fillingDate: {filling_date_str}) saved to Neo4j (queued in transaction).") # Clarified logging
    return True
    

def process_symbol_logic(
    symbol_to_process: str,
    current_period_back_val: int,
    fdm_module_instance: FinancialDataModule,
    openai_client_instance: Optional[OpenAI],
    neo4j_driver_instance: Optional[Driver],
    app_config: Config
):
    start_ts = time.time()
    logger.info(f"Starting analysis for symbol: {symbol_to_process}, period_back: {current_period_back_val}")

    prospective_fmp_filling_date_str = None

    # STEP 1: Determine prospective_fmp_filling_date_str
    try:
        target_statements = fdm_module_instance.get_financial_statements(
            symbol=symbol_to_process, statement="income-statement",
            period_param=period, limit=1 + current_period_back_val
        )
        if target_statements and len(target_statements) > current_period_back_val:
            target_statement_for_date = target_statements[current_period_back_val]
            date_val = target_statement_for_date.get("fillingDate") or target_statement_for_date.get("date")
            if date_val: prospective_fmp_filling_date_str = date_val[:10]
        
        if prospective_fmp_filling_date_str:
            logger.info(f"Prospective FMP filling date for {symbol_to_process} is {prospective_fmp_filling_date_str}")
        else:
            logger.warning(f"Could not determine prospective FMP filling date for {symbol_to_process}. Cache check will be less targeted or skipped.")
    except Exception as e_date_fetch:
        logger.error(f"Error during initial FMP call for filling date for {symbol_to_process}: {e_date_fetch}.")
        prospective_fmp_filling_date_str = None # Ensure it's None

    # STEP 2: Check Neo4j cache
    if neo4j_driver_instance: # Check if driver is available before attempting to use it
        # Define transformer inside the conditional block to ensure neo4j_driver_instance is valid for its scope
        def transform_and_check_analysis_report(record_cursor_ar):
            single_record_ar = record_cursor_ar.single()
            if not single_record_ar or not single_record_ar["ar"]: return None
            
            report_node_data = dict(single_record_ar["ar"])
            metadata_dict = {}
            analysis_dict = {} # Initialize analysis_dict

            if 'metadata_json' in report_node_data and isinstance(report_node_data['metadata_json'], str):
                try: metadata_dict = json.loads(report_node_data['metadata_json'])
                except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding metadata_json for {symbol_to_process}: {e}"); return None
            else: logger.warning(f"Cache: Missing or invalid metadata_json for {symbol_to_process}."); return None

            analysis_as_of_date_str = metadata_dict.get("as_of_date")
            report_node_filling_date_obj = report_node_data.get("fillingDate")
            report_node_filling_date_str = report_node_filling_date_obj.iso_format()[:10] if report_node_filling_date_obj and hasattr(report_node_filling_date_obj, 'iso_format') else None

            logger.info(f"Cache Candidate for {symbol_to_process}: Node.fillingDate={report_node_filling_date_str}, Metadata.as_of_date={analysis_as_of_date_str}")

            # Core comparison: Prospective FMP fillingDate vs. cached analysis's as_of_date
            if prospective_fmp_filling_date_str and analysis_as_of_date_str and \
               prospective_fmp_filling_date_str == analysis_as_of_date_str:
                logger.info(f"CACHE HIT: Prospective FMP Date ({prospective_fmp_filling_date_str}) matches Cached Analysis As-Of-Date ({analysis_as_of_date_str}).")
                
                # Reconstruct the full report object
                full_cached_report = {
                    "metadata": metadata_dict,
                    "analysis": {}, # Initialize
                    # Include other top-level fields from the node if they were stored directly
                    "prompt_tokens": report_node_data.get("prompt_tokens"),
                    "completion_tokens": report_node_data.get("completion_tokens"),
                    "total_tokens": report_node_data.get("total_tokens"),
                    "analysis_generated_at": report_node_data.get("analysis_generated_at").iso_format() if hasattr(report_node_data.get('analysis_generated_at'), 'iso_format') else None,
                    "model_used": report_node_data.get("model_used"),
                    "symbol_processing_duration_total": report_node_data.get("symbol_processing_duration"), # Assuming 'symbol_processing_duration' was the key on the node
                    "fmp_data_for_analysis": {} # Placeholder, or load if stored
                }

                if 'analysis_json' in report_node_data and isinstance(report_node_data['analysis_json'], str):
                    try: full_cached_report['analysis'] = json.loads(report_node_data['analysis_json'])
                    except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding analysis_json for {symbol_to_process}: {e}"); # Keep analysis as {}
                
                # If fmp_data_snapshot_json was stored, load it
                #if 'fmp_data_snapshot_json' in report_node_data and isinstance(report_node_data['fmp_data_snapshot_json'], str):
                #    try: full_cached_report['fmp_data_for_analysis'] = json.loads(report_node_data['fmp_data_snapshot_json'])
                #    except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding fmp_data_snapshot_json for {symbol_to_process}: {e}");

                return full_cached_report
            
            logger.info(f"CACHE MISS or Date Mismatch for {symbol_to_process}.")
            return None

        cypher_for_cache_check = ""
        params_for_cache_check = {"symbol_param": symbol_to_process}
        if prospective_fmp_filling_date_str:
            cypher_for_cache_check = "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($date_param)}) RETURN ar LIMIT 1"
            params_for_cache_check["date_param"] = prospective_fmp_filling_date_str
        else: # Fallback if prospective date couldn't be determined, get latest and check its as_of_date
            cypher_for_cache_check = "MATCH (ar:AnalysisReport {symbol: $symbol_param}) RETURN ar ORDER BY ar.fillingDate DESC LIMIT 1"

        existing_analysis_report = neo4j_driver_instance.execute_query(
            query_=cypher_for_cache_check, database_=None, 
            result_transformer_=transform_and_check_analysis_report, **params_for_cache_check
        )

        if existing_analysis_report: # If transformer returned a valid, reconstructed report
            logger.info(f"Using cached analysis report for {symbol_to_process}.")
            return existing_analysis_report # THIS IS THE CACHE HIT RETURN
    else:
        logger.info("Neo4j driver not available or prospective FMP date unknown. Skipping cache check.")


    logger.info(f"Proceeding with full FMP data fetch and OpenAI analysis for {symbol_to_process} (Prospective FMP Date: {prospective_fmp_filling_date_str or 'Unknown'}).")
    
    fmp_company_data = None
    actual_fmp_filling_date_str = None 

    try:
        # STEP 3: Fetch full FMP data
        fmp_company_data = fdm_module_instance.execute_all(
            symbol_param=symbol_to_process,
            current_period_back=current_period_back_val,
            current_period_type=period
        )
        
        date_val_pkg = (fmp_company_data.get("metadata_package", {}).get("fmp_filling_date") or
                       (fmp_company_data["income_statement"][0].get("fillingDate") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None) or
                       (fmp_company_data["income_statement"][0].get("date") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None))
        if not date_val_pkg:
            logger.error(f"CRITICAL: No FMP filling date in full package for {symbol_to_process}.")
            return {"status": "fmp_error_no_date_full_pkg", "symbol": symbol_to_process, "fmp_data": fmp_company_data or {}}
        actual_fmp_filling_date_str = date_val_pkg[:10]
        logger.info(f"Actual FMP filling date from full package for {symbol_to_process} is {actual_fmp_filling_date_str}")

        # Optional: Save raw FMP data to disk
        # ... (code to save fmp_company_data to file) ...

        # STEP 4: OpenAI Analysis
        if not openai_client_instance:
            logger.warning(f"OpenAI client not available. Skipping OpenAI for {symbol_to_process}.")
            return {"status": "data_only_no_openai", "symbol": symbol_to_process, "fmp_data": fmp_company_data}

        question = f"Perform a detailed {period} financial and fundamental analysis for ({symbol_to_process}) company using the provided data."
        instructions = load_prompt_instruction(PROMPT_PATH)
        fmp_company_data_string = json.dumps(fmp_company_data, ensure_ascii=False, default=str)
        
        logger.info(f"Requesting OpenAI analysis for {symbol_to_process} (FMP fillingDate: {actual_fmp_filling_date_str})...")
        generated_analysis_json = None 
        response_obj_for_metadata = None # To store the successful response object

        try:
            response = openai_client_instance.chat.completions.create(
                model=getattr(app_config, 'openai_model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"{period.capitalize()} Company financial information (JSON):\n{fmp_company_data_string}\n\nQuestion: {question}"}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=8192
            )
            response_obj_for_metadata = response # Save for metadata if successful
            message_content = response.choices[0].message.content
            #print(message_content)
            if message_content:
                try: 
                    generated_analysis_json = json.loads(message_content)
                    logger.info(f"Successfully parsed OpenAI JSON response for {symbol_to_process}.")
                except json.JSONDecodeError as jde:
                    logger.error(f"OpenAI: Invalid JSON for {symbol_to_process}: {jde}. Resp: {message_content[:200]}");
                    generated_analysis_json = {"error_openai_json": str(jde), "raw_response": message_content}
            else:
                logger.error(f"OpenAI: Empty response for {symbol_to_process}. Finish: {response.choices[0].finish_reason}");
                generated_analysis_json = {"error_openai_empty": f"Empty. Reason: {response.choices[0].finish_reason}"}

        except (APIError, APIConnectionError, RateLimitError, BadRequestError) as e_api:
            logger.error(f"OpenAI API error for {symbol_to_process}: {e_api}", exc_info=False)
            generated_analysis_json = {"error_openai_api": str(e_api)}
        except Exception as e_openai_proc:
            logger.error(f"OpenAI processing error for {symbol_to_process}: {e_openai_proc}", exc_info=True)
            generated_analysis_json = {"error_openai_processing": str(e_openai_proc)}
        
        if generated_analysis_json is None:
             generated_analysis_json = {"error_openai_unknown": "Analysis was not populated after call attempt."}

        # Add metadata to generated_analysis_json
        if isinstance(generated_analysis_json, dict):
            if not any(k.startswith("error_") for k in generated_analysis_json) and response_obj_for_metadata:
                # Only add these if OpenAI call was successful and we have the response object
                generated_analysis_json['prompt_tokens'] = response_obj_for_metadata.usage.prompt_tokens
                generated_analysis_json['completion_tokens'] = response_obj_for_metadata.usage.completion_tokens
                generated_analysis_json['total_tokens'] = response_obj_for_metadata.usage.total_tokens
                generated_analysis_json['model_used'] = response_obj_for_metadata.model
            
            generated_analysis_json['analysis_generated_at'] = dt_class.now(datetime.timezone.utc).isoformat()
            generated_analysis_json['symbol_processing_duration_total'] = time.time() - start_ts
            
            current_meta = generated_analysis_json.get('metadata', {})
            if not isinstance(current_meta, dict): current_meta = {}
            current_meta['ticker'] = symbol_to_process
            current_meta['fillingDate'] = actual_fmp_filling_date_str # For Neo4j node key
            current_meta['as_of_date'] = actual_fmp_filling_date_str  # For cache comparison consistency
            current_meta['calendarYear'] = fmp_company_data.get("metadata_package", {}).get("fmp_calendar_year")
            generated_analysis_json['metadata'] = current_meta
            
            generated_analysis_json['fmp_data_for_analysis'] = fmp_company_data
        
        # ... (Optional save generated_analysis_json to disk) ...

        # STEP 5: Save new analysis to Neo4j (only if no errors from OpenAI)
        if neo4j_driver_instance and isinstance(generated_analysis_json, dict) and \
           not any(k.startswith("error_") for k in generated_analysis_json): # Check for any error key
            try:
                with neo4j_driver_instance.session(database_=None) as session:
                    session.execute_write(
                        save_analysis_to_neo4j,
                        symbol_param=symbol_to_process,
                        analysis_report_data=generated_analysis_json
                    )
            except Exception as e_neo_save:
                logger.error(f"Error saving NEW analysis to Neo4j for {symbol_to_process}: {e_neo_save}", exc_info=True)
                print(f"NEO4J SAVE ERROR: {type(e_neo_save).__name__}: {e_neo_save}")
                if isinstance(generated_analysis_json, dict): generated_analysis_json['error_neo4j_save'] = str(e_neo_save)
        
        if isinstance(generated_analysis_json, dict) and not any(k.startswith("error_") for k in generated_analysis_json):
             logger.info(f"SUCCESS (new analysis by OpenAI): {symbol_to_process} processed in {time.time() - start_ts:.2f} seconds.")
        else:
             logger.warning(f"ISSUES processing {symbol_to_process}. Final result object: {str(generated_analysis_json)[:200]}...")
        return generated_analysis_json

    except RuntimeError as e_runtime:
        logger.error(f"RUNTIME ERROR for {symbol_to_process} during full FMP fetch: {e_runtime}", exc_info=True)
        return {"status": "runtime_error_fmp_full", "symbol": symbol_to_process, "error": str(e_runtime), "fmp_data_on_error": fmp_company_data}
    except Exception as e_main:
        logger.error(f"OVERALL FAILED for {symbol_to_process} in main block: {e_main}", exc_info=True)
        return {"status": "overall_failure_main_process", "symbol": symbol_to_process, "error": str(e_main)}
