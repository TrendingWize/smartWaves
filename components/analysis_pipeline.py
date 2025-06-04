# analysis_pipeline.py (Refactored)

from __future__ import annotations
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError
from .financial_data_module import FinancialDataModule, Config

try:
    from neo4j import GraphDatabase, Driver, unit_of_work
except ImportError:
    GraphDatabase = None; Driver = None; unit_of_work = None
    print("WARNING: neo4j driver not installed.")

# --------------------------------------------------------------------------- #
# Logging & Constants
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("annual")

# --------------------------------------------------------------------------- #
# Helper: get or create all resource singletons in Streamlit session state
# --------------------------------------------------------------------------- #
def get_analysis_resources():
    if not hasattr(st.session_state, "APP_CONFIG"):
        st.session_state.APP_CONFIG = Config(
            fmp_key=st.secrets.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", ""),
            openai_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            openai_model=st.secrets.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            neo4j_uri=st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI"),
            neo4j_user=st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER"),
            neo4j_password=st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD"),
        )
    config = st.session_state.APP_CONFIG

    if not hasattr(st.session_state, "FDM_MODULE_INSTANCE"):
        st.session_state.FDM_MODULE_INSTANCE = FinancialDataModule(config)

    if not hasattr(st.session_state, "OPENAI_CLIENT_INSTANCE") and config.openai_key:
        st.session_state.OPENAI_CLIENT_INSTANCE = OpenAI(api_key=config.openai_key)

    if not hasattr(st.session_state, "NEO4J_DRIVER_INSTANCE") and config.neo4j_uri and GraphDatabase:
        st.session_state.NEO4J_DRIVER_INSTANCE = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )
        try:
            st.session_state.NEO4J_DRIVER_INSTANCE.verify_connectivity()
        except Exception as e:
            logger.error(f"Neo4j connectivity failed: {e}")

    return (
        st.session_state.APP_CONFIG,
        st.session_state.FDM_MODULE_INSTANCE,
        st.session_state.OPENAI_CLIENT_INSTANCE,
        st.session_state.NEO4J_DRIVER_INSTANCE
    )

# --------------------------------------------------------------------------- #
# Prompt Instruction Loader with Error Handling
# --------------------------------------------------------------------------- #
def load_prompt_instruction(md_path: str) -> str:
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt instruction file not found: {md_path}")
        return "INSTRUCTIONS FILE NOT FOUND. Please check configuration."

# --------------------------------------------------------------------------- #
# Neo4j: Get & Save Analysis Report
# --------------------------------------------------------------------------- #
@unit_of_work(timeout=10)
def get_analysis_from_neo4j(tx, symbol_param: str, filling_date_str: str) -> Optional[Dict[str, Any]]:
    query = (
        "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($filling_date_str)}) "
        "RETURN ar"
    )
    result = tx.run(query, symbol_param=symbol_param, filling_date_str=filling_date_str)
    record = result.single()
    if record and record["ar"]:
        report_data = dict(record["ar"])
        if 'metadata' in report_data and isinstance(report_data['metadata'], str):
            try: report_data['metadata'] = json.loads(report_data['metadata'])
            except json.JSONDecodeError: pass
        if 'analysis' in report_data and isinstance(report_data['analysis'], str):
            try: report_data['analysis'] = json.loads(report_data['analysis'])
            except json.JSONDecodeError: pass
        if 'metadata' in report_data and isinstance(report_data['metadata'], dict):
            for key, val in report_data['metadata'].items():
                if hasattr(val, 'iso_format'):
                    report_data['metadata'][key] = val.iso_format()
        if hasattr(report_data.get('analysis_generated_at'), 'iso_format'):
            report_data['analysis_generated_at'] = report_data['analysis_generated_at'].iso_format()
        return report_data
    return None

@unit_of_work(timeout=30)
def save_analysis_to_neo4j(tx, symbol_param: str, analysis_report_data: Dict[str, Any]):
    metadata_block = analysis_report_data.get("metadata", {})
    analysis_block = analysis_report_data.get("analysis", {})
    filling_date_str = metadata_block.get("fillingDate") 

    if not filling_date_str:
        logger.error(f"Cannot save analysis for {symbol_param} to Neo4j: missing fillingDate in metadata.")
        raise ValueError(f"Missing fillingDate in metadata for symbol {symbol_param} during Neo4j save.")

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
    tx.run(company_cypher, **company_params)

    params_for_ar = {
        "symbol_param": symbol_param,
        "filling_date_str": filling_date_str,
        "metadata_json_str": json.dumps(metadata_block),
        "analysis_json_str": json.dumps(analysis_block),
        "prompt_tokens": analysis_report_data.get("prompt_tokens"),
        "completion_tokens": analysis_report_data.get("completion_tokens"),
        "total_tokens": analysis_report_data.get("total_tokens"),
        "analysis_generated_at_str": analysis_report_data.get("analysis_generated_at"),
        "model_used": analysis_report_data.get("model_used"),
        "symbol_processing_duration": analysis_report_data.get("symbol_processing_duration_total"),
        "calendarYear": analysis_report_data.get("metadata", {}).get("calendarYear"),
    }

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
        "MATCH (c:Company {symbol: $symbol_param}) "
        "MERGE (c)-[:HAS_ANALYSIS_REPORT]->(ar) "
        "RETURN ar"
    )
    tx.run(query_ar, **params_for_ar)
    logger.info(f"Analysis for {symbol_param} (fillingDate: {filling_date_str}) saved to Neo4j.")

# --------------------------------------------------------------------------- #
# Main: AI Analysis Generation Function
# --------------------------------------------------------------------------- #
def generate_ai_report(
    symbol_to_process: str,
    report_period: str,
    period_back_offset: int = 0
) -> Optional[Dict[str, Any]]:
    config, fdm, openai_client, neo4j_driver = get_analysis_resources()
    PROMPT_PATH = os.path.join(os.path.dirname(__file__), "annual_analysis_instructions.md")
    instructions = load_prompt_instruction(PROMPT_PATH)
    return process_symbol_logic(
        symbol_to_process=symbol_to_process,
        current_period_back_val=period_back_offset,
        report_period=report_period,
        fdm_module_instance=fdm,
        openai_client_instance=openai_client,
        neo4j_driver_instance=neo4j_driver,
        app_config=config,
        prompt_instructions=instructions,
        prompt_path=PROMPT_PATH
    )

# --------------------------------------------------------------------------- #
# Core Symbol Processing/Analysis
# --------------------------------------------------------------------------- #
def process_symbol_logic(
    symbol_to_process: str,
    current_period_back_val: int,
    report_period: str,
    fdm_module_instance: FinancialDataModule,
    openai_client_instance: Optional[OpenAI],
    neo4j_driver_instance: Optional[Any],
    app_config: Config,
    prompt_instructions: str,
    prompt_path: str
):
    start_ts = time.time()
    logger.info(f"Starting analysis for symbol: {symbol_to_process}, period: {report_period}, period_back: {current_period_back_val}")

    prospective_fmp_filling_date_str = None
    try:
        target_statements = fdm_module_instance.get_financial_statements(
            symbol=symbol_to_process, statement="income-statement",
            period_param=report_period, limit=1 + current_period_back_val
        )
        if target_statements and len(target_statements) > current_period_back_val:
            target_statement_for_date = target_statements[current_period_back_val]
            date_val = target_statement_for_date.get("fillingDate") or target_statement_for_date.get("date")
            if date_val: prospective_fmp_filling_date_str = date_val[:10]
    except Exception as e_date_fetch:
        logger.error(f"Error during FMP call for filling date for {symbol_to_process}: {e_date_fetch}.")
        prospective_fmp_filling_date_str = None

    # Neo4j cache check
    if neo4j_driver_instance and prospective_fmp_filling_date_str:
        def transform_and_check_analysis_report(record_cursor_ar):
            single_record_ar = record_cursor_ar.single()
            if not single_record_ar or not single_record_ar["ar"]: return None
            report_node_data = dict(single_record_ar["ar"])
            metadata_dict = {}
            if 'metadata_json' in report_node_data and isinstance(report_node_data['metadata_json'], str):
                try: metadata_dict = json.loads(report_node_data['metadata_json'])
                except json.JSONDecodeError: logger.error(f"Cache: Error decoding metadata_json for {symbol_to_process}."); return None
            analysis_as_of_date_str = metadata_dict.get("as_of_date")
            report_node_filling_date_obj = report_node_data.get("fillingDate")
            report_node_filling_date_str = report_node_filling_date_obj.iso_format()[:10] if report_node_filling_date_obj and hasattr(report_node_filling_date_obj, 'iso_format') else None
            if prospective_fmp_filling_date_str and analysis_as_of_date_str and prospective_fmp_filling_date_str == analysis_as_of_date_str:
                full_cached_report = {
                    "metadata": metadata_dict,
                    "analysis": {},
                    "prompt_tokens": report_node_data.get("prompt_tokens"),
                    "completion_tokens": report_node_data.get("completion_tokens"),
                    "total_tokens": report_node_data.get("total_tokens"),
                    "analysis_generated_at": report_node_data.get("analysis_generated_at").iso_format() if hasattr(report_node_data.get('analysis_generated_at'), 'iso_format') else None,
                    "model_used": report_node_data.get("model_used"),
                    "symbol_processing_duration_total": report_node_data.get("symbol_processing_duration"),
                    "fmp_data_for_analysis": {}
                }
                if 'analysis_json' in report_node_data and isinstance(report_node_data['analysis_json'], str):
                    try: full_cached_report['analysis'] = json.loads(report_node_data['analysis_json'])
                    except json.JSONDecodeError: logger.error(f"Cache: Error decoding analysis_json for {symbol_to_process}.")
                return full_cached_report
            return None

        cypher_for_cache_check = "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($date_param)}) RETURN ar LIMIT 1"
        params_for_cache_check = {"symbol_param": symbol_to_process, "date_param": prospective_fmp_filling_date_str}
        existing_analysis_report = neo4j_driver_instance.execute_query(
            query_=cypher_for_cache_check, database_=None, 
            result_transformer_=transform_and_check_analysis_report, **params_for_cache_check
        )
        if existing_analysis_report:
            logger.info(f"Using cached analysis report for {symbol_to_process}.")
            return existing_analysis_report

    logger.info(f"Proceeding with FMP data fetch and OpenAI analysis for {symbol_to_process} (Prospective FMP Date: {prospective_fmp_filling_date_str or 'Unknown'}).")

    fmp_company_data = None
    actual_fmp_filling_date_str = None
    try:
        fmp_company_data = fdm_module_instance.execute_all(
            symbol_param=symbol_to_process,
            current_period_back=current_period_back_val,
            current_period_type=report_period
        )
        date_val_pkg = (
            fmp_company_data.get("metadata_package", {}).get("fmp_filling_date") or
            (fmp_company_data["income_statement"][0].get("fillingDate") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None) or
            (fmp_company_data["income_statement"][0].get("date") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None)
        )
        if not date_val_pkg:
            logger.error(f"No FMP filling date in package for {symbol_to_process}.")
            return {"status": "fmp_error_no_date_full_pkg", "symbol": symbol_to_process, "fmp_data": fmp_company_data or {}}
        actual_fmp_filling_date_str = date_val_pkg[:10]

        if not openai_client_instance:
            logger.warning(f"OpenAI client not available. Skipping OpenAI for {symbol_to_process}.")
            return {"status": "data_only_no_openai", "symbol": symbol_to_process, "fmp_data": fmp_company_data}

        question = f"Perform a detailed {report_period} financial and fundamental analysis for ({symbol_to_process}) company using the provided data."
        fmp_company_data_string = json.dumps(fmp_company_data, ensure_ascii=False, default=str)

        logger.info(f"Requesting OpenAI analysis for {symbol_to_process} (FMP fillingDate: {actual_fmp_filling_date_str})...")
        generated_analysis_json = None
        response_obj_for_metadata = None

        try:
            response = openai_client_instance.chat.completions.create(
                model=app_config.openai_model,
                messages=[
                    {"role": "system", "content": prompt_instructions},
                    {"role": "user", "content": f"{report_period.capitalize()} Company financial information (JSON):\n{fmp_company_data_string}\n\nQuestion: {question}"}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=8192
            )
            response_obj_for_metadata = response
            message_content = response.choices[0].message.content
            if message_content:
                try:
                    generated_analysis_json = json.loads(message_content)
                    logger.info(f"Successfully parsed OpenAI JSON response for {symbol_to_process}.")
                except json.JSONDecodeError as jde:
                    logger.error(f"OpenAI: Invalid JSON for {symbol_to_process}: {jde}. Resp: {message_content[:200]}")
                    generated_analysis_json = {"error_openai_json": str(jde), "raw_response": message_content}
            else:
                logger.error(f"OpenAI: Empty response for {symbol_to_process}. Reason: {response.choices[0].finish_reason}")
                generated_analysis_json = {"error_openai_empty": f"Empty. Reason: {response.choices[0].finish_reason}"}
        except (APIError, APIConnectionError, RateLimitError, BadRequestError) as e_api:
            logger.error(f"OpenAI API error for {symbol_to_process}: {e_api}")
            generated_analysis_json = {"error_openai_api": str(e_api)}
        except Exception as e_openai_proc:
            logger.exception(f"OpenAI processing error for {symbol_to_process}: {e_openai_proc}")
            generated_analysis_json = {"error_openai_processing": str(e_openai_proc)}

        if generated_analysis_json is None:
            generated_analysis_json = {"error_openai_unknown": "Analysis was not populated after call attempt."}

        if isinstance(generated_analysis_json, dict):
            if not any(k.startswith("error_") for k in generated_analysis_json) and response_obj_for_metadata:
                generated_analysis_json['prompt_tokens'] = response_obj_for_metadata.usage.prompt_tokens
                generated_analysis_json['completion_tokens'] = response_obj_for_metadata.usage.completion_tokens
                generated_analysis_json['total_tokens'] = response_obj_for_metadata.usage.total_tokens
                generated_analysis_json['model_used'] = response_obj_for_metadata.model

            generated_analysis_json['analysis_generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            generated_analysis_json['symbol_processing_duration_total'] = time.time() - start_ts

            current_meta = generated_analysis_json.get('metadata',
