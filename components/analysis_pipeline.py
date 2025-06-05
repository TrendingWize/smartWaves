# analysis_pipeline.py (refactored, robust, and clean)
from __future__ import annotations
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime as dt_class
from typing import Any, Dict, Optional

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # If not running in Streamlit

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError
except ImportError:
    OpenAI = None

try:
    from neo4j import GraphDatabase, Driver, unit_of_work
except ImportError:
    GraphDatabase = None
    Driver = None
    unit_of_work = lambda *a, **k: (lambda f: f)

from .financial_data_module import FinancialDataModule, Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("annual")

# ---------------------------------------------------------------------------- #
# Resource/context loader (A, C)
# ---------------------------------------------------------------------------- #
def get_analysis_resources():
    # If running in Streamlit, keep resources per session; else local vars.
    config = None
    fdm = None
    openai_client = None
    neo4j_driver = None

    # Streamlit or script
    if st is not None and hasattr(st, "session_state"):
        sess = st.session_state
        if not hasattr(sess, "APP_CONFIG"):
            sess.APP_CONFIG = Config(
                fmp_key=st.secrets.get("FMP_API_KEY", "") or os.getenv("FMP_API_KEY", ""),
                openai_key=st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
                openai_model=st.secrets.get("OPENAI_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                neo4j_uri=st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI", ""),
                neo4j_user=st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER", ""),
                neo4j_password=st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", ""),
            )
        config = sess.APP_CONFIG

        if not hasattr(sess, "FDM_MODULE_INSTANCE"):
            sess.FDM_MODULE_INSTANCE = FinancialDataModule(config)
        fdm = sess.FDM_MODULE_INSTANCE

        if not hasattr(sess, "OPENAI_CLIENT_INSTANCE") and config.openai_key and OpenAI:
            sess.OPENAI_CLIENT_INSTANCE = OpenAI(api_key=config.openai_key)
        openai_client = getattr(sess, "OPENAI_CLIENT_INSTANCE", None)

        if not hasattr(sess, "NEO4J_DRIVER_INSTANCE") and config.neo4j_uri and GraphDatabase:
            sess.NEO4J_DRIVER_INSTANCE = GraphDatabase.driver(
                config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
            )
        neo4j_driver = getattr(sess, "NEO4J_DRIVER_INSTANCE", None)
    else:
        # Non-Streamlit/script context: local scope
        config = Config(
            fmp_key=os.getenv("FMP_API_KEY", ""),
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_user=os.getenv("NEO4J_USER", ""),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        )
        fdm = FinancialDataModule(config)
        openai_client = OpenAI(api_key=config.openai_key) if config.openai_key and OpenAI else None
        neo4j_driver = (
            GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            if config.neo4j_uri and GraphDatabase
            else None
        )
    return config, fdm, openai_client, neo4j_driver

# ---------------------------------------------------------------------------- #
# Prompt loader (E)
# ---------------------------------------------------------------------------- #
def load_prompt_instruction(md_path: str) -> str:
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt instruction file not found: {md_path}")
        return "INSTRUCTIONS FILE NOT FOUND. Please check configuration."

# ---------------------------------------------------------------------------- #
# Main pipeline entry point
# ---------------------------------------------------------------------------- #
def generate_ai_report(
    symbol_to_process: str,
    report_period: str,
    period_back_offset: int = 0
) -> Optional[Dict[str, Any]]:
    config, fdm, openai_client, neo4j_driver = get_analysis_resources()

    # Load prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "annual_analysis_instructions.md")
    instructions = load_prompt_instruction(prompt_path)

    try:
        report_data = process_symbol_logic(
            symbol_to_process=symbol_to_process,
            current_period_back_val=period_back_offset,
            report_period=report_period,
            fdm_module_instance=fdm,
            openai_client_instance=openai_client,
            neo4j_driver_instance=neo4j_driver,
            app_config=config,
            prompt_instructions=instructions,
        )
        return report_data
    except Exception as e:
        logger.exception(f"Exception in generate_ai_report: {e}")
        return {"status": "error_exception", "symbol": symbol_to_process, "message": str(e)}

# ---------------------------------------------------------------------------- #
# Neo4j UOW for fetch/save
# ---------------------------------------------------------------------------- #
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
            try:
                report_data['metadata'] = json.loads(report_data['metadata'])
            except json.JSONDecodeError:
                pass
        if 'analysis' in report_data and isinstance(report_data['analysis'], str):
            try:
                report_data['analysis'] = json.loads(report_data['analysis'])
            except json.JSONDecodeError:
                pass
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
    logger.info(f"Analysis for {symbol_param} (fillingDate: {filling_date_str}) saved to Neo4j (queued in transaction).")
    return True

# ---------------------------------------------------------------------------- #
# Core Analysis Logic (now fully parameterized, no globals)
# ---------------------------------------------------------------------------- #
def process_symbol_logic(
    symbol_to_process: str,
    current_period_back_val: int,
    report_period: str,
    fdm_module_instance: FinancialDataModule,
    openai_client_instance: Optional[OpenAI],
    neo4j_driver_instance: Optional[Driver],
    app_config: Config,
    prompt_instructions: str,
):
    start_ts = time.time()
    logger.info(f"Starting analysis for symbol: {symbol_to_process}, period_back: {current_period_back_val}")

    prospective_fmp_filling_date_str = None

    # STEP 1: Determine prospective_fmp_filling_date_str
    try:
        target_statements = fdm_module_instance.get_financial_statements(
            symbol=symbol_to_process, statement="income-statement",
            period_param=report_period, limit=1 + current_period_back_val
        )
        if target_statements and len(target_statements) > current_period_back_val:
            target_statement_for_date = target_statements[current_period_back_val]
            date_val = target_statement_for_date.get("fillingDate") or target_statement_for_date.get("date")
            if date_val:
                prospective_fmp_filling_date_str = date_val[:10]
        if prospective_fmp_filling_date_str:
            logger.info(f"Prospective FMP filling date for {symbol_to_process} is {prospective_fmp_filling_date_str}")
        else:
            logger.warning(f"Could not determine prospective FMP filling date for {symbol_to_process}.")
    except Exception as e_datdef process_symbol_logic(
    symbol_to_process: str,
    current_period_back_val: int,
    report_period: str,
    fdm_module_instance,
    openai_client_instance,
    neo4j_driver_instance,
    app_config,
    prompt_instructions: str,
):
    import time, json
    from datetime import datetime as dt_class
    logger = logging.getLogger("annual")
    start_ts = time.time()
    logger.info(f"Starting analysis for symbol: {symbol_to_process}, period_back: {current_period_back_val}")

    # STEP 1: Determine prospective_fmp_filling_date_str
    prospective_fmp_filling_date_str = None
    try:
        target_statements = fdm_module_instance.get_financial_statements(
            symbol=symbol_to_process, statement="income-statement",
            period_param=report_period, limit=1 + current_period_back_val
        )
        if target_statements and len(target_statements) > current_period_back_val:
            target_statement_for_date = target_statements[current_period_back_val]
            date_val = target_statement_for_date.get("fillingDate") or target_statement_for_date.get("date")
            if date_val:
                prospective_fmp_filling_date_str = date_val[:10]
        if prospective_fmp_filling_date_str:
            logger.info(f"Prospective FMP filling date for {symbol_to_process} is {prospective_fmp_filling_date_str}")
        else:
            logger.warning(f"Could not determine prospective FMP filling date for {symbol_to_process}.")
    except Exception as e_date_fetch:
        logger.error(f"Error during initial FMP call for filling date for {symbol_to_process}: {e_date_fetch}.")
        prospective_fmp_filling_date_str = None

    # STEP 2: Check Neo4j cache
    existing_analysis_report = None
    if neo4j_driver_instance:
        def transform_and_check_analysis_report(record_cursor_ar):
            single_record_ar = record_cursor_ar.single()
            if not single_record_ar or not single_record_ar.get("ar"):
                return None
            report_node_data = dict(single_record_ar["ar"])
            metadata_dict, analysis_dict = {}, {}
            if 'metadata_json' in report_node_data and isinstance(report_node_data['metadata_json'], str):
                try:
                    metadata_dict = json.loads(report_node_data['metadata_json'])
                except json.JSONDecodeError as e:
                    logger.error(f"Cache: Error decoding metadata_json for {symbol_to_process}: {e}")
                    return None
            else:
                logger.warning(f"Cache: Missing or invalid metadata_json for {symbol_to_process}.")
                return None
            if 'analysis_json' in report_node_data and isinstance(report_node_data['analysis_json'], str):
                try:
                    analysis_dict = json.loads(report_node_data['analysis_json'])
                except json.JSONDecodeError as e:
                    logger.error(f"Cache: Error decoding analysis_json for {symbol_to_process}: {e}")
                    analysis_dict = {}
            analysis_as_of_date_str = metadata_dict.get("as_of_date")
            report_node_filling_date_obj = report_node_data.get("fillingDate")
            report_node_filling_date_str = (
                report_node_filling_date_obj.iso_format()[:10]
                if report_node_filling_date_obj and hasattr(report_node_filling_date_obj, 'iso_format')
                else None
            )
            if prospective_fmp_filling_date_str and analysis_as_of_date_str and \
               prospective_fmp_filling_date_str == analysis_as_of_date_str:
                logger.info(
                    f"CACHE HIT: Prospective FMP Date ({prospective_fmp_filling_date_str}) matches Cached Analysis As-Of-Date ({analysis_as_of_date_str})."
                )
                return {
                    "metadata": metadata_dict,
                    "analysis": analysis_dict,
                    "prompt_tokens": report_node_data.get("prompt_tokens"),
                    "completion_tokens": report_node_data.get("completion_tokens"),
                    "total_tokens": report_node_data.get("total_tokens"),
                    "analysis_generated_at": (
                        report_node_data.get("analysis_generated_at").iso_format()
                        if hasattr(report_node_data.get("analysis_generated_at"), "iso_format")
                        else report_node_data.get("analysis_generated_at")
                    ),
                    "model_used": report_node_data.get("model_used"),
                    "symbol_processing_duration_total": report_node_data.get("symbol_processing_duration"),
                    "fmp_data_for_analysis": {},
                }
            return None

        cypher_for_cache_check = ""
        params_for_cache_check = {"symbol_param": symbol_to_process}
        if prospective_fmp_filling_date_str:
            cypher_for_cache_check = (
                "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($date_param)}) RETURN ar LIMIT 1"
            )
            params_for_cache_check["date_param"] = prospective_fmp_filling_date_str
        else:
            cypher_for_cache_check = (
                "MATCH (ar:AnalysisReport {symbol: $symbol_param}) RETURN ar ORDER BY ar.fillingDate DESC LIMIT 1"
            )
        existing_analysis_report = neo4j_driver_instance.execute_query(
            query_=cypher_for_cache_check,
            database_=None,
            result_transformer_=transform_and_check_analysis_report,
            **params_for_cache_check
        )
        if existing_analysis_report:
            logger.info(f"CACHE HIT: Using cached analysis report for {symbol_to_process}.")
            return existing_analysis_report
        else:
            logger.info(f"CACHE MISS: No cached analysis for {symbol_to_process} (filling date: {prospective_fmp_filling_date_str})")
    else:
        logger.info("Neo4j driver not available or prospective FMP date unknown. Skipping cache check.")

    # STEP 3: Fetch full FMP data
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
            logger.error(f"CRITICAL: No FMP filling date in full package for {symbol_to_process}.")
            return {"status": "fmp_error_no_date_full_pkg", "symbol": symbol_to_process, "fmp_data": fmp_company_data or {}}
        actual_fmp_filling_date_str = date_val_pkg[:10]
        logger.info(f"Fetched FMP data for {symbol_to_process}. Actual filling date: {actual_fmp_filling_date_str}")
    except Exception as e_fmp:
        logger.error(f"Failed FMP data fetch for {symbol_to_process}: {e_fmp}", exc_info=True)
        return {"status": "fmp_error_fetch", "symbol": symbol_to_process, "error": str(e_fmp)}

    # STEP 4: OpenAI Analysis
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
            model=getattr(app_config, 'openai_model', 'gpt-4o'),
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
            logger.error(f"OpenAI: Empty response for {symbol_to_process}. Finish: {response.choices[0].finish_reason}")
            generated_analysis_json = {"error_openai_empty": f"Empty. Reason: {response.choices[0].finish_reason}"}

    except Exception as e_openai_proc:
        logger.error(f"OpenAI processing error for {symbol_to_process}: {e_openai_proc}", exc_info=True)
        generated_analysis_json = {"error_openai_processing": str(e_openai_proc)}

    logger.info(f"OpenAI analysis keys for {symbol_to_process}: {list(generated_analysis_json.keys()) if isinstance(generated_analysis_json, dict) else 'not a dict'}")

    if generated_analysis_json is None:
        generated_analysis_json = {"error_openai_unknown": "Analysis was not populated after call attempt."}

    # Add metadata to generated_analysis_json
    if isinstance(generated_analysis_json, dict):
        if not any(k.startswith("error_") for k in generated_analysis_json) and response_obj_for_metadata:
            generated_analysis_json['prompt_tokens'] = getattr(response_obj_for_metadata.usage, "prompt_tokens", None)
            generated_analysis_json['completion_tokens'] = getattr(response_obj_for_metadata.usage, "completion_tokens", None)
            generated_analysis_json['total_tokens'] = getattr(response_obj_for_metadata.usage, "total_tokens", None)
            generated_analysis_json['model_used'] = getattr(response_obj_for_metadata, "model", None)

        generated_analysis_json['analysis_generated_at'] = dt_class.now().isoformat()
        generated_analysis_json['symbol_processing_duration_total'] = time.time() - start_ts

        current_meta = generated_analysis_json.get('metadata', {})
        if not isinstance(current_meta, dict):
            current_meta = {}
        current_meta['ticker'] = symbol_to_process
        current_meta['fillingDate'] = actual_fmp_filling_date_str
        current_meta['as_of_date'] = actual_fmp_filling_date_str
        current_meta['calendarYear'] = fmp_company_data.get("metadata_package", {}).get("fmp_calendar_year")
        generated_analysis_json['metadata'] = current_meta
        generated_analysis_json['fmp_data_for_analysis'] = fmp_company_data

    # STEP 5: Save new analysis to Neo4j (only if no errors from OpenAI)
    if (
        neo4j_driver_instance
        and isinstance(generated_analysis_json, dict)
        and not any(k.startswith("error_") for k in generated_analysis_json)
    ):
        try:
            with neo4j_driver_instance.session(database_=None) as session:
                session.execute_write(
                    save_analysis_to_neo4j,
                    symbol_param=symbol_to_process,
                    analysis_report_data=generated_analysis_json
                )
            logger.info(f"Saved new analysis report for {symbol_to_process} to Neo4j (filling date: {actual_fmp_filling_date_str})")
        except Exception as e_neo_save:
            logger.error(f"Error saving NEW analysis to Neo4j for {symbol_to_process}: {e_neo_save}", exc_info=True)
            if isinstance(generated_analysis_json, dict):
                generated_analysis_json['error_neo4j_save'] = str(e_neo_save)

    if isinstance(generated_analysis_json, dict) and not any(k.startswith("error_") for k in generated_analysis_json):
        logger.info(f"SUCCESS (new analysis by OpenAI): {symbol_to_process} processed in {time.time() - start_ts:.2f} seconds.")
    else:
        logger.warning(f"ISSUES processing {symbol_to_process}. Final result object: {str(generated_analysis_json)[:200]}...")

    return generated_analysis_json

