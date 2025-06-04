from .db import get_neo4j_driver
from typing import Any, List, Dict 
import pandas as pd
import plotly.express as px

from .utils_helpers import (
    format_value,
    calculate_delta,
    _arrow,
    R_display_metric_card,
    get_nearest_aggregate_similarities,
    fetch_financial_details_for_companies,
    fetch_income_statement_data,
    fetch_sector_list,
    fetch_company_preview
)


import streamlit as st
import sqlite3
import hashlib
import uuid # For session tokens
from styles import load_global_css

DATABASE_NAME = 'users.db'

# --- Database Setup ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_user_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_approved INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            current_session_token TEXT
        )
    ''')
    # Ensure admin user exists
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    admin = cursor.fetchone()
    if not admin:
        hashed_password = hash_password("1234")
        cursor.execute(
            "INSERT INTO users (username, password_hash, is_approved, is_admin) VALUES (?, ?, ?, ?)",
            ("admin", hashed_password, 1, 1)
        )
    conn.commit()
    conn.close()

# --- Password Hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password_hash, provided_password):
    return stored_password_hash == hash_password(provided_password)

# --- User Management ---
def add_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError: # Username already exists
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_unapproved_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE is_approved = 0 AND is_admin = 0")
    users = cursor.fetchall()
    conn.close()
    return users

def approve_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_approved = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def update_session_token(username, token):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET current_session_token = ? WHERE username = ?", (token, username))
    conn.commit()
    conn.close()

# --- Session State Initialization ---
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'session_token' not in st.session_state: # For this browser session
        st.session_state.session_token = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home" # Default page

# --- Login / Logout ---
def login_user(username, password):
    user = get_user(username)
    if user and verify_password(user['password_hash'], password):
        if not user['is_approved']:
            st.error("Your account is pending approval from an administrator.")
            return False
        
        # Generate a new session token for this login
        new_session_token = str(uuid.uuid4())
        update_session_token(username, new_session_token)

        st.session_state.logged_in = True
        st.session_state.username = user['username']
        st.session_state.is_admin = bool(user['is_admin'])
        st.session_state.session_token = new_session_token # Store in browser session
        st.success(f"Welcome back, {username}!")
        st.rerun() # Rerun to update UI
        return True
    st.error("Invalid username or password.")
    return False

def logout_user():
    if st.session_state.username:
        update_session_token(st.session_state.username, None) # Clear token in DB
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.session_state.session_token = None
    st.success("You have been logged out.")
    st.rerun()

# --- Concurrent Login Check ---
def check_concurrent_login():
    if st.session_state.logged_in and st.session_state.username and st.session_state.session_token:
        user = get_user(st.session_state.username)
        if user and user['current_session_token'] != st.session_state.session_token:
            st.warning("You have been logged out because this account was logged in from another device/browser.")
            logout_user() # This will call rerun
            return True # Was logged out
    return False # Not logged out by this check

def plot_financial_performance_charts_row(
    df_dict: Dict[str, pd.DataFrame], # Dict where key is metric name, value is DataFrame with 'period' and value col
    chart_titles: Dict[str, str],    # Dict mapping metric name to display chart title
    value_col_names: Dict[str, str], # Dict mapping metric name to its value column name (e.g., 'value', 'value_percent')
    is_percent_dict: Dict[str, bool] # Dict mapping metric name to boolean if it's a percentage
):
    active_charts_data = []
    for metric_name, df in df_dict.items():
        if df is not None and not df.empty and 'period' in df.columns and value_col_names.get(metric_name) in df.columns:
            if pd.api.types.is_numeric_dtype(df[value_col_names[metric_name]]):
                 active_charts_data.append({
                     "metric_name": metric_name,
                     "df": df,
                     "title": chart_titles.get(metric_name, metric_name.replace("_"," ").title()),
                     "value_col": value_col_names.get(metric_name),
                     "is_percent": is_percent_dict.get(metric_name, False)
                 })
    
    if not active_charts_data:
        # st.caption("No data for this chart group.") # Can be too noisy
        return

    num_charts_to_plot = len(active_charts_data)
    cols = st.columns(num_charts_to_plot)

    for i, chart_data in enumerate(active_charts_data):
        with cols[i]:
            df_plot = chart_data["df"].copy() # Work with a copy
            value_col = chart_data["value_col"]

            # Ensure periods are sorted (CRUCIAL for line charts to connect correctly)
            try:
                if all(isinstance(p, str) and p.isdigit() and len(p) == 4 for p in df_plot['period']):
                    df_plot['period_num'] = pd.to_numeric(df_plot['period'])
                    df_plot = df_plot.sort_values(by='period_num').drop(columns=['period_num'])
                # Add more sophisticated date parsing/sorting here if 'period' can be "YYYY-MM-DD", "Q1 2023" etc.
                # Example for "YYYY-MM-DD":
                # elif all(isinstance(p, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', p) for p in df_plot['period']):
                #    df_plot['period_dt'] = pd.to_datetime(df_plot['period'])
                #    df_plot = df_plot.sort_values(by='period_dt').drop(columns=['period_dt'])
            except Exception as e_sort:
                st.caption(f"Note: Could not sort periods for {chart_data['title']}. Chart might appear unsorted. Error: {e_sort}")


            chart_is_line = len(df_plot) >= 4 # Adjusted heuristic for line chart
            chart_kwargs = {"x": 'period', "y": value_col, "title": chart_data["title"]}

            if chart_is_line:
                chart_kwargs["markers"] = True
                fig = px.line(df_plot, **chart_kwargs)
            else:
                chart_kwargs["text_auto"] = True
                fig = px.bar(df_plot, **chart_kwargs)
            
            yaxis_title = value_col.replace("_"," ").title()
            yaxis_ticksuffix = "%" if chart_data["is_percent"] else None
            yaxis_tickformat = None if chart_data["is_percent"] else "$,.2s"

            fig.update_layout(title_x=0.5, title_font_size=14, 
                              yaxis_title=yaxis_title, 
                              yaxis_ticksuffix=yaxis_ticksuffix, yaxis_tickformat=yaxis_tickformat,
                              height=320, margin=dict(t=40,b=20,l=10,r=10), xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
            
def display_ai_analysis_dashboard(analysis_data: Dict[str, Any], metadata: Dict[str, Any], company_symbol: str):
    load_global_css()

    # ... (Dashboard Overview Cards section remains the same) ...
    st.markdown("<div class='section-header-ai'>Dashboard Overview</div>", unsafe_allow_html=True)
    val_summary_data = analysis_data.get("valuation_analysis", {}).get("valuation_summary", {})
    current_price_val = metadata.get("last_price")
    card_metrics_config = [
        {"label": "Fair Value (Base Est.)", "value_path": ["valuation_analysis", "valuation_summary", "fair_value_estimate_base"], "type": "currency"},
        {"label": "Current Price", "value": current_price_val, "type": "currency"},
        {"label": "Recommendation", "value_path": ["investment_thesis_summary", "overall_recommendation"], "type": "text_strong"},
        {"label": "P/E (TTM)", "value_path": ["fundamental_analysis", "valuation_ratios", "pe_ratio_ttm"], "type": "number"},
        {"label": "Revenue CAGR (Hist.)", "value_path": ["growth_prospects", "historical_growth_summary", "revenue_cagr", "rate_percent"], "type": "percent"},
        {"label": "ROE (TTM)", "value_path": ["fundamental_analysis", "profitability_ratios", "roe_ttm"], "type": "percent"},
    ]
    active_cards_list = [] # Logic to populate active_cards_list remains same
    for card_conf in card_metrics_config:
        val = card_conf.get("value") 
        if val is None and "value_path" in card_conf: 
            temp_val = analysis_data
            try:
                for p_key in card_conf["value_path"]: temp_val = temp_val[p_key]
                val = temp_val
            except: val = None
        if val is not None and str(val).strip() != "" and str(val).lower() != 'n/a':
            active_cards_list.append({"label": card_conf["label"], "value": val, "type": card_conf["type"]})
    if active_cards_list: # Card display logic remains same
        cols = st.columns(len(active_cards_list))
        for i, card_info in enumerate(active_cards_list):
            with cols[i]:
                val_str = str(card_info["value"]) 
                if card_info["type"] == "currency" and isinstance(card_info["value"], (int, float)): val_str = f"${card_info['value']:,.2f}"
                elif card_info["type"] == "percent" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.1f}%"
                elif card_info["type"] == "number" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.2f}"
                elif card_info["type"] == "text_strong": val_str = f"<strong>{card_info['value']}</strong>"
                st.markdown(f"""<div class="metric-card-ai"><div class="stMetricLabel">{card_info['label']}</div><div class="stMetricValue">{val_str}</div></div>""", unsafe_allow_html=True)
    else: st.info("Key overview metrics not available.")


    # --- Financial Performance Section (Modified for grouped charts) ---
    fin_perf_data = analysis_data.get("financial_performance", {})
    if fin_perf_data:
        st.markdown(f"<div class='section-header-ai'>Financial Performance ({company_symbol})</div>", unsafe_allow_html=True)

        # --- Chart Group 1: Key Absolute Values ---
        st.markdown("<div class='subsection-header-ai'>Key Performance Indicators (Absolute Values)</div>", unsafe_allow_html=True)
        abs_metrics_df_dict = {}
        abs_metrics_titles = {}
        abs_metrics_value_cols = {}
        abs_metrics_is_percent = {}

        for metric_key, display_title in [
            ("revenues", "Revenue"), 
            ("profitability_metrics.grossProfit", "Gross Profit"), # Assuming grossProfit is nested like this, or adjust
            ("profitability_metrics.operating_income", "Operating Income"),
            ("profitability_metrics.net_income", "Net Income")
        ]:
            data_path = metric_key.split('.')
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data:
                    current_data = current_data[p_key]
                else:
                    valid_path = False; break
            
            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value' in df.columns: # Assuming primary value column is 'value'
                    abs_metrics_df_dict[metric_key] = df
                    abs_metrics_titles[metric_key] = display_title
                    abs_metrics_value_cols[metric_key] = 'value'
                    abs_metrics_is_percent[metric_key] = False
        
        plot_financial_performance_charts_row(abs_metrics_df_dict, abs_metrics_titles, abs_metrics_value_cols, abs_metrics_is_percent)
        
        # Display explanations for this group if available
        for metric_key_orig, _ in [("revenues", ""), ("profitability_metrics.grossProfit", ""), ("profitability_metrics.operating_income", ""), ("profitability_metrics.net_income", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data
            valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)


        # --- Chart Group 2: Key Margins ---
        st.markdown("<div class='subsection-header-ai'>Key Profitability Margins (%)</div>", unsafe_allow_html=True)
        margin_metrics_df_dict = {}
        margin_metrics_titles = {}
        margin_metrics_value_cols = {}
        margin_metrics_is_percent = {}

        for metric_key, display_title in [
            ("gross_margin", "Gross Margin"),
            ("ebitda_margin", "EBITDA Margin"),
            ("net_margin", "Net Margin"),
            ("profitability_metrics.roic", "ROIC") # Example if ROIC is here and has "values"
        ]:
            data_path = metric_key.split('.') # Handles potential nesting like profitability_metrics.roic
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data: current_data = current_data[p_key]
                else: valid_path = False; break

            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value_percent' in df.columns: # Margins usually have 'value_percent'
                    margin_metrics_df_dict[metric_key] = df
                    margin_metrics_titles[metric_key] = display_title
                    margin_metrics_value_cols[metric_key] = 'value_percent'
                    margin_metrics_is_percent[metric_key] = True
        
        plot_financial_performance_charts_row(margin_metrics_df_dict, margin_metrics_titles, margin_metrics_value_cols, margin_metrics_is_percent)

        # Display explanations for margins
        for metric_key_orig, _ in [("gross_margin", ""), ("ebitda_margin", ""), ("net_margin", ""), ("profitability_metrics.roic", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data; valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)

    # --- Render other main analysis sections using the generic renderer ---
    sections_in_order_config = [
        # ("financial_performance", "Financial Performance"), # Handled specially above
        ("fundamental_analysis", "Fundamental Ratios (TTM)"),
        ("valuation_analysis", "Valuation Analysis"),
        ("growth_prospects", "Growth Prospects & Outlook"),
        ("competitive_position", "Competitive Landscape"),
        ("peers_comparison", "Peer Group Benchmarking"),
        ("revenue_segmentation", "Revenue Segmentation Insights"),
        ("risk_factors", "Key Risk Factors"),
        ("scenario_analysis", "Scenario Analysis (Valuation)"),
        ("shareholder_returns_analysis", "Shareholder Returns & Capital Allocation"),
        ("investment_thesis_summary", "Investment Thesis & Recommendation"),
    ]

    for section_key, section_display_title in sections_in_order_config:
        section_data_content = analysis_data.get(section_key)
        if section_data_content:
            st.markdown(f"<div class='section-header-ai'>{section_display_title}</div>", unsafe_allow_html=True)
            with st.container():
                render_generic_section(section_data_content, [section_key])
    
    st.markdown("---")
    if st.checkbox("Show Full Raw AI Analysis JSON", value=False, key="show_raw_ai_json_cb_v4"):
        st.json(analysis_data)
        
def get_sentiment_html_enhanced(classification_text, text_prefix=""):
    if not classification_text or not isinstance(classification_text, str): return ""
    icon = ""; style_class = ""
    cl_lower = classification_text.lower()
    if any(s in cl_lower for s in ["positive", "bullish", "outperform", "favorable"]): icon, style_class = "✅", "sentiment-positive"
    elif any(s in cl_lower for s in ["negative", "bearish", "underperform", "unfavorable", "risks"]): icon, style_class = "❌", "sentiment-negative"
    elif any(s in cl_lower for s in ["neutral", "mixed", "cautiously"]): icon, style_class = "⚠️", "sentiment-neutral"
    return f"<span class='{style_class}'>{icon} {text_prefix}{classification_text}</span>" if icon else f"<span>({classification_text})</span>"

def format_currency_short(value):
    if pd.isna(value) or not isinstance(value, (int, float)): return "N/A"
    if abs(value) >= 1_000_000_000_000: return f"${value/1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000: return f"${value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000: return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000: return f"${value/1_000:.1f}K"
    return f"${value:,.2f}"

def format_generic_value(value, key_context=""):
    if pd.isna(value) or value is None: return "N/A" # Added explicit None check
    val_str = str(value)
    key_lower = key_context.lower()
    if isinstance(value, (int, float)):
        if any(s in key_lower for s in ["price", "value", "amount", "capitalization", "ebitda", "income", "debt", "equity", "assets", "cash"]) and \
           not any(s in key_lower for s in ["percent", "rate", "margin", "yield", "ratio", "cagr", "beta", "multiple", "growth", "coverage", "turnover", "cycle"]): # More specific exclusion
            return format_currency_short(value)
        elif any(s in key_lower for s in ["percent", "rate", "margin", "yield", "cagr", "growth"]):
            return f"{value:.2f}%"
        elif isinstance(value, float):
            return f"{value:.2f}" # Default for other floats
    return val_str # Return original string if not a number or no specific format matched

# --- Recursive Section Renderer (MUST BE DEFINED BEFORE display_ai_analysis_dashboard) ---
def render_generic_section(data_to_render: Any, current_path_keys: List[str]):
    if data_to_render is None: return # Added None check

    if isinstance(data_to_render, list):
        if not data_to_render:
            st.markdown(f"<div class='kpi-item'>  •  No items listed.</div>", unsafe_allow_html=True)
            return
        
        # Special handling for peers_comparison.comparison_table
        if current_path_keys and current_path_keys[-1] == "comparison_table" and all(isinstance(item, dict) for item in data_to_render):
            try:
                df_comparison = pd.DataFrame(data_to_render)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Could not render comparison table: {e}")
                for item_dict in data_to_render: # Fallback to list rendering
                    st.markdown(f"<div class='kpi-item'>▪️ {json.dumps(item_dict)}</div>", unsafe_allow_html=True)
            return # Handled comparison_table

        for item_dict in data_to_render: # Assuming list of dicts for most other cases
            if isinstance(item_dict, dict):
                summary_parts = []
                item_classification_html = ""
                if "item_text" in item_dict: summary_parts.append(item_dict["item_text"])
                elif "name" in item_dict and "ticker" in item_dict: summary_parts.append(f"{item_dict['name']} ({item_dict.get('ticker', 'N/A')})")
                elif "segment_name" in item_dict: summary_parts.append(f"{item_dict['segment_name']}: {item_dict.get('revenue_percentage', 'N/A')}% ({format_currency_short(item_dict.get('revenue_amount'))})")
                elif "region" in item_dict: summary_parts.append(f"{item_dict['region']}: {item_dict.get('revenue_percentage', 'N/A')}% ({format_currency_short(item_dict.get('revenue_amount'))})")
                else:
                    for k_item, v_item in item_dict.items():
                        if k_item != "classification": summary_parts.append(f"{str(k_item).replace('_',' ').title()}: {format_generic_value(v_item, k_item)}")
                if "classification" in item_dict: item_classification_html = get_sentiment_html_enhanced(item_dict["classification"])
                st.markdown(f"<div class='kpi-item'>▪️ {', '.join(summary_parts)} {item_classification_html}</div>", unsafe_allow_html=True)
            elif isinstance(item_dict, str):
                st.markdown(f"<div class='kpi-item'>▪️ {item_dict}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='kpi-item'>▪️ {str(item_dict)}</div>", unsafe_allow_html=True)
        return

    if isinstance(data_to_render, dict):
        for key, value in data_to_render.items():
            if value == "__PROCESSED__": continue
            display_key = str(key).replace("_", " ").title()
            new_path_keys = current_path_keys + [key]

            if key == "values" and isinstance(value, list) and value and isinstance(value[0], dict) and "period" in value[0]:
                parent_section_title = str(current_path_keys[-1]).replace("_"," ").title() if current_path_keys else "Trend"
                df_values = pd.DataFrame(value)
                possible_value_cols = ['value', 'value_percent', 'growth_percent', 'rate_percent']
                value_col_name = next((col for col in possible_value_cols if col in df_values.columns and pd.api.types.is_numeric_dtype(df_values[col])), None)
                if value_col_name and 'period' in df_values.columns:
                    try:
                        if all(isinstance(p, str) and p.isdigit() and len(p) == 4 for p in df_values['period']):
                            df_values['period_num'] = pd.to_numeric(df_values['period'])
                            df_values = df_values.sort_values(by='period_num').drop(columns=['period_num'])
                    except: pass
                    chart_is_line = len(df_values) >= 5
                    chart_kwargs = {"x": 'period', "y": value_col_name, "title": parent_section_title}
                    if chart_is_line: chart_kwargs["markers"] = True; fig = px.line(df_values, **chart_kwargs)
                    else: chart_kwargs["text_auto"] = True; fig = px.bar(df_values, **chart_kwargs)
                    is_percent_chart_val = any(s in value_col_name.lower() for s in ["percent", "rate", "margin"])
                    fig.update_layout(title_x=0.5, yaxis_title=value_col_name.replace("_"," ").title(), 
                                      yaxis_ticksuffix="%" if is_percent_chart_val else None, 
                                      yaxis_tickformat=None if is_percent_chart_val else "$,.2s",
                                      height=350, margin=dict(t=50,b=20,l=20,r=20), xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"**{display_key}:**")
                    st.dataframe(df_values, hide_index=True, use_container_width=True)
                continue

            is_text_block = any(key.endswith(suffix) for suffix in ["explanation", "commentary", "summary", "justification", "overview", "assumptions", "note"])
            if is_text_block and isinstance(value, str):
                classification_html = ""
                base_key_for_class = key
                for suffix in ["explanation", "commentary", "summary", "justification", "overview", "assumptions", "note"]:
                    if key.endswith(suffix): base_key_for_class = key[:-len(suffix)]; break
                possible_class_keys = [f"{base_key_for_class}classification", f"{base_key_for_class}_classification"]
                if "cagr_calculated" in key: possible_class_keys.append(key.replace("_calculated", "_classification"))
                found_class_val = None
                for class_key_attempt in possible_class_keys:
                    if class_key_attempt in data_to_render and data_to_render[class_key_attempt] != "__PROCESSED__":
                        found_class_val = data_to_render[class_key_attempt]
                        data_to_render[class_key_attempt] = "__PROCESSED__"; break 
                if found_class_val: classification_html = get_sentiment_html_enhanced(found_class_val)
                st.markdown(f"**{display_key}:** {classification_html}", unsafe_allow_html=True)
                st.markdown(f"<div class='explanation-box'>{value}</div>", unsafe_allow_html=True)
                continue
            
            if key.endswith("classification") and value == "__PROCESSED__": continue

            # Specific handling for structures like *_cagr_calculated
            if "_cagr_calculated" in key and isinstance(value, dict):
                st.markdown(f"<div class='subsection-header-ai'>{display_key}</div>", unsafe_allow_html=True)
                for k_cagr, v_cagr in value.items():
                    if k_cagr == "calculation_note" and v_cagr:
                         st.markdown(f"<div class='explanation-box'><em>Note:</em> {v_cagr}</div>", unsafe_allow_html=True)
                    elif k_cagr != "classification":
                         st.markdown(f"<div class='kpi-item'><strong>{str(k_cagr).replace('_',' ').title()}:</strong> {format_generic_value(v_cagr, k_cagr)}</div>", unsafe_allow_html=True)
                cagr_classification = value.get("classification")
                if cagr_classification:
                    st.markdown(f"<div class='kpi-item'>{get_sentiment_html_enhanced(cagr_classification, 'Overall Assessment: ')}</div>", unsafe_allow_html=True)
                continue

            if isinstance(value, (dict, list)):
                if value and not (key == "values" and isinstance(value, list)): 
                    st.markdown(f"<div class='subsection-header-ai'>{display_key}</div>", unsafe_allow_html=True)
                render_generic_section(value, new_path_keys)
            else:
                val_str = format_generic_value(value, key)
                st.markdown(f"<div class='kpi-item'><strong>{display_key}:</strong> {val_str}</div>", unsafe_allow_html=True)
        return
    
    st.markdown(f"{str(data_to_render)}")


# --- New Helper for plotting a group of related time series charts ---
def plot_financial_performance_charts_row(
    df_dict: Dict[str, pd.DataFrame], # Dict where key is metric name, value is DataFrame with 'period' and value col
    chart_titles: Dict[str, str],    # Dict mapping metric name to display chart title
    value_col_names: Dict[str, str], # Dict mapping metric name to its value column name (e.g., 'value', 'value_percent')
    is_percent_dict: Dict[str, bool] # Dict mapping metric name to boolean if it's a percentage
):
    active_charts_data = []
    for metric_name, df in df_dict.items():
        if df is not None and not df.empty and 'period' in df.columns and value_col_names.get(metric_name) in df.columns:
            if pd.api.types.is_numeric_dtype(df[value_col_names[metric_name]]):
                 active_charts_data.append({
                     "metric_name": metric_name,
                     "df": df,
                     "title": chart_titles.get(metric_name, metric_name.replace("_"," ").title()),
                     "value_col": value_col_names.get(metric_name),
                     "is_percent": is_percent_dict.get(metric_name, False)
                 })
    
    if not active_charts_data:
        # st.caption("No data for this chart group.") # Can be too noisy
        return

    num_charts_to_plot = len(active_charts_data)
    cols = st.columns(num_charts_to_plot)

    for i, chart_data in enumerate(active_charts_data):
        with cols[i]:
            df_plot = chart_data["df"].copy() # Work with a copy
            value_col = chart_data["value_col"]

            # Ensure periods are sorted (CRUCIAL for line charts to connect correctly)
            try:
                if all(isinstance(p, str) and p.isdigit() and len(p) == 4 for p in df_plot['period']):
                    df_plot['period_num'] = pd.to_numeric(df_plot['period'])
                    df_plot = df_plot.sort_values(by='period_num').drop(columns=['period_num'])
                # Add more sophisticated date parsing/sorting here if 'period' can be "YYYY-MM-DD", "Q1 2023" etc.
                # Example for "YYYY-MM-DD":
                # elif all(isinstance(p, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', p) for p in df_plot['period']):
                #    df_plot['period_dt'] = pd.to_datetime(df_plot['period'])
                #    df_plot = df_plot.sort_values(by='period_dt').drop(columns=['period_dt'])
            except Exception as e_sort:
                st.caption(f"Note: Could not sort periods for {chart_data['title']}. Chart might appear unsorted. Error: {e_sort}")


            chart_is_line = len(df_plot) >= 4 # Adjusted heuristic for line chart
            chart_kwargs = {"x": 'period', "y": value_col, "title": chart_data["title"]}

            if chart_is_line:
                chart_kwargs["markers"] = True
                fig = px.line(df_plot, **chart_kwargs)
            else:
                chart_kwargs["text_auto"] = True
                fig = px.bar(df_plot, **chart_kwargs)
            
            yaxis_title = value_col.replace("_"," ").title()
            yaxis_ticksuffix = "%" if chart_data["is_percent"] else None
            yaxis_tickformat = None if chart_data["is_percent"] else "$,.2s"

            fig.update_layout(title_x=0.5, title_font_size=14, 
                              yaxis_title=yaxis_title, 
                              yaxis_ticksuffix=yaxis_ticksuffix, yaxis_tickformat=yaxis_tickformat,
                              height=320, margin=dict(t=40,b=20,l=10,r=10), xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

# --- Main Display Function for the Dashboard Content ---
def display_ai_analysis_dashboard(analysis_data: Dict[str, Any], metadata: Dict[str, Any], company_symbol: str):
    load_global_css()
    # ... (Dashboard Overview Cards section remains the same) ...
    st.markdown("<div class='section-header-ai'>Dashboard Overview</div>", unsafe_allow_html=True)
    val_summary_data = analysis_data.get("valuation_analysis", {}).get("valuation_summary", {})
    current_price_val = metadata.get("last_price")
    card_metrics_config = [
        {"label": "Fair Value (Base Est.)", "value_path": ["valuation_analysis", "valuation_summary", "fair_value_estimate_base"], "type": "currency"},
        {"label": "Current Price", "value": current_price_val, "type": "currency"},
        {"label": "Recommendation", "value_path": ["investment_thesis_summary", "overall_recommendation"], "type": "text_strong"},
        {"label": "P/E (TTM)", "value_path": ["fundamental_analysis", "valuation_ratios", "pe_ratio_ttm"], "type": "number"},
        {"label": "Revenue CAGR (Hist.)", "value_path": ["growth_prospects", "historical_growth_summary", "revenue_cagr", "rate_percent"], "type": "percent"},
        {"label": "ROE (TTM)", "value_path": ["fundamental_analysis", "profitability_ratios", "roe_ttm"], "type": "percent"},
    ]
    active_cards_list = [] # Logic to populate active_cards_list remains same
    for card_conf in card_metrics_config:
        val = card_conf.get("value") 
        if val is None and "value_path" in card_conf: 
            temp_val = analysis_data
            try:
                for p_key in card_conf["value_path"]: temp_val = temp_val[p_key]
                val = temp_val
            except: val = None
        if val is not None and str(val).strip() != "" and str(val).lower() != 'n/a':
            active_cards_list.append({"label": card_conf["label"], "value": val, "type": card_conf["type"]})
    if active_cards_list: # Card display logic remains same
        cols = st.columns(len(active_cards_list))
        for i, card_info in enumerate(active_cards_list):
            with cols[i]:
                val_str = str(card_info["value"]) 
                if card_info["type"] == "currency" and isinstance(card_info["value"], (int, float)): val_str = f"${card_info['value']:,.2f}"
                elif card_info["type"] == "percent" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.1f}%"
                elif card_info["type"] == "number" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.2f}"
                elif card_info["type"] == "text_strong": val_str = f"<strong>{card_info['value']}</strong>"
                st.markdown(f"""<div class="metric-card-ai"><div class="stMetricLabel">{card_info['label']}</div><div class="stMetricValue">{val_str}</div></div>""", unsafe_allow_html=True)
    else: st.info("Key overview metrics not available.")


    # --- Financial Performance Section (Modified for grouped charts) ---
    fin_perf_data = analysis_data.get("financial_performance", {})
    if fin_perf_data:
        st.markdown(f"<div class='section-header-ai'>Financial Performance ({company_symbol})</div>", unsafe_allow_html=True)

        # --- Chart Group 1: Key Absolute Values ---
        st.markdown("<div class='subsection-header-ai'>Key Performance Indicators (Absolute Values)</div>", unsafe_allow_html=True)
        abs_metrics_df_dict = {}
        abs_metrics_titles = {}
        abs_metrics_value_cols = {}
        abs_metrics_is_percent = {}

        for metric_key, display_title in [
            ("revenues", "Revenue"), 
            ("profitability_metrics.grossProfit", "Gross Profit"), # Assuming grossProfit is nested like this, or adjust
            ("profitability_metrics.operating_income", "Operating Income"),
            ("profitability_metrics.net_income", "Net Income")
        ]:
            data_path = metric_key.split('.')
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data:
                    current_data = current_data[p_key]
                else:
                    valid_path = False; break
            
            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value' in df.columns: # Assuming primary value column is 'value'
                    abs_metrics_df_dict[metric_key] = df
                    abs_metrics_titles[metric_key] = display_title
                    abs_metrics_value_cols[metric_key] = 'value'
                    abs_metrics_is_percent[metric_key] = False
        
        plot_financial_performance_charts_row(abs_metrics_df_dict, abs_metrics_titles, abs_metrics_value_cols, abs_metrics_is_percent)
        
        # Display explanations for this group if available
        for metric_key_orig, _ in [("revenues", ""), ("profitability_metrics.grossProfit", ""), ("profitability_metrics.operating_income", ""), ("profitability_metrics.net_income", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data
            valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)


        # --- Chart Group 2: Key Margins ---
        st.markdown("<div class='subsection-header-ai'>Key Profitability Margins (%)</div>", unsafe_allow_html=True)
        margin_metrics_df_dict = {}
        margin_metrics_titles = {}
        margin_metrics_value_cols = {}
        margin_metrics_is_percent = {}

        for metric_key, display_title in [
            ("gross_margin", "Gross Margin"),
            ("ebitda_margin", "EBITDA Margin"),
            ("net_margin", "Net Margin"),
            ("profitability_metrics.roic", "ROIC") # Example if ROIC is here and has "values"
        ]:
            data_path = metric_key.split('.') # Handles potential nesting like profitability_metrics.roic
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data: current_data = current_data[p_key]
                else: valid_path = False; break

            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value_percent' in df.columns: # Margins usually have 'value_percent'
                    margin_metrics_df_dict[metric_key] = df
                    margin_metrics_titles[metric_key] = display_title
                    margin_metrics_value_cols[metric_key] = 'value_percent'
                    margin_metrics_is_percent[metric_key] = True
        
        plot_financial_performance_charts_row(margin_metrics_df_dict, margin_metrics_titles, margin_metrics_value_cols, margin_metrics_is_percent)

        # Display explanations for margins
        for metric_key_orig, _ in [("gross_margin", ""), ("ebitda_margin", ""), ("net_margin", ""), ("profitability_metrics.roic", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data; valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)

        # --- You can add more chart groups here for other financial performance aspects ---
        # e.g., Growth Rates (Revenue Growth, EBITDA Growth, EPS Growth)
        # e.g., Cash Flow metrics (Operating Cash Flow, Free Cash Flow)

        # Fallback to render_generic_section for any remaining parts of financial_performance
        # Be careful not to re-render what's already charted above.
        # This requires a more complex state or passing already processed keys.
        # For now, this example focuses on specific chart groups.
        # You might need to create a list of keys already handled and skip them in render_generic_section.

    # --- Render other main analysis sections using the generic renderer ---
    sections_in_order_config = [
        # ("financial_performance", "Financial Performance"), # Handled specially above
        ("fundamental_analysis", "Fundamental Ratios (TTM)"),
        ("valuation_analysis", "Valuation Analysis"),
        ("growth_prospects", "Growth Prospects & Outlook"),
        ("competitive_position", "Competitive Landscape"),
        ("peers_comparison", "Peer Group Benchmarking"),
        ("revenue_segmentation", "Revenue Segmentation Insights"),
        ("risk_factors", "Key Risk Factors"),
        ("scenario_analysis", "Scenario Analysis (Valuation)"),
        ("shareholder_returns_analysis", "Shareholder Returns & Capital Allocation"),
        ("investment_thesis_summary", "Investment Thesis & Recommendation"),
    ]

    for section_key, section_display_title in sections_in_order_config:
        section_data_content = analysis_data.get(section_key)
        if section_data_content:
            st.markdown(f"<div class='section-header-ai'>{section_display_title}</div>", unsafe_allow_html=True)
            with st.container():
                render_generic_section(section_data_content, [section_key])
    
    st.markdown("---")
    if st.checkbox("Show Full Raw AI Analysis JSON", value=False, key="show_raw_ai_json_cb_v4"):
        st.json(analysis_data)
        
        
__all__ = [
    "get_neo4j_driver",
    "format_value",
    "calculate_delta",
    "_arrow",
    "R_display_metric_card",
    "fetch_income_statement_data",
    "fetch_financial_details_for_companies"
]
