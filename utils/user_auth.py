# Utility functions (e.g., auth, DB, hashing)
# smart_waves/utils.py
import streamlit as st
import sqlite3
import hashlib
import uuid # For session tokens
from typing import Dict, List, Tuple, Any, Optional
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
