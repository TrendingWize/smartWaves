# components/company_profile.py

import streamlit as st
import pandas as pd # Though we might not need pandas much for this display
from utils.utils_helpers  import get_neo4j_driver 

@st.cache_data(ttl="1h") # Cache the profile data
def fetch_company_profile_data(_driver, symbol: str):
    if not _driver:
        return None
    
    query = """
    MATCH (c:Company)
    WHERE c.symbol = $sym
    RETURN c
    """
    # In a real scenario, you might return c.propertyName AS propertyName for all desired fields
    # For this example, we'll assume the node 'c' contains all the keys directly as properties.
    # If not, you'll need to adjust the RETURN statement in the query.
    # Example: RETURN c.symbol as symbol, c.companyName as companyName, ...

    try:
        with _driver.session(database="neo4j") as session:
            result = session.run(query, sym=symbol)
            record = result.single() # Expecting only one company for a given symbol
            if record and record["c"]:
                # Convert Neo4j Node to dictionary
                # The properties of a Node object are accessed directly
                profile_data = dict(record["c"].items())
                return profile_data
            else:
                return None
    except Exception as e:
        st.error(f"Error fetching company profile for {symbol}: {e}")
        return None

def display_company_profile(profile_data):
    """Displays the company profile data in a structured way."""
    if not profile_data:
        st.warning("Company profile data not available.")
        return

    # --- Header Section ---
    col1, col2 = st.columns([1, 4])
    with col1:
        if profile_data.get("image"):
            st.image(profile_data["image"], width=150)
    with col2:
        st.title(profile_data.get("companyName", "N/A"))
        st.caption(f"{profile_data.get('exchangeFullName', '')} ({profile_data.get('exchange', '')}): {profile_data.get('symbol', '')} | {profile_data.get('sector', '')} | {profile_data.get('industry', '')}")
        if profile_data.get("website"):
            st.markdown(f"🌐 [{profile_data['website']}]({profile_data['website']})")
    
    st.markdown("---")

    # --- Key Financials & Stock Info ---
    st.subheader("📈 Stock & Financial Highlights")
    cols_stock = st.columns(4)
    
    def display_profile_metric(column, label, value, currency="", help_text=None):
        val_str = "N/A"
        if value is not None:
            if isinstance(value, (int, float)):
                if abs(value) >= 1_000_000_000_000: val_str = f"{value / 1_000_000_000_000:.2f}T"
                elif abs(value) >= 1_000_000_000: val_str = f"{value / 1_000_000_000:.2f}B"
                elif abs(value) >= 1_000_000: val_str = f"{value / 1_000_000:.2f}M"
                elif abs(value) >= 1_000: val_str = f"{value / 1_000:.2f}K"
                else: val_str = f"{value:,.2f}" # Default to 2 decimal places for price/change like values
            else:
                val_str = str(value)
        
        if currency and value is not None:
            val_str = f"{currency} {val_str}"
            
        column.metric(label=label, value=val_str, help=help_text)

    with cols_stock[0]:
        display_profile_metric(st, "Price", profile_data.get("price"), currency=profile_data.get("currency",""))
        display_profile_metric(st, "Change", profile_data.get("change"), currency=profile_data.get("currency",""))
        if profile_data.get("changePercentage") is not None:
             st.markdown(f"**% Change:** {profile_data['changePercentage']:+.2f}%")


    with cols_stock[1]:
        display_profile_metric(st, "Market Cap", profile_data.get("marketCap"), currency=profile_data.get("currency",""))
        display_profile_metric(st, "Beta", profile_data.get("beta"))
        
    with cols_stock[2]:
        display_profile_metric(st, "Volume", profile_data.get("volume"))
        display_profile_metric(st, "Avg Volume", profile_data.get("averageVolume"))

    with cols_stock[3]:
        display_profile_metric(st, "52 Week Range", profile_data.get("range"))
        display_profile_metric(st, "Last Dividend", profile_data.get("lastDividend"), currency=profile_data.get("currency",""))

    st.markdown("---")
    
    # --- Company Overview ---
    st.subheader("🏢 Company Overview")
    if profile_data.get("description"):
        st.markdown(f"**Description:** {profile_data['description']}")
        st.markdown("") # Spacer

    cols_overview = st.columns(2)
    with cols_overview[0]:
        st.markdown(f"**CEO:** {profile_data.get('ceo', 'N/A')}")
        st.markdown(f"**Country:** {profile_data.get('country', 'N/A')}")
        st.markdown(f"**Full Time Employees:** {profile_data.get('fullTimeEmployees', 'N/A')}")
        st.markdown(f"**IPO Date:** {profile_data.get('ipoDate', 'N/A')}")
    with cols_overview[1]:
        st.markdown(f"**Address:** {profile_data.get('address', 'N/A')}, {profile_data.get('city', 'N/A')}, {profile_data.get('state', 'N/A')} {profile_data.get('zip', 'N/A')}")
        st.markdown(f"**Phone:** {profile_data.get('phone', 'N/A')}")
        st.markdown(f"**CIK:** {profile_data.get('cik', 'N/A')}")
        st.markdown(f"**ISIN:** {profile_data.get('isin', 'N/A')}")
        st.markdown(f"**CUSIP:** {profile_data.get('cusip', 'N/A')}")

    st.markdown("---")

    # --- Trading Information ---
    st.subheader("ℹ️ Trading Information")
    cols_trading = st.columns(3)
    with cols_trading[0]:
        st.markdown(f"**Actively Trading:** {'Yes' if profile_data.get('isActivelyTrading') else 'No'}")
    with cols_trading[1]:
        st.markdown(f"**Is ETF:** {'Yes' if profile_data.get('isEtf') else 'No'}")
    with cols_trading[2]:
        st.markdown(f"**Is Fund:** {'Yes' if profile_data.get('isFund') else 'No'}")
        # st.markdown(f"**Is ADR:** {'Yes' if profile_data.get('isAdr') else 'No'}") # Uncomment if needed

def company_profile_tab_content(symbol_from_main_input):
    """Main function to drive the company profile tab."""
    if not symbol_from_main_input:
        st.info("Please enter a company symbol in the main search bar.")
        return

    # st.header(f"Company Profile: {symbol_from_main_input}") # Title is now part of display_company_profile
    
    neo_driver = get_neo4j_driver()
    if not neo_driver:
        return # Error usually shown by get_neo4j_driver

    profile_data = fetch_company_profile_data(neo_driver, symbol_from_main_input)
    
    if profile_data:
        display_company_profile(profile_data)
    else:
        st.warning(f"No profile data found for {symbol_from_main_input}.")
