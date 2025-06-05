import streamlit as st
st.set_page_config(page_title="Technical Analysis", layout="wide")

import json, textwrap, datetime as dt, re
from pathlib import Path
from base64 import b64encode

import requests, pandas as pd
from PIL import Image
import google.generativeai as genai
import openai
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ── App config ──────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cairo&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── secrets / constants ─────────────────────────────────────────────────────
FMP_API_KEY     = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY  = st.secrets.get("GOOGLE_API_KEY")
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# ── Date Helper ─────────────────────────────────────────────────────────────
def get_start_date_months_ago(end_date: dt.date, months_back: int) -> str:
    return (end_date - relativedelta(months=months_back)).isoformat()

# ── Data Fetch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(tkr: str, d_from: str, d_to: str) -> pd.DataFrame:
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{tkr}?apikey={FMP_API_KEY}&from={d_from}&to={d_to}")
    data = requests.get(url, timeout=30).json()
    rows = data.get("historical", []) if isinstance(data, dict) else data
    if not rows: raise ValueError("No price data")
    df = pd.DataFrame(rows).rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[["open", "high", "low", "close", "volume"]].sort_index()

# ── Chart Generator ─────────────────────────────────────────────────────────
def save_composite_chart_plotly(df, tkr, frame, chart_type="candlestick"):
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["sma50"] = df["close"].rolling(window=50).mean()
    df["sma100"] = df["close"].rolling(window=100).mean()

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    min_price = df["close"].replace(0, pd.NA).min()
    max_price = df["close"].max()
    price_range_ratio = max_price / min_price if min_price and min_price > 0 else 1
    use_log = price_range_ratio > 10 and min_price > 1
    if use_log:
        df["close"] = df["close"].clip(lower=1.01)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("Price & SMAs", "Volume", "MACD", "RSI"))

    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick',
                                     increasing_line_color='limegreen', decreasing_line_color='red'), row=1, col=1)
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode='lines',
                                 line=dict(color="limegreen"), name="Close Price"), row=1, col=1)
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=df.index, y=df["close"], marker_color="darkcyan", name="Close Price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], name="SMA20", line=dict(color="gold")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma100"], name="SMA100", line=dict(color="blue")), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="lightgray"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="royalblue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="orangered")), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram", marker_color="darkgray"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="seagreen")), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="crimson", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="dodgerblue", row=4, col=1)

    fig.update_layout(
        height=900, showlegend=True,
        title=f"{tkr} ({frame}) – LLM-Optimized Technical Chart",
        xaxis_rangeslider_visible=False, template="plotly_white"
    )

    fig.update_yaxes(type="log" if use_log else "linear", row=1, col=1)
    if use_log:
        fig.update_yaxes(type="log", row=1, col=1, tickformat=".2f", dtick=0.30103)

    if use_log:
        fig.update_yaxes(
        type="log",
        row=1, col=1,
        tickformat=".2f",          # Shows two decimal places
        dtick=0.30103              # Log scale step: 10^0.30103 ≈ 2
    )

    if use_log:
        fig.update_yaxes(
        type="log",
        row=1, col=1,
        tickformat=".2f",          # Shows two decimal places
        dtick=0.30103              # Log scale step: 10^0.30103 ≈ 2
    )

    return fig

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – LLM Chart")

c_sym, c_from, c_to, c_frame, c_type = st.columns([2, 2, 2, 2, 2])
ticker_symbol = c_sym.text_input("Ticker Symbol", "AAPL").upper().strip()
start_date = c_from.selectbox("Start Date", [dt.date.today() - relativedelta(months=i) for i in range(1, 501)] - dt.timedelta(days=180))
end_date = c_to.selectbox("End Date", [dt.date.today() - relativedelta(months=i) for i in range(0, 500)]) - relativedelta(months=i) for i in range(0, 500)))
frame = c_frame.selectbox("Indicator Frame", ["Daily", "Weekly", "Monthly"])
chart_type = c_type.selectbox("Chart Type", ["candlestick", "line", "bar"])

if st.button("🔍 Generate Chart"):
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            with st.spinner("Fetching data and plotting chart …"):
                df = get_ohlcv(ticker_symbol, start_date.isoformat(), end_date.isoformat())
                fig = save_composite_chart_plotly(df, ticker_symbol, frame, chart_type)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
