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

# ── Data Fetch ─────────────────────────────────────────────────────────────
def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily":
        return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    return df.resample(rule).agg(agg).dropna()

# ── Data Fetch ─────────────────────────────────────────────────────────────(ttl=3_600)
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
    # ── Technical Indicators ──
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

    # ── Log scale decision ──
    min_price = df["close"].replace(0, pd.NA).min()
    max_price = df["close"].max()
    price_range_ratio = max_price / min_price if min_price and min_price > 0 else 1
    use_log = price_range_ratio > 10 and min_price > 1


    # ── Plotly Subplots ──
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("Price & SMAs", "Volume", "MACD", "RSI"))

    # ── Chart Type ──
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick',
                                     increasing_line_color='limegreen',
                                     decreasing_line_color='red'), row=1, col=1)
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode='lines',
                                 line=dict(color="limegreen"), name="Close Price"), row=1, col=1)
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=df.index, y=df["close"], marker_color="darkcyan", name="Close Price"), row=1, col=1)

    # ── Overlays & Indicators ──
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

    # ── Layout ──
    fig.update_layout(
        height=900,
        showlegend=True,
        title=f"{tkr} ({frame}) – LLM-Optimized Technical Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    # Apply log or linear Y-axis to the price chart (top subplot)
    fig.update_yaxes(type="log" if use_log else "linear", row=1, col=1)

    if use_log:
        fig.update_yaxes(
            type="log",
            row=1, col=1,
            tickformat=".2f",          # Shows two decimal places
            dtick=0.10000              # Log scale step: 10^0.30103 ≈ 2
        )



    return fig


def ask_gpt(img, prompt):
    image_data = b64encode(open(img, "rb").read()).decode()
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a technical analysis expert."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]}
        ],
        temperature=0.7
    ).choices[0].message.content


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – LLM Chart")

c_sym, c_from, c_to, c_frame, c_type = st.columns([2, 2, 2, 2, 2])
ticker_symbol = c_sym.text_input("Ticker Symbol", "AAPL").upper().strip()

# List of start/end options (dates)
start_options = [dt.date.today() - relativedelta(months=i) for i in range(1, 501)]
end_options = [dt.date.today() - relativedelta(months=i) for i in range(0, 500)]

# Dropdowns
start_date = c_from.selectbox("Start Date", start_options)
end_date = c_to.selectbox("End Date", end_options)

frame = c_frame.selectbox("Indicator Frame", ["Daily", "Weekly", "Monthly"])
chart_type = c_type.selectbox("Chart Type", ["candlestick", "line", "bar"])

run_btn = col_btn.button(" Generate AI Analysis")

if run_btn:
    if date_from >= date_to:
        st.error("From-date must precede To-date")
        st.stop()

    if not (FMP_API_KEY and GOOGLE_API_KEY and OPENAI_API_KEY):
        st.info("Add FMP_API_KEY, GOOGLE_API_KEY, and OPENAI_API_KEY to secrets")
        st.stop()

    try:
        with st.spinner("Fetching price data …"):
            raw = get_ohlcv(tkr, date_from.isoformat(), date_to.isoformat())
            df = add_indicators(resample(raw, frame))

        # Save + show chart
        cmp = save_composite_chart_plotly(df, tkr, frame)
        st.subheader("🖼️ Composite Price Chart (Python-generated)")
        st.image(cmp, caption=f"{tkr} – {frame} Composite Chart", use_column_width=True)

        # Prompt
        prompt = textwrap.dedent(f"""
        **Role:** Expert Market Technician specializing in pure price action analysis. You master Elliott Wave theory, Wyckoff methodology, 
        and classical chart patterns. All analysis must derive exclusively from **price structure, volume, and market geometry**. Response in {c_language} language.

        **Absolute Rules:**  
        ✅ Prioritize: Candlestick patterns, volume-profile, swing structure, institutional accumulation/distribution signs and chart indicators.

        #### 1. **Market Structure Framework**
        - **Trend Identification:** swing highs/lows, Wyckoff phase  
        - **Critical Levels:** 3 support/resistance zones, major pivots

        #### 2. **Pattern Recognition**
        - **Classical Patterns:** e.g., H&S, Triangles  
        - **Elliott Wave Count:** likely wave, alt count %  
        - **Candlestick Signals:** clusters of reversals

        #### 3. **Projections & Risk Zones**
        - **Price Targets:** +30d/+60d/+252d  
        - **Failure Scenarios:** invalidation levels, stop zones

        #### 4. **Synthesis**
        - **Bias:** Bullish/Base/Bearish (1–5 scale)  
        - **Entry Triggers:** breakout + volume  
        - **Timeframe Alignment:** conflicts if any
        """)

        # Tabs
        tabs = st.tabs(["Gemini", "GPT"])
        with tabs[0]:
            with st.spinner("Gemini is thinking …"):
                gemini_response = ask_gemini(cmp, prompt)
                if c_language.lower() == "arabic":
                    st.markdown(
                        f"""<div dir=\"rtl\" style=\"text-align: right; font-family: 'Cairo', sans-serif;\">{gemini_response}</div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(gemini_response)

        with tabs[1]:
            with st.spinner("GPT is thinking …"):
                gpt_response = ask_gpt(cmp, prompt)
                if c_language.lower() == "arabic":
                    st.markdown(
                        f"""<div dir=\"rtl\" style=\"text-align: right; font-family: 'Cairo', sans-serif;\">{gpt_response}</div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(gpt_response)

    except Exception as e:
        st.error(f"Error: {e}")

