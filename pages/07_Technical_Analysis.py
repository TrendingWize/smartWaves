# smart_waves/pages/07_Technical_Analysis.py

import streamlit as st
st.set_page_config(page_title="Technical Analysis", layout="wide")

import json, textwrap, datetime as dt, re
from pathlib import Path
from base64 import b64encode

import requests, pandas as pd, pandas_ta as ta
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
from PIL import Image
import google.generativeai as genai
import openai

# ── App config ──────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cairo&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── secrets / constants ─────────────────────────────────────────────────────
FMP_API_KEY     = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY  = st.secrets.get("GOOGLE_API_KEY")
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")
OUT_DIR         = Path("charts"); OUT_DIR.mkdir(exist_ok=True)

openai.api_key = OPENAI_API_KEY

# ── data helpers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(tkr:str, d_from:str, d_to:str) -> pd.DataFrame:
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{tkr}?apikey={FMP_API_KEY}&from={d_from}&to={d_to}")
    data = requests.get(url, timeout=30).json()
    rows = data.get("historical", []) if isinstance(data, dict) else data
    if not rows: raise ValueError("No price data")
    df = pd.DataFrame(rows).rename(columns=str.lower)
    if "price" in df and "close" not in df: df["close"] = df["price"]
    for c in ("open", "high", "low", "close"): df[c] = df.get(c, df["close"])
    df["volume"] = df.get("volume", 0)
    return (df.assign(date=lambda d: pd.to_datetime(d["date"]))
              .set_index("date").sort_index()
              [["open", "high", "low", "close", "volume"]])

def resample(df, frame):
    if frame == "Daily": return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()

def add_indicators(df):
    df["sma20"]  = ta.sma(df["close"], 20)
    df["sma50"]  = ta.sma(df["close"], 50)
    df["sma100"] = ta.sma(df["close"], 100)
    df = pd.concat([df, ta.macd(df["close"])], axis=1)
    df["rsi"] = ta.rsi(df["close"], 14)
    return df

def save_composite_chart(df, tkr, frame):
    fig, ax = plt.subplots(figsize=(12, 6)); ax.set_yscale("log")
    ax.plot(df.index, df["close"], label="Close")
    for sma, l in [("sma20", "SMA20"), ("sma50", "SMA50"), ("sma100", "SMA100")]:
        ax.plot(df.index, df[sma], label=l)
    ax.legend(); ax.set_title(f"{tkr} ({frame})")
    path = OUT_DIR / f"{tkr}_{frame}_cmp.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)

def ask_gemini(img, prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    return model.generate_content([prompt, Image.open(img)]).text.strip()

def ask_gpt(img, prompt):
    image_data = b64encode(open(img, "rb").read()).decode()
    return openai.ChatCompletion.create(
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

def tv_chart(sym, interval, theme, height, width, autosize):
    props = {
        "symbol": sym, "interval": interval, "theme": theme, "style": "1",
        "locale": "en", "timezone": "Etc/UTC", "allow_symbol_change": True,
        "support_host": "https://www.tradingview.com", "height": height
    }
    if autosize: props["autosize"] = True
    else: props["width"] = width or 800
    outer = f"height:{height}px;" + (f'width:{width or 800}px;' if not autosize else "")
    html(f"""
    <div class="tradingview-widget-container" style="{outer}">
      <div id="tv_widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
              async type="text/javascript">{json.dumps(props, separators=(',',':'))}</script>
    </div>
    """, height=height, width=None if autosize else width or 800, scrolling=False)

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Interactive & AI Insights")

# language selector
c_language = st.selectbox("Language", ["Arabic", "English"])

# layout
c_sym, c_from, c_to, c_frm, c_int = st.columns([3, 2, 2, 2, 2])
tv_symbol = c_sym.text_input("TradingView Symbol", "NASDAQ:AAPL").upper().strip()
today = dt.date.today()
date_from = c_from.date_input("From", today - dt.timedelta(days=365))
date_to   = c_to.date_input("To", today)
frame     = c_frm.selectbox("Indicator frame", ["Daily", "Weekly", "Monthly"])
tv_int    = c_int.selectbox("TV interval", ["1", "15", "30", "60", "D", "W", "M"], 4)

height = st.slider("Chart height (px)", 400, 1000, 750)
col_auto, col_theme, col_btn = st.columns([1, 3, 1])
autosz  = col_auto.checkbox("Autosize width", True)
theme   = col_theme.radio("Theme", ["auto", "light", "dark"], horizontal=True)
run_btn = col_btn.button("💡 Generate AI Analysis")

# tradingview chart
chart_theme = "dark" if theme == "auto" and st.get_option("theme.base") == "dark" else (theme if theme != "auto" else "light")
tv_chart(tv_symbol, tv_int, chart_theme, height, None if autosz else 800, autosz)

# AI Analysis
if run_btn:
    if date_from >= date_to:
        st.error("From-date must precede To-date")
        st.stop()

    if not (FMP_API_KEY and GOOGLE_API_KEY and OPENAI_API_KEY):
        st.info("Add FMP_API_KEY, GOOGLE_API_KEY, and OPENAI_API_KEY to secrets")
        st.stop()

    tkr = re.split(r"[:/]", tv_symbol)[-1]
    try:
        with st.spinner("Fetching price data …"):
            raw = get_ohlcv(tkr, date_from.isoformat(), date_to.isoformat())
            df = add_indicators(resample(raw, frame))
        cmp = save_composite_chart(df, tkr, frame)

        prompt = textwrap.dedent(f"""
        **Role:** Expert Market Technician specializing in pure price action analysis. You master Elliott Wave theory, Wyckoff methodology, 
        and classical chart patterns. All analysis must derive exclusively from **price structure, volume, and market geometry**. Response in {c_language} language.

        **Absolute Rules:**  
        ✅ Prioritize: Candlestick patterns, volume-profile, swing structure, and institutional accumulation/distribution signs.

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

        tabs = st.tabs(["Gemini", "GPT"])
        with tabs[0]:
            with st.spinner("Gemini is thinking …"):
                gemini_response = ask_gemini(cmp, prompt)
                if c_language.lower() == "arabic":
                    st.markdown(
                        f"""<div dir="rtl" style="text-align: right; font-family: 'Cairo', sans-serif;">{gemini_response}</div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(gemini_response)

        with tabs[1]:
            with st.spinner("GPT is thinking …"):
                gpt_response = ask_gpt(cmp, prompt)
                if c_language.lower() == "arabic":
                    st.markdown(
                        f"""<div dir="rtl" style="text-align: right; font-family: 'Cairo', sans-serif;">{gpt_response}</div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(gpt_response)

    except Exception as e:
        st.error(f"Error: {e}")
