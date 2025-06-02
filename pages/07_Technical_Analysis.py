# smart_waves/pages/07_Technical_Analysis.py
import json, textwrap, datetime as dt
from pathlib import Path

import requests, pandas as pd, pandas_ta as ta
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.components.v1 import html
from PIL import Image
import google.generativeai as genai

# ░░ Streamlit config ░░
st.set_page_config(page_title="Technical Analysis", layout="wide")

# ░░ Secrets / constants ░░
FMP_API_KEY    = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
OUT_DIR        = Path("charts"); OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download price history
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{ticker}?apikey={FMP_API_KEY}&from={date_from}&to={date_to}")
    data = requests.get(url, timeout=30).json()

    rows = data.get("historical", []) if isinstance(data, dict) else data
    if not rows:
        raise ValueError("No price data returned")

    df = pd.DataFrame(rows).rename(columns=str.lower)
    if "price" in df and "close" not in df:
        df["close"] = df["price"]

    for col in ("open", "high", "low", "close"):
        df[col] = df.get(col, df["close"])
    df["volume"] = df.get("volume", 0)

    return (
        df.assign(date=lambda d: pd.to_datetime(d["date"]))
          .set_index("date")
          .sort_index()
          [["open","high","low","close","volume"]]
    )

def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily":
        return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    agg  = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(rule).agg(agg).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicators & composite (Gemini)
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma20"]  = ta.sma(df["close"], 20)
    df["sma50"]  = ta.sma(df["close"], 50)
    df["sma100"] = ta.sma(df["close"], 100)
    df = pd.concat([df, ta.macd(df["close"])], axis=1)
    df["rsi"]    = ta.rsi(df["close"], 14)
    return df

def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yscale("log")
    ax.plot(df.index, df["close"], label="Close")
    ax.plot(df.index, df["sma20"],  label="SMA20")
    ax.plot(df.index, df["sma50"],  label="SMA50")
    ax.plot(df.index, df["sma100"], label="SMA100")
    ax.legend(); ax.set_title(f"{ticker} ({frame})")
    path = OUT_DIR / f"{ticker}_{frame}_composite.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Gemini summary
# ─────────────────────────────────────────────────────────────────────────────
def ask_gemini(img_path: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    return model.generate_content([prompt, Image.open(img_path)]).text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 4. TradingView Advanced-Chart embed
# ─────────────────────────────────────────────────────────────────────────────
def tradingview_chart(symbol: str,
                      interval: str = "D",
                      theme: str = "light",
                      height: int = 550,
                      width: int | None = None,
                      autosize: bool = True):
    """Embed TradingView Advanced Chart.  
       • autosize=True  → fills parent width (Streamlit column)  
       • autosize=False → fixed width passed via `width` arg."""
    props = {
        "autosize": autosize,
        "symbol": symbol,
        "interval": interval,
        "theme": theme,
        "style": "1",
        "locale": "en",
        "timezone": "Etc/UTC",
        "allow_symbol_change": True,
        "support_host": "https://www.tradingview.com",
    }

    outer_style = f"height:{height}px;"
    if not autosize and width:
        outer_style += f"width:{width}px;"

    html_code = f"""
    <div class="tradingview-widget-container" style="{outer_style}">
      <div id="tv_widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
              async type="text/javascript">
      {json.dumps(props, separators=(",", ":"))}
      </script>
    </div>
    """
    html(html_code, height=height, width=width if not autosize else None, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Interactive & AI Insights")

symbol = st.text_input("Ticker", value="AAPL").upper().strip()
today = dt.date.today(); default_from = today - dt.timedelta(days=365)
c1, c2 = st.columns(2)
date_from = c1.date_input("From", default_from)
date_to   = c2.date_input("To",   today)

frame   = st.selectbox("Indicator frame", ["Daily","Weekly","Monthly"])
tv_int  = st.selectbox("TV interval", ["1","15","30","60","D","W","M"], index=4)
theme   = st.radio("Theme", ["auto","light","dark"], horizontal=True)
height  = st.slider("Chart height (px)", 400, 1000, 600)
autosz  = st.checkbox("Autosize width", value=True)
run_btn = st.button("🚀 Generate")

if run_btn:
    if date_from >= date_to:
        st.error("From-date must precede To-date"); st.stop()
    if not (FMP_API_KEY and GOOGLE_API_KEY):
        st.info("Add FMP_API_KEY & GOOGLE_API_KEY to secrets"); st.stop()

    if theme == "auto":
        theme = "dark" if st.get_option("theme.base") == "dark" else "light"

    with st.spinner("Fetching data …"):
        raw = get_ohlcv(symbol, date_from.isoformat(), date_to.isoformat())
        df  = add_indicators(resample(raw, frame))

    st.subheader("🔹 Interactive Chart")
    tradingview_chart(exchange = st.selectbox("Exchange", ["NASDAQ", "NYSE", "AMEX"]),
    symbol = f"{exchange}:{symbol}",
                      interval=tv_int,
                      theme=theme,
                      height=height,
                      autosize=autosz,
                      width=None if autosz else 800)   # change 800 if you want different fixed width

    # Gemini commentary
    comp = save_composite_chart(df, symbol, frame)
    prompt = textwrap.dedent(f"""
        You are a professional market technician.
        Analyse this {frame.lower()} composite chart of {symbol} (log close, SMAs, volume, MACD, RSI).
        Discuss trend, momentum, support/resistance. Give price targets for +30, +60, +252 trading days (bullish/base/bearish).
    """)
    with st.spinner("Gemini is thinking …"):
        st.subheader("🧠 Gemini Commentary")
        st.markdown(ask_gemini(comp, prompt))
