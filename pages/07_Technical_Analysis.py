# smart_waves/pages/07_Technical_Analysis.py
import json, textwrap, datetime as dt, re
from pathlib import Path

import requests, pandas as pd, pandas_ta as ta
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.components.v1 import html
from PIL import Image
import google.generativeai as genai

st.set_page_config(page_title="Technical Analysis", layout="wide")

# Secrets / constants
FMP_API_KEY    = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
OUT_DIR        = Path("charts"); OUT_DIR.mkdir(exist_ok=True)

# ─── 1. DATA HELPERS ────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(ticker: str, d_from: str, d_to: str) -> pd.DataFrame:
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{ticker}?apikey={FMP_API_KEY}&from={d_from}&to={d_to}")
    data = requests.get(url, timeout=30).json()
    rows = data.get("historical", []) if isinstance(data, dict) else data
    if not rows: raise ValueError("No price data returned")

    df = pd.DataFrame(rows).rename(columns=str.lower)
    if "price" in df and "close" not in df: df["close"] = df["price"]
    for c in ("open","high","low","close"): df[c] = df.get(c, df["close"])
    df["volume"] = df.get("volume", 0)

    return (df.assign(date=lambda d: pd.to_datetime(d["date"]))
              .set_index("date").sort_index()
              [["open","high","low","close","volume"]])

def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily": return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    agg  = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(rule).agg(agg).dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma20"]  = ta.sma(df["close"], 20)
    df["sma50"]  = ta.sma(df["close"], 50)
    df["sma100"] = ta.sma(df["close"], 100)
    df = pd.concat([df, ta.macd(df["close"])], axis=1)
    df["rsi"]    = ta.rsi(df["close"], 14)
    return df

def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yscale("log"); ax.plot(df.index, df["close"], label="Close")
    for sma, lbl in [("sma20","SMA20"), ("sma50","SMA50"), ("sma100","SMA100")]:
        ax.plot(df.index, df[sma], label=lbl)
    ax.legend(); ax.set_title(f"{ticker} ({frame})")
    path = OUT_DIR / f"{ticker}_{frame}_composite.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)

# ─── 2. GEMINI ──────────────────────────────────────────────────────────────
def ask_gemini(img_path: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    return model.generate_content([prompt, Image.open(img_path)]).text.strip()

# ─── 3. TRADINGVIEW WIDGET ──────────────────────────────────────────────────
def tradingview_chart(symbol: str, interval: str, theme: str,
                      height: int, width: int | None, autosize: bool):
    props = {
        "symbol": symbol,
        "interval": interval,
        "theme": theme,
        "style": "1",
        "locale": "en",
        "timezone": "Etc/UTC",
        "allow_symbol_change": True,
        "support_host": "https://www.tradingview.com",
        "height": height,
    }
    if autosize: props["autosize"] = True
    else:        props["width"]    = width or 800

    outer_style = f"height:{height}px;" + (f"width:{(width or 800)}px;" if not autosize else "")
    html_code = f"""
    <div class="tradingview-widget-container" style="{outer_style}">
      <div id="tv_widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
              async type="text/javascript">{json.dumps(props, separators=(",", ":"))}</script>
    </div>
    """
    html(html_code, height=height, width=None if autosize else (width or 800), scrolling=False)

# ─── 4. UI ──────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Interactive & AI Insights")

col_sym, col_from, col_to, col_frm, col_int = st.columns([3,2,2,2,2])
tv_symbol = col_sym.text_input("TradingView Symbol", "NASDAQ:AAPL").upper().strip()
today = dt.date.today()
date_from = col_from.date_input("From", today - dt.timedelta(days=365))
date_to   = col_to.date_input("To",   today)
frame   = col_frm.selectbox("Indicator frame", ["Daily","Weekly","Monthly"])
tv_int  = col_int.selectbox("TV interval", ["1","15","30","60","D","W","M"], 4)

height  = st.slider("Chart height (px)", 400, 1000, 750)
autosz  = st.checkbox("Autosize width", True)
theme   = st.radio("Theme", ["auto","light","dark"], horizontal=True)
run_btn = st.button("💡 Generate AI Analysis")

# show chart right away
st.subheader("🔹 Interactive Chart")
chart_theme = "dark" if theme=="auto" and st.get_option("theme.base")=="dark" else (theme if theme!="auto" else "light")
tradingview_chart(tv_symbol, tv_int, chart_theme, height, None if autosz else 800, autosz)

# --- analysis only when user clicks ---------------------------------------
if run_btn:
    if date_from >= date_to:
        st.error("From-date must precede To-date"); st.stop()
    if not (FMP_API_KEY and GOOGLE_API_KEY):
        st.info("Add FMP_API_KEY & GOOGLE_API_KEY to secrets"); st.stop()

    plain_tkr = re.split(r"[:/]", tv_symbol)[-1]  # last part after ':' or '/'

    try:
        with st.spinner("Fetching price data …"):
            raw = get_ohlcv(plain_tkr, date_from.isoformat(), date_to.isoformat())
            df  = add_indicators(resample(raw, frame))

        comp = save_composite_chart(df, plain_tkr, frame)
        prompt = textwrap.dedent(f"""
            You are a professional market technician.
            Analyse this {frame.lower()} composite chart of {plain_tkr}
            (log close, SMAs, volume, MACD, RSI). Provide trend, support/resistance,
            and price targets for +30, +60, +252 trading days (bullish / base / bearish).
        """)
        with st.spinner("Gemini is thinking …"):
            st.subheader("🧠 Gemini Commentary")
            st.markdown(ask_gemini(comp, prompt))

    except Exception as exc:
        st.error(f"Error: {exc}")
