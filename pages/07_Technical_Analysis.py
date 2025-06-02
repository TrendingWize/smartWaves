import os, json, textwrap, datetime as dt
from pathlib import Path

import requests, pandas as pd, pandas_ta as ta
import matplotlib.pyplot as plt, matplotlib.dates as mdates
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
plt.rcParams["axes.titlepad"] = 6

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data download + resample
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

    if "price" in df and "close" not in df:  # minimal endpoint
        df["close"] = df["price"]
    for col in ("open", "high", "low", "close"):
        if col not in df: df[col] = df["close"]
    if "volume" not in df: df["volume"] = 0

    return (df.assign(date=lambda d: pd.to_datetime(d["date"]))
              .set_index("date")
              .sort_index())[
                ["open", "high", "low", "close", "volume"]
              ]

def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily":
        return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    return df.resample(rule).agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicators
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma20"]  = ta.sma(df["close"], 20)
    df["sma50"]  = ta.sma(df["close"], 50)
    df["sma100"] = ta.sma(df["close"], 100)
    df = pd.concat([df, ta.macd(df["close"])], axis=1)
    df["rsi"]    = ta.rsi(df["close"], 14)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. Matplotlib composite (kept for Gemini only)
# ─────────────────────────────────────────────────────────────────────────────
def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["close"], label="Close (log)")
    ax.set_yscale("log"); ax.legend(); fig.tight_layout()
    path = OUT_DIR / f"{ticker}_{frame}_composite.png"
    fig.savefig(path, dpi=120); plt.close(fig); return str(path)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Gemini analysis
# ─────────────────────────────────────────────────────────────────────────────
def ask_gemini(img_path: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    return model.generate_content([prompt, Image.open(img_path)]).text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Interactive Chart")

symbol = st.text_input("Ticker (e.g., NVDA)", value="AAPL").upper().strip()
today = dt.date.today(); default_from = today - dt.timedelta(days=365)
col_from, col_to = st.columns(2)
date_from = col_from.date_input("From", default_from)
date_to   = col_to.date_input("To",   today)
frame = st.selectbox("Time frame", ["Daily", "Weekly", "Monthly"], index=1)
run   = st.button("🚀 Generate Report")

if run:
    if date_from >= date_to:
        st.error("⚠️ From-date must be earlier than To-date"); st.stop()
    if not (FMP_API_KEY and GOOGLE_API_KEY):
        st.info("Set FMP_API_KEY & GOOGLE_API_KEY in secrets"); st.stop()

    try:
        with st.spinner("Fetching data …"):
            raw = get_ohlcv(symbol, date_from.isoformat(), date_to.isoformat())
            df  = add_indicators(resample(raw, frame))

        # --- Lightweight Charts embed -----------------------------------
        candle = [
            {"time": d.strftime("%Y-%m-%d"),
             "open": float(o), "high": float(h),
             "low": float(l), "close": float(c)}
            for d, o, h, l, c in df[["open","high","low","close"]].itertuples()
        ]
        sma20 = [{"time": d.strftime("%Y-%m-%d"), "value": float(v)}
                 for d, v in df["sma20"].dropna().items()]
        sma50 = [{"time": d.strftime("%Y-%m-%d"), "value": float(v)}
                 for d, v in df["sma50"].dropna().items()]
        sma100= [{"time": d.strftime("%Y-%m-%d"), "value": float(v)}
                 for d, v in df["sma100"].dropna().items()]

        chart_json = {
            "candles": candle,
            "sma20": sma20, "sma50": sma50, "sma100": sma100,
            "title": f"{symbol} — {frame}"
        }
        html_code = f"""
        <div id="tvchart" style="width:100%;height:460px;"></div>
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <script>
        const data   = {json.dumps(chart_json)};
        const chart  = LightweightCharts.createChart(document.getElementById('tvchart'), {{
            height: 460,
            layout: {{ textColor:'#000', background: {{ type:'solid', color:'#fff' }} }},
            timeScale: {{ timeVisible: true, secondsVisible: false }},
        }});
        const candleSeries = chart.addCandlestickSeries();
        candleSeries.setData(data.candles);
        const sma20 = chart.addLineSeries({{ color:'#008000', lineWidth:1 }});
        sma20.setData(data.sma20);
        const sma50 = chart.addLineSeries({{ color:'#0000ff', lineWidth:1 }});
        sma50.setData(data.sma50);
        const sma100 = chart.addLineSeries({{ color:'#d00000', lineWidth:1 }});
        sma100.setData(data.sma100);
        chart.applyOptions({{ watermark: {{ visible:true, text:data.title, fontSize:14 }} }});
        window.addEventListener('resize', () => {{
            chart.resize(document.getElementById('tvchart').clientWidth, 460);
        }});
        </script>
        """
        html(html_code, height=480)

        # --- keep composite for Gemini only ------------------------------
        composite_path = save_composite_chart(df, symbol, frame)
        prompt = textwrap.dedent(f"""
            You are a professional market technician.
            Analyse this {frame.lower()} composite chart of {symbol} (log close, SMAs 20/50/100, volume, MACD, RSI).
            Comment on trend, momentum, support/resistance. Provide price targets for +30, +60, +252 trading days under bullish, base, bearish scenarios.
        """)
        with st.spinner("Gemini is thinking …"):
            st.subheader("🧠 Gemini Commentary")
            st.markdown(ask_gemini(composite_path, prompt))

    except Exception as e:
        st.error(f"Error: {e}")
