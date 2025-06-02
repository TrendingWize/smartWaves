import os
import datetime as dt
from pathlib import Path
import textwrap

import requests
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from PIL import Image
import google.generativeai as genai

# ░░ Streamlit config ░░
st.set_page_config(page_title="Technical Analysis", layout="wide")

# ░░ Secrets / constants ░░
FMP_API_KEY    = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
OUT_DIR        = Path("charts")
OUT_DIR.mkdir(exist_ok=True)
plt.rcParams["axes.titlepad"] = 6

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data download + resample
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/"
        f"{ticker}?apikey={FMP_API_KEY}&from={date_from}&to={date_to}"
    )
    data = requests.get(url, timeout=30).json()

    # --- normalise response ---------------------------------------------
    if isinstance(data, dict):
        rows = data.get("historical", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    if not rows:
        raise ValueError(f"No price records for {ticker} in range.")

    df = pd.DataFrame(rows).rename(columns=str.lower)

    # --- harmonise columns ----------------------------------------------
    if "price" in df.columns and "close" not in df.columns:
        df["close"] = df["price"]

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            df[col] = df["close"]

    if "volume" not in df.columns:
        df["volume"] = 0

    return (
        df.assign(date=lambda d: pd.to_datetime(d["date"]))
          .set_index("date")
          .sort_index()
        )[["open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily":
        return df
    rule = "W-FRI" if frame == "Weekly" else "M"
    agg  = {"open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()

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
# 3. Composite chart generator
# ─────────────────────────────────────────────────────────────────────────────
def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(4, 1, hspace=0.05, height_ratios=[3, 1, 1.5, 1])

    # Price + SMAs
    ax_price = fig.add_subplot(gs[0])
    ax_price.set_yscale("log")
    ax_price.plot(df.index, df["close"], label="Close")
    ax_price.plot(df.index, df["sma20"],  label="SMA 20")
    ax_price.plot(df.index, df["sma50"],  label="SMA 50")
    ax_price.plot(df.index, df["sma100"], label="SMA 100")
    ax_price.set_ylabel("Price (log)")
    ax_price.legend(loc="upper left")

    # Volume
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_vol.bar(df.index, df["volume"], color="gray")
    ax_vol.set_ylabel("Vol")

    # MACD
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
    ax_macd.plot(df.index, df["MACD_12_26_9"], label="MACD")
    ax_macd.plot(df.index, df["MACDs_12_26_9"], label="Signal")
    ax_macd.bar(df.index, df["MACDh_12_26_9"], alpha=0.3)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")

    # RSI
    ax_rsi = fig.add_subplot(gs[3], sharex=ax_price)
    ax_rsi.plot(df.index, df["rsi"])
    ax_rsi.axhline(70, linestyle="--")
    ax_rsi.axhline(30, linestyle="--")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)

    locator   = {"Daily": mdates.MonthLocator(),
                 "Weekly": mdates.YearLocator(),
                 "Monthly": mdates.YearLocator()}
    formatter = {"Daily": mdates.DateFormatter("%b-%y"),
                 "Weekly": mdates.DateFormatter("%Y"),
                 "Monthly": mdates.DateFormatter("%Y")}
    ax_rsi.xaxis.set_major_locator(locator[frame])
    ax_rsi.xaxis.set_major_formatter(formatter[frame])

    for ax in [ax_price, ax_vol, ax_macd]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f"{ticker} | {frame} composite", y=0.93)
    fig.tight_layout()
    path = OUT_DIR / f"{ticker}_{frame}_composite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Gemini analysis
# ─────────────────────────────────────────────────────────────────────────────
def ask_gemini(image_path: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    img   = Image.open(image_path)
    return model.generate_content([prompt, img]).text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Composite Chart")

symbol = st.text_input("Ticker (e.g., NVDA):", value="AAPL").upper().strip()

today        = dt.date.today()
default_from = today - dt.timedelta(days=365)
col_from, col_to = st.columns(2)
date_from = col_from.date_input("From", default_from)
date_to   = col_to.date_input("To",   today)

frame = st.selectbox("Time frame", ["Daily", "Weekly", "Monthly"], index=1)
run   = st.button("🚀 Generate Report")

if run:
    if date_from >= date_to:
        st.error("⚠️ **From** date must be earlier than **To** date.")
        st.stop()

    if not (FMP_API_KEY and GOOGLE_API_KEY):
        st.info("Set FMP_API_KEY and GOOGLE_API_KEY in Streamlit secrets.")
        st.stop()

    try:
        with st.spinner("Fetching price data …"):
            raw_df = get_ohlcv(symbol, date_from.isoformat(), date_to.isoformat())
            df     = resample(raw_df, frame)
            df     = add_indicators(df)
            chart  = save_composite_chart(df, symbol, frame)

        st.image(chart, use_column_width=True)

        prompt = textwrap.dedent(f"""
            You are a professional market technician.
            Analyse this {frame.lower()} composite chart of {symbol} (log close, 20/50/100 SMAs, volume, MACD, RSI).
            Comment on trend, momentum, support/resistance. Provide price targets for +30, +60, +252 trading days under bullish, base, bearish scenarios.
        """)
        with st.spinner("Gemini is thinking …"):
            summary = ask_gemini(chart, prompt)
        st.subheader("🧠 Gemini Commentary")
        st.markdown(summary)

    except Exception as e:
        st.error(f"Error: {e}")
