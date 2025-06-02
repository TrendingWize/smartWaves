import os
import datetime as dt
from pathlib import Path

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

    # ---- normalise to a list of dicts -----------------------------------
    if isinstance(data, dict):
        rows = data.get("historical", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    if not rows:
        raise ValueError(
            f"No price records returned for {ticker} between {date_from} and {date_to}"
        )

    df = pd.DataFrame(rows).rename(columns=str.lower)

    # ---- harmonise column names ----------------------------------------
    if "close" in df.columns and "close" not in df.columns:
        df["close"] = df["close"]

    # if high/low/open missing, synthesize from close so indicators work
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]

    if "volume" not in df.columns:
        df["volume"] = 0

    df = (
        df.assign(date=lambda d: pd.to_datetime(d["date"]))
          .set_index("date")
          .sort_index()
    )

    return df[["open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    if frame == "Daily":
        out = df
    else:
        rule = "W-FRI" if frame == "Weekly" else "M"
        agg  = {"open": "first", "high": "max", "low": "min",
                "price": "last", "volume": "sum"}
        out = df.resample(rule).agg(agg).dropna()
    return out.rename(columns=str.title)  # Open/High/…


# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicators
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"]  = ta.sma(df["price"], 20)
    df["SMA50"]  = ta.sma(df["price"], 50)
    df["SMA100"] = ta.sma(df["price"], 100)
    df = pd.concat([df, ta.macd(df["price"])], axis=1)
    df["RSI"]    = ta.rsi(df["price"], 14)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composite chart generator
# ─────────────────────────────────────────────────────────────────────────────
def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(4, 1, hspace=0.05, height_ratios=[3, 1, 1.5, 1])

    ax_price = fig.add_subplot(gs[0])
    ax_price.set_yscale("log")
    ax_price.plot(df.index, df["price"], label="price")
    ax_price.plot(df.index, df["SMA20"],  label="SMA 20")
    ax_price.plot(df.index, df["SMA50"],  label="SMA 50")
    ax_price.plot(df.index, df["SMA100"], label="SMA 100")
    ax_price.set_ylabel("Price (log)")
    ax_price.legend(loc="upper left")

    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_vol.bar(df.index, df["Volume"], color="gray")
    ax_vol.set_ylabel("Vol")

    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
    ax_macd.plot(df.index, df["MACD_12_26_9"], label="MACD")
    ax_macd.plot(df.index, df["MACDs_12_26_9"], label="Signal")
    ax_macd.bar(df.index, df["MACDh_12_26_9"], alpha=0.3)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")

    ax_rsi = fig.add_subplot(gs[3], sharex=ax_price)
    ax_rsi.plot(df.index, df["RSI"])
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

today = dt.date.today()
default_from = today - dt.timedelta(days=365)
col_from, col_to = st.columns(2)
date_from = col_from.date_input("From", default_from)
date_to   = col_to.date_input("To", today)

frame = st.selectbox("Time frame", ["Daily", "Weekly", "Monthly"], index=1)

run = st.button("🚀 Generate Report")

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

        prompt = textwrap.dedent(
            f"""
            You are a professional market technician.
            Analyse this {frame.lower()} composite chart of {symbol} (price log-scale, 20/50/100 SMAs, volume, MACD, RSI).
            Comment on trend, momentum, support/resistance. Provide price targets for +30, +60, +252 trading days under bullish/base/bearish scenarios.
            """
        )
        with st.spinner("Gemini is thinking …"):
            summary = ask_gemini(chart, prompt)
        st.subheader("🧠 Gemini Commentary")
        st.markdown(summary)

    except Exception as e:
        st.error(f"Error: {e}")
