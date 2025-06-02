import os
import datetime as dt
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

# ░░ Environment / Secrets ░░
FMP_API_KEY     = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY  = st.secrets.get("GOOGLE_API_KEY")
OUT_DIR         = "charts"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["axes.titlepad"] = 6


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download DAILY data once, then resample to W-Fri or Month-End as needed
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_daily_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame covering <years>*252 sessions."""
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{ticker}?apikey={FMP_API_KEY}&timeseries={years*252}")
    raw = requests.get(url, timeout=30).json().get("historical", [])
    if not raw:
        raise ValueError(f"No data returned for {ticker}")

    df = (pd.DataFrame(raw)
            .rename(columns=str.lower)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .set_index("date")
            .sort_index())

    if "adjclose" in df and not df["adjclose"].isna().all():
        df["close"] = df["adjclose"]

    return df[["open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    """Return df resampled to 'daily', 'weekly', or 'monthly'."""
    if frame == "Daily":
        return df
    rule = "W-FRI" if frame == "Weekly" else "M"           # month-end
    agg  = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    return (df.resample(rule).agg(agg).dropna()
              .rename(columns=str.title))  # Open/High/…


# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicators
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"]  = ta.sma(df["Close"], 20)
    df["SMA50"]  = ta.sma(df["Close"], 50)
    df["SMA100"] = ta.sma(df["Close"], 100)
    df = pd.concat([df, ta.macd(df["Close"])], axis=1)
    df["RSI"]    = ta.rsi(df["Close"], 14)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composite chart generator
# ─────────────────────────────────────────────────────────────────────────────
def save_composite_chart(df: pd.DataFrame, ticker: str, frame: str) -> str:
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.05, height_ratios=[3, 1, 1.5, 1])

    ax_price = fig.add_subplot(gs[0])
    ax_price.set_yscale("log")
    ax_price.plot(df.index, df["Close"], label="Close")
    ax_price.plot(df.index, df["SMA20"], label="SMA 20")
    ax_price.plot(df.index, df["SMA50"], label="SMA 50")
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

    # x-axis ticks per frame
    locator = {"Daily": mdates.MonthLocator(),
               "Weekly": mdates.YearLocator(),
               "Monthly": mdates.YearLocator()}
    formatter = {"Daily": mdates.DateFormatter("%b-%y"),
                 "Weekly": mdates.DateFormatter("%Y"),
                 "Monthly": mdates.DateFormatter("%Y")}
    ax_rsi.xaxis.set_major_locator(locator[frame])
    ax_rsi.xaxis.set_major_formatter(formatter[frame])

    for ax in [ax_price, ax_vol, ax_macd]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f"{ticker} | {frame} Price, Volume & Indicators", y=0.93)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{ticker}_{frame}_composite.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gemini analysis
# ─────────────────────────────────────────────────────────────────────────────
def ask_gemini(image_path: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    img = Image.open(image_path)
    return model.generate_content([prompt, img]).text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Composite Chart")

symbol = st.text_input("Ticker (e.g., NVDA):", value="AAPL").upper().strip()
years  = st.slider("History (years)", 1, 10, 5)
frame  = st.selectbox("Time frame", ["Daily", "Weekly", "Monthly"], index=1)

if symbol and FMP_API_KEY and GOOGLE_API_KEY:
    try:
        with st.spinner("Fetching & crunching data …"):
            daily_df   = get_daily_ohlcv(symbol, years)
            frame_df   = resample(daily_df, frame)
            enriched   = add_indicators(frame_df)
            chart_path = save_composite_chart(enriched, symbol, frame)

        st.image(chart_path, use_column_width=True)

        prompt = (
            f"You are a professional market technician. Analyse this {frame.lower()} "
            f"{symbol} composite chart (price on log scale with 20/50/100 SMAs, "
            "volume, MACD, RSI). Discuss trend, momentum, support/resistance, "
            "Elliott wave structure, and provide price targets for the next "
            "30, 60 and 252 trading days under bullish, base and bearish scenarios."
        )
        with st.spinner("Gemini is thinking …"):
            summary = ask_gemini(chart_path, prompt)
            st.subheader("🧠 Gemini Commentary")
            st.markdown(summary)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please set FMP and Google API keys in Streamlit secrets.")
