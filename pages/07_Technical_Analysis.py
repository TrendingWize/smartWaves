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

# ── secrets / constants ─────────────────────────────────────────────────────
FMP_API_KEY    = st.secrets.get("FMP_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
OUT_DIR        = Path("charts"); OUT_DIR.mkdir(exist_ok=True)

# ── data helpers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_ohlcv(tkr:str, d_from:str, d_to:str)->pd.DataFrame:
    url=(f"https://financialmodelingprep.com/api/v3/historical-price-full/"
         f"{tkr}?apikey={FMP_API_KEY}&from={d_from}&to={d_to}")
    data=requests.get(url,timeout=30).json()
    rows=data.get("historical",[]) if isinstance(data,dict) else data
    if not rows: raise ValueError("No price data")
    df=pd.DataFrame(rows).rename(columns=str.lower)
    if "price" in df and "close" not in df: df["close"]=df["price"]
    for c in ("open","high","low","close"): df[c]=df.get(c,df["close"])
    df["volume"]=df.get("volume",0)
    return (df.assign(date=lambda d:pd.to_datetime(d["date"]))
              .set_index("date").sort_index()
              [["open","high","low","close","volume"]])

def resample(df,frame):
    if frame=="Daily": return df
    rule="W-FRI" if frame=="Weekly" else "M"
    agg={"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(rule).agg(agg).dropna()

def add_indicators(df):
    df["sma20"]=ta.sma(df["close"],20)
    df["sma50"]=ta.sma(df["close"],50)
    df["sma100"]=ta.sma(df["close"],100)
    df=pd.concat([df,ta.macd(df["close"])],axis=1)
    df["rsi"]=ta.rsi(df["close"],14); return df

def save_composite_chart(df,tkr,frame):
    fig,ax=plt.subplots(figsize=(12,6)); ax.set_yscale("log")
    ax.plot(df.index,df["close"],label="Close")
    for sma,l in [("sma20","SMA20"),("sma50","SMA50"),("sma100","SMA100")]:
        ax.plot(df.index,df[sma],label=l)
    ax.legend(); ax.set_title(f"{tkr} ({frame})")
    p=OUT_DIR/f"{tkr}_{frame}_cmp.png"
    fig.tight_layout(); fig.savefig(p,dpi=120); plt.close(fig); return str(p)

def ask_gemini(img,prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    mdl=genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    return mdl.generate_content([prompt,Image.open(img)]).text.strip()

def tv_chart(sym,interval,theme,height,width,autosize):
    props={"symbol":sym,"interval":interval,"theme":theme,"style":"1",
           "locale":"en","timezone":"Etc/UTC","allow_symbol_change":True,
           "support_host":"https://www.tradingview.com","height":height}
    if autosize: props["autosize"]=True
    else: props["width"]=width or 800
    outer=f"height:{height}px;"+(f'width:{width or 800}px;' if not autosize else "")
    html(f"""
    <div class="tradingview-widget-container" style="{outer}">
      <div id="tv_widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
              async type="text/javascript">{json.dumps(props, separators=(',',':'))}</script>
    </div>
    """,height=height,width=None if autosize else width or 800,scrolling=False)

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("📈 Technical Analysis – Interactive & AI Insights")

c_sym,c_from,c_to,c_frm,c_int=st.columns([3,2,2,2,2])
tv_symbol=c_sym.text_input("TradingView Symbol","NASDAQ:AAPL").upper().strip()
today=dt.date.today()
date_from=c_from.date_input("From",today-dt.timedelta(days=365))
date_to  =c_to.date_input("To",today)
frame   =c_frm.selectbox("Indicator frame",["Daily","Weekly","Monthly"])
tv_int  =c_int.selectbox("TV interval",["1","15","30","60","D","W","M"],4)

height=st.slider("Chart height (px)",400,1000,750)

# new row: autosize | theme | button
col_auto,col_theme,col_btn=st.columns([1,3,1])
autosz=col_auto.checkbox("Autosize width",True)
theme =col_theme.radio("Theme",["auto","light","dark"],horizontal=True)
run_btn=col_btn.button("💡 Generate AI Analysis")

# interactive chart (always)
chart_theme="dark" if theme=="auto" and st.get_option("theme.base")=="dark" else (theme if theme!="auto" else "light")
tv_chart(tv_symbol,tv_int,chart_theme,height,None if autosz else 800,autosz)

# analysis only on click
if run_btn:
    if date_from>=date_to:
        st.error("From-date must precede To-date"); st.stop()
    if not (FMP_API_KEY and GOOGLE_API_KEY):
        st.info("Add FMP_API_KEY & GOOGLE_API_KEY to secrets"); st.stop()

    tkr=re.split(r"[:/]",tv_symbol)[-1]
    try:
        with st.spinner("Fetching price data …"):
            raw=get_ohlcv(tkr,date_from.isoformat(),date_to.isoformat())
            df =add_indicators(resample(raw,frame))
        cmp=save_composite_chart(df,tkr,frame)
        prompt=textwrap.dedent(f"""
**Role:** Expert Market Technician specializing in pure price action analysis. You master Elliott Wave theory, Wyckoff methodology, 
and classical chart patterns. All analysis must derive exclusively from **price structure, volume, and market geometry**.  

**Absolute Rules:**  
✅ Prioritize: Candlestick patterns, volume-profile, swing structure, and institutional accumulation/distribution signs  

---

**Analysis Task:**  
Analyze the composite chart of {c_sym} (Price + Volume only). Structure your response:  

#### 1. **Market Structure Framework**  
- **Trend Identification:**  
  - Primary trend direction (bullish/bearish/neutral) using *swing highs/lows*  
  - Key market phase (Accumulation/Markup/Distribution/Decline) via **Wyckoff principles**  
- **Critical Levels:**  
  - 3 near-term support/resistance zones (price + volume confluence)  
  - Major historical pivots affecting current structure  

#### 2. **Pattern Recognition**  
- **Classical Patterns:**  
  - Flagged instances of: Cup & Handle, Head & Shoulders, Triangles, Channels  
  - Pattern validity assessment (volume confirmation, breakout strength)  
- **Elliott Wave Count:**  
  - Current probable wave position (e.g., "Wave 3 of impulse")  
  - Alternate counts with confidence %  
- **Candlestick Signals:**  
  - High-probability reversal/continuation clusters (e.g., 3-bar plays)  

#### 3. **Projections & Risk Zones**  
- **Price Targets (+30d/+60d/+252d):**  
  - Measured moves from patterns  
  - Fibonacci extensions (impulse waves)  
  - Volume-profile HVN/LVN targets  
- **Failure Scenarios:**  
  - Key breakdown levels invalidating analysis  
  - Stop-loss placement zones (price + volume voids)  

#### 4. **Synthesis**  
- **Current Bias:** Bullish/Base/Bearish (scale 1-5 conviction)  
- **Optimal Entry Triggers:**  
  - Breakout/retest levels with volume filters  
  - Early reversal signals (wick rejections, volume spikes)  
- **Timeframe Alignment:**  
  - Conflicting signals across timeframes (if any)  

        """)
        with st.spinner("Gemini is thinking …"):
            st.subheader("🧠 Gemini Commentary")
            st.markdown(ask_gemini(cmp,prompt))
    except Exception as e:
        st.error(f"Error: {e}")
