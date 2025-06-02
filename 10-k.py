#!/usr/bin/env python
"""
10-k.py  ──  Download the latest 10-K via Financial Modeling Prep,
summarise it with OpenAI, and save a coloured Markdown + HTML report.
"""

from __future__ import annotations
import streamlit as st
import asyncio
import json
import os
import re
import sys
import textwrap
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import markdown2
import nltk
import numpy as np
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.data import find as _nltk_find

# ───────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("test_analysis"); OUTPUT_DIR.mkdir(exist_ok=True)

FMP_API_KEY     = os.getenv("FMP_API_KEY","Aw0rlddPHSnxmi3VmZ6jN4u3b2vvUvxn")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY") or sys.exit("Set OPENAI_API_KEY")

EMBED_MODEL         = "text-embedding-3-small"
EMBED_TOKEN_LIMIT   = 800
EMBED_BATCH_SIZE    = 128
CHAT_MODEL          = "gpt-4o-mini"
MAX_CONCURRENT_Q    = 5          # parallel GPT calls
REQUEST_TIMEOUT     = 30.0

SEC_HEADERS = {
    "User-Agent": "SmartWave/1.0 (+https://smartwave.example; contact@smartwave.example)",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}

QUESTIONS = [
    # General Insights
    "What are the main growth opportunities highlighted in the filing?",
    "Summarize the company's competitive advantages mentioned.",
    "What are the key risks mentioned in the filing?",
    "What are the company's stated strategic priorities and future plans?",
    # Business Overview (Item 1)
    "Provide a summary of the company’s business model, operations, and primary revenue sources.",
    "What markets or geographies does the company primarily operate in?",
    "What new products, services, or initiatives are mentioned in the business overview?",
    # Risk Factors (Item 1A)
    "Summarize the main risk factors described in the filing.",
    "Highlight any emerging risks or industry-specific concerns noted in the document.",
    # Legal Proceedings (Item 3)
    "What legal proceedings are disclosed in the filing? Are there any significant lawsuits or regulatory actions mentioned?",
    # MD&A (Item 7)
    "What insights can be drawn from the Management’s Discussion and Analysis (MD&A)?",
    "What are the key financial trends and operational highlights noted by management?",
    "How has the company performed compared to its stated goals or projections?",
    "What concerns or challenges does management foresee for the next fiscal year?",
    # Financials (Item 8)
    "Provide a summary of the financial performance metrics and key takeaways from the filing's financial statements.",
    "What changes are noticeable in revenue, profitability, or cash flows compared to prior years?",
    "Are there any unusual or noteworthy accounting items mentioned in the financial statements?",
    # Notes
    "Summarize the significant accounting policies and estimates used by the company.",
    "Highlight any major changes in accounting methods or adjustments disclosed in the notes.",
    "What contingent liabilities or off-balance-sheet arrangements are disclosed in the notes?",
    # Executive Compensation (Item 11)
    "What details are provided about executive compensation structures and performance incentives?",
    "Are there any controversies or shareholder concerns related to executive pay?",
    # Corporate Governance (Item 12)
    "What insights can be gathered about the company’s governance practices and board composition?",
    "Are there any notable shareholder proposals or governance concerns mentioned?",
    # Forward-Looking
    "What forward-looking statements are included in the filing, and what assumptions do they rely on?",
    "Are there any specific warnings or disclaimers about forward-looking information?"
]

# ───────────────────────────────────────────────────────────────────────────
# 1) SET-UP
# ───────────────────────────────────────────────────────────────────────────
try:
    import h2  # noqa: F401
    _HTTP2 = True
except ImportError:
    _HTTP2 = False

import openai
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

def _ensure_nltk(pkg:str)->None:
    try:_nltk_find(f"tokenizers/{pkg}")
    except LookupError:nltk.download(pkg,quiet=True)
for _p in ("punkt","punkt_tab"):_ensure_nltk(_p)
warnings.filterwarnings("ignore",category=XMLParsedAsHTMLWarning)

enc = tiktoken.get_encoding("cl100k_base")

# ───────────────────────────────────────────────────────────────────────────
# 2) HELPERS
# ───────────────────────────────────────────────────────────────────────────
def _ticker()->str:
    t=os.getenv("TICKER_SYMBOL"); 
    if not t: sys.exit("Set TICKER_SYMBOL env var."); 
    return t.upper()

async def _json(client,url): r=await client.get(url,timeout=REQUEST_TIMEOUT); r.raise_for_status(); return r.json()

async def _latest_10k_url(client, ticker)->str:
    today=datetime.utcnow().date(); frm=today-timedelta(days=400)
    url=(f"https://financialmodelingprep.com/stable/sec-filings-search/symbol"
         f"?symbol={ticker}&from={frm}&to={today}&page=0&limit=100&apikey={FMP_API_KEY}")
    data=await _json(client,url)
    filings=[f for f in data if str(f.get("formType","")).startswith("10-K")]
    if not filings: raise RuntimeError("No 10-K in last 400 days.")
    latest=max(filings,key=lambda f:f.get("filingDate",f.get("acceptedDate","")))
    link=latest.get("finalLink") or latest.get("link")
    if not link: raise RuntimeError("10-K lacks link field.")
    return link.strip()

def _extract_text(html:str)->str:
    soup=BeautifulSoup(html,"lxml")
    for tag in soup(["script","style","table"]): tag.decompose()
    return re.sub(r"\n{2,}","\n",soup.get_text("\n")).strip()

def _chunks(text:str)->List[str]:
    sents=nltk.tokenize.sent_tokenize(text)
    lens=np.fromiter((len(enc.encode(s)) for s in sents),dtype=np.int32)
    out,buf,buf_len=[],[],0
    for s,l in zip(sents,lens):
        if l>EMBED_TOKEN_LIMIT:
            toks=enc.encode(s)
            for i in range(0,len(toks),EMBED_TOKEN_LIMIT):
                out.append(enc.decode(toks[i:i+EMBED_TOKEN_LIMIT])); 
            continue
        if buf_len+l>EMBED_TOKEN_LIMIT:
            out.append(" ".join(buf)); buf,buf_len=[],0
        buf.append(s); buf_len+=l
    if buf: out.append(" ".join(buf))
    return out

async def _embed(chunks:List[str])->None:
    for i in range(0,len(chunks),EMBED_BATCH_SIZE):
        await openai_client.embeddings.create(model=EMBED_MODEL,
                                              input=chunks[i:i+EMBED_BATCH_SIZE])

async def _ask_gpt(question:str, context:str, ticker:str)->str:
    prompt=textwrap.dedent(f"""
        Answer the following question about {ticker}'s latest 10-K filing.
        Be concise (4-6 lines max) and factual.
        Question: {question}
        <FILING>
        {context}
        </FILING>
    """).strip()

    resp=await openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.3,
            max_tokens=300,
            messages=[
                {"role":"system","content":"You are a precise financial analyst."},
                {"role":"user","content":prompt}
            ])
    return resp.choices[0].message.content.strip()

async def _answer_questions(ticker:str, context:str)->dict[str,str]:
    sem=asyncio.Semaphore(MAX_CONCURRENT_Q)
    async def _worker(q):
        async with sem: return q, await _ask_gpt(q,context,ticker)
    tasks=[_worker(q) for q in QUESTIONS]
    results=await asyncio.gather(*tasks)
    return dict(results)

def _color(txt:str,hex_:str)->str: return f'<span style="color:{hex_}">{txt}</span>'

def _markdown(ticker:str, answers:dict[str,str])->str:
    md=[f"# **{ticker} 10-K Q&A**\n"]
    colors=["#0057e7","#008000","#d00000","#aa00aa"]  # rotate hues
    for i,(q,ans) in enumerate(answers.items(),1):
        col=colors[i%len(colors)]
        md.append(f"### Q{i}. {_color(q,col)}\n")
        md.extend(f"- {line}" for line in ans.splitlines() if line.strip())
        md.append("")
    return "\n".join(md)

# ───────────────────────────────────────────────────────────────────────────
# 3) MAIN
# ───────────────────────────────────────────────────────────────────────────
async def main()->None:
    ticker=_ticker()

    async with httpx.AsyncClient(http2=_HTTP2,follow_redirects=True) as client:
        sec_url=await _latest_10k_url(client,ticker)
        print("Downloading:",sec_url)
        html_resp=await client.get(sec_url,headers=SEC_HEADERS,timeout=REQUEST_TIMEOUT)
        html_resp.raise_for_status()
        text=_extract_text(html_resp.text)

    # optional embedding step
    await _embed(_chunks(text))

    context=" ".join(text.split()[:12000])  # truncate for GPT
    answers=await _answer_questions(ticker,context)

    md=_markdown(ticker,answers)
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path=OUTPUT_DIR/f"{ticker}_10-K_QA_{ts}.md"
    html_path=OUTPUT_DIR/f"{ticker}_10-K_QA_{ts}.html"
    md_path.write_text(md,"utf-8")
    html_path.write_text(markdown2.markdown(md,extras=["tables"]),"utf-8")
    print("\n✓ Markdown:",md_path,"\n✓ HTML:",html_path)

# ───────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    asyncio.run(main())

# ───────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    asyncio.run(main())
