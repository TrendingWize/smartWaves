#!/usr/bin/env python
"""
10-k.py ─ Semantic-search Q&A over latest 10-K

Requires in requirements.txt:
    httpx[http2]>=0.27.0   beautifulsoup4>=4.12
    lxml>=5.2             nltk>=3.8
    numpy>=1.26           tiktoken>=0.6
    openai>=1.25          markdown2>=2.4
"""

from __future__ import annotations

import asyncio, json, os, re, sys, textwrap, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import httpx, markdown2, nltk, numpy as np, tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.data import find as _nltk_find
from pathlib import Path              # keep this import if you removed it

import streamlit as st
# ── CONFIG ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("test_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

FMP_API_KEY = st.secrets.get("FMP_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

EMBED_MODEL       = "text-embedding-3-small"
CHAT_MODEL        = "gpt-4o-mini"
EMBED_TOKEN_LIMIT = 800
EMBED_BATCH       = 128
TOP_K             = 5          # chunks retrieved per question
MAX_CONCURRENT_Q  = 5
REQUEST_TIMEOUT   = 30.0

SEC_HEADERS = {
    "User-Agent": "SmartWave/1.0 (+https://smartwave.example; contact@smartwave.example)",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}

QUESTIONS = [   # ← exactly the list you provided
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

# ── LIB / ENV INIT ────────────────────────────────────────────────────────
try: import h2; _HTTP2 = True
except ImportError: _HTTP2 = False

import openai
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

def _nltk_dl(pkg:str)->None:
    try: _nltk_find(f"tokenizers/{pkg}")
    except LookupError: nltk.download(pkg, quiet=True)
for p in ("punkt","punkt_tab"): _nltk_dl(p)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
enc = tiktoken.get_encoding("cl100k_base")

# ── HELPERS ───────────────────────────────────────────────────────────────
def _ticker()->str:
    t = os.getenv("TICKER_SYMBOL")
    if not t: sys.exit("Set TICKER_SYMBOL")
    return t.upper()

async def _json(c: httpx.AsyncClient, url:str):
    r = await c.get(url, timeout=REQUEST_TIMEOUT); r.raise_for_status(); return r.json()

async def _latest_10k_url(c: httpx.AsyncClient, tk:str)->str:
    today = datetime.utcnow().date(); frm = today - timedelta(days=400)
    url   = ( "https://financialmodelingprep.com/stable/sec-filings-search/symbol"
             f"?symbol={tk}&from={frm}&to={today}&page=0&limit=100&apikey={FMP_API_KEY}")
    data = await _json(c, url)
    filings = [f for f in data if str(f.get("formType","")).startswith("10-K")]
    if not filings:  raise RuntimeError("No 10-K in last 400d.")
    latest = max(filings, key=lambda f:f.get("filingDate", f.get("acceptedDate","")))
    return (latest.get("finalLink") or latest.get("link")).strip()

def _plain_text(html:str)->str:
    soup = BeautifulSoup(html,"lxml")
    for tag in soup(["script","style","table"]): tag.decompose()
    return re.sub(r"\n{2,}","\n", soup.get_text("\n")).strip()

def _chunk(text:str)->Tuple[List[str], np.ndarray]:
    sents = nltk.tokenize.sent_tokenize(text)
    lens  = np.fromiter((len(enc.encode(s)) for s in sents), dtype=np.int32)
    chunks, buf, buf_len = [], [], 0
    for s,l in zip(sents,lens):
        if l>EMBED_TOKEN_LIMIT:
            toks = enc.encode(s)
            for i in range(0,len(toks),EMBED_TOKEN_LIMIT):
                chunks.append(enc.decode(toks[i:i+EMBED_TOKEN_LIMIT]))
            continue
        if buf_len + l > EMBED_TOKEN_LIMIT:
            chunks.append(" ".join(buf)); buf,buf_len=[],0
        buf.append(s); buf_len += l
    if buf: chunks.append(" ".join(buf))
    return chunks, lens  # lens only used for info

async def _embed_chunks(chunks:List[str])->np.ndarray:
    vecs=[]
    for i in range(0,len(chunks),EMBED_BATCH):
        res = await openai_client.embeddings.create(model=EMBED_MODEL,
                                                    input=chunks[i:i+EMBED_BATCH])
        vecs.extend([d.embedding for d in res.data])
    vecs = np.array(vecs, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # normalise
    return vecs

async def _embed_text(text:str)->np.ndarray:
    res = await openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    v   = np.array(res.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)

def _top_k(sim_vec:np.ndarray, doc_matrix:np.ndarray, k:int)->List[int]:
    scores = doc_matrix @ sim_vec   # cosine because pre-normalised
    return np.argpartition(-scores, k)[:k]

async def _answer_one(q:str, tk:str,
                      doc_chunks:List[str], doc_vecs:np.ndarray)->Tuple[str,str]:
    q_vec = await _embed_text(q)
    idxs  = _top_k(q_vec, doc_vecs, TOP_K)
    context = "\n\n".join(doc_chunks[i] for i in idxs)
    prompt = textwrap.dedent(f"""
        You are a precise financial analyst.
        Use ONLY the context provided to answer the question. youre resonse **MUST** be as Markdown format.
        Use appropriat Markdown formating for the answer like title, headings, tables etc... questions always as a heading (#). 
        after the answer add classfication (positiv, negative, sever negative etc...) use green color for positive and red colour for negative and sever negative.
        Question: {q}
        <CONTEXT>
        {context}
        </CONTEXT>
    """).strip()
    rsp = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        max_tokens=4000,
        messages=[{"role":"user","content":prompt}]
    )
    return q, rsp.choices[0].message.content.strip()

async def _qa_all(tk:str, doc_chunks:List[str], doc_vecs:np.ndarray)->dict[str,str]:
    sem = asyncio.Semaphore(MAX_CONCURRENT_Q)
    async def _worker(question):
        async with sem:
            return await _answer_one(question, tk, doc_chunks, doc_vecs)
    return dict(await asyncio.gather(*[_worker(q) for q in QUESTIONS]))

def _color(t:str,c:str)->str: return f'<span style="color:{c}">{t}</span>'

def _markdown(tk:str, answers:dict[str,str])->str:
    cols = ["#0057e7","#008000","#d00000","#aa00aa"]
    md=[f"# **{tk} 10-K Semantic Q&A**\n"]
    for i,(q,a) in enumerate(answers.items(),1):
        md.append(f"### Q{i}. {_color(q,cols[i%len(cols)])}\n")
        md.extend(f"- {ln}" for ln in a.splitlines() if ln.strip())
        md.append("")
    return "\n".join(md)

# ── MAIN ──────────────────────────────────────────────────────────────────
async def main()->None:
    tk=_ticker()

    async with httpx.AsyncClient(http2=_HTTP2, follow_redirects=True) as c:
        sec = await _latest_10k_url(c, tk)
        print("Fetching:", sec)
        html = await c.get(sec, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
        html.raise_for_status()
        text = _plain_text(html.text)

    chunks, _ = _chunk(text)
    print(f"{len(chunks)} chunks → embedding …")
    doc_vecs = await _embed_chunks(chunks)

    answers = await _qa_all(tk, chunks, doc_vecs)
    md = _markdown(tk, answers)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path   = OUTPUT_DIR / f"{tk}_10-K_QA_{ts}.md"
    html_path = OUTPUT_DIR / f"{tk}_10-K_QA_{ts}.html"
    md_path.write_text(md,"utf-8")
    html_path.write_text(markdown2.markdown(md, extras=["tables"]),"utf-8")
    print("✓ Written:", md_path, html_path, sep="\n")

if __name__=="__main__":
    asyncio.run(main())
