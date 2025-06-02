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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import markdown2
import nltk
import numpy as np
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.data import find as _nltk_find

# ────────────────────────────────────────────
# 0)  CONFIGURATION
# ────────────────────────────────────────────
OUTPUT_DIR = Path("test_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
EMBED_TOKEN_LIMIT = 800
EMBED_BATCH_SIZE = 128

CHAT_MODEL = "gpt-4o-mini"
MAX_CHAT_TOKENS = 4096

REQUEST_TIMEOUT = 30.0

FMP_API_KEY = st.secrets.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", "")

SEC_HEADERS = {
    "User-Agent": "SmartWave/1.0 (+https://smartwave.example; contact@smartwave.example)",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}

# ────────────────────────────────────────────
# 1)  ENVIRONMENT SET-UP
# ────────────────────────────────────────────
try:
    import h2  # noqa: F401
    _HTTP2 = True
except ImportError:
    _HTTP2 = False

import openai  # openai-python ≥1.x

openai_client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or sys.exit("Set OPENAI_API_KEY")
)

# NLTK resources
def _ensure_nltk(model: str) -> None:
    try:
        _nltk_find(f"tokenizers/{model}")
    except LookupError:
        nltk.download(model, quiet=True)

for _m in ("punkt", "punkt_tab"):
    _ensure_nltk(_m)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
enc = tiktoken.get_encoding("cl100k_base")

# ────────────────────────────────────────────
# 2)  HELPERS
# ────────────────────────────────────────────
def _ticker() -> str:
    t = os.getenv("TICKER_SYMBOL")
    if not t:
        sys.exit("ERROR: set TICKER_SYMBOL env var.")
    return t.upper()


async def _fetch_json(client: httpx.AsyncClient, url: str) -> list | dict:
    r = await client.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


async def _latest_10k_url_fmp(client: httpx.AsyncClient, ticker: str) -> str:
    """
    Call /sec-filings-search/symbol for the last 400 days
    and return the latest 10-K finalLink (or link).
    """
    today = datetime.utcnow().date()
    frm   = today - timedelta(days=400)

    url = (
        "https://financialmodelingprep.com/stable/sec-filings-search/symbol"
        f"?symbol={ticker}"
        f"&from={frm.isoformat()}&to={today.isoformat()}"
        f"&page=0&limit=100&apikey={FMP_API_KEY}"
    )
    data = await _fetch_json(client, url)
    if not data:
        raise RuntimeError("FMP filings search returned empty list.")

    # isolate 10-K (or amended 10-K/A) and pick the most recent by filingDate
    filings_10k = [
        f for f in data
        if isinstance(f.get("formType"), str) and f["formType"].startswith("10-K")
    ]
    if not filings_10k:
        raise RuntimeError("No 10-K filing found in the last 400 days.")

    latest = max(
        filings_10k,
        key=lambda f: f.get("filingDate", f.get("acceptedDate", "")),
    )

    # prefer finalLink if present, else link
    link = latest.get("finalLink") or latest.get("link")
    if not link:
        raise RuntimeError("Filing object lacks a usable SEC link.")
    return link.strip()


def _extract_text(body: str) -> str:
    soup = BeautifulSoup(body, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    txt = soup.get_text("\n")
    return re.sub(r"\n{2,}", "\n", txt).strip()


def _chunk_sentences(text: str) -> List[str]:
    sents = nltk.tokenize.sent_tokenize(text)
    lens = np.fromiter((len(enc.encode(s)) for s in sents), dtype=np.int32)

    chunks, buf, buf_len = [], [], 0
    for s, ln in zip(sents, lens):
        if ln > EMBED_TOKEN_LIMIT:
            toks = enc.encode(s)
            for i in range(0, len(toks), EMBED_TOKEN_LIMIT):
                chunks.append(enc.decode(toks[i : i + EMBED_TOKEN_LIMIT]))
            continue
        if buf_len + ln > EMBED_TOKEN_LIMIT:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s); buf_len += ln
    if buf:
        chunks.append(" ".join(buf))
    return chunks


async def _embed(chunks: List[str]) -> None:
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        await openai_client.embeddings.create(
            model=EMBED_MODEL, input=chunks[i : i + EMBED_BATCH_SIZE]
        )


def _color(txt: str, hex_: str) -> str:
    return f'<span style="color:{hex_}">{txt}</span>'


def _markdown(ticker: str, sections: list[tuple[str, str, str]]) -> str:
    head = f"# **{ticker} 10-K Analysis**\n\n"
    out: list[str] = []
    for title, txt, hex_c in sections:
        out.append(f"## {title}\n")
        for line in txt.splitlines():
            if line.strip():
                out.append(f"- {_color(line.strip(), hex_c)}")
        out.append("")
    return head + "\n".join(out)


async def _analyse(ticker: str, context: str) -> dict[str, str]:
    prompt = textwrap.dedent(
        f"""
        Summarise the following 10-K filing for {ticker}.
        Respond with valid JSON containing exactly these keys:
        summary, risks, opportunities — each value is 3-5 bullet lines
        separated by \\n (plain text).

        <FILING>
        {context}
        </FILING>
        """
    ).strip()

    chat = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a precise financial analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(chat.choices[0].message.content)


# ────────────────────────────────────────────
# 3)  MAIN
# ────────────────────────────────────────────
async def main() -> None:
    ticker = _ticker()

    async with httpx.AsyncClient(http2=_HTTP2, follow_redirects=True) as client:
        sec_url = await _latest_10k_url_fmp(client, ticker)
        print(f"Downloading filing … {sec_url}")
        html = await client.get(sec_url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
        html.raise_for_status()
        text = _extract_text(html.text)

    chunks = _chunk_sentences(text)
    await _embed(chunks)  # optional semantic store

    excerpt = " ".join(text.split()[:12000])  # keep within GPT context
    analysis = await _analyse(ticker, excerpt)

    md = _markdown(
        ticker,
        [
            ("Summary",        analysis["summary"],       "#008000"),
            ("Key Risks",      analysis["risks"],         "#d00000"),
            ("Opportunities",  analysis["opportunities"], "#0057e7"),
        ],
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path   = OUTPUT_DIR / f"{ticker}_10-K_{ts}.md"
    html_path = OUTPUT_DIR / f"{ticker}_10-K_{ts}.html"

    md_path.write_text(md, "utf-8")
    html_path.write_text(markdown2.markdown(md, extras=["tables"]), "utf-8")

    print(f"\n✓ Markdown: {md_path}")
    print(f"✓ HTML   : {html_path}")


if __name__ == "__main__":
    asyncio.run(main())
