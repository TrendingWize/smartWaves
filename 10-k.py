#!/usr/bin/env python
"""
10-k.py  ──  Download the latest 10-K for a given TICKER_SYMBOL,
summarise it with OpenAI, and write a coloured Markdown + HTML report.

Dependencies (add to requirements.txt):
    httpx[http2]>=0.27.0
    beautifulsoup4>=4.12
    lxml>=5.2
    nltk>=3.8
    numpy>=1.26
    tiktoken>=0.6
    openai>=1.25
    markdown2>=2.4
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import httpx
import markdown2
import nltk
import numpy as np
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.data import find as _nltk_find
from tqdm.asyncio import tqdm_asyncio

# ────────────────────────────────────────────
# 0)  CONFIGURATION
# ────────────────────────────────────────────
OUTPUT_DIR = Path("test_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
EMBED_TOKEN_LIMIT = 800  # max tokens per embedding chunk
EMBED_BATCH_SIZE = 128

CHAT_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo-0125"
MAX_CHAT_TOKENS = 4096      # truncate context if it gets too large

REQUEST_TIMEOUT = 30.0

# SEC requires a descriptive UA
SEC_HEADERS = {
    "User-Agent": "SmartWave/1.0 (+https://smartwave.example; contact@smartwave.example)",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}

# ────────────────────────────────────────────
# 1)  ENVIRONMENT & LIBRARY SET-UP
# ────────────────────────────────────────────
try:
    import h2  # noqa: F401  # enables HTTP/2 support in httpx
    _HTTP2 = True
except ImportError:
    _HTTP2 = False

# openai-python 1.x
import openai
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    sys.exit("ERROR: set OPENAI_API_KEY in environment.")
openai_client = openai.AsyncOpenAI(api_key=openai_key)

# Ensure NLTK tokenisers are available
def _ensure_nltk(model: str) -> None:
    try:
        _nltk_find(f"tokenizers/{model}")
    except LookupError:
        nltk.download(model, quiet=True)

for m in ("punkt", "punkt_tab"):
    _ensure_nltk(m)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")

# ────────────────────────────────────────────
# 2)  HELPER FUNCTIONS
# ────────────────────────────────────────────
def _ticker_from_env() -> str:
    ticker = os.getenv("TICKER_SYMBOL")
    if not ticker:
        sys.exit("ERROR: TICKER_SYMBOL env var is required.")
    return ticker.upper()


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def _latest_10k_url(cik: str) -> str:
    """Return the inline-XBRL (htm) link for most recent 10-K filing."""
    # SEC's fast-search JSON endpoint
    url = (
        f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    )
    r = httpx.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    filings = data["filings"]["recent"]
    for idx, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][idx].replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession}/{data['tickers'][0].lower()}-{accession}-"
                "xbrl.htm"
            )
    raise RuntimeError("No 10-K found.")


def _extract_text_from_html(body: str) -> str:
    soup = BeautifulSoup(body, "lxml")  # HTML parser is fine for inline-XBRL
    # crude drop of tables / scripts
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)  # collapse blank lines
    return text.strip()


def _chunk_sentences(text: str, token_limit: int) -> List[str]:
    """Return a list of sentence chunks, each ≤ token_limit."""
    sents = nltk.tokenize.sent_tokenize(text)
    # vectorise token counts
    lengths = np.fromiter((len(enc.encode(s)) for s in sents), dtype=np.int32)

    chunks, buf, buf_len = [], [], 0
    for sent, sent_len in zip(sents, lengths):
        if sent_len > token_limit:  # rare: chop long sentence
            subtoks = enc.encode(sent)
            for i in range(0, len(subtoks), token_limit):
                chunk = enc.decode(subtoks[i : i + token_limit])
                chunks.append(chunk)
            continue

        if buf_len + sent_len > token_limit:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent); buf_len += sent_len

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _color(txt: str, hex_: str) -> str:
    return f'<span style="color:{hex_}">{txt}</span>'


def _build_markdown(ticker: str, sections: list[tuple[str, str, str]]) -> str:
    header = f"# **{ticker} 10-K Analysis**\n\n"
    body_lines: list[str] = []
    for title, txt, hex_c in sections:
        body_lines.append(f"## {title}\n")
        for line in txt.splitlines():
            if line.strip():
                body_lines.append(f"- {_color(line.strip(), hex_c)}")
        body_lines.append("")  # blank line
    return header + "\n".join(body_lines)


async def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        res = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        vectors.extend([d.embedding for d in res.data])
    return vectors


async def _generate_analysis(ticker: str, context: str) -> dict[str, str]:
    prompt = textwrap.dedent(
        f"""
        You are an equity research analyst. Summarise the following 10-K filing
        for {ticker}. Your reply MUST be valid JSON with keys:
            summary        – 3-5 bullet lines (one string, lines separated by \\n)
            risks          – 3-5 bullets
            opportunities  – 3-5 bullets

        Use plain text; no Markdown bullets inside values.
        Filing begins below delimited by <FILING></FILING>.

        <FILING>
        {context}
        </FILING>
        """
    ).strip()

    # keep context within model limits
    prompt_tokens = len(enc.encode(prompt))
    if prompt_tokens > MAX_CHAT_TOKENS - 1024:
        # truncate from the end (least relevant)
        excess = prompt_tokens - (MAX_CHAT_TOKENS - 1024)
        keep = enc.encode(prompt)[:-excess]
        prompt = enc.decode(keep)

    chat = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        max_tokens=1024,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a precise financial analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(chat.choices[0].message.content)


# ────────────────────────────────────────────
# 3)  MAIN ORCHESTRATION
# ────────────────────────────────────────────
async def main() -> None:
    ticker = _ticker_from_env()

    # resolve CIK
    cik_lookup = httpx.get(
        f"https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT
    ).json()
    ticker_map = {v["ticker"]: v["cik_str"] for v in cik_lookup.values()}
    if ticker not in ticker_map:
        sys.exit(f"Ticker {ticker} not found in SEC list.")
    cik = str(ticker_map[ticker])

    filing_url = _latest_10k_url(cik)
    print(f"Downloading filing … ({filing_url})")

    async with httpx.AsyncClient(http2=_HTTP2, follow_redirects=True) as client:
        html = await _fetch_text(client, filing_url)

    text = _extract_text_from_html(html)
    chunks = _chunk_sentences(text, token_limit=EMBED_TOKEN_LIMIT)

    # optional: embed (not used further but left in for future semantic work)
    print(f"Embedding {len(chunks)} chunks …")
    await _embed_chunks(chunks)

    # use the first ~12 000 words as context (enough for GPT-4 mini)
    context_excerpt = " ".join(text.split()[:12000])
    analysis = await _generate_analysis(ticker, context_excerpt)

    # build coloured Markdown
    sections = [
        ("Summary",        analysis["summary"],       "#008000"),
        ("Key Risks",      analysis["risks"],         "#d00000"),
        ("Opportunities",  analysis["opportunities"], "#0057e7"),
    ]
    md = _build_markdown(ticker, sections)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    md_path = OUTPUT_DIR / f"{ticker}_10-K_{ts}.md"
    md_path.write_text(md, encoding="utf-8")

    html_path = OUTPUT_DIR / f"{ticker}_10-K_{ts}.html"
    html_path.write_text(markdown2.markdown(md, extras=["tables"]), encoding="utf-8")

    print(f"\n✓ Markdown written to {md_path}")
    print(f"✓ HTML written to {html_path}")


# ────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
