#!/usr/bin/env python3
################################################################################
# File: rag_web_interface.py          (Supabase / pgvector demo – v2025‑05‑06)
################################################################################
"""
Mini Flask app that answers misophonia questions with Retrieval‑Augmented
Generation (gpt-4.1-mini-2025-04-14 + Supabase pgvector).

### Patch 2  (2025‑05‑06)
• **Embeddings** now created with **text‑embedding‑ada‑002** (1536‑D).  
• Similarity is re‑computed client‑side with a **plain cosine function** so the
  ranking no longer depends on pgvector's built‑in distance or any RPC
  threshold quirks.

The rest of the grounded‑answer logic (added in Patch 1) is unchanged.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path        # (unused but left in to mirror original)
from typing import Dict, List
import json

from dotenv import load_dotenv
from flask import Flask, jsonify, request, make_response
from openai import OpenAI
from supabase import create_client
from flask_compress import Compress
from flask_cors import CORS

# ────────────────────────────── configuration ───────────────────────────── #

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PORT           = int(os.getenv("PORT", 8080))

if not (OPENAI_API_KEY and SUPABASE_URL and SUPABASE_KEY):
    raise SystemExit(
        "❌  Required env vars: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger("rag_app")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)
app.config['COMPRESS_ALGORITHM'] = 'gzip'
Compress(app)

# ────────────────────────────── helper functions ────────────────────────── #


def embed(text: str) -> List[float]:
    """
    Return OpenAI embedding vector for *text* using text‑embedding‑ada‑002.

    ada‑002 has 1536 dimensions and is inexpensive yet solid for similarity.
    """
    resp = oa.embeddings.create(
        model="text-embedding-ada-002",
        input=text[:8192],  # safety slice
    )
    return resp.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Plain cosine similarity between two equal‑length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


# Add in-memory embedding cache
_qcache = {}
def embed_cached(text):
    if text in _qcache: return _qcache[text]
    vec = embed(text)
    _qcache[text] = vec
    return vec


# Add regex patterns for bibliography detection
_DOI_RE   = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', re.I)
_YEAR_RE  = re.compile(r'\b(19|20)\d{2}\b')

def looks_like_refs(text: str) -> bool:
    """
    Return True if this chunk is likely just a bibliography list:
      • more than 12 DOIs, or
      • more than 15 year mentions.
    """
    doi_count  = len(_DOI_RE.findall(text))
    year_count = len(_YEAR_RE.findall(text))
    return doi_count > 12 or year_count > 15


def semantic_search(
    query: str,
    *,
    limit: int = 8,
    threshold: float = 0.0,
) -> List[Dict]:
    """
    Retrieve candidate chunks via the pgvector RPC, then re-rank with an
    **explicit cosine similarity** so the final score is always in
    **[-100 … +100] percent**.

    Why the extra work?
    -------------------
    •  The SQL function returns a raw inner-product that can be > 1.  
       (embeddings are *not* unit-length.)  
    •  By pulling the real 1 536-D vectors and re-computing a cosine we get a
       true, bounded similarity that front-end code can safely show.

    The -100 … +100 range is produced by:  
        pct = clamp(cosine × 100, -100, 100)
    """
    # 1. Embed the query once and keep it cached
    q_vec = embed_cached(query)

    # 2. Fast ANN search in Postgres (over-fetch 4× so we can re-rank)
    rows = (
        sb.rpc(
            "match_research_chunks",
            {
                "query_embedding": q_vec,
                "match_threshold": threshold,
                "match_count": limit * 4,
            },
        )
        .execute()
        .data
    ) or []

    # 3. Filter out bibliography-only chunks
    rows = [r for r in rows if not looks_like_refs(r["text"])]

    if not rows:
        return []

    # 4. Fetch document metadata (title, authors …) in one round-trip
    doc_ids = {r["document_id"] for r in rows}
    meta = {
        d["id"]: d
        for d in (
            sb.table("research_documents")
              .select("id,title,authors,year,journal,doi,abstract,keywords,research_topics,source_pdf")
              .in_("id", list(doc_ids))
              .execute()
              .data
            or []
        )
    }

    # 5. Pull embeddings once and compute **plain cosine** (no scaling)
    chunk_ids = [r["id"] for r in rows]

    emb_rows = (
        sb.table("research_chunks")
          .select("id, embedding")
          .in_("id", chunk_ids)
          .execute()
          .data
    ) or []

    emb_map: Dict[str, List[float]] = {}
    for e in emb_rows:
        raw = e["embedding"]
        if isinstance(raw, list):                    # list[Decimal]
            emb_map[e["id"]] = [float(x) for x in raw]
        elif isinstance(raw, str) and raw.startswith('['):   # TEXT  "[…]"
            emb_map[e["id"]] = [float(x) for x in raw.strip('[]').split(',')]

    for r in rows:
        vec = emb_map.get(r["id"])
        if vec:                                     # we now have the real vector
            cos = cosine_similarity(q_vec, vec)
            r["similarity"] = round(cos * 100, 1)   # –100…+100 % (or 0…100 %)
        else:                                       # fallback if something failed
            dist = float(r.get("similarity", 1.0))  # 0…2 cosine-distance
            r["similarity"] = round((1.0 - dist) * 100, 1)

        r["doc"] = meta.get(r["document_id"], {})

    # 6. Keep the top *limit* rows after proper re-ranking
    ranked = sorted(rows, key=lambda x: x["similarity"], reverse=True)[:limit]
    return ranked


# ──────────────────────── NEW RAG‑PROMPT HELPERS ───────────────────────── #

MAX_PROMPT_CHARS: int = 24_000  # ~6 k tokens @ 4 chars/token heuristic


def trim_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Fail‑safe guard: ensure concatenated chunk texts remain under the
    MAX_PROMPT_CHARS budget.  Keeps highest‑similarity chunks first.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)
    output: List[Dict] = []
    total_chars = 0
    for c in sorted_chunks:
        chunk_len = len(c["text"])
        if total_chars + chunk_len > MAX_PROMPT_CHARS:
            break
        output.append(c)
        total_chars += chunk_len
    return output


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Build a structured prompt that asks GPT to:
      • answer in Markdown with short intro + numbered list of key points
      • cite inline like [1], [2] …
      • finish with a Bibliography that includes the *paper title*
    """
    snippet_lines, biblio_lines = [], []
    for i, c in enumerate(chunks, 1):
        snippet_lines.append(
            f"[{i}] \"{c['text'].strip()}\" "
            f"(pp. {c['page_start']}-{c['page_end']})"
        )

        d = c["doc"]
        title   = d.get("title", "Untitled")
        authors = ", ".join(d.get("authors") or ["Unknown"])
        journal = d.get("journal", "Unknown journal")
        year    = d.get("year", "n.d.")
        pages   = f"pp. {c['page_start']}-{c['page_end']}"
        doi_raw = d.get("doi")
        doi_md  = f"[doi:{doi_raw}](https://doi.org/{doi_raw})" if doi_raw else ""

        # Title now comes first ↓↓↓
        biblio_lines.append(
            f"[{i}] *{title}* · {authors} · {journal} ({year}) · {pages} {doi_md}"
        )

    prompt_parts = [
        "You are Misophonia Companion, a highly knowledgeable and empathetic AI assistant built to support clinicians, researchers, and individuals managing misophonia.",
        "You draw on evidence from peer-reviewed literature, clinical guidelines, and behavioral science.",
        "Your responses are clear, thoughtful, and grounded in the provided context.",
        "====",
        "QUESTION:",
        question,
        "====",
        "CONTEXT:",
        *snippet_lines,
        "====",
        "INSTRUCTIONS:",
        "• Write your answer in **Markdown**.",
        "• Begin with a concise summary (2–3 sentences).",
        "• Then elaborate on key points using well-structured paragraphs.",
        "• Provide relevant insights or suggestions (e.g., clinical, behavioral, emotional, or research-related).",
        "• If helpful, use lists, subheadings, or analogies to enhance understanding.",
        "• Use a professional and empathetic tone.",
        "• Cite sources inline like [1], [2] etc.",
        "• After the answer, include a 'BIBLIOGRAPHY:' section that lists each source exactly as provided below.",
        "• If none of the context answers the question, reply: \"I'm sorry, I don't have sufficient information to answer that.\"",
        "====",
        "BEGIN OUTPUT",
        "ANSWER:",
        "",  # where the model writes the main response
        "BIBLIOGRAPHY:",
        *biblio_lines,
    ]


    return '\n'.join(prompt_parts)


def extract_citations(answer: str) -> List[str]:
    """
    Parse numeric citations (e.g., "[1]", "[2]") from the answer text.
    Returns unique citation numbers in ascending order.
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(citations), key=int)


# ──────────────────────────────── routes ────────────────────────────────── #

@app.post("/search")
def search():
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        # Retrieve semantic matches (client‑side cosine re‑ranked)
        raw_matches = semantic_search(question, limit=int(payload.get("limit", 8)))

        if not raw_matches:
            return jsonify(
                {
                    "answer": "I'm sorry, I don't have sufficient information to answer that.",
                    "citations": [],
                    "results": [],
                }
            )

        # ──────────────────── TRIM CHUNKS TO BUDGET ──────────────────── #
        chunks = trim_chunks(raw_matches)

        # ──────────────────── BUILD PROMPT & CALL LLM ─────────────────── #
        prompt = build_prompt(question, chunks)

        completion = oa.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer_text: str = completion.choices[0].message.content.strip()

        # ──────────────────── EXTRACT CITATIONS ──────────────────────── #
        citations = extract_citations(answer_text)

        # Remove embedding vectors before sending back to the browser
        for m in raw_matches:
            m.pop("embedding", None)

        # ──────────────────── RETURN JSON ─────────────────────────────── #
        response = jsonify(
            {
                "answer": answer_text,
                "citations": citations,
                "results": raw_matches,
            }
        )
        response.headers['Connection'] = 'keep-alive'
        return response
    except Exception as exc:  # noqa: BLE001
        log.exception("search failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/stats")
def stats():
    """Tiny ops endpoint—count total chunks."""
    resp = sb.table("research_chunks").select("id", count="exact").execute()
    return jsonify({"total_chunks": resp.count})


# ──────────────────────────────── main ─────────────────────────────────── #

if __name__ == "__main__":
    log.info("Starting Flask on 0.0.0.0:%s …", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=True)
