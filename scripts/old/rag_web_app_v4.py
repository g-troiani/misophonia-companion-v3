#!/usr/bin/env python3
################################################################################
# File: rag_web_interface.py          (Supabase / pgvector demo – v2025‑05‑06)
################################################################################
"""
Mini Flask app that answers misophonia questions with Retrieval‑Augmented
Generation (GPT‑4o + Supabase pgvector).

### Patch 2  (2025‑05‑06)
• **Embeddings** now created with **text‑embedding‑ada‑002** (1536‑D).  
• Similarity is re‑computed client‑side with a **plain cosine function** so the
  ranking no longer depends on pgvector's built‑in distance or any RPC
  threshold quirks.

The rest of the grounded‑answer logic (added in Patch 1) is unchanged.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path        # (unused but left in to mirror original)
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from openai import OpenAI
from supabase import create_client

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


def semantic_search(
    query: str,
    *,
    limit: int = 8,
    threshold: float = 0.0,
) -> List[Dict]:
    """
    Retrieve candidate chunks through the existing pgvector RPC (fast ANN),
    **then** re‑rank them with an explicit cosine calculation so the ordering
    is 100 % transparent and reproducible.

    If you want to ignore pgvector entirely you could fetch all embeddings and
    compute locally, but for demo purposes the hybrid approach is far faster.
    """
    q_vec = embed(query)

    # First pass: ANN search in Postgres to narrow down to *limit*×4 rows.
    rows = (
        sb.rpc(
            "match_research_chunks",
            {
                "query_embedding": q_vec,
                "match_threshold": threshold,      # 0 returns everything pg finds
                "match_count": limit * 4,          # over‑fetch, we'll re‑rank
            },
        )
        .execute()
        .data
    )
    if not rows:
        return []

    # Second pass: join metadata & compute **plain cosine** ourselves
    for r in rows:
        # ensure we have the parent document
        if "doc" not in r:
            r["doc"] = (
                sb.table("research_documents")
                .select(
                    """
                    title, authors, year, journal, doi, abstract,
                    keywords, research_topics, source_pdf
                    """
                )
                .eq("id", r["document_id"])
                .single()
                .execute()
                .data
            )

        # pgvector RPC usually returns similarity & embedding already;
        # we overwrite similarity with our own cosine, falling back if needed
        if isinstance(r.get("embedding"), list):
            r["similarity"] = cosine_similarity(q_vec, r["embedding"])
        else:
            # If embedding isn't returned, use pg's value (already cosine dist)
            r["similarity"] = r.get("similarity", 0.0)

    # Order by our freshly computed similarity
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
    Prompt instructs GPT to:
    • answer with inline numeric citations ([1], [2]…)
    • finish with a full bibliography list, one line per source
    """
    snippet_lines, biblio_lines = [], []
    for i, c in enumerate(chunks, 1):
        snippet_lines.append(f"[{i}] \"{c['text'].strip()}\" (pp. {c['page_start']}-{c['page_end']})")

        d = c["doc"]
        authors = ", ".join(d.get("authors") or ["Unknown"])
        pages   = f"pp. {c['page_start']}-{c['page_end']}"
        doi     = f" DOI link" if d.get("doi") else ""
        biblio_lines.append(
            f"[{i}] {authors} · {d.get('journal', 'Unknown')} ({d.get('year','n.d.')}) · {pages}{doi}"
        )

    prompt = (
        "You are an expert assistant.\n\n"
        "Context chunks:\n" + "\n".join(snippet_lines) + "\n\n"
        "User question:\n" + question + "\n\n"
        "Instructions:\n"
        "• Answer strictly with information in the chunks.\n"
        "• Cite each chunk you use inline like [1], [2] …\n"
        "• After your answer, add a section titled \"Bibliography:\" that lists all "
        "sources in the same numeric order, one per line, exactly as supplied below.\n"
        "• If none of the chunks answer the question, say: "
        "\"I'm sorry, I don't have sufficient information to answer that.\"\n\n"
        "Bibliography lines (use verbatim):\n" + "\n".join(biblio_lines) + "\n\n"
        "Answer:"
    )
    return prompt


def extract_citations(answer: str) -> List[str]:
    """
    Parse numeric citations (e.g., "[1]", "[2]") from the answer text.
    Returns unique citation numbers in ascending order.
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(citations), key=int)


# ──────────────────────────────── routes ────────────────────────────────── #

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Misophonia RAG Demo</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet">
  <style>
    body { padding-top: 4rem; }
    #results .card + .card { margin-top: .5rem; }
  </style>
</head>
<body class="container">
  <h1 class="mb-4">Misophonia Research Q&A</h1>

  <form id="qform" class="row g-3">
    <div class="col-10">
      <input id="query" class="form-control" placeholder="Ask a question…" required>
    </div>
    <div class="col">
      <button class="btn btn-primary w-100">Search</button>
    </div>
  </form>

  <hr>
  <div id="answer" class="mb-4"></div>
  <div id="results"></div>

<script>
const form = document.getElementById('qform');
form.addEventListener('submit', async e => {
  e.preventDefault();
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  document.getElementById('answer').innerHTML = '<em>Loading…</em>';
  document.getElementById('results').innerHTML = '';
  try {
    const r = await fetch('/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q}),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    document.getElementById('answer').innerHTML =
      '<p>' + d.answer
               .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
               .replace(/\\n/g,'<br>')
           + '</p>';
    (d.results || []).forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-header">
          <strong>[${i+1}] ${s.doc.title || 'Untitled'}</strong>
          <small class="text-muted float-end">${(s.similarity*100).toFixed(1)} % sim.</small>
        </div>
        <div class="card-body">
          <blockquote class="blockquote mb-2">${s.text}</blockquote>
          <p class="mb-1">
            <em>
              ${s.doc.title || 'Untitled'} ·
              ${(s.doc.authors || []).join(', ') || 'Unknown authors'} ·
              ${s.doc.journal || ''} (${s.doc.year || 'n.d.'}) ·
              pp. ${s.page_start}-${s.page_end}
            </em>
          </p>
          ${s.doc.doi ? `<a href="https://doi.org/${s.doc.doi}" target="_blank">DOI link</a>` : ''}
        </div>`;
      document.getElementById('results').appendChild(card);
    });
  } catch (err) {
    console.error(err);
    document.getElementById('answer').textContent = 'Error: ' + err.message;
  }
});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


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
        return jsonify(
            {
                "answer": answer_text,
                "citations": citations,
                "results": raw_matches,
            }
        )
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
