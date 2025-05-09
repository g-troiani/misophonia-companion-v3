#!/usr/bin/env python3
################################################################################
# File: rag_web_app.py                 (Supabase / pgvector demo – v2025‑05‑06)
################################################################################
"""
Mini Flask app that answers misophonia questions with Retrieval‑Augmented
Generation (GPT‑4o + Supabase pgvector), now including page ranges in citations.

✓  Vector search in Postgres via an RPC (`match_research_chunks` returning pages)
✓  GPT‑4o answer with numeric citations
✓  Single‑file Bootstrap front‑end (auto‑served)

Run:

$ pip install flask python-dotenv supabase openai
$ export OPENAI_API_KEY=...
$ export SUPABASE_URL=...
$ export SUPABASE_SERVICE_ROLE_KEY=...
$ python rag_web_app.py
→ open http://127.0.0.1:8080
"""
from __future__ import annotations

import logging
import os
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
    """Return OpenAI embedding vector for *text*."""
    resp = oa.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8192],  # safety slice
    )
    return resp.data[0].embedding


def semantic_search(
    query: str,
    *,
    limit: int = 8,
    threshold: float = 0.78,
) -> List[Dict]:
    """Top‑*k* nearest chunks, enriched with their document metadata and page info."""
    vec = embed(query)
    rows = (
        sb.rpc(
            "match_research_chunks",        # RPC returning page_start & page_end
            {
                "query_embedding": vec,
                "match_threshold": threshold,
                "match_count": limit,
            },
        )
        .execute()
        .data
    )
    if not rows:
        return []

    # attach parent‑document metadata
    for r in rows:
        doc = (
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
        r["doc"] = doc
    return rows


def build_prompt(question: str, sources: List[Dict]) -> str:
    """Assemble a compact RAG prompt with numbered snippets + full bib entries incl. pages."""
    snippets = []
    bib = []
    for i, s in enumerate(sources, start=1):
        page_info = f"(pp. {s['page_start']}-{s['page_end']})" if s.get('page_start') is not None else ""
        snippets.append(f"[{i}] “{s['text'].strip()}” {page_info}")
        d = s["doc"]
        authors = ", ".join(d.get("authors") or ["Unknown"])
        pages = f"{s['page_start']}-{s['page_end']}" if s.get('page_start') is not None else ""
        bib.append(
            f"[{i}] {authors} ({d.get('year','n.d.')}). *{d.get('title','Unknown')}*, "
            f"{d.get('journal','Unknown journal')}, pp. {pages}"
            + (f", DOI: {d['doi']}" if d.get('doi') else "")
        )
    return (
        "You are an academic assistant specialised in misophonia. Answer the question **only**"
        " using the excerpts; cite sources like [1], [2] …\n\n"
        f"Question:\n{question}\n\n"
        "Excerpts:\n" + "\n".join(snippets) +
        "\n\nBibliography:\n" + "\n".join(bib) +
        "\n\nAnswer:"
    )


def generate_answer(prompt: str) -> str:
    """Return GPT‑4o completion text."""
    res = oa.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return res.choices[0].message.content


# ──────────────────────────────── routes ────────────────────────────────── #

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Misophonia RAG Demo</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet">
  <style>
    body { padding-top: 4rem; }
    #results .card + .card { margin-top: .5rem; }
  </style>
</head>
<body class="container">
  <h1 class="mb-4">Misophonia Research Q&A</h1>

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
      '<p>' + d.response.replace(/\\n/g,'<br>') + '</p>';
    d.results.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-header">
          <strong>[${i+1}] ${s.doc.title || 'Untitled'}</strong>
          <small class="text-muted float-end">${(s.similarity*100).toFixed(1)} % sim.</small>
        </div>
        <div class="card-body">
          <blockquote class="blockquote mb-2">${s.text}</blockquote>
          <p class="mb-1">
            <em>${(s.doc.authors || []).join(', ') || 'Unknown authors'} · ${s.doc.journal || ''} (${s.doc.year || 'n.d.'}) · pp. ${s.page_start}-${s.page_end}</em>
          </p>
          ${s.doc.doi ? `<a href="https://doi.org/${s.doc.doi}" target="_blank">DOI link</a>` : ''}
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
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        matches = semantic_search(query, limit=int(payload.get("limit", 6)))
        if not matches:
            return jsonify({"response": "No relevant sources found.", "results": []})
        answer = generate_answer(build_prompt(query, matches))

        # remove heavy data before sending
        for m in matches:
            m.pop("embedding", None)

        return jsonify({"response": answer, "results": matches})
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
