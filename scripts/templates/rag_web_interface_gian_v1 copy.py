#!/usr/bin/env python3
################################################################################
# File: scripts/rag_web_interface.py        (Supabase edition – v2025‑05‑06 R2)
################################################################################
"""
Interactive RAG demo (Flask) that queries a Supabase vector store
(`research_chunks`) instead of Firebase / Firestore.

Changes in R2 ­­­→  ▸ removes the /stats endpoint + front‑end fetch  
                   ▸ cleaner citation helpers (no "Unknown" placeholders)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from supabase import create_client


# ───────────────────────────── configuration ───────────────────────────── #

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


# ------------------------------------------------------------------
# Bibliographic fields we want to pull from research_documents
DOC_FIELDS = (
    "title, authors, year, journal, doi"
)

# All columns we need from research_chunks  +  the join
CHUNK_SELECT = (
    "text, page_start, page_end, "
    f"research_documents({DOC_FIELDS})"
)
# ------------------------------------------------------------------

if not OPENAI_API_KEY:
    print("⛔  OPENAI_API_KEY missing — aborting")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)


# ───────────────────────────── Supabase helpers ─────────────────────────── #

def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


sb = init_supabase()        # single client reused everywhere


# ──────────────────────── metadata / citation helpers ────────────────────── #

def _first(meta: Dict[str, Any], keys: Sequence[str]):
    """Return first non‑empty, non‑placeholder value in *meta* for any of *keys*."""
    for k in keys:
        v = meta.get(k)
        if v and str(v).strip() and not str(v).lower().startswith("unknown"):
            return v
    return None


def _first_author(meta: Dict[str, Any]) -> str | None:
    if (a := _first(meta, ["primary_author", "author", "creator"])):
        return str(a)
    if (lst := meta.get("authors")):
        if isinstance(lst, list) and lst:
            return str(lst[0])
        # if it's a string "Foo, Bar" take first token
        if isinstance(lst, str) and lst.strip() and not lst.lower().startswith("unknown"):
            return lst.split(",")[0].strip()
    return None


def _page(meta: Dict[str, Any]) -> str | None:
    if (p := _first(meta, ["page", "page_number"])):
        return str(p)
    if (rng := _first(meta, ["pages", "page_range"])):
        # Accept list or "12–15" / "12-15"
        if isinstance(rng, list) and rng:
            return str(rng[0])
        if isinstance(rng, str):
            return rng.split("–")[0].split("-")[0]
    return None


def make_citation(meta: Dict[str, Any]) -> str:
    """
    Compact citation for UI cards – omits any empty element.
    Format:  Author Year – Title. Journal. doi:… p.X
    """
    parts: list[str] = []

    if (a := _first_author(meta)):
        parts.append(a)

    if (y := _first(meta, ["year", "pub_year"])):
        parts.append(str(y))

    if (t := _first(meta, ["title", "document_title", "section"])):
        parts.append(f"– {t.strip()}")

    if (j := _first(meta, ["journal", "source"])):
        parts.append(j)

    if (doi := _first(meta, ["doi"])):
        parts.append(f"doi:{doi}")

    if (pg := _page(meta)):
        parts.append(f"p.{pg}")

    return " ".join(parts) if parts else "No citation metadata"


def long_citation(meta: Dict[str, Any]) -> str:
    """
    Longer citation fed to GPT (always includes Title if available).
    """
    auth  = _first_author(meta) or ""
    year  = _first(meta, ["year", "pub_year"]) or ""
    title = _first(meta, ["title", "document_title", "section"]) or ""
    jour  = _first(meta, ["journal", "source"]) or ""
    doi   = _first(meta, ["doi"]) or ""
    pg    = _page(meta) or ""

    bits = []
    if auth or year:
        bits.append(f"{auth} ({year}).".strip())
    if title:
        bits.append(title)
    if jour:
        bits.append(jour)
    if doi:
        bits.append(f"doi:{doi}")
    if pg:
        bits.append(f"p.{pg}")

    return " ".join(bits)


# keep alias for any legacy code
pretty_source = make_citation


# ────────────────────────────── embedding helpers ────────────────────────── #

def generate_embedding(text: str) -> List[float] | None:
    """Generate an Ada‑002 embedding for *text*."""
    try:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return resp.data[0].embedding
    except Exception as exc:                                  # pragma: no cover
        log.error("Embedding failed: %s", exc)
        return None


# ─────────────────────────────── search logic ────────────────────────────── #

def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Server‑side vector search via RPC.
    """
    vec = generate_embedding(query)
    if vec is None:
        return []

    try:
        resp = (
            sb.rpc(
                "search_research_chunks",
                {
                    "query_embedding": vec,
                    "match_count": limit,
                    "similarity_threshold": 0.6,
                },
            ).execute()
        )
        rows = resp.data or []
        for r in rows:
            r["similarity"] = float(r.get("similarity", 0.0))
            r["match_type"] = f"Semantic ({r['similarity']:.4f})"
            r["chunk_id"]   = r.pop("id", "unknown")
            r["source"]     = make_citation(r.get("metadata", {}))
        return rows
    except Exception as exc:
        log.error("RPC search failed: %s", exc)
        return []


def keyword_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Simple client‑side keyword fallback over a capped set of chunks.
    """
    try:
        resp = (
            sb.table("research_chunks")
            .select(CHUNK_SELECT)
            .limit(2_000)
            .execute()
        )
        all_chunks = resp.data or []
    except Exception as exc:
        log.error("Keyword search fetch failed: %s", exc)
        return []

    stop  = {"the","and","or","in","of","to","a","is","that","for","on","with"}
    terms = {t for t in query.lower().split() if t not in stop}
    if not terms:
        return []

    results: List[Dict[str, Any]] = []
    for ch in all_chunks:
        txt = str(ch.get("text", "")).lower()
        if not txt:
            continue

        matches = sum(1 for t in terms if t in txt)
        if matches == 0:
            continue

        score = matches / len(terms)
        if query.lower() in txt:
            score += 0.3
        if score < 0.3:
            continue

        meta = ch.get("metadata") or {}
        results.append(
            {
                "chunk_id": meta.get("chunk_id", "unknown"),
                "text": ch.get("text", ""),
                "metadata": meta,
                "similarity": float(score),
                "match_type": f"Keyword ({score:.4f})",
                "source": make_citation(meta),
            }
        )

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:limit]


def comprehensive_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    sem = semantic_search(query, limit)
    kw  = keyword_search(query, limit)

    combined: List[Dict[str, Any]] = []
    seen = set()

    for r in sem + kw:
        cid = r.get("chunk_id")
        if cid not in seen:
            combined.append(r)
            seen.add(cid)

    combined.sort(key=lambda r: r["similarity"], reverse=True)
    return combined[:limit]


# ──────────────────────────── RAG generation ─────────────────────────────── #

def generate_rag_response(
    query: str,
    docs: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> str:
    """Produce the final GPT‑4o answer with inline citations."""
    try:
        snippets = [
            f"Source {i} — {long_citation(d.get('metadata', {}))}\n{d.get('text','')}\n"
            for i, d in enumerate(docs, 1)
        ]

        prompt = f"""You are a helpful AI assistant specialised in misophonia research.

Answer the question below using *only* the provided research snippets.

Question: "{query}"

Snippets:
{''.join(snippets)}

Give a concise, structured answer with numbered citations like [1], [2] … .
If evidence is insufficient, say so explicitly."""
        comp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You answer misophonia questions strictly from the provided sources, citing them.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return comp.choices[0].message.content
    except Exception as exc:                                                  # pragma: no cover
        log.error("RAG generation failed: %s", exc)
        return f"Error generating response: {exc}"


# ───────────────────────────── Flask endpoints ───────────────────────────── #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not data.get("query"):
            return jsonify({"error": "No query provided"}), 400

        query = data["query"].strip()
        limit = int(data.get("limit", 5))

        results = comprehensive_search(query, limit)

        if not results:
            return jsonify(
                {"error": "No relevant documents found for your query", "results": []}
            ), 404

        # After you get results from the database:
        sources = []

        for r in results:
            # PostgREST nests the 1‑to‑1 join as an OBJECT, not a list
            doc = r.get("research_documents") or {}

            # authors is a Postgres text[] → Python list; stringify for HTML
            authors = ", ".join(doc["authors"]) if isinstance(doc.get("authors"), list) else doc.get("authors")

            # page range: prefer the 'pages' column; otherwise build it
            page_range = (
                doc.get("pages")
                or f"{r.get('page_start', '?')}-{r.get('page_end', '?')}"
            )

            sources.append(
                {
                    "chunk"      : r.get("text", ""),         # the snippet
                    "title"      : doc.get("title")   or "Unknown Title",
                    "section"    : "Unknown Section",    # add later if you store it
                    "authors"    : authors            or "Unknown Authors",
                    "journal"    : doc.get("journal") or "Unknown Journal",
                    "year"       : doc.get("year")    or "Unknown",
                    "volume"     : doc.get("volume"),
                    "issue"      : doc.get("issue"),
                    "doi"        : doc.get("doi"),
                    "page_range" : page_range,
                }
            )

        answer = generate_rag_response(query, results)
        return jsonify({"results": sources, "response": answer})
    except Exception as e:
        logging.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    try:
        # 1) total rows in the vector table ----------------------------
        total_chunks = (
            sb.table("research_chunks")
              .select("id", count="exact")
              .execute()
              .count
            or 0
        )

        return jsonify({"total_chunks": total_chunks}), 200

    except Exception as e:
        logging.exception("Stats endpoint failed")
        return jsonify({"error": str(e)}), 500


# ────────────────────────────── bootstrap html ───────────────────────────── #

if __name__ == "__main__":
    tpl_dir   = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    index_path = os.path.join(tpl_dir, "index.html")

    if not os.path.exists(index_path):
        with open(index_path, "w") as fp:
            fp.write(
"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Misophonia Research RAG Interface</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{padding:20px;background:#f8f9fa}
    .container{max-width:1000px;margin:0 auto}
    .header{text-align:center;margin-bottom:30px}
    .search-box{margin-bottom:20px}
    .results-container{margin-top:20px}
    .result-card{margin-bottom:15px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.1)}
    .result-card .card-header{font-weight:bold;display:flex;justify-content:space-between}
    .loading{text-align:center;padding:20px;display:none}
    .response-container{margin-top:30px;padding:20px;background:#fff;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.1)}
    pre { background:#f8f9fa;border:0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Misophonia Research RAG Interface</h1>
      <p class="text-muted">Search across research documents with semantic and keyword retrieval</p>
    </div>

    <div class="search-box">
      <div class="input-group mb-3">
        <input type="text" id="search-input" class="form-control form-control-lg"
               placeholder="Ask a question about misophonia…" aria-label="Search query">
        <button class="btn btn-primary" type="button" id="search-button">Search</button>
      </div>
      <div class="form-text">Try questions about treatments, neurological basis, symptoms, or coping strategies</div>
    </div>

    <div class="loading" id="loading">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading…</span>
      </div>
      <p>Searching research documents and generating response…</p>
    </div>

    <div id="response-area" style="display:none">
      <div class="response-container">
        <h3>Research‑Based Answer</h3>
        <div id="response-content"></div>
      </div>

      <div class="results-container">
        <h3>Source Documents</h3>
        <div id="results-list"></div>
      </div>
    </div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded",()=>{
  const $q      = document.getElementById("search-input");
  const $btn    = document.getElementById("search-button");
  const $load   = document.getElementById("loading");
  const $resp   = document.getElementById("response-area");
  const $respCt = document.getElementById("response-content");
  const $list   = document.getElementById("results-list");

  async function search(){
    const query=$q.value.trim();
    if(!query) return;

    $load.style.display="block";
    $resp.style.display="none";

    try{
      const res=await fetch("/search",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({query,limit:5})
      });
      if(!res.ok){
        let msg=`Server error (${res.status})`;
        try{msg=(await res.json()).error||msg;}catch{}
        throw new Error(msg);
      }
      const data=await res.json();
      const answerHtml=String(data.response||"").replace(/\\n/g,"<br>");
      $respCt.innerHTML=`<p>${answerHtml}</p>`;
      $list.innerHTML="";

      (data.results||[]).forEach((r,i)=>{
        const card=document.createElement("div");
        card.className="card result-card";
        const src = data.results[i].metadata;
        card.innerHTML=`
          <div class="card-header bg-light d-flex justify-content-between align-items-center"
               data-bs-toggle="collapse" data-bs-target="#chunk-${i}">
            <span>${r.source}</span>
            <span class="badge bg-primary opacity-75">${r.match_type}</span>
          </div>
          <div id="chunk-${i}" class="collapse show">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-baseline">
                <strong>
                  ${src.journal} (${src.year})
                </strong>
                ${src.doi ? `<a href="https://doi.org/${src.doi}" target="_blank" class="badge bg-primary">DOI</a>` : ''}
              </div>

              <h4 class="card-title">${src.title}</h4>
              <small class="text-muted">
                ${src.section}
              </small>

              <p class="mb-0">
                <em>${src.authors}</em><br>
                ${src.volume ? `${src.journal} ${src.volume}${src.issue ? `(${src.issue})` : ''}:` : ''}
                ${src.page_range}
              </p>
              
              <div class="mb-2 small text-muted">Similarity: ${(r.similarity??0).toFixed(4)}</div>
              <pre class="small text-muted mb-0" style="white-space:pre-wrap;">${r.text||""}</pre>
            </div>
          </div>`;
        $list.appendChild(card);
      });

    }catch(err){
      $respCt.innerHTML=`<div class="alert alert-warning">${err.message}</div>`;
      $list.innerHTML="";
    }finally{
      $load.style.display="none";
      $resp.style.display="block";
    }
  }

  $btn.addEventListener("click",search);
  $q.addEventListener("keypress",e=>{if(e.key==="Enter")search();});
});
</script>
</body>
</html>"""
            )

    log.info("Starting Flask on 0.0.0.0:8080 …")
    app.run(host="0.0.0.0", port=8080, debug=True)
