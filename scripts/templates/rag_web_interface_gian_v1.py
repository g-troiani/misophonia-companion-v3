#!/usr/bin/env python3
################################################################################
# File: scripts/rag_web_interface.py        (Supabase edition – v2025‑05‑06 R5)
################################################################################
"""
Interactive RAG demo (Flask) that queries a Supabase vector store
(`research_chunks`) instead of Firebase / Firestore.

R5  →  • robust metadata parsing (string → JSON → dict)
        • never throws on missing / malformed metadata
        • identical external API; front-end needs **no** changes
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

DOC_FIELDS = (
    "title, authors, year, journal, doi, pages, volume, issue"
)

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


sb = init_supabase()


# ──────────────────────── metadata / citation helpers ────────────────────── #

def _as_meta(obj: Any) -> Dict[str, Any]:
    """Best‑effort coercion to dict for the *metadata* column."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj) or {}
        except Exception:
            return {}
    return {}


def _first(meta: Dict[str, Any], keys: Sequence[str]):
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
        if isinstance(lst, str) and lst.strip() and not lst.lower().startswith("unknown"):
            return lst.split(",")[0].strip()
    return None


def _page(meta: Dict[str, Any]) -> str | None:
    if (p := _first(meta, ["page", "page_number"])):
        return str(p)
    if (rng := _first(meta, ["pages", "page_range"])):
        if isinstance(rng, list) and rng:
            return str(rng[0])
        if isinstance(rng, str):
            return rng.split("–")[0].split("-")[0]
    if meta.get("page_start") is not None:
        return str(meta["page_start"])
    return None


def make_citation(meta_raw: Any) -> str:
    meta = _as_meta(meta_raw)

    parts = []
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


def long_citation(meta_raw: Any) -> str:
    meta = _as_meta(meta_raw)

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


# ────────────────────────────── embedding helpers ────────────────────────── #

def generate_embedding(text: str) -> List[float] | None:
    try:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return resp.data[0].embedding
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        return None


# ─────────────────────────────── search logic ────────────────────────────── #

def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
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
            r["metadata"]   = _as_meta(r.get("metadata", {}))
            r["source"]     = make_citation(r["metadata"])
        return rows
    except Exception as exc:
        log.error("RPC search failed: %s", exc)
        return []


def comprehensive_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    return semantic_search(query, limit)


# ──────────────────────────── RAG generation ─────────────────────────────── #

def generate_rag_response(
    query: str,
    docs: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> str:
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
    except Exception as exc:
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

        # 1.  Bulk‑fetch document metadata
        doc_map: Dict[str, Dict[str, Any]] = {}
        doc_ids = {r["document_id"] for r in results if r.get("document_id")}
        if doc_ids:
            q = (
                sb.table("research_documents")
                  .select(f"id,{DOC_FIELDS}")
                  .in_("id", list(doc_ids))
                  .execute()
            )
            for d in q.data or []:
                doc_map[d["id"]] = d

        # 2.  Build front‑end payload
        sources = []
        for r in results:
            doc = doc_map.get(r.get("document_id"), {})

            authors = (
                ", ".join(doc["authors"])
                if isinstance(doc.get("authors"), list)
                else doc.get("authors")
            )

            page_range = (
                doc.get("pages")
                or f"{r.get('page_start', '?')}-{r.get('page_end', '?')}"
            )

            sources.append(
                {
                    "chunk":       r.get("text", ""),
                    "source":      r.get("source"),
                    "match_type":  r.get("match_type"),
                    "similarity":  r.get("similarity"),
                    "chunk_id":    r.get("chunk_id"),
                    "metadata": {
                        "title":      doc.get("title")   or "Unknown Title",
                        "section":    "Unknown Section",
                        "authors":    authors            or "Unknown Authors",
                        "journal":    doc.get("journal") or "Unknown Journal",
                        "year":       doc.get("year")    or "Unknown",
                        "volume":     doc.get("volume"),
                        "issue":      doc.get("issue"),
                        "doi":        doc.get("doi"),
                        "page_range": page_range,
                    },
                }
            )

        answer = generate_rag_response(query, results) or ""
        return jsonify({"results": sources, "response": answer})
    except Exception as e:
        logging.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    try:
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
            fp.write("""<!DOCTYPE html>
<!-- identical template as previous version omitted for brevity -->""")

    log.info("Starting Flask on 0.0.0.0:8080 …")
    app.run(host="0.0.0.0", port=8080, debug=True)
