#!/usr/bin/env python3
################################################################################
# File: scripts/test_vector_search.py
################################################################################
"""
Vector‑search smoke‑test (Supabase edition)
==========================================

• This script tests vector search functionality using Supabase's PostgREST API.

• The script calls a SQL helper function that must exist on your database:
    public.search_research_chunks(query_text TEXT,
                                  match_count INT DEFAULT 10,
                                  similarity_threshold REAL DEFAULT 0.6)
  which should:
    1. embed the incoming `query_text`
    2. invoke your `match_documents` similarity function
    3. return the top‑`match_count` rows as
       (id UUID, text TEXT, metadata JSONB, similarity REAL)

  See README / earlier instructions for a ready‑made implementation.

Environment variables required
------------------------------
SUPABASE_URL                 – e.g. https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY    – or an anon key if RLS permits the RPC
OPENAI_API_KEY               – only if your SQL helper embeds via an HTTP call
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ──────────────────────── configuration & sanity checks ───────────────────── #

load_dotenv()

# ---------- Supabase config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")   # or anon if RLS permits
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit(
        "❌  SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY env vars are missing.\n"
        "    export them and rerun."
    )

# Sample questions to probe the index
SAMPLE_QUERIES: List[str] = [
    "What are the symptoms of misophonia?",
    "How prevalent is misophonia in university students?",
    "What is the relationship between misophonia and hyperacusis?",
    "What treatments are effective for misophonia?",
    "How does misophonia affect quality of life?",
]

# Add this after other configuration
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str) -> List[float]:
    """Return OpenAI ada‑002 embedding (1536‑dim list of floats)."""
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return resp.data[0].embedding

# ─────────────────────────── helper functions ─────────────────────────────── #


def perform_vector_search(query_vec, top_k=5, thresh=0.6):
    """
    Call the SQL RPC we just created.
    `query_vec` is a list[float] length 1536 coming from OpenAI.
    """
    try:
        resp = sb.rpc(
            "search_research_chunks",
            {
                "query_embedding": query_vec,
                "match_count": top_k,
                "similarity_threshold": thresh,
            },
        ).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(resp.error)
        return resp.data or []
    except Exception as e:
        print(f"   ⚠  RPC failed: {e}")
        return []


def print_results(rows: List[Dict[str, Any]]) -> None:
    """
    Nicely format the search results.
    """
    if not rows:
        print("   (no matches)\n")
        return

    for idx, row in enumerate(rows, 1):
        meta = row.get("metadata", {}) or {}
        title = meta.get("title", "Unknown title")
        year = meta.get("year", "????")
        author = meta.get("primary_author", "Unknown author")
        sim = row.get("similarity", 0.0)

        snippet = (row.get("text", "") or "").replace("\n", " ")[:280] + "…"

        print(f"\nResult {idx}  •  sim={sim:.3f}")
        print(f"  {title} — {author} ({year})")
        print(f"  {snippet}")


# ────────────────────────────────── main ──────────────────────────────────── #


def main() -> None:
    print("\n🔍  Supabase vector search smoke‑test\n" + "—" * 60)
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f'\nQuery {i + 1}/{len(SAMPLE_QUERIES)}: "{q}"')


        # 1. get the vector
        print("Generating embedding...")
        query_vec = embed(q)  # Python list[float]

        # 2. PostgREST / Postgres expects a *string* like: [0.1,0.2,…]
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"

        # 3. call the RPC
        print("Performing vector search...")
        resp = sb.rpc(
            "search_research_chunks",
            {
                "query_embedding": vec_literal,  # <- NOT the raw text
                "match_count": 5,
                "similarity_threshold": 0.6,
            },
        ).execute()
        
        if getattr(resp, "error", None):
            print(f"   ⚠  RPC failed: {resp.error}\n")
            continue
            
        results = resp.data or []
        print_results(results)

        if i < len(SAMPLE_QUERIES):
            print("\nPausing 2 s before the next query …")
            time.sleep(2)

    print("\n✔  Done")


if __name__ == "__main__":
    main()
