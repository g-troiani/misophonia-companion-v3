#!/usr/bin/env python3
###############################################################################
# File: scripts/resilient_batch_embedding_generator.py
###############################################################################
"""
Resilient Batch Embedding Generator — Supabase edition (token‑safe v3‑fix)
==========================================================================

• Pulls rows from `research_chunks` whose `embedding` is NULL
  **and** `chunking_strategy = 'token_window'`.
• Uses the *actual* token count stored in each row, so even weird
  `/C111/C102/…` glyph dumps are handled correctly.
• Sends slices that respect both:
      – a *row* cutoff (`--commit`, default 100) and
      – the model's 8 192‑token limit (we stop at ≤ 8 000).
• UPSERTs vectors back to Supabase and writes a JSON run‑report.

Environment
-----------
SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY  must be set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm

# ───────────────────────────── configuration ────────────────────────────── #

load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL  = "text-embedding-ada-002"

MODEL_TOKEN_LIMIT = 8_192         # model hard limit
TOKEN_GUARD       = 200           # safety margin
MAX_TOTAL_TOKENS  = MODEL_TOKEN_LIMIT - TOKEN_GUARD   # 8 000

DEFAULT_BATCH_ROWS  = 5_000       # rows fetched from Supabase per outer loop
DEFAULT_COMMIT_ROWS = 100         # max rows per single OpenAI request

MAX_RETRIES = 5
RETRY_DELAY = 2                   # seconds between retries

openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ───────────────────────────── Supabase helpers ─────────────────────────── #

def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def count_processed_chunks(sb) -> int:
    res = (
        sb.table("research_chunks")
        .select("id", count="exact")
        .not_.is_("embedding", "null")
        .execute()
    )
    return res.count or 0


def fetch_unprocessed_chunks(
    sb, *, limit: int, offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Grab up to *limit* rows whose embedding IS NULL.
    We also fetch token_start/end so we can trust the real token length.
    """
    first, last = offset, offset + limit - 1          # PostgREST uses inclusive range
    res = (
        sb.table("research_chunks")
        .select("id,text,token_start,token_end")
        .eq("chunking_strategy", "token_window")
        .is_("embedding", "null")
        .range(first, last)
        .execute()
    )
    return res.data or []

# ───────────────────────────  OpenAI interaction  ───────────────────────── #

def generate_embedding_batch(texts: List[str]) -> List[List[float]]:
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            resp = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as exc:
            attempt += 1
            log.warning("Embedding batch %s/%s failed: %s",
                        attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_DELAY)
    raise RuntimeError("OpenAI embedding batch failed after retries")

# ───────────────────────── token‑safe slicing helpers ────────────────────── #

def chunk_tokens(row: Dict[str, Any]) -> int:
    """
    Return the best‑guess token length for a chunk.

    Priority:
    1.  Use token_end – token_start + 1  *if present and plausible*.
        (Some PDFs have token_end values that balloon into 80‑90 k range;
         we treat anything 0 < t ≤ 16 384 as plausible.)
    2.  Fallback: estimate by words × 0.75, which is roughly the OpenAI
        tokenizer ratio for English prose.  Cap to the model limit so we
        never falsely mark a chunk as oversize.
    """
    try:
        t = int(row["token_end"]) - int(row["token_start"]) + 1
        if 0 < t <= 16_384:        # sane bound (2× GPT‑4o context)
            return t
    except Exception:
        pass

    txt = row.get("text", "") or ""
    approx = int(len(txt.split()) * 0.75) + 1
    return min(max(1, approx), MODEL_TOKEN_LIMIT)   # never exceed limit


def safe_slices(
    rows: List[Dict[str, Any]],
    max_rows: int,
) -> List[List[Dict[str, Any]]]:
    """
    Break *rows* into slices obeying BOTH limits:
        • len(slice) ≤ max_rows
        • Σ tokens(slice) ≤ MAX_TOTAL_TOKENS
    """
    slices: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    cur_tokens = 0

    for r in rows:
        txt = (r.get("text") or "").replace("\x00", "")
        if not txt.strip():
            continue

        t = chunk_tokens(r)

        if t > MAX_TOTAL_TOKENS:
            log.warning("Chunk %s is > %s tokens — skipping.", r["id"], MAX_TOTAL_TOKENS)
            continue

        need_new_slice = (
            len(current) >= max_rows or
            cur_tokens + t > MAX_TOTAL_TOKENS
        )
        if need_new_slice and current:
            slices.append(current)
            current = []
            cur_tokens = 0

        current.append({"id": r["id"], "text": txt})
        cur_tokens += t

    if current:
        slices.append(current)
    return slices


def embed_slice(sb, slice_rows: List[Dict[str, Any]]) -> int:
    """Generate embeddings for *slice_rows* and UPSERT them back. Returns #rows updated."""
    embeds = generate_embedding_batch([r["text"] for r in slice_rows])
    ok = 0
    for row, emb in zip(slice_rows, embeds):
        attempt = 0
        while attempt < MAX_RETRIES:
            res = (
                sb.table("research_chunks")
                .update({"embedding": emb})
                .eq("id", row["id"])
                .execute()
            )
            if getattr(res, "error", None):
                attempt += 1
                log.warning("Update %s failed (%s/%s): %s",
                            row["id"], attempt, MAX_RETRIES, res.error)
                time.sleep(RETRY_DELAY)
            else:
                ok += 1
                break
    return ok

# ────────────────────────────────── main ────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embed research_chunks where embedding IS NULL (token‑safe)"
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_ROWS,
                    help="rows fetched from Supabase per outer loop (default 5000)")
    ap.add_argument("--commit", type=int, default=DEFAULT_COMMIT_ROWS,
                    help="row‑cutoff per single OpenAI request (default 100)")
    ap.add_argument("--skip", type=int, default=0,
                    help="row offset for pagination / parallel runs")
    ap.add_argument("--model", default=EMBEDDING_MODEL,
                    help="override embedding model name")
    ap.add_argument("--model-limit", type=int, default=MODEL_TOKEN_LIMIT,
                    help="override model token limit (hard max context)")
    args = ap.parse_args()

    # allow quick overrides without editing the file
    global EMBEDDING_MODEL, MODEL_TOKEN_LIMIT, MAX_TOTAL_TOKENS
    EMBEDDING_MODEL   = args.model
    MODEL_TOKEN_LIMIT = args.model_limit
    MAX_TOTAL_TOKENS  = MODEL_TOKEN_LIMIT - TOKEN_GUARD

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    sb = init_supabase()
    log.info("Rows already embedded: %s", count_processed_chunks(sb))

    loop = 0
    total_embedded = 0
    while True:
        loop += 1
        rows = fetch_unprocessed_chunks(sb, limit=args.batch_size, offset=args.skip)
        if not rows:
            log.info("✨  Done — no more rows needing embeddings.")
            break

        log.info("Loop %s — fetched %s rows needing embeddings.", loop, len(rows))
        slices = safe_slices(rows, args.commit)
        log.info("Created %s token‑safe OpenAI requests.", len(slices))

        for sl in tqdm(slices, desc=f"Embedding loop {loop}", unit="batch"):
            try:
                total_embedded += embed_slice(sb, sl)
            except Exception as exc:
                log.error("Slice failed: %s", exc)

        log.info("Loop %s complete – total embedded so far: %s", loop, total_embedded)

    # -------------  JSON report  ------------- #
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "batch_size": args.batch_size,
        "commit": args.commit,
        "skip": args.skip,
        "total_embedded": total_embedded,
        "total_with_embeddings": count_processed_chunks(sb),
    }
    fname = f"supabase_embedding_report_{ts}.json"
    with open(fname, "w") as fp:
        json.dump(report, fp, indent=2)
    log.info("Report saved to %s", fname)


if __name__ == "__main__":        # pragma: no cover
    main()
