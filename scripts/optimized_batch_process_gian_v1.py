#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  File: scripts/optimized_batch_process.py   (token‑window v3: NUL‑safe)
# ──────────────────────────────────────────────────────────────────────────────
"""
Batch‑upload pre‑parsed research papers into Supabase
using a fixed‑token sliding window (768 tokens, 20 % overlap).

INPUT  directories
──────────────────
documents/research/json   one JSON per paper (metadata + per‑page text)
documents/research/txt    flat .txt version (kept for future re‑chunking)

DB schema expected
──────────────────
  research_documents  (one row per paper)
  research_chunks     (token_window strategy)

Chunking parameters
───────────────────
  • window      768 tokens
  • overlap     154 tokens   (20 %)
  • step        614 tokens   (window − overlap)

Sentence‑friendly rule
──────────────────────
After *window* tokens are gathered we keep adding UNTIL the next token
ends with `.`, `!`, or `?` —or the file ends, or we exceed
window + 256 tokens.  This avoids mid‑sentence splits whenever possible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

# ───────────────────────────── env / constants ────────────────────────────── #

load_dotenv()

REPO_ROOT      = Path(__file__).resolve().parent.parent
JSON_DIR       = REPO_ROOT / "documents" / "research" / "json"
TXT_DIR        = REPO_ROOT / "documents" / "research" / "txt"   # kept but unused here

SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

DEFAULT_WINDOW   = 768
DEFAULT_OVERLAP  = int(DEFAULT_WINDOW * 0.20)     # 154
DEFAULT_STEP     = DEFAULT_WINDOW - DEFAULT_OVERLAP
SENT_END_RE      = re.compile(r"[.!?]$")

# ─────────────────────────── NUL‑byte scrubber ───────────────────────────── #

def scrub_nuls(obj: Any) -> Any:
    """Recursively remove ASCII‑NUL characters from any str inside *obj*."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, list):
        return [scrub_nuls(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_nuls(v) for k, v in obj.items()}
    return obj

# ───────────────────────────────── helpers ────────────────────────────────── #

def discover_json_files() -> List[Path]:
    return sorted(JSON_DIR.rglob("*.json"))

def load_paper(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf‑8"))

def concat_sections(sections: Sequence[Dict[str, Any]]) -> tuple[list[str], list[int]]:
    """
    Return (tokens, token_to_page)  ↦  token_to_page[i] = 1‑based page number.
    """
    tokens: list[str] = []
    page_map: list[int] = []
    for sec in sections:
        page = sec.get("page_number") or 1
        sec_tokens = sec.get("text", "").split()
        tokens.extend(sec_tokens)
        page_map.extend([page] * len(sec_tokens))
    return tokens, page_map

def sliding_window_chunks(
    tokens: List[str],
    page_map: List[int],
    *,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """Fixed‑token windows with overlap, but avoid cutting sentences."""
    step = max(1, window - overlap)
    chunks: list[dict[str, Any]] = []
    i = 0
    safety_extra = 256

    while i < len(tokens):
        start = i
        end   = min(len(tokens), start + window)

        # extend to sentence boundary if possible
        while (
            end < len(tokens)
            and not SENT_END_RE.search(tokens[end - 1])
            and (end - start) < window + safety_extra
        ):
            end += 1

        text_slice  = " ".join(tokens[start:end])
        page_start  = page_map[start]
        page_end    = page_map[end - 1]
        chunk_idx   = len(chunks)

        chunks.append(
            {
                "chunk_index":  chunk_idx,
                "token_start":  start,
                "token_end":    end - 1,
                "page_start":   page_start,
                "page_end":     page_end,
                "text":         scrub_nuls(text_slice),
            }
        )
        if end == len(tokens):
            break
        i += step

    return chunks

# ────────────────────────────── Supabase I/O ─────────────────────────────── #

def get_processed_pdfs(sb) -> set[str]:
    """
    Return the set of source_pdf paths already present in research_documents.
    """
    try:
        rows = sb.table("research_documents").select("source_pdf").execute().data or []
        return {row["source_pdf"] for row in rows}
    except Exception as e:
        print(f"[get_processed_pdfs] {e}")
        return set()

BIB_KEYS = {
    "doc_type", "title", "authors", "year",
    "journal", "doi", "abstract", "keywords",
    "research_topics", "source_pdf",
}

def insert_document(sb, raw_meta: Dict[str, Any]) -> str:
    """
    Insert into research_documents; returns UUID.  Re‑use row if DOI matches.
    """
    raw_meta = scrub_nuls(raw_meta)
    doi = raw_meta.get("doi")

    # ── 1.  reuse if DOI already there
    if doi:
        q = (
            sb.table("research_documents")
            .select("id")
            .eq("doi", doi)
            .limit(1)
            .execute()
        )
        if q.data:
            return q.data[0]["id"]

    # ── 2.  trim to bibliographic subset
    meta_small = {k: raw_meta.get(k) for k in BIB_KEYS}

    payload = {
        **meta_small,
        # defaults / housekeeping
        "doc_type":   meta_small.get("doc_type", "scientific paper"),
        "authors":    meta_small.get("authors") or [],
        "keywords":   meta_small.get("keywords") or [],
        "research_topics": meta_small.get("research_topics") or [],
        "metadata":   meta_small,                 # stored as jsonb
        "created_at": datetime.utcnow().isoformat(),
    }
    res = sb.table("research_documents").insert(payload).execute()
    if getattr(res, "error", None):
        raise RuntimeError(res.error)
    return res.data[0]["id"]

def insert_chunks(sb, doc_id: str, json_path: Path, chunks: Sequence[Dict[str, Any]]):
    max_batch = 500
    rows = [
        {
            "document_id":       doc_id,
            "chunk_index":       ch["chunk_index"],
            "token_start":       ch["token_start"],
            "token_end":         ch["token_end"],
            "page_start":        ch["page_start"],
            "page_end":          ch["page_end"],
            "text":              ch["text"],
            "metadata":          json.loads(json.dumps({}).replace('\u0000', '')),  # extra per‑chunk data (now clean)
            "chunking_strategy": "token_window",
            "source_file":       json_path.as_posix(),  # ← renamed column
            "created_at":        datetime.utcnow().isoformat(),
        }
        for ch in chunks
    ]
    for i in range(0, len(rows), max_batch):
        sb.table("research_chunks").insert(rows[i : i + max_batch]).execute()

# ─────────────────────────────── main process ────────────────────────────── #

def process_one_paper(
    sb,
    json_path: Path,
    *,
    window: int,
    overlap: int,
) -> tuple[bool, str]:
    try:
        paper = load_paper(json_path)
        sections = paper.get("sections") or []
        if not sections:
            return False, "No sections/text found."

        tokens, page_map = concat_sections(sections)
        if not tokens:
            return False, "Empty token list."

        chunks = sliding_window_chunks(tokens, page_map, window=window, overlap=overlap)
        doc_id = insert_document(sb, paper)
        insert_chunks(sb, doc_id, json_path, chunks)
        return True, f"{len(chunks)} chunks"
    except Exception as e:
        return False, str(e)

# ─────────────────────────────────── CLI ─────────────────────────────────── #

def main() -> None:
    p = argparse.ArgumentParser(description="Upload 768‑token window chunks to Supabase")
    p.add_argument("--batch-size", type=int, default=20, help="papers per run (0 = all)")
    p.add_argument("--window",     type=int, default=DEFAULT_WINDOW, help="tokens per chunk")
    p.add_argument("--overlap",    type=int, default=DEFAULT_OVERLAP, help="token overlap")
    p.add_argument("--selection",  choices=["sequential", "random"], default="sequential")
    args = p.parse_args()

    if not (SUPABASE_URL and SUPABASE_KEY):
        sys.exit("❌  Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    all_json = discover_json_files()
    processed_pdfs = get_processed_pdfs(sb)
    remaining = [
        p for p in all_json
        if scrub_nuls(load_paper(p)).get("source_pdf") not in processed_pdfs
    ]

    if args.selection == "random" and args.batch_size and len(remaining) > args.batch_size:
        remaining = random.sample(remaining, args.batch_size)
    elif args.batch_size:
        remaining = remaining[: args.batch_size]

    if not remaining:
        print("✨  Nothing new to process.")
        return

    step = args.window - args.overlap
    print(f"Uploading {len(remaining)} papers …  (window={args.window}, overlap={args.overlap}, step={step})")
    ok = fail = 0
    for jp in tqdm(remaining, desc="Papers"):
        success, msg = process_one_paper(
            sb,
            jp,
            window=args.window,
            overlap=args.overlap,
        )
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  ! {jp.name} → {msg}")

    print(f"\nDone ✔   success={ok}   failed={fail}")

if __name__ == "__main__":
    main()
