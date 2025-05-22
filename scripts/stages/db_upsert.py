#!/usr/bin/env python3
"""
Stage 6 — create / update `research_documents` + `research_chunks` rows.
If `do_embed=True` we immediately hand-off to Stage 7 for embeddings.
"""
from __future__ import annotations
import json, logging, os, pathlib, sys
from datetime import datetime
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

META_FIELDS = ["doc_type","title","authors","year","journal","doi","abstract",
               "keywords","research_topics","peer_reviewed","open_access",
               "license","open_access_status"]

log = logging.getLogger(__name__)

# ─── Supabase & sanitiser helpers ──────────────────────────────────
def scrub_nuls(obj:Any)->Any:
    if isinstance(obj,str):  return obj.replace("\x00","")
    if isinstance(obj,list): return [scrub_nuls(x) for x in obj]
    if isinstance(obj,dict): return {k:scrub_nuls(v) for k,v in obj.items()}
    return obj

def init_supabase():
    if not (SUPABASE_URL and SUPABASE_KEY):
        sys.exit("⛔  SUPABASE env vars missing")
    return create_client(SUPABASE_URL,SUPABASE_KEY)

def upsert_document(sb,meta:Dict[str,Any])->str:
    meta = scrub_nuls(meta)
    doi = meta.get("doi")
    if doi:
        existing = (
            sb.table("research_documents")
            .select("id")
            .eq("doi", doi)
            .limit(1)
            .execute()
            .data
        )
        if existing:
            doc_id = existing[0]["id"]
            sb.table("research_documents").update(meta).eq("id", doc_id).execute()
            return doc_id
    res = sb.table("research_documents").insert(meta).execute()
    if hasattr(res, "error") and res.error:
        raise RuntimeError(f"Document insert failed: {res.error}")
    return res.data[0]["id"]

def insert_chunks(sb,doc_id:str,chunks:Sequence[Dict[str,Any]],src_json:pathlib.Path):
    ts = datetime.utcnow().isoformat()
    inserted_ids = []
    
    rows = [
        {
            "document_id": doc_id,
            "chunk_index": ch["chunk_index"],
            "token_start": ch["token_start"],
            "token_end": ch["token_end"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": scrub_nuls(ch["text"]),
            "metadata": scrub_nuls(ch.get("metadata", {})),
            "chunking_strategy": "token_window",
            "source_file": str(src_json),
            "created_at": ts,
        }
        for ch in chunks
    ]
    
    for i in range(0, len(rows), 500):
        batch = rows[i : i + 500]
        try:
            res = sb.table("research_chunks").insert(batch).execute()
            if hasattr(res, "error") and res.error:
                log.error("Supabase insert failed → %s", res.error)
                raise RuntimeError(f"Chunk insert failed: {res.error}")
            # Collect the inserted IDs
            inserted_ids.extend([r["id"] for r in res.data])
        except Exception as e:
            log.error(f"Failed to insert chunk batch {i//500 + 1}: {e}")
            raise
    
    return inserted_ids

def upsert(json_doc:pathlib.Path, chunks:List[Dict[str,Any]]|None,
           *, do_embed:bool=False)->None:
    sb=init_supabase()

    data=json.loads(json_doc.read_text())
    row={k:data.get(k) for k in META_FIELDS}|{"source_pdf":data.get("source_pdf", str(json_doc))}
    
    doc_id=upsert_document(sb, row)

    if not chunks:
        log.info("No chunks to insert for %s", json_doc.stem)
        return

    inserted_ids = insert_chunks(sb, doc_id, chunks, json_doc)
    log.info("↑ %s chunks inserted for %s", len(chunks), json_doc.stem)

    if do_embed and inserted_ids:
        # Import here to avoid circular dependency
        from stages import embed_vectors
        # Pass the IDs of chunks we just inserted
        log.info("Embedding %s chunks...", len(inserted_ids))
        # embed_vectors expects to fetch chunks by ID, so we just trigger it
        embed_vectors.main()  # This will find all chunks with NULL embeddings

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p=argparse.ArgumentParser()
    p.add_argument("json",type=pathlib.Path)
    p.add_argument("--chunks",type=pathlib.Path)
    p.add_argument("--embed",action="store_true")
    args=p.parse_args()
    
    chunks = None
    if args.chunks and args.chunks.exists():
        chunks = json.loads(args.chunks.read_text())
    
    upsert(args.json, chunks, do_embed=args.embed)