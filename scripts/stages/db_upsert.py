#!/usr/bin/env python3
"""
Stage 6 — Optimized database operations with batching and connection pooling.
"""
from __future__ import annotations
import json, logging, os, pathlib, sys
from datetime import datetime
from typing import Any, Dict, List, Sequence, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

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

# Thread-safe connection pool
class SupabasePool:
    """Thread-safe Supabase client pool."""
    def __init__(self, size: int = 10):
        self.size = size
        self._clients = []
        self._lock = threading.Lock()
        self._initialized = False
    
    def get(self):
        """Get a client from the pool."""
        with self._lock:
            if not self._initialized:
                self._initialize()
            # Round-robin selection
            import random
            return self._clients[random.randint(0, self.size - 1)]
    
    def _initialize(self):
        """Initialize client pool."""
        for _ in range(self.size):
            self._clients.append(init_supabase())
        self._initialized = True

# Global pool
_sb_pool = SupabasePool()

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

# Optimized batch operations
def insert_chunks_optimized(sb, doc_id: str, chunks: Sequence[Dict[str, Any]], 
                          src_json: pathlib.Path, batch_size: int = 1000):
    """Insert chunks with larger batches for better performance."""
    ts = datetime.utcnow().isoformat()
    inserted_ids = []
    
    # Prepare all rows
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
    
    # Insert in larger batches
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            res = sb.table("research_chunks").insert(batch).execute()
            if hasattr(res, "error") and res.error:
                log.error("Batch insert failed: %s", res.error)
                # Fall back to smaller batches
                for j in range(0, len(batch), 100):
                    mini_batch = batch[j:j + 100]
                    mini_res = sb.table("research_chunks").insert(mini_batch).execute()
                    if hasattr(mini_res, "data"):
                        inserted_ids.extend([r["id"] for r in mini_res.data])
            else:
                inserted_ids.extend([r["id"] for r in res.data])
        except Exception as e:
            log.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
            # Try individual inserts as last resort
            for row in batch:
                try:
                    single_res = sb.table("research_chunks").insert(row).execute()
                    if hasattr(single_res, "data") and single_res.data:
                        inserted_ids.append(single_res.data[0]["id"])
                except:
                    pass
    
    return inserted_ids

def insert_chunks(sb,doc_id:str,chunks:Sequence[Dict[str,Any]],src_json:pathlib.Path):
    """Original interface using optimized implementation."""
    return insert_chunks_optimized(sb, doc_id, chunks, src_json, batch_size=500)

async def upsert_batch_async(
    documents: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> None:
    """Upsert multiple documents concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def upsert_one(doc_data):
        async with semaphore:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: upsert(
                    doc_data["json_path"],
                    doc_data["chunks"],
                    do_embed=doc_data.get("do_embed", False)
                )
            )
    
    tasks = [upsert_one(doc) for doc in documents]
    
    from tqdm.asyncio import tqdm_asyncio
    await tqdm_asyncio.gather(*tasks, desc="Upserting to database")

# Keep original interface but use optimized implementation
def upsert(json_doc: pathlib.Path, chunks: List[Dict[str, Any]] | None,
           *, do_embed: bool = False) -> None:
    """Original interface with optimized implementation."""
    sb = _sb_pool.get()  # Use connection from pool
    
    data = json.loads(json_doc.read_text())
    row = {k: data.get(k) for k in META_FIELDS} | {
        "source_pdf": data.get("source_pdf", str(json_doc))
    }
    
    doc_id = upsert_document(sb, row)
    
    if not chunks:
        log.info("No chunks to insert for %s", json_doc.stem)
        return
    
    # Use optimized batch insert
    inserted_ids = insert_chunks_optimized(sb, doc_id, chunks, json_doc)
    log.info("↑ %s chunks inserted for %s", len(chunks), json_doc.stem)
    
    if do_embed and inserted_ids:
        from stages import embed_vectors
        embed_vectors.main()

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