#!/usr/bin/env python3
"""
Stage 7 — Fill NULL embeddings in research_chunks (token-safe batch generator)
"""
from __future__ import annotations
# ... full resilient script body copied verbatim ...
#  ───────────────────────── configuration ─────────────────────────
import argparse, json, logging, os, sys, time
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm
load_dotenv()
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL="text-embedding-ada-002"
MODEL_TOKEN_LIMIT=8192
TOKEN_GUARD=200
MAX_TOTAL_TOKENS=MODEL_TOKEN_LIMIT-TOKEN_GUARD
DEFAULT_BATCH_ROWS=5000
DEFAULT_COMMIT_ROWS=100
MAX_RETRIES=5
RETRY_DELAY=2
openai_client=OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s")
log=logging.getLogger(__name__)
# ---------------- Supabase helpers / fetch rows / token helpers -------------
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE creds missing"); sys.exit(1)
    return create_client(SUPABASE_URL,SUPABASE_KEY)
def count_processed_chunks(sb)->int:
    res=sb.table("research_chunks").select("id",count="exact").not_.is_("embedding","null").execute()
    return res.count or 0
def fetch_unprocessed_chunks(sb,*,limit:int,offset:int=0)->List[Dict[str,Any]]:
    first,last=offset,offset+limit-1
    res=sb.table("research_chunks").select("id,text,token_start,token_end")\
        .eq("chunking_strategy","token_window").is_("embedding","null")\
        .range(first,last).execute()
    return res.data or []
def generate_embedding_batch(texts:List[str])->List[List[float]]:
    attempt=0
    while attempt<MAX_RETRIES:
        try:
            resp=openai_client.embeddings.create(model=EMBEDDING_MODEL,input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:
            attempt+=1; log.warning("Embedding batch %s/%s failed: %s",attempt,MAX_RETRIES,exc)
            time.sleep(RETRY_DELAY)
    raise RuntimeError("OpenAI embedding batch failed after retries")
def chunk_tokens(row:Dict[str,Any])->int:
    try:
        t=int(row["token_end"])-int(row["token_start"])+1
        if 0<t<=16384: return t
    except: pass
    approx=int(len((row.get("text") or "").split())*0.75)+1
    return min(max(1,approx),MODEL_TOKEN_LIMIT)
def safe_slices(rows:List[Dict[str,Any]],max_rows:int)->List[List[Dict[str,Any]]]:
    slices,current,cur_tokens=[],[],0
    for r in rows:
        txt=(r.get("text") or "").replace("\x00","")
        if not txt.strip(): continue
        t=chunk_tokens(r)
        if t>MAX_TOTAL_TOKENS:
            log.warning("Chunk %s is too long – skipping.",r["id"]); continue
        need_new=len(current)>=max_rows or cur_tokens+t>MAX_TOTAL_TOKENS
        if need_new and current:
            slices.append(current); current=[]; cur_tokens=0
        current.append({"id":r["id"],"text":txt}); cur_tokens+=t
    if current: slices.append(current)
    return slices
def embed_slice(sb,slice_rows:List[Dict[str,Any]])->int:
    embeds=generate_embedding_batch([r["text"] for r in slice_rows])
    ok=0
    for row,emb in zip(slice_rows,embeds):
        attempt=0
        while attempt<MAX_RETRIES:
            res=sb.table("research_chunks").update({"embedding":emb}).eq("id",row["id"]).execute()
            if getattr(res,"error",None):
                attempt+=1; log.warning("Update %s failed (%s/%s): %s",row["id"],attempt,MAX_RETRIES,res.error)
                time.sleep(RETRY_DELAY)
            else: ok+=1; break
    return ok
# ------------------------------ main ----------------------------------------
def main()->None:
    # make the module-level constants writable in this function *before* we read them
    global EMBEDDING_MODEL, MODEL_TOKEN_LIMIT, MAX_TOTAL_TOKENS
    
    ap=argparse.ArgumentParser(description="Embed research_chunks where embedding IS NULL (token-safe)")
    ap.add_argument("--batch-size",type=int,default=DEFAULT_BATCH_ROWS)
    ap.add_argument("--commit",type=int,default=DEFAULT_COMMIT_ROWS)
    ap.add_argument("--skip",type=int,default=0)
    ap.add_argument("--model",default=EMBEDDING_MODEL)
    ap.add_argument("--model-limit",type=int,default=MODEL_TOKEN_LIMIT)
    args=ap.parse_args()
    EMBEDDING_MODEL=args.model; MODEL_TOKEN_LIMIT=args.model_limit
    MAX_TOTAL_TOKENS=MODEL_TOKEN_LIMIT-TOKEN_GUARD
    if not OPENAI_API_KEY: log.error("OPENAI_API_KEY missing"); sys.exit(1)
    sb=init_supabase()
    log.info("Rows already embedded: %s",count_processed_chunks(sb))
    loop,total=0,0
    while True:
        loop+=1
        rows=fetch_unprocessed_chunks(sb,limit=args.batch_size,offset=args.skip)
        if not rows: log.info("✨  Done — no more rows."); break
        log.info("Loop %s — fetched %s rows.",loop,len(rows))
        slices=safe_slices(rows,args.commit)
        log.info("Created %s token-safe OpenAI requests.",len(slices))
        for sl in tqdm(slices,desc=f"Embedding loop {loop}",unit="batch"):
            try: total+=embed_slice(sb,sl)
            except Exception as exc: log.error("Slice failed: %s",exc)
        log.info("Loop %s complete – total embedded so far: %s",loop,total)
    ts=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report={"timestamp":ts,"batch_size":args.batch_size,"commit":args.commit,
            "skip":args.skip,"total_embedded":total,
            "total_with_embeddings":count_processed_chunks(sb)}
    fname=f"supabase_embedding_report_{ts}.json"
    with open(fname,"w") as fp: json.dump(report,fp,indent=2)
    log.info("Report saved to %s",fname)
if __name__=="__main__": main() 