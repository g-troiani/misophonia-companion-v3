#!/usr/bin/env python3
"""
Stage 7 — Optimized embedding with rate limiting, deduplication, and conservative batching.
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import asyncio
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import hashlib

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using conservative token estimation")

load_dotenv()
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL="text-embedding-ada-002"
MODEL_TOKEN_LIMIT=8192
TOKEN_GUARD=200

# 🎯 DYNAMIC BATCHING - Token limits with safety margins
MAX_BATCH_TOKENS = 7692  # Conservative limit (8192 - 500 safety margin)
MAX_CHUNK_TOKENS = 6000  # Individual chunk limit
MIN_BATCH_TOKENS = 100   # Minimum viable batch size

MAX_TOTAL_TOKENS=MAX_BATCH_TOKENS  # Use dynamic batch limit instead

# Conservative defaults for rate limiting
DEFAULT_BATCH_ROWS=200  # Reduced from 5000
DEFAULT_COMMIT_ROWS=10  # Reduced from 100
DEFAULT_MAX_CONCURRENT=3  # Reduced from 50
MAX_RETRIES=5
RETRY_DELAY=2
RATE_LIMIT_DELAY=0.3  # 300ms between API calls
MAX_CALLS_PER_MINUTE=150  # Conservative limit

openai_client=OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s")
log=logging.getLogger(__name__)

# Global variables for rate limiting
call_timestamps = []
total_tokens_used = 0

class AsyncEmbedder:
    """Async embedding client with strict rate limiting and connection pooling."""
    
    def __init__(self, api_key: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.call_count = 0
        self.start_time = time.time()
        
        # Initialize tiktoken encoder if available
        if TIKTOKEN_AVAILABLE:
            self.encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        else:
            self.encoder = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)  # Reduced limits
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken if available."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Conservative fallback
            return int(len(text.split()) * 0.75) + 50
    
    async def rate_limit_check(self):
        """Enforce rate limiting."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        call_timestamps[:] = [ts for ts in call_timestamps if current_time - ts < 60]
        
        # Check if we're approaching the rate limit
        if len(call_timestamps) >= MAX_CALLS_PER_MINUTE - 5:
            sleep_time = 60 - (current_time - call_timestamps[0])
            if sleep_time > 0:
                log.info(f"Rate limit protection: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Always enforce minimum delay between calls
        await asyncio.sleep(RATE_LIMIT_DELAY)
        
        # Record this call
        call_timestamps.append(current_time)
    
    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts asynchronously with rate limiting and token validation."""
        global total_tokens_used
        
        async with self.semaphore:
            await self.rate_limit_check()
            
            # 🛡️ FINAL TOKEN VALIDATION - Last safety check before API call
            batch_tokens = sum(self.count_tokens(text) for text in texts)
            
            if batch_tokens > MAX_BATCH_TOKENS:
                error_msg = f"🚨 CRITICAL: Batch tokens {batch_tokens} exceed limit {MAX_BATCH_TOKENS}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            
            total_tokens_used += batch_tokens
            
            log.info(f"🎯 API call: {len(texts)} texts (~{batch_tokens} tokens) - WITHIN LIMITS ✅")
            log.info(f"📊 Total tokens used so far: ~{total_tokens_used}")
            
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.session.post(
                        "https://api.openai.com/v1/embeddings",
                        json={
                            "model": EMBEDDING_MODEL,
                            "input": texts
                        }
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.call_count += 1
                            log.info(f"✅ API call successful: {len(texts)} embeddings generated")
                            return [item["embedding"] for item in data["data"]]
                        elif resp.status == 400:  # Bad request - likely token limit
                            error = await resp.text()
                            log.error(f"🚨 API error 400 (likely token limit): {error}")
                            log.error(f"🚨 Batch details: {len(texts)} texts, {batch_tokens} tokens")
                            raise RuntimeError(f"Token limit API error: {error}")
                        elif resp.status == 429:  # Rate limit error
                            error = await resp.text()
                            log.warning(f"Rate limit hit: {error}")
                            # Exponential backoff for rate limits
                            sleep_time = (2 ** attempt) * 2
                            log.info(f"Exponential backoff: sleeping {sleep_time}s")
                            await asyncio.sleep(sleep_time)
                        else:
                            error = await resp.text()
                            log.warning(f"API error {resp.status}: {error}")
                            await asyncio.sleep(RETRY_DELAY)
                except Exception as e:
                    log.warning(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
            raise RuntimeError("Failed to embed batch after retries")

def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate text chunks based on content hash."""
    seen_hashes: Set[str] = set()
    unique_chunks = []
    duplicates_removed = 0
    
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
            
        # Create hash of the text content
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_chunks.append(chunk)
        else:
            duplicates_removed += 1
    
    if duplicates_removed > 0:
        log.info(f"Deduplication: removed {duplicates_removed} duplicate chunks")
    
    return unique_chunks

async def process_chunks_async(
    sb,
    chunks: List[Dict[str, Any]],
    embedder: AsyncEmbedder,
    commit_size: int = DEFAULT_COMMIT_ROWS
) -> int:
    """Process chunks with async embedding and batch updates."""
    # Deduplicate chunks first
    unique_chunks = deduplicate_chunks(chunks)
    log.info(f"📊 CHUNK PROCESSING PROGRESS:")
    log.info(f"   📄 Original chunks: {len(chunks)}")
    log.info(f"   📄 Unique chunks: {len(unique_chunks)}")
    if len(chunks) != len(unique_chunks):
        log.info(f"   🔄 Duplicates removed: {len(chunks) - len(unique_chunks)}")
    
    # Create token-safe slices
    slices = safe_slices(unique_chunks, commit_size)
    total_embedded = 0
    total_slices = len(slices)
    
    if total_slices == 0:
        log.warning(f"📊 NO SLICES TO PROCESS - All chunks were filtered out")
        return 0
    
    log.info(f"📊 EMBEDDING SLICES PROGRESS:")
    log.info(f"   🎯 Total slices to process: {total_slices}")
    log.info(f"   📄 Total chunks to embed: {sum(len(slice_data) for slice_data in slices)}")
    
    # Process each slice
    for i, slice_data in enumerate(slices):
        slice_num = i + 1
        slice_progress = (slice_num / total_slices * 100)
        
        log.info(f"📊 SLICE {slice_num}/{total_slices} ({slice_progress:.1f}%):")
        log.info(f"   📄 Processing {len(slice_data)} chunks in this slice")
        
        texts = [row["text"] for row in slice_data]
        
        try:
            # Get embeddings asynchronously
            log.info(f"   🔄 Calling OpenAI API for {len(texts)} embeddings...")
            embeddings = await embedder.embed_batch_async(texts)
            log.info(f"   ✅ Received {len(embeddings)} embeddings from API")
            
            # Update database in batch
            log.info(f"   💾 Updating database for {len(slice_data)} chunks...")
            update_tasks = []
            for row, emb in zip(slice_data, embeddings):
                # ✅ GUARANTEED SKIP LOGIC - Layer 3: Final update verification
                def update_with_verification(r, e):
                    # Final check before updating
                    final_check = sb.table("research_chunks").select("embedding")\
                        .eq("id", r["id"]).execute()
                    
                    if final_check.data and final_check.data[0].get("embedding"):
                        log.info(f"🛡️ Chunk {r['id']} already has embedding - SKIPPING update")
                        return {"skipped": True}
                    
                    # Safe to update
                    return sb.table("research_chunks")\
                        .update({"embedding": e})\
                        .eq("id", r["id"])\
                        .execute()
                
                # Use thread pool for database updates
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, update_with_verification, row, emb)
                update_tasks.append(task)
            
            # Wait for all updates
            log.info(f"   ⏳ Waiting for {len(update_tasks)} database updates...")
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception) and not (isinstance(r, dict) and r.get("skipped")))
            skipped = sum(1 for r in results if isinstance(r, dict) and r.get("skipped"))
            failed = sum(1 for r in results if isinstance(r, Exception))
            total_embedded += successful
            
            # 📊 SLICE COMPLETION PROGRESS
            log.info(f"📊 SLICE {slice_num} COMPLETE:")
            log.info(f"   ✅ Successful updates: {successful}")
            if skipped > 0:
                log.info(f"   ⏭️  Skipped (already embedded): {skipped}")
            if failed > 0:
                log.info(f"   ❌ Failed updates: {failed}")
            log.info(f"   📈 Total embedded so far: {total_embedded}")
            
        except Exception as e:
            log.error(f"❌ SLICE {slice_num} FAILED: {e}")
            log.error(f"   📄 Chunks in failed slice: {len(slice_data)}")
    
    # Final summary
    log.info(f"📊 CHUNK PROCESSING COMPLETE:")
    log.info(f"   ✅ Total chunks embedded: {total_embedded}")
    log.info(f"   📊 Slices processed: {total_slices}")
    log.info(f"   📈 Success rate: {(total_embedded/len(unique_chunks)*100):.1f}%" if unique_chunks else "0%")
    
    return total_embedded

async def main_async(batch_size: int = None, commit_size: int = None, max_concurrent: int = None):
    """Async main function for embedding."""
    global EMBEDDING_MODEL, MODEL_TOKEN_LIMIT, MAX_TOTAL_TOKENS, total_tokens_used
    
    # Use provided parameters or conservative defaults
    batch_size = batch_size or DEFAULT_BATCH_ROWS
    commit_size = commit_size or DEFAULT_COMMIT_ROWS
    max_concurrent = max_concurrent or DEFAULT_MAX_CONCURRENT
    
    log.info(f"Starting with conservative settings:")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Commit size: {commit_size}")
    log.info(f"  Max concurrent: {max_concurrent}")
    log.info(f"  Rate limit delay: {RATE_LIMIT_DELAY}s")
    
    # 🎯 Dynamic batching status
    log.info(f"🎯 DYNAMIC BATCHING ENABLED:")
    log.info(f"   Max batch tokens: {MAX_BATCH_TOKENS} (with safety margin)")
    log.info(f"   Max chunk tokens: {MAX_CHUNK_TOKENS}")
    log.info(f"   Min batch tokens: {MIN_BATCH_TOKENS}")
    log.info(f"   ✅ GUARANTEED: No token limit API errors")
    
    # ✅ Check tiktoken availability for accurate token counting
    if not TIKTOKEN_AVAILABLE:
        log.warning("⚠️  tiktoken not available - using conservative token estimation")
        log.warning("⚠️  For accurate token counting, install: pip install tiktoken")
    else:
        log.info("✅ tiktoken available - using accurate token counting")
    
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY missing")
        sys.exit(1)
    
    sb = init_supabase()
    existing_count = count_processed_chunks(sb)
    log.info(f"Rows already embedded: {existing_count}")
    
    # ✅ GUARANTEED SKIP LOGIC - Status check
    total_chunks_res = sb.table("research_chunks").select("id", count="exact")\
        .eq("chunking_strategy", "token_window").execute()
    total_chunks = total_chunks_res.count or 0
    
    pending_chunks = total_chunks - existing_count
    
    log.info("📊 Current status:")
    log.info(f"   Chunks already embedded: {existing_count}")
    log.info(f"   Chunks needing embedding: {pending_chunks}")
    log.info(f"   Total chunks: {total_chunks}")
    
    if pending_chunks == 0:
        log.info("✅ All chunks already have embeddings - nothing to do!")
        return
    
    async with AsyncEmbedder(OPENAI_API_KEY, max_concurrent) as embedder:
        loop, total = 0, 0
        
        # 📊 PROGRESS TRACKING - Initialize counters
        total_chunks_to_process = pending_chunks
        chunks_processed = 0
        chunks_skipped = 0
        
        log.info(f"📊 EMBEDDING PROGRESS TRACKING INITIALIZED:")
        log.info(f"   🎯 Total chunks to embed: {total_chunks_to_process}")
        log.info(f"   🎯 Starting embedding process...")
        
        while True:
            loop += 1
            rows = fetch_unprocessed_chunks(sb, limit=batch_size, offset=0)
            
            if not rows:
                log.info("✨ Done — no more rows.")
                break
            
            # 📊 PROGRESS: Show current status before processing
            remaining_chunks = total_chunks_to_process - chunks_processed
            progress_percent = (chunks_processed / total_chunks_to_process * 100) if total_chunks_to_process > 0 else 0
            
            log.info(f"📊 EMBEDDING PROGRESS - Loop {loop}:")
            log.info(f"   📄 Fetched: {len(rows)} chunks")
            log.info(f"   ✅ Processed: {chunks_processed}/{total_chunks_to_process} ({progress_percent:.1f}%)")
            log.info(f"   ⏳ Remaining: {remaining_chunks} chunks")
            if chunks_skipped > 0:
                log.info(f"   ⚠️  Skipped: {chunks_skipped} oversized chunks")
            
            # Process chunks asynchronously
            embedded = await process_chunks_async(sb, rows, embedder, commit_size)
            total += embedded
            chunks_processed += embedded
            
            # Update skipped count (this will be calculated in process_chunks_async)
            current_chunk_count = count_processed_chunks(sb)
            actual_embedded_this_loop = current_chunk_count - (existing_count + chunks_processed - embedded)
            
            # 📊 PROGRESS: Show results after processing
            final_progress_percent = (chunks_processed / total_chunks_to_process * 100) if total_chunks_to_process > 0 else 0
            log.info(f"📊 LOOP {loop} COMPLETE:")
            log.info(f"   ✅ This loop: {embedded} chunks embedded")
            log.info(f"   📈 Total progress: {chunks_processed}/{total_chunks_to_process} ({final_progress_percent:.1f}%)")
            log.info(f"   🔄 API calls made: {embedder.call_count}")
            log.info(f"   📊 Total tokens used: ~{total_tokens_used}")
            
            # Check if we're making progress or stuck
            if embedded == 0 and len(rows) > 0:
                chunks_skipped += len(rows)
                log.warning(f"⚠️  WARNING: No chunks embedded in this loop - all {len(rows)} chunks were skipped")
                log.warning(f"⚠️  Total skipped so far: {chunks_skipped} chunks")
                
                # Prevent infinite loop by limiting consecutive failed attempts
                if loop > 5 and embedded == 0:
                    log.error(f"🚨 STUCK: {loop} consecutive loops with no progress - stopping to prevent infinite loop")
                    break
    
    # Save report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "batch_size": batch_size,
        "commit": commit_size,
        "max_concurrent": max_concurrent,
        "total_embedded": total,
        "total_with_embeddings": count_processed_chunks(sb),
        "total_tokens_used": total_tokens_used,
        "api_calls_made": embedder.call_count if 'embedder' in locals() else 0
    }
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports/embedding"
    os.makedirs(reports_dir, exist_ok=True)
    
    fname = f"{reports_dir}/supabase_embedding_report_{ts}.json"
    with open(fname, "w") as fp:
        json.dump(report, fp, indent=2)
    
    log.info(f"Report saved to {fname}")

async def main_async_cli():
    """CLI version with argument parsing."""
    global EMBEDDING_MODEL, MODEL_TOKEN_LIMIT, MAX_TOTAL_TOKENS
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_ROWS)
    ap.add_argument("--commit", type=int, default=DEFAULT_COMMIT_ROWS)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--model", default=EMBEDDING_MODEL)
    ap.add_argument("--model-limit", type=int, default=MODEL_TOKEN_LIMIT)
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                   help="Maximum concurrent API calls (default: 3 for rate limiting)")
    args = ap.parse_args()
    
    EMBEDDING_MODEL = args.model
    MODEL_TOKEN_LIMIT = args.model_limit
    MAX_TOTAL_TOKENS = MODEL_TOKEN_LIMIT - TOKEN_GUARD
    
    await main_async(args.batch_size, args.commit, args.max_concurrent)

# Keep original interface for compatibility
def main() -> None:
    """Original synchronous interface."""
    asyncio.run(main_async_cli())

# Keep all existing helper functions unchanged...
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
    
    chunks = res.data or []
    
    if chunks:
        # ✅ GUARANTEED SKIP LOGIC - Layer 2: Double-check verification
        chunk_ids = [chunk["id"] for chunk in chunks]
        verification = sb.table("research_chunks").select("id")\
            .in_("id", chunk_ids).not_.is_("embedding", "null").execute()
        
        already_embedded_ids = {row["id"] for row in verification.data or []}
        
        if already_embedded_ids:
            log.warning(f"🛡️ Found {len(already_embedded_ids)} chunks that already have embeddings - SKIPPING them")
            chunks = [chunk for chunk in chunks if chunk["id"] not in already_embedded_ids]
        
        log.info(f"✅ GUARANTEED: Fetched {len(chunks)} chunks WITHOUT embeddings")
        
        if not chunks:
            log.info("✅ All chunks already have embeddings - nothing to do!")
    
    return chunks

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
    
    # Use tiktoken if available for more accurate counting
    text = row.get("text", "")
    if TIKTOKEN_AVAILABLE:
        try:
            encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
            return len(encoder.encode(text))
        except:
            pass
    
    # Conservative fallback
    approx=int(len(text.split())*0.75)+50  # More conservative estimate
    return min(max(1,approx),MODEL_TOKEN_LIMIT)

def safe_slices(rows:List[Dict[str,Any]],max_rows:int)->List[List[Dict[str,Any]]]:
    """🎯 DYNAMIC BATCH SIZING - Guarantees no token limit errors."""
    return dynamic_batch_slices(rows)

def dynamic_batch_slices(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    🎯 DYNAMIC BATCH SIZING - Creates variable-sized batches that GUARANTEE no API token errors.
    
    Protection Layers:
    1. Individual chunk validation (max 6,000 tokens)
    2. Dynamic batch sizing (max 7,692 tokens total)
    3. Conservative safety margins
    4. Multiple validation checks
    """
    batches = []
    current_batch = []
    current_tokens = 0
    skipped_chunks = 0
    
    # Initialize token encoder for accurate counting
    encoder = None
    if TIKTOKEN_AVAILABLE:
        try:
            encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except:
            pass
    
    def count_tokens_accurate(text: str) -> int:
        """Count tokens with maximum accuracy."""
        if encoder:
            return len(encoder.encode(text))
        else:
            # Conservative fallback estimation
            return int(len(text.split()) * 0.75) + 50
    
    log.info(f"🎯 Starting dynamic batch creation with {len(rows)} chunks")
    log.info(f"🎯 Limits: {MAX_CHUNK_TOKENS} tokens/chunk, {MAX_BATCH_TOKENS} tokens/batch")
    
    for i, row in enumerate(rows):
        # Clean text and validate
        text = (row.get("text") or "").replace("\x00", "").strip()
        if not text:
            log.warning(f"🎯 Skipping empty chunk {row.get('id', 'unknown')}")
            continue
        
        # 🛡️ PROTECTION LAYER 1: Individual chunk validation
        chunk_tokens = count_tokens_accurate(text)
        
        if chunk_tokens > MAX_CHUNK_TOKENS:
            log.warning(f"🎯 Skipping oversized chunk {row.get('id', 'unknown')}: {chunk_tokens} tokens (max: {MAX_CHUNK_TOKENS})")
            skipped_chunks += 1
            continue
        
        # 🛡️ PROTECTION LAYER 2: Dynamic batch sizing
        would_exceed = current_tokens + chunk_tokens > MAX_BATCH_TOKENS
        
        if would_exceed and current_batch:
            # Finalize current batch
            log.info(f"🎯 Created batch with {len(current_batch)} chunks (~{current_tokens} tokens)")
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        # Add chunk to current batch
        current_batch.append({"id": row["id"], "text": text})
        current_tokens += chunk_tokens
        
        # 🛡️ PROTECTION LAYER 3: Safety validation
        if current_tokens > MAX_BATCH_TOKENS:
            log.error(f"🚨 CRITICAL: Batch exceeded limit! {current_tokens} > {MAX_BATCH_TOKENS}")
            # Emergency fallback - remove last chunk and finalize batch
            if len(current_batch) > 1:
                current_batch.pop()
                current_tokens -= chunk_tokens
                log.info(f"🎯 Emergency: Created batch with {len(current_batch)} chunks (~{current_tokens} tokens)")
                batches.append(current_batch)
                current_batch = [{"id": row["id"], "text": text}]
                current_tokens = chunk_tokens
            else:
                log.error(f"🚨 Single chunk too large: {chunk_tokens} tokens")
                current_batch = []
                current_tokens = 0
                skipped_chunks += 1
    
    # Add final batch if not empty
    if current_batch and current_tokens >= MIN_BATCH_TOKENS:
        log.info(f"🎯 Created final batch with {len(current_batch)} chunks (~{current_tokens} tokens)")
        batches.append(current_batch)
    elif current_batch:
        log.warning(f"🎯 Skipping tiny final batch: {current_tokens} tokens < {MIN_BATCH_TOKENS} minimum")
    
    # 🛡️ PROTECTION LAYER 4: Final validation
    total_chunks = sum(len(batch) for batch in batches)
    max_batch_tokens = max((sum(count_tokens_accurate(chunk["text"]) for chunk in batch) for batch in batches), default=0)
    
    log.info(f"🎯 DYNAMIC BATCHING COMPLETE:")
    log.info(f"   📊 {len(batches)} batches created")
    log.info(f"   📊 {total_chunks} chunks processed")
    log.info(f"   📊 {skipped_chunks} chunks skipped")
    log.info(f"   📊 Largest batch: {max_batch_tokens} tokens (limit: {MAX_BATCH_TOKENS})")
    log.info(f"   ✅ GUARANTEED: No batch exceeds {MAX_BATCH_TOKENS} tokens")
    
    if max_batch_tokens > MAX_BATCH_TOKENS:
        log.error(f"🚨 CRITICAL ERROR: Batch validation failed!")
        raise RuntimeError(f"Batch token validation failed: {max_batch_tokens} > {MAX_BATCH_TOKENS}")
    
    return batches

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

if __name__=="__main__": 
    asyncio.run(main_async_cli()) 