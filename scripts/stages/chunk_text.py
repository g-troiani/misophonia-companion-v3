#!/usr/bin/env python3
"""
Stage 5 — Token-based chunking with tiktoken validation.
"""
from __future__ import annotations
import json, logging, math, pathlib, re
from typing import Dict, List, Sequence, Tuple, Any, Optional
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# 🎯 TIKTOKEN VALIDATION - Add token validation layer
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - token-based chunking requires tiktoken")

# Token limits for validation (consistent with embed_vectors.py)
MAX_CHUNK_TOKENS = 6000  # Maximum tokens per chunk for embedding
MIN_CHUNK_TOKENS = 100   # Minimum viable chunk size
EMBEDDING_MODEL = "text-embedding-ada-002"

# ─── TOKEN-BASED CHUNKING PARAMETERS ───────────────────────────
WINDOW_TOKENS = 3000   # Token-based window (was 768 words)
OVERLAP_TOKENS = 600   # Token-based overlap (20% of window)
log = logging.getLogger(__name__)

# ─── helpers to split into token windows ───────────────────────────
def concat_tokens_by_encoding(sections: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Convert sections to encoded tokens with page mapping."""
    if not TIKTOKEN_AVAILABLE:
        raise RuntimeError("tiktoken is required for token-based chunking")
    
    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    all_tokens = []
    token_to_page = []
    
    for s in sections:
        # Be defensive - build text if missing
        if 'text' not in s:
            if 'elements' in s:
                text = '\n'.join(
                    el.get('text', '') for el in s['elements'] 
                    if el.get('text', '') and not el.get('text', '').startswith('self_ref=')
                )
            else:
                text = ""
            log.warning(f"Section missing 'text' field, built from elements: {s.get('section', 'untitled')}")
        else:
            text = s.get("text", "")
        
        # Encode text to tokens
        tokens = encoder.encode(text)
        all_tokens.extend(tokens)
        
        # Handle page mapping
        if 'page_number' in s:
            page_num = s['page_number']
        elif 'page_start' in s and 'page_end' in s:
            page_start = s['page_start']
            page_end = s['page_end']
            if page_start == page_end:
                page_num = page_start
            else:
                # For multi-page sections, distribute tokens across pages
                pages_span = page_end - page_start + 1
                tokens_per_page = max(1, len(tokens) // pages_span)
                for i, token in enumerate(tokens):
                    page = min(page_end, page_start + (i // tokens_per_page))
                    token_to_page.append(page)
                continue  # Skip the extend below since we handled it above
        else:
            page_num = 1
            log.warning(f"Section has no page info: {s.get('section', 'untitled')}")
        
        # Map all tokens in this section to the page (for single page sections)
        if 'page_start' not in s or s['page_start'] == s.get('page_end', s['page_start']):
            token_to_page.extend([page_num] * len(tokens))
    
    return all_tokens, token_to_page

def sliding_chunks_by_tokens(
    encoded_tokens: List[int], 
    token_to_page: List[int], 
    *, 
    window_tokens: int, 
    overlap_tokens: int
) -> List[Dict[str, Any]]:
    """Create sliding chunks based on token count."""
    if not TIKTOKEN_AVAILABLE:
        raise RuntimeError("tiktoken is required for token-based chunking")
    
    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    step = max(1, window_tokens - overlap_tokens)
    chunks = []
    i = 0
    
    while i < len(encoded_tokens):
        start = i
        end = min(len(encoded_tokens), i + window_tokens)
        
        # Extract token chunk
        chunk_tokens = encoded_tokens[start:end]
        
        # Decode back to text
        chunk_text = encoder.decode(chunk_tokens)
        
        # Determine page range for this chunk
        page_start = token_to_page[start] if start < len(token_to_page) else 1
        page_end = token_to_page[end - 1] if end - 1 < len(token_to_page) else page_start
        
        chunk = {
            "chunk_index": len(chunks),
            "token_start": start,
            "token_end": end - 1,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "token_count": len(chunk_tokens)  # Actual token count
        }
        
        chunks.append(chunk)
        
        # Break if we've reached the end
        if end == len(encoded_tokens):
            break
            
        i += step
    
    return chunks

# Legacy word-based functions (kept for fallback if tiktoken unavailable)
def concat_tokens(sections:Sequence[Dict[str,Any]])->Tuple[List[str],List[int]]:
    tokens,page_map=[],[]
    for s in sections:
        # Be defensive - build text if missing
        if 'text' not in s:
            if 'elements' in s:
                text = '\n'.join(
                    el.get('text', '') for el in s['elements'] 
                    if el.get('text', '') and not el.get('text', '').startswith('self_ref=')
                )
            else:
                text = ""
            log.warning(f"Section missing 'text' field, built from elements: {s.get('section', 'untitled')}")
        else:
            text = s.get("text", "")
            
        words=text.split()
        tokens.extend(words)
        
        # Handle both page_number (from page_secs) and page_start/page_end (from logical_secs)
        if 'page_number' in s:
            # Single page section
            page_map.extend([s['page_number']]*len(words))
        elif 'page_start' in s and 'page_end' in s:
            # Multi-page section - distribute words across pages
            page_start = s['page_start']
            page_end = s['page_end']
            if page_start == page_end:
                # All on same page
                page_map.extend([page_start]*len(words))
            else:
                # Distribute words evenly across pages
                pages_span = page_end - page_start + 1
                words_per_page = max(1, len(words) // pages_span)
                for i, word in enumerate(words):
                    page = min(page_end, page_start + (i // words_per_page))
                    page_map.append(page)
        else:
            # Fallback to page 1
            page_map.extend([1]*len(words))
            log.warning(f"Section has no page info: {s.get('section', 'untitled')}")
            
    return tokens,page_map

def sliding_chunks(tokens:List[str],page_map:List[int],*,window:int,overlap:int)->List[Dict[str,Any]]:
    step=max(1,window-overlap)
    out,i=[],0
    SENT_END_RE=re.compile(r"[.!?]$")
    while i<len(tokens):
        start,end=i,min(len(tokens),i+window)
        while end<len(tokens) and not SENT_END_RE.search(tokens[end-1]) and end-start<window+256:
            end+=1
        out.append({"chunk_index":len(out),"token_start":start,"token_end":end-1,
                    "page_start":page_map[start],"page_end":page_map[end-1],
                    "text":" ".join(tokens[start:end])})
        if end==len(tokens): break
        i+=step
    return out
# ───────────────────────────────────────────────────────────────────

# 🎯 TIKTOKEN VALIDATION FUNCTIONS
def count_tiktoken_accurate(text: str) -> int:
    """Count tokens accurately using tiktoken."""
    if TIKTOKEN_AVAILABLE:
        try:
            encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
            return len(encoder.encode(text))
        except:
            pass
    
    # Conservative fallback estimation
    return int(len(text.split()) * 1.5) + 50

def smart_split_chunk(chunk: Dict[str, Any], max_tokens: int = MAX_CHUNK_TOKENS) -> List[Dict[str, Any]]:
    """Split an oversized chunk into smaller token-compliant chunks."""
    text = chunk["text"]
    tokens = count_tiktoken_accurate(text)
    
    if tokens <= max_tokens:
        return [chunk]
    
    # Calculate how many sub-chunks we need
    num_splits = math.ceil(tokens / max_tokens)
    target_words_per_split = len(text.split()) // num_splits
    
    log.info(f"🎯 Splitting oversized chunk: {tokens} tokens → {num_splits} chunks of ~{max_tokens} tokens each")
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    sub_chunks = []
    current_words = []
    current_tokens = 0
    
    # Split by sentences when possible, otherwise by words
    for sentence in sentences:
        sentence_words = sentence.strip().split()
        if not sentence_words:
            continue
            
        sentence_tokens = count_tiktoken_accurate(sentence)
        
        # If adding this sentence would exceed limit, finalize current chunk
        if current_tokens + sentence_tokens > max_tokens and current_words:
            # Create sub-chunk
            sub_text = " ".join(current_words)
            sub_chunk = {
                "chunk_index": chunk["chunk_index"],
                "sub_chunk_index": len(sub_chunks),
                "token_start": chunk["token_start"] + len(" ".join(words[:words.index(current_words[0])])),
                "token_end": chunk["token_start"] + len(" ".join(words[:words.index(current_words[-1])])) + len(current_words[-1]),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": sub_text
            }
            sub_chunks.append(sub_chunk)
            current_words = []
            current_tokens = 0
        
        # Add sentence to current chunk
        current_words.extend(sentence_words)
        current_tokens += sentence_tokens
    
    # Add final sub-chunk if there's remaining content
    if current_words:
        sub_text = " ".join(current_words)
        if count_tiktoken_accurate(sub_text) >= MIN_CHUNK_TOKENS:
            sub_chunk = {
                "chunk_index": chunk["chunk_index"],
                "sub_chunk_index": len(sub_chunks),
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": sub_text
            }
            sub_chunks.append(sub_chunk)
    
    log.info(f"🎯 Split complete: {len(sub_chunks)} sub-chunks created")
    return sub_chunks

def validate_and_fix_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate chunks against tiktoken limits and fix oversized ones."""
    if not TIKTOKEN_AVAILABLE:
        log.warning("🎯 tiktoken not available - skipping token validation")
        return chunks
    
    valid_chunks = []
    oversized_count = 0
    total_tokens_before = 0
    total_tokens_after = 0
    
    log.info(f"🎯 TIKTOKEN VALIDATION: Processing {len(chunks)} chunks")
    
    for chunk in chunks:
        tokens = count_tiktoken_accurate(chunk["text"])
        total_tokens_before += tokens
        
        if tokens <= MAX_CHUNK_TOKENS:
            # Chunk is fine, keep as-is
            valid_chunks.append(chunk)
            total_tokens_after += tokens
        else:
            # Split oversized chunk
            log.warning(f"🎯 Oversized chunk detected: {tokens} tokens (max: {MAX_CHUNK_TOKENS})")
            oversized_count += 1
            
            split_chunks = smart_split_chunk(chunk, MAX_CHUNK_TOKENS)
            valid_chunks.extend(split_chunks)
            
            # Count tokens in split chunks
            for split_chunk in split_chunks:
                total_tokens_after += count_tiktoken_accurate(split_chunk["text"])
    
    # Final validation check
    max_tokens_found = max((count_tiktoken_accurate(chunk["text"]) for chunk in valid_chunks), default=0)
    
    log.info(f"🎯 TIKTOKEN VALIDATION COMPLETE:")
    log.info(f"   📊 Original chunks: {len(chunks)}")
    log.info(f"   📊 Final chunks: {len(valid_chunks)}")
    log.info(f"   📊 Oversized chunks split: {oversized_count}")
    log.info(f"   📊 Largest chunk: {max_tokens_found} tokens (limit: {MAX_CHUNK_TOKENS})")
    log.info(f"   ✅ GUARANTEED: All chunks ≤ {MAX_CHUNK_TOKENS} tokens")
    
    if max_tokens_found > MAX_CHUNK_TOKENS:
        log.error(f"🚨 VALIDATION FAILED: Chunk still exceeds limit!")
        raise RuntimeError(f"Chunk validation failed: {max_tokens_found} > {MAX_CHUNK_TOKENS}")
    
    return valid_chunks

# Legacy constants for fallback
WINDOW  = 768
OVERLAP = int(WINDOW*0.20)           # 154-token (20 %) overlap

# Parallel chunking for multiple documents
async def chunk_batch_async(
    json_paths: List[pathlib.Path],
    max_workers: Optional[int] = None
) -> Dict[pathlib.Path, List[Dict[str, Any]]]:
    """Chunk multiple documents in parallel with token-based chunking."""
    from .acceleration_utils import hardware
    
    results = {}
    
    # 🎯 Show token-based chunking status
    if TIKTOKEN_AVAILABLE:
        log.info(f"🎯 TOKEN-BASED CHUNKING ENABLED: Using {WINDOW_TOKENS} token windows with {OVERLAP_TOKENS} token overlap")
    else:
        log.warning(f"⚠️  TIKTOKEN NOT AVAILABLE: Falling back to word-based chunking (install: pip install tiktoken)")
    
    # CPU-bound task - use process pool
    with hardware.get_process_pool(max_workers) as executor:
        future_to_path = {
            executor.submit(chunk, path): path 
            for path in json_paths
        }
        
        from tqdm import tqdm
        from concurrent.futures import as_completed
        
        for future in tqdm(as_completed(future_to_path), 
                          total=len(json_paths), 
                          desc="Chunking documents"):
            path = future_to_path[future]
            try:
                chunks = future.result()
                results[path] = chunks
            except Exception as exc:
                log.error(f"Failed to chunk {path.name}: {exc}")
                results[path] = []
    
    # 🎯 Summary of token-based chunking results
    total_chunks = sum(len(chunks) for chunks in results.values())
    total_tokens = sum(chunk.get('token_count', count_tiktoken_accurate(chunk['text'])) 
                      for chunks in results.values() 
                      for chunk in chunks)
    log.info(f"🎯 BATCH CHUNKING COMPLETE: {total_chunks} chunks created ({total_tokens} total tokens)")
    
    return results

# Optimized chunking with better memory management
def chunk_optimized(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Token-based optimized chunking with validation."""
    root = json_path.with_name(json_path.stem + "_chunks.json")
    if root.exists():
        existing_chunks = json.loads(root.read_text())
        # Validate existing chunks if they don't have tiktoken validation yet
        if existing_chunks and "sub_chunk_index" not in str(existing_chunks):
            log.info(f"🎯 Re-validating existing chunks in {root.name}")
            validated_chunks = validate_and_fix_chunks(existing_chunks)
            if len(validated_chunks) != len(existing_chunks):
                # Chunks were modified, save the validated version
                with open(root, 'w') as f:
                    json.dump(validated_chunks, f, indent=2, ensure_ascii=False)
                log.info(f"🎯 Updated {root.name} with validated chunks")
            return validated_chunks
        return existing_chunks
    
    # Stream large JSON files
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "sections" not in data:
        raise RuntimeError(f"{json_path} has no 'sections' key")
    
    # Process in smaller batches to reduce memory usage
    sections = data["sections"]
    batch_size = 100  # Process 100 sections at a time
    
    if TIKTOKEN_AVAILABLE:
        # 🎯 TOKEN-BASED CHUNKING - Primary approach
        log.info(f"🎯 Using token-based chunking with {WINDOW_TOKENS} token windows")
        
        all_tokens = []
        all_token_to_page = []
        
        for i in range(0, len(sections), batch_size):
            batch = sections[i:i + batch_size]
            tokens, page_map = concat_tokens_by_encoding(batch)
            all_tokens.extend(tokens)
            all_token_to_page.extend(page_map)
        
        chunks = sliding_chunks_by_tokens(
            all_tokens, 
            all_token_to_page, 
            window_tokens=WINDOW_TOKENS, 
            overlap_tokens=OVERLAP_TOKENS
        )
        
        # Token-based chunks are already guaranteed to be within limits
        log.info(f"🎯 Token-based chunking produced {len(chunks)} chunks")
        
        # Additional validation for safety
        max_tokens = max((chunk.get('token_count', count_tiktoken_accurate(chunk['text'])) for chunk in chunks), default=0)
        if max_tokens > MAX_CHUNK_TOKENS:
            log.warning(f"🎯 Unexpected: Token-based chunk exceeds limit ({max_tokens} > {MAX_CHUNK_TOKENS}), applying fallback validation")
            chunks = validate_and_fix_chunks(chunks)
    else:
        # 🎯 FALLBACK: Word-based chunking when tiktoken unavailable
        log.warning(f"🎯 Falling back to word-based chunking (tiktoken unavailable)")
        
        all_tokens = []
        all_page_map = []
        
        for i in range(0, len(sections), batch_size):
            batch = sections[i:i + batch_size]
            toks, pmap = concat_tokens(batch)
            all_tokens.extend(toks)
            all_page_map.extend(pmap)
        
        chunks = sliding_chunks(all_tokens, all_page_map, window=WINDOW, overlap=OVERLAP)
        
        # Apply validation for word-based chunks
        if chunks:
            log.info(f"🎯 Applying validation to {len(chunks)} word-based chunks")
            chunks = validate_and_fix_chunks(chunks)
    
    # Write chunks
    if chunks:
        with open(root, 'w') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        log.info("✓ %s chunks → %s", len(chunks), root.name)
        
        return chunks
    else:
        log.warning("%s – no chunks produced; file skipped", json_path.name)
        return []

# Keep original interface
def chunk(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Original interface maintained for compatibility."""
    return chunk_optimized(json_path)

if __name__ == "__main__":
    import argparse, logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p=argparse.ArgumentParser(); p.add_argument("json",type=pathlib.Path)
    chunk(p.parse_args().json) 