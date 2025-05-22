#!/usr/bin/env python3
"""
Stage 5 — split the clean body text into **overlapping 768-token windows**
and dump a side-car `<n>_chunks.json`.
"""
from __future__ import annotations
import json, logging, math, pathlib, re
from typing import Dict, List, Sequence, Tuple, Any

# ─── helpers to split into token windows ───────────────────────────
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

WINDOW  = 768
OVERLAP = int(WINDOW*0.20)           # 154-token (20 %) overlap
log = logging.getLogger(__name__)

def chunk(json_path:pathlib.Path)->List[Dict[str,any]]:
    root=json_path.with_name(json_path.stem+"_chunks.json")
    if root.exists():                              # idempotent
        return json.loads(root.read_text())
    data=json.loads(json_path.read_text())
    if "sections" not in data:          # <- defensive
        raise RuntimeError(f"{json_path} has no 'sections' key – enrichment step lost context")
    toks,pmap=concat_tokens(data["sections"])
    chunks=sliding_chunks(toks,pmap,window=WINDOW,overlap=OVERLAP)
    if chunks:                                    # ← new guard
        root.write_text(json.dumps(chunks,indent=2,ensure_ascii=False))
        log.info("✓ %s chunks → %s", len(chunks), root.name)
    else:
        log.warning("%s – no chunks produced; file skipped", json_path.name)
    return chunks

if __name__ == "__main__":
    import argparse, logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p=argparse.ArgumentParser(); p.add_argument("json",type=pathlib.Path)
    chunk(p.parse_args().json) 