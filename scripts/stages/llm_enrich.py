#!/usr/bin/env python3
"""
Stage 4 — read an existing JSON, make **one** GPT-4 call that fills every
bibliographic field, and merge it back.
"""
from __future__ import annotations
import json, logging, pathlib, re, os
from textwrap import dedent
from typing import Any, Dict, List

# ─── minimal shared helpers ────────────────────────────────────────
def _authors(val:Any)->List[str]:
    if val is None: return []
    if isinstance(val,list): return [str(a).strip() for a in val if a]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())

META_FIELDS_CORE = [
    "doc_type","title","authors","year","journal","doi","abstract",
    "keywords","research_topics",
]
EXTRA_MD_FIELDS = ["peer_reviewed","open_access","license","open_access_status"]
META_FIELDS     = META_FIELDS_CORE+EXTRA_MD_FIELDS
_DEF_META_TEMPLATE = {**{k:None for k in META_FIELDS_CORE},
                      "doc_type":"scientific paper",
                      "authors":[], "keywords":[], "research_topics":[],
                      "peer_reviewed":None,"open_access":None,
                      "license":None,"open_access_status":None}

def merge_meta(*sources:Dict[str,Any])->Dict[str,Any]:
    merged=_DEF_META_TEMPLATE.copy()
    for src in sources:
        for k in META_FIELDS:
            v=src.get(k)
            if v not in (None,"",[],{}): merged[k]=v
    merged["authors"]=_authors(merged["authors"])
    merged["keywords"]=merged["keywords"] or []
    merged["research_topics"]=merged["research_topics"] or []
    return merged
# ───────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = "gpt-4.1-mini-2025-04-14"
log            = logging.getLogger(__name__)

def _first_words(txt:str,n:int=3000)->str: return " ".join(txt.split()[:n])

def _gpt(text:str)->Dict[str,Any]:
    if not OPENAI_API_KEY: return {}
    cli=OpenAI(api_key=OPENAI_API_KEY)
    prompt=dedent(f"""
        Extract all metadata fields ({', '.join(META_FIELDS)}) from the paper
        below and return ONE JSON object.  Text:  {text}
    """)
    rsp=cli.chat.completions.create(model=MODEL,temperature=0,
            messages=[{"role":"system","content":"metadata extractor"},
                      {"role":"user","content":prompt}])
    raw=rsp.choices[0].message.content
    m=re.search(r"{[\s\S]*}",raw)
    return json.loads(m.group(0) if m else "{}")

# ─── public function ───────────────────────────────────────────────────────
def enrich(json_path:pathlib.Path)->None:
    data=json.loads(json_path.read_text())
    # reconstruct the body text from the new structure
    full=" ".join(
        el.get("text","") for sec in data["sections"] for el in sec.get("elements",[])
    )
    new_meta=_gpt(_first_words(full))
    # ✅ merge – keep everything that was already there
    data.update(new_meta)                 # <- adds/overwrites meta fields
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), 'utf-8')
    log.info("✓ metadata enriched → %s", json_path.name)

if __name__ == "__main__":
    import argparse, logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p=argparse.ArgumentParser(); p.add_argument("json",type=pathlib.Path)
    enrich(p.parse_args().json) 