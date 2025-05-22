#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Shared helpers/constants for EVERY stage of the Misophonia pipeline
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sys
from collections import Counter
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

# third-party deps (same graceful-fail block as the monolith)
try:
    import PyPDF2
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from openai import OpenAI
    from supabase import create_client
except ImportError as exc:       # pragma: no cover
    sys.exit(f"Missing dependency: {exc}.  Run `pip install -r requirements.txt`.")

load_dotenv()

# ---------------------------------------------------------------------------
#  Global constants and paths
# ---------------------------------------------------------------------------
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

REPO_ROOT  = pathlib.Path(__file__).resolve().parents[2]  # /scripts/stages/..
TXT_DIR    = REPO_ROOT / "documents" / "research" / "txt"
JSON_DIR   = REPO_ROOT / "documents" / "research" / "json"
ARTEFACTS  = REPO_ROOT / "scripts" / "artefacts"
ARTEFACTS.mkdir(parents=True, exist_ok=True)

DEFAULT_WINDOW   = 768
DEFAULT_OVERLAP  = int(DEFAULT_WINDOW * 0.20)   # 154
GPT_MODEL_META   = "gpt-4.1-mini-2025-04-14"
EMBED_MODEL      = "text-embedding-ada-002"

# ---------------------------------------------------------------------------
#  Latin-1 replacement table  (unchanged)
# ---------------------------------------------------------------------------
_LATIN1_REPLACEMENTS = {
    0x82: "‚", 0x83: "ƒ", 0x84: "„", 0x85: "…", 0x86: "†", 0x87: "‡",
    0x88: "ˆ", 0x89: "‰", 0x8A: "Š", 0x8B: "‹", 0x8C: "Œ", 0x8E: "Ž",
    0x91: "'", 0x92: "'", 0x93: '"', 0x94: '"', 0x95: "•", 0x96: "–",
    0x97: "—", 0x98: "˜", 0x99: "™", 0x9A: "š", 0x9B: "›", 0x9C: "œ",
    0x9E: "ž", 0x9F: "Ÿ",
}
_TRANSLATE_LAT1 = str.maketrans(_LATIN1_REPLACEMENTS)

_WS_RE          = re.compile(r"[ \t]+\n")
_HEADING_TYPES  = {"title", "heading", "header", "subtitle", "subheading"}
_DOI_RE         = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_JOURNAL_RE     = re.compile(
    r"(Journal|Revista|Proceedings|Annals|Neuroscience|Psychiatry|"
    r"Psychology|Nature|Science)[^\n]{0,120}", re.I)
_ABSTRACT_RE    = re.compile(r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:"
                             r"\n[A-Z][^\n]{0,60}\n|\Z)", re.S)
_KEYWORDS_RE    = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)

META_FIELDS_CORE = [
    "doc_type", "title", "authors", "year",
    "journal", "doi", "abstract", "keywords", "research_topics",
]
EXTRA_MD_FIELDS = ["peer_reviewed", "open_access", "license", "open_access_status"]
META_FIELDS      = META_FIELDS_CORE + EXTRA_MD_FIELDS

_DEF_META_TEMPLATE: Dict[str, Any] = {
    **{k: None for k in META_FIELDS_CORE},
    "doc_type": "scientific paper",
    "authors": [],
    "keywords": [],
    "research_topics": [],
    "peer_reviewed": None,
    "open_access": None,
    "license": None,
    "open_access_status": None,
}

# ---------------------------------------------------------------------------
#  Basic utilities
# ---------------------------------------------------------------------------
def latin1_scrub(txt:str)->str:        return txt.translate(_TRANSLATE_LAT1)
def normalize_ws(txt:str)->str:        return _WS_RE.sub("\n", txt).strip()
def pct_ascii_letters(txt:str)->float: return sum(ch.isascii() and ch.isalpha() for ch in txt) / max(1,len(txt))
def needs_ocr(txt:str)->bool:          return (not txt.strip()) or ("\x00" in txt) or (pct_ascii_letters(txt) < 0.15)

def scrub_nuls(obj:Any)->Any:
    if isinstance(obj,str):  return obj.replace("\x00","")
    if isinstance(obj,list): return [scrub_nuls(x) for x in obj]
    if isinstance(obj,dict): return {k:scrub_nuls(v) for k,v in obj.items()}
    return obj

# ---------------------------------------------------------------------------
#  PDF extraction helpers (Docling → Unstructured → PyPDF2 cascade)
# ---------------------------------------------------------------------------
def _make_converter()->DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    return DocumentConverter({InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

def elements_from_unstructured(pdf:pathlib.Path)->List[Dict[str,Any]]:
    from unstructured.partition.pdf import partition_pdf
    els = partition_pdf(str(pdf), strategy="hi_res")
    pages:Dict[int,List[Dict[str,str]]] = {}
    for el in els:
        pn = getattr(el.metadata,"page_number",1)
        pages.setdefault(pn,[]).append({"type":el.category or "paragraph",
                                        "text":normalize_ws(str(el))})
    out=[]
    for pn,items in sorted(pages.items()):
        out.append({"section":f"Page {pn}","page_number":pn,
                    "text":"\n".join(i["text"] for i in items),
                    "elements":items})
    return out

def elements_from_pypdf(pdf:pathlib.Path)->List[Dict[str,Any]]:
    out=[]
    with open(pdf,"rb") as fh:
        for pn,pg in enumerate(PyPDF2.PdfReader(fh).pages,1):
            try:    raw = pg.extract_text() or ""
            except: raw = ""
            raw = normalize_ws(raw)
            paragraphs=[p for p in re.split(r"\n{2,}",raw) if p.strip()]
            els=[{"type":"paragraph","text":p} for p in paragraphs]
            out.append({"section":f"Page {pn}","page_number":pn,
                        "text":"\n".join(paragraphs),"elements":els})
    return out

def elements_from_docling(doc)->List[Dict[str,Any]]:
    pages:Dict[int,List[Dict[str,str]]]={}
    for item,_ in doc.iterate_items():
        pg  = getattr(item.prov[0],"page_no",1)
        lbl = (getattr(item,"label","") or "").upper()
        if   lbl in ("TITLE","SECTION_HEADER","HEADER"): typ="heading"
        elif lbl=="LIST_ITEM":  typ="list_item"
        elif lbl=="TABLE":      typ="table"
        else:                   typ="paragraph"
        pages.setdefault(pg,[]).append({"type":typ,"text":str(getattr(item,"text",item)).strip()})
    out=[]
    for pn in sorted(pages):
        out.append({"section":f"Page {pn}","page_number":pn,
                    "text":"\n".join(el["text"] for el in pages[pn]),
                    "elements":pages[pn]})
    return out

# ---------------------------------------------------------------------------
def render_sections_as_markdown(sections:List[Dict[str,Any]])->str:
    md=[]
    for sec in sections:
        md.append(f"# {sec.get('section','(Untitled Section)')}")
        if "page_start" in sec and "page_end" in sec:
            md.append(f"*Pages {sec['page_start']}–{sec['page_end']}*")
        for el in sec.get("elements",[]):
            typ=(el.get("type") or "paragraph").lower()
            txt=el.get("text","").rstrip()
            if   typ=="list_item": md.append(f"- {txt}")
            elif typ=="table":     md.append(txt)
            else:                  md.append(txt)
        md.append("")
    return "\n".join(md).strip()

def group_sections_by_headings(page_secs:Sequence[Dict[str,Any]])->List[Dict[str,Any]]:
    logical:List[Dict[str,Any]]=[]
    current:Dict[str,Any]|None=None
    last_page=1
    for psec in page_secs:
        pn=psec.get("page_number") or last_page
        last_page=pn
        for el in psec.get("elements",[]):
            kind=(el.get("type") or "").lower()
            text=el.get("text","").strip()
            if kind in _HEADING_TYPES and text:
                if current: logical.append(current)
                current={"section":text,"page_start":pn,"page_end":pn,"elements":[el.copy()]}
            else:
                if not current:
                    current={"section":"(untitled)","page_start":pn,
                             "page_end":pn,"elements":[]}
                current["elements"].append(el.copy())
                current["page_end"]=pn
    if current and not logical: logical.append(current)
    return logical

# ---------------------------------------------------------------------------
#  Metadata helpers
# ---------------------------------------------------------------------------
def _authors(val:Any)->List[str]:
    if val is None: return []
    if isinstance(val,list): return [str(a).strip() for a in val if a]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())

def extract_bib_from_filename(pdf:pathlib.Path)->Dict[str,Any]:
    stem=pdf.stem
    m=re.search(r"\b(19|20)\d{2}\b",stem)
    year=int(m.group(0)) if m else None
    if m:
        author=stem[:m.start()].strip()
        title=stem[m.end():].strip(" -_")
    else:
        parts=stem.split(" ",1)
        author=parts[0]
        title=parts[1] if len(parts)==2 else None
    return {"authors":[author] if author else [],"year":year,"title":title}

def extract_bib_from_header(txt:str)->Dict[str,Any]:
    meta={}
    if m:=_DOI_RE.search(txt):       meta["doi"]=m.group(0)
    if m:=_JOURNAL_RE.search(txt):   meta["journal"]=" ".join(m.group(0).split())
    if m:=_ABSTRACT_RE.search(txt):  meta["abstract"]=" ".join(m.group(1).split())
    if m:=_KEYWORDS_RE.search(txt):
        kws=[k.strip(" ;.,") for k in re.split(r"[;,]",m.group(1)) if k.strip()]
        meta["keywords"]=kws; meta["research_topics"]=kws
    return meta

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

# ---------------------------------------------------------------------------
#  LLM enrichment helpers
# ---------------------------------------------------------------------------
def _extract_first_n_words(txt:str,n:int=3000)->str:
    return " ".join(txt.split()[:n])

def generate_metadata(client:OpenAI,text:str,model:str=GPT_MODEL_META)->Dict[str,Any]:
    prompt=dedent(f"""\
       Extract the following metadata from this scientific paper and return exactly one JSON object with keys:
         • doc_type (e.g. "Scientific Paper, Blog Article, Interview, etc.")
         • title • authors • year • journal • DOI • abstract • keywords
         • research_topics • peer_reviewed • open_access • license • open_access_status

       If a field is not present, set it to null or an empty array. Here is the paper's full text:

       {text}
    """)
    resp=client.chat.completions.create(
        model=model,temperature=0,
        messages=[{"role":"system","content":"You are an assistant that extracts structured metadata from scientific papers."},
                  {"role":"user","content":prompt}])
    content=resp.choices[0].message.content
    m=re.search(r"{[\s\S]*}",content)
    if m: content=m.group(0)
    return json.loads(content)

# ---------------------------------------------------------------------------
#  extract_pdf  (entire original body, unchanged)
# ---------------------------------------------------------------------------
def extract_pdf(
    pdf:pathlib.Path,
    txt_dir:pathlib.Path,
    json_dir:pathlib.Path,
    conv:DocumentConverter,
    *,
    overwrite:bool=False,
    ocr_lang:str="eng",
    keep_markup:bool=True,
    docling_only:bool=False,
    stats:Counter|None=None,
    enrich_llm:bool=True,
)->pathlib.Path:
    txt_path=txt_dir/f"{pdf.stem}.txt"
    json_path=json_dir/f"{pdf.stem}.json"
    if not overwrite and txt_path.exists() and json_path.exists():
        return json_path
    # ---- Docling first ----------------------------------------------------
    page_secs=[]
    bundle=None
    try:
        bundle=conv.convert(str(pdf))
        if keep_markup:
            page_secs=elements_from_docling(bundle.document)
        else:
            full=bundle.document.export_to_text(page_break_marker="\f")
            page_secs=[{"section":"Full document","page_number":1,
                        "text":full,"elements":[{"type":"paragraph","text":full}]}]
    except Exception as exc:
        logging.warning("Docling failed on %s → %s", pdf.name, exc)
    # ---- Unstructured fallback -------------------------------------------
    if not page_secs and not docling_only:
        try: page_secs=elements_from_unstructured(pdf)
        except Exception as exc:
            logging.warning("unstructured failed on %s → %s", pdf.name, exc)
    # ---- PyPDF2 last resort ----------------------------------------------
    if not page_secs and not docling_only:
        logging.info("PyPDF2 fallback on %s", pdf.name)
        page_secs=elements_from_pypdf(pdf)
    if not page_secs:
        raise RuntimeError("No text extracted from PDF")
    # ---- latin-1 scrub + OCR repair --------------------------------------
    for sec in page_secs:
        sec["text"]=latin1_scrub(sec.get("text",""))
        for el in sec.get("elements",[]): el["text"]=latin1_scrub(el.get("text",""))
        pn=sec.get("page_number")
        if pn and needs_ocr(sec["text"]):
            try:
                from pdfplumber import open as pdfopen
                import pytesseract
                with pdfopen(str(pdf)) as doc:
                    pil=doc.pages[pn-1].to_image(resolution=300).original
                ocr_txt=normalize_ws(pytesseract.image_to_string(pil,lang=ocr_lang))
                if ocr_txt:
                    sec["text"]=ocr_txt
                    sec["elements"]=[{"type":"paragraph","text":p}
                                     for p in re.split(r"\n{2,}",ocr_txt) if p.strip()]
                    if stats is not None: stats["ocr_pages"] += 1
            except Exception: pass
    logical_secs=group_sections_by_headings(page_secs) if keep_markup else page_secs
    header_txt=" ".join(s["text"] for s in page_secs[:2])[:8000]
    heuristic_meta=merge_meta(
        bundle.document.metadata.model_dump() if bundle and hasattr(bundle.document,"metadata") else {},
        extract_bib_from_filename(pdf),
        extract_bib_from_header(header_txt),
    )
    llm_meta={}
    if enrich_llm and OPENAI_API_KEY:
        try:
            oa=OpenAI(api_key=OPENAI_API_KEY)
            llm_meta=generate_metadata(oa, _extract_first_n_words(" ".join(s["text"] for s in logical_secs)))
        except Exception as exc:
            logging.warning("LLM metadata extraction failed on %s → %s", pdf.name, exc)
    meta=merge_meta(heuristic_meta,llm_meta)
    payload={**meta,"created_at":datetime.utcnow().isoformat()+"Z",
             "source_pdf":str(pdf.resolve()),"sections":logical_secs}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    md_text=render_sections_as_markdown(logical_secs) if keep_markup else \
        "\n".join("# "+s.get("section","(No title)")+"\n\n"+s.get("text","") for s in logical_secs)
    txt_path.write_text(md_text,"utf-8")
    if stats is not None: stats["processed"] += 1
    return json_path

# ---------------------------------------------------------------------------
#  Chunk helpers
# ---------------------------------------------------------------------------
def concat_tokens(sections:Sequence[Dict[str,Any]])->tuple[List[str],List[int]]:
    tokens,page_map=[],[]
    for s in sections:
        page=s.get("page_number") or 1
        words=s.get("text","").split()
        tokens.extend(words)
        page_map.extend([page]*len(words))
    return tokens,page_map

def sliding_chunks(tokens:List[str],page_map:List[int],*,window:int,overlap:int)->List[Dict[str,Any]]:
    step=max(1,window-overlap)
    out=[]
    i=0
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

# ---------------------------------------------------------------------------
#  Supabase helpers
# ---------------------------------------------------------------------------
def init_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
    return create_client(SUPABASE_URL,SUPABASE_SERVICE_KEY)

def upsert_document(sb,meta:Dict[str,Any])->str:
    meta=scrub_nuls(meta)
    doi=meta.get("doi")
    if doi:
        existing=sb.table("research_documents").select("id").eq("doi",doi).limit(1).execute().data
        if existing:
            doc_id=existing[0]["id"]
            sb.table("research_documents").update(meta).eq("id",doc_id).execute()
            return doc_id
    res=sb.table("research_documents").insert(meta).execute()
    if getattr(res,"error",None): raise RuntimeError(res.error)
    return res.data[0]["id"]

def insert_chunks(sb,doc_id:str,chunks:Sequence[Dict[str,Any]],src_json:pathlib.Path):
    ts=datetime.utcnow().isoformat()
    rows=[{"document_id":doc_id,**ch,"metadata":{},"chunking_strategy":"token_window",
           "source_file":str(src_json),"created_at":ts,"text":scrub_nuls(ch["text"])}
          for ch in chunks]
    for i in range(0,len(rows),500):
        sb.table("research_chunks").insert(rows[i:i+500]).execute() 