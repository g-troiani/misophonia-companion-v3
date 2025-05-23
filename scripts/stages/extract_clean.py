#!/usr/bin/env python3
"""
Stage 1-2 — *Extract PDF → clean text → logical sections*  
Optimized version with concurrent processing support.
Outputs `<n>.json` (+ a pretty `.txt`) in `documents/research/{json,txt}/`.
This file is **fully self-contained** – no import from `stages.common`.
"""
from __future__ import annotations

import json, logging, os, pathlib, re, sys
from collections import Counter
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Sequence, Optional
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ─── helpers formerly in common.py ──────────────────────────────────
_LATIN1_REPLACEMENTS = {  # windows-1252 fallback
    0x82: "‚",
    0x83: "ƒ",
    0x84: "„",
    0x85: "…",
    0x86: "†",
    0x87: "‡",
    0x88: "ˆ",
    0x89: "‰",
    0x8A: "Š",
    0x8B: "‹",
    0x8C: "Œ",
    0x8E: "Ž",
    0x91: "'",
    0x92: "'",
    0x93: '"',
    0x94: '"',
    0x95: "•",
    0x96: "–",
    0x97: "—",
    0x98: "˜",
    0x99: "™",
    0x9A: "š",
    0x9B: "›",
    0x9C: "œ",
    0x9E: "ž",
    0x9F: "Ÿ",
}
_TRANSLATE_LAT1 = str.maketrans(_LATIN1_REPLACEMENTS)
def latin1_scrub(txt:str)->str: return txt.translate(_TRANSLATE_LAT1)

_WS_RE = re.compile(r"[ \t]+\n")
def normalize_ws(txt:str)->str:
    return re.sub(r"[ \t]{2,}", " ", _WS_RE.sub("\n", txt)).strip()

def pct_ascii_letters(txt:str)->float:
    letters=sum(ch.isascii() and ch.isalpha() for ch in txt)
    return letters/max(1,len(txt))

def needs_ocr(txt:str)->bool:
    return (not txt.strip()) or ("\x00" in txt) or (pct_ascii_letters(txt)<0.15)

def scrub_nuls(obj:Any)->Any:
    if isinstance(obj,str):  return obj.replace("\x00","")
    if isinstance(obj,list): return [scrub_nuls(x) for x in obj]
    if isinstance(obj,dict): return {k:scrub_nuls(v) for k,v in obj.items()}
    return obj

def _make_converter()->DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    return DocumentConverter({InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
# ───────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client            # only needed if you later push rows
from tqdm import tqdm

# ─── optional heavy deps ----------------------------------------------------
try:
    import PyPDF2                             # raw fallback
    from unstructured.partition.pdf import partition_pdf
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError as exc:                    # pragma: no cover
    sys.exit(f"Missing dependency → {exc}.  Run `pip install -r requirements.txt`.")

# ─── env / paths ------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REPO_ROOT  = pathlib.Path(__file__).resolve().parents[2]
TXT_DIR    = REPO_ROOT / "documents" / "research" / "txt"
JSON_DIR   = REPO_ROOT / "documents" / "research" / "json"
TXT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

GPT_META_MODEL = "gpt-4.1-mini-2025-04-14"

log = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           helper utilities                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_LATIN = {0x91:"'",0x92:"'",0x93:'"',0x94:'"',0x95:"•",0x96:"–",0x97:"—"}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_JOURNAL_RE = re.compile(r"(Journal|Proceedings|Annals|Psychiatry|Psychology|Nature|Science)[^\n]{0,120}", re.I)
_ABSTRACT_RE = re.compile(r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:\n[A-Z][^\n]{0,60}\n|\Z)", re.S)
_KEYWORDS_RE = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)
_HEADING_TYPES = {"title","heading","header","subtitle","subheading"}

# ─── minimal bib helpers ───────────────────────────────────────────────────
def _authors(val)->List[str]:
    if val is None: return []
    if isinstance(val,list): return [str(x).strip() for x in val if x]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())

_DEF_META = {
    "doc_type":"scientific paper", "title":None,"authors":[],"year":None,"journal":None,"doi":None,
    "abstract":None,"keywords":[],"research_topics":[],
    "peer_reviewed":None,"open_access":None,"license":None,"open_access_status":None,
}

def merge_meta(*sources:Dict[str,Any])->Dict[str,Any]:
    m=_DEF_META.copy()
    for src in sources:
        for k,v in src.items():
            if v not in (None,"",[],{}): m[k]=v
    m["authors"]=_authors(m["authors"])
    m["keywords"]=m["keywords"] or []
    m["research_topics"]=m["research_topics"] or []
    return m

def bib_from_filename(pdf:pathlib.Path)->Dict[str,Any]:
    s=pdf.stem
    m=re.search(r"\b(19|20)\d{2}\b",s)
    yr=int(m.group(0)) if m else None
    if m:
        auth=s[:m.start()].strip(); title=s[m.end():].strip(" -_")
    else:
        parts=s.split(" ",1); auth=parts[0]; title=parts[1] if len(parts)==2 else None
    return {"authors":[auth] if auth else [],"year":yr,"title":title}

def bib_from_header(txt:str)->Dict[str,Any]:
    md={}
    if m:=_DOI_RE.search(txt): md["doi"]=m.group(0)
    if m:=_JOURNAL_RE.search(txt): md["journal"]=" ".join(m.group(0).split())
    if m:=_ABSTRACT_RE.search(txt): md["abstract"]=" ".join(m.group(1).split())
    if m:=_KEYWORDS_RE.search(txt):
        kws=[k.strip(" ;.,") for k in re.split(r"[;,]",m.group(1)) if k.strip()]
        md["keywords"]=kws; md["research_topics"]=kws
    return md

# ─── GPT enrichment ────────────────────────────────────────────────────────
def _first_words(txt:str,n:int=3000)->str: return " ".join(txt.split()[:n])

def gpt_metadata(text:str)->Dict[str,Any]:
    if not OPENAI_API_KEY: return {}
    cli=OpenAI(api_key=OPENAI_API_KEY)
    prompt=dedent(f"""
        Extract metadata (doc_type,title,authors,year,journal,doi,abstract,
        keywords,research_topics,peer_reviewed,open_access,license,
        open_access_status) from this paper and return one JSON object.
        Text:
        {text}
    """)
    rsp=cli.chat.completions.create(model=GPT_META_MODEL,temperature=0,
        messages=[{"role":"system","content":"Structured metadata extractor"},
                  {"role":"user","content":prompt}])
    txt=rsp.choices[0].message.content
    json_txt=re.search(r"{[\s\S]*}",txt)
    return json.loads(json_txt.group(0) if json_txt else "{}")        # type: ignore[arg-type]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         core extraction logic                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _docling_elements(doc)->List[Dict[str,Any]]:
    p:Dict[int,List[Dict[str,str]]]={}
    for it,_ in doc.iterate_items():
        pg=getattr(it.prov[0],"page_no",1)
        lbl=(getattr(it,"label","") or "").upper()
        typ="heading" if lbl in ("TITLE","SECTION_HEADER","HEADER") else \
            "list_item" if lbl=="LIST_ITEM" else \
            "table" if lbl=="TABLE" else "paragraph"
        p.setdefault(pg,[]).append({"type":typ,"text":str(getattr(it,"text",it)).strip()})
    out=[]
    for pn in sorted(p):
        out.append({"section":f"Page {pn}","page_number":pn,
                    "text":"\n".join(el["text"] for el in p[pn]),
                    "elements":p[pn]})
    return out

def _unstructured_elements(pdf:pathlib.Path)->List[Dict[str,Any]]:
    els=partition_pdf(str(pdf),strategy="hi_res")
    pages:Dict[int,List[Dict[str,str]]]={}
    for el in els:
        pn=getattr(el.metadata,"page_number",1)
        pages.setdefault(pn,[]).append({"type":el.category or "paragraph",
                                        "text":normalize_ws(str(el))})
    return [{"section":f"Page {pn}","page_number":pn,
             "text":"\n".join(e["text"] for e in it),"elements":it}
            for pn,it in sorted(pages.items())]

def _pypdf_elements(pdf:pathlib.Path)->List[Dict[str,Any]]:
    out=[]
    with open(pdf,"rb") as fh:
        for pn,pg in enumerate(PyPDF2.PdfReader(fh).pages,1):
            raw=normalize_ws(pg.extract_text() or "")
            paras=[p for p in re.split(r"\n{2,}",raw) if p.strip()]
            out.append({"section":f"Page {pn}","page_number":pn,
                        "text":"\n".join(paras),
                        "elements":[{"type":"paragraph","text":p} for p in paras]})
    return out

def _group_by_headings(page_secs:Sequence[Dict[str,Any]])->List[Dict[str,Any]]:
    out=[]; cur=None; last=1
    for s in page_secs:
        pn=s.get("page_number",last); last=pn
        for el in s.get("elements",[]):
            kind=(el.get("type")or"").lower(); txt=el.get("text","").strip()
            if kind in _HEADING_TYPES and txt:
                if cur: out.append(cur)
                cur={"section":txt,"page_start":pn,"page_end":pn,"elements":[el.copy()]}
            else:
                if not cur:
                    cur={"section":"(untitled)","page_start":pn,"page_end":pn,"elements":[]}
                cur["elements"].append(el.copy()); cur["page_end"]=pn
    if cur and not out: out.append(cur)
    return out

def _sections_md(sections:List[Dict[str,Any]])->str:
    md=[]
    for s in sections:
        md.append(f"# {s.get('section','(Untitled)')}")
        for el in s.get("elements",[]): md.append(el.get("text",""))
        md.append("")
    return "\n".join(md).strip()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         Inline extract_pdf from common                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_pdf(
    pdf: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    conv: DocumentConverter,
    *,
    overwrite: bool = False,
    ocr_lang: str = "eng",
    keep_markup: bool = True,
    docling_only: bool = False,
    stats: Counter | None = None,
    enrich_llm: bool = True,
) -> pathlib.Path:
    """Return path to JSON payload ready for downstream stages."""
    txt_path = txt_dir / f"{pdf.stem}.txt"
    json_path = json_dir / f"{pdf.stem}.json"
    if not overwrite and txt_path.exists() and json_path.exists():
        return json_path

    # –– 1. Docling
    page_secs: List[Dict[str, Any]] = []
    bundle = None
    try:
        bundle = conv.convert(str(pdf))
        if keep_markup:
            page_secs = _docling_elements(bundle.document)
        else:
            full = bundle.document.export_to_text(page_break_marker="\f")
            page_secs = [{
                "section": "Full document",
                "page_number": 1,
                "text": full,
                "elements": [{"type": "paragraph", "text": full}],
            }]
    except Exception as exc:
        log.warning("Docling failed on %s → %s", pdf.name, exc)

    # –– 2. Unstructured fallback
    if not page_secs and not docling_only:
        try:
            page_secs = _unstructured_elements(pdf)
        except Exception as exc:
            log.warning("unstructured failed on %s → %s", pdf.name, exc)

    # –– 3. PyPDF last resort
    if not page_secs and not docling_only:
        log.info("PyPDF2 fallback on %s", pdf.name)
        page_secs = _pypdf_elements(pdf)

    if not page_secs:
        raise RuntimeError("No text extracted from PDF")

    # Latin-1 scrub + OCR repair
    for sec in page_secs:
        sec["text"] = latin1_scrub(sec.get("text", ""))
        for el in sec.get("elements", []):
            el["text"] = latin1_scrub(el.get("text", ""))
        pn = sec.get("page_number")
        need_ocr_before = bool(pn and needs_ocr(sec["text"]))
        if need_ocr_before:
            try:
                from pdfplumber import open as pdfopen
                import pytesseract

                with pdfopen(str(pdf)) as doc:
                    pil = doc.pages[pn - 1].to_image(resolution=300).original
                ocr_txt = normalize_ws(
                    pytesseract.image_to_string(pil, lang=ocr_lang)
                )
                if ocr_txt:
                    sec["text"] = ocr_txt
                    sec["elements"] = [
                        {"type": "paragraph", "text": p}
                        for p in re.split(r"\n{2,}", ocr_txt)
                        if p.strip()
                    ]
                    if stats is not None:
                        stats["ocr_pages"] += 1
            except Exception:
                pass

    # when markup is disabled we already have final sections
    logical_secs = (
        _group_by_headings(page_secs) if keep_markup else page_secs
    )

    # CRITICAL: Ensure EVERY section has a 'text' field
    # This is required by the chunking stage
    for sec in logical_secs:
        if 'elements' in sec:
            # Build text from elements if not already present
            element_texts = []
            for el in sec.get("elements", []):
                el_text = el.get("text", "")
                # Skip internal docling metadata that starts with self_ref=
                if el_text and not el_text.startswith('self_ref='):
                    element_texts.append(el_text)
            
            # Always set text field (overwrite if exists to ensure consistency)
            sec["text"] = "\n".join(element_texts)
        elif 'text' not in sec:
            # Fallback for sections without elements
            sec["text"] = ""
    
    # Also ensure page_secs have text (for debugging/consistency)
    for sec in page_secs:
        if 'text' not in sec and 'elements' in sec:
            element_texts = []
            for el in sec.get("elements", []):
                el_text = el.get("text", "")
                if el_text and not el_text.startswith('self_ref='):
                    element_texts.append(el_text)
            sec["text"] = "\n".join(element_texts)

    header_txt = " ".join(s["text"] for s in page_secs[:2])[:8000]
    heuristic_meta = merge_meta(
        bundle.document.metadata.model_dump()
        if "bundle" in locals() and hasattr(bundle.document, "metadata")
        else {},
        bib_from_filename(pdf),
        bib_from_header(header_txt),
    )

    # single GPT call for **all** metadata fields -----------------------
    llm_meta: Dict[str, Any] = {}
    if enrich_llm and OPENAI_API_KEY:
        try:
            oa = OpenAI(api_key=OPENAI_API_KEY)
            llm_meta = gpt_metadata(
                _first_words(" ".join(s["text"] for s in logical_secs))
            )
        except Exception as exc:
            log.warning("LLM metadata extraction failed on %s → %s", pdf.name, exc)

    meta = merge_meta(heuristic_meta, llm_meta)

    payload = {
        **meta,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_pdf": str(pdf.resolve()),
        "sections": logical_secs,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    # --- Markdown serialisation (rich) ---
    md_text = _sections_md(logical_secs) if keep_markup else "\n".join(
        "# " + s.get("section", "(No title)") + "\n\n" + s.get("text", "") for s in logical_secs
    )
    txt_path.write_text(md_text, "utf-8")

    if stats is not None:
        stats["processed"] += 1

    return json_path

# New: Thread-safe converter pool
class ConverterPool:
    """Thread-safe pool of DocumentConverter instances."""
    def __init__(self, size: int = None):
        self.size = size or mp.cpu_count()
        self._converters = []
        self._lock = mp.Lock()
        self._initialized = False
    
    def get(self):
        """Get a converter from the pool."""
        with self._lock:
            if not self._initialized:
                self._initialize()
            return self._converters[mp.current_process()._identity[0] % self.size]
    
    def _initialize(self):
        """Lazy initialize converters."""
        for _ in range(self.size):
            self._converters.append(_make_converter())
        self._initialized = True

# Global converter pool
_converter_pool = ConverterPool()

def extract_pdf_concurrent(
    pdf: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    *,
    overwrite: bool = False,
    ocr_lang: str = "eng",
    keep_markup: bool = True,
    docling_only: bool = False,
    stats: Counter | None = None,
    enrich_llm: bool = True,
) -> pathlib.Path:
    """Thread-safe version of extract_pdf that uses converter pool."""
    conv = _converter_pool.get()
    return extract_pdf(
        pdf, txt_dir, json_dir, conv,
        overwrite=overwrite,
        ocr_lang=ocr_lang,
        keep_markup=keep_markup,
        docling_only=docling_only,
        stats=stats,
        enrich_llm=enrich_llm
    )

async def extract_batch_async(
    pdfs: List[pathlib.Path],
    *,
    overwrite: bool = False,
    keep_markup: bool = True,
    ocr_lang: str = "eng",
    enrich_llm: bool = True,
    max_workers: Optional[int] = None,
) -> List[pathlib.Path]:
    """Extract multiple PDFs concurrently."""
    from .acceleration_utils import hardware
    
    stats = Counter()
    results = []
    
    # Use process pool for CPU-bound extraction
    with hardware.get_process_pool(max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(
                extract_pdf_concurrent,
                pdf, TXT_DIR, JSON_DIR,
                overwrite=overwrite,
                ocr_lang=ocr_lang,
                keep_markup=keep_markup,
                enrich_llm=enrich_llm,
                stats=stats
            ): pdf
            for pdf in pdfs
        }
        
        # Process completed tasks
        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="Extracting PDFs"):
            pdf = future_to_pdf[future]
            try:
                json_path = future.result()
                results.append(json_path)
            except Exception as exc:
                log.error(f"Failed to extract {pdf.name}: {exc}")
                
    log.info(f"Extraction complete: {stats['processed']} processed, {stats['ocr_pages']} OCR pages")
    return results

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          public entry-point                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ---------------------------------------------------------------------------
#  Stage-1/2 wrapper (now paper-thin)
# ---------------------------------------------------------------------------
def run_one(
    pdf: pathlib.Path,
    *,
    overwrite: bool = False,
    keep_markup: bool = True,
    ocr_lang: str = "eng",
    enrich_llm: bool = True,
    conv=None,
    stats: Counter | None = None,
) -> pathlib.Path:
    conv = conv or _make_converter()
    out = extract_pdf(
        pdf,
        TXT_DIR,
        JSON_DIR,
        conv,
        overwrite=overwrite,
        ocr_lang=ocr_lang,
        keep_markup=keep_markup,
        enrich_llm=enrich_llm,
        stats=stats,
    )
    return out

def json_path_for(pdf:pathlib.Path)->pathlib.Path:
    """Helper if Stage 1 is disabled but its artefact exists."""
    return JSON_DIR / f"{pdf.stem}.json"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("pdf",type=pathlib.Path)
    args=p.parse_args()
    print(run_one(args.pdf)) 