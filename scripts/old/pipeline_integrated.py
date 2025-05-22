#!/usr/bin/env python3
"""
Integrated Misophonia-Research Pipeline ▸ v1.2  (2025-05-16)
================================================================
One **single-command** script that drives a PDF through the *same* seven
stages the legacy multi-file tool-chain covered — now with **one GPT call**
that populates *all* bibliographic fields at once.

1. **Extract**
   Docling parses every page, falling back in order to
   `unstructured.partition.pdf` *then* a plain PyPDF2 pass. Each page is
   OCR-patched if it contains mostly garbled or empty text.

2. **Clean & Structure**:
   Latin-1 "mojibake" repair and whitespace
   normalisation, then heading detection that groups page blocks into
   logical *sections* (`Introduction`, `Methods`, …).

3. **Heuristic metadata draft**:
   cues from the filename + first-page
   header build a bibliographic skeleton (title, authors, year, DOI,
   journal …).

4. **LLM enrichment (single GPT-4-mini call)**:
   one request returns **all**
   final keys and merges them on top of the heuristic draft:
       • doc_type ‧ title ‧ authors ‧ year ‧ journal ‧ doi  
       • abstract ‧ keywords ‧ research_topics  
       • peer_reviewed ‧ open_access ‧ license ‧ open_access_status

5. **Chunk**:
   the clean body text is split into 768-token windows with
   20 % overlap while respecting sentence boundaries.

6. **Upsert rows**:
   a paper-level row goes to `research_documents`, each
   chunk to `research_chunks` (with inherited metadata).

7. **Embed**:
   every chunk is embedded with `text-embedding-ada-002` and
   the 1 536-D vector is stored in Supabase (pgvector).

The order **Extract → Enrich → Chunk → Embed** guarantees that chunks
always carry the *final* metadata before they are vectorised.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
from collections import Counter
from datetime import datetime
import random
from textwrap import dedent

from typing import Any, Dict, List, Sequence
from rich.console import Console
from rich.logging import RichHandler

# ────────────── third-party deps (graceful fail) ──────────────── #
try:
    from dotenv import load_dotenv
    import PyPDF2
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from openai import OpenAI
    from supabase import create_client
    from tqdm import tqdm
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}.  Run `pip install -r requirements.txt`.\n"
    )

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ───────────────────────── configuration ───────────────────────── #
DEFAULT_WINDOW = 768
DEFAULT_OVERLAP = int(DEFAULT_WINDOW * 0.20)  # 154
GPT_MODEL_META = "gpt-4.1-mini-2025-04-14"
EMBED_MODEL = "text-embedding-ada-002"

TXT_DIR = pathlib.Path("documents/research/txt")
JSON_DIR = pathlib.Path("documents/research/json")

# ────────────────────── stage-toggle flags ─────────────────────── #
#   Flip any of these to False to skip the corresponding step
#   without touching CLI flags.
RUN_EXTRACT      = True   # stages 1-2: extract + clean/structure
RUN_LLM_ENRICH   = True   # stage 4: single GPT call
RUN_CHUNK        = True   # stage 5: 768-token windows
RUN_DB           = True   # stage 6: upsert rows to Supabase
RUN_EMBED        = True   # stage 7: generate embeddings

# ───────────────────────────── logging ──────────────────────────── #
_CONSOLE = Console(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True)],
)
log = logging.getLogger("pipeline")

# ─────────────────────── extraction helper stack ────────────────── #

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
    0x91: "‘",
    0x92: "’",
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

_WS_RE = re.compile(r"[ \t]+\n")
_HEADING_TYPES = {"title", "heading", "header", "subtitle", "subheading"}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_JOURNAL_RE = re.compile(
    r"(Journal|Revista|Proceedings|Annals|Neuroscience|Psychiatry|Psychology|Nature|Science)[^\n]{0,120}",
    re.I,
)
_ABSTRACT_RE = re.compile(
    r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:\n[A-Z][^\n]{0,60}\n|\Z)", re.S
)
_KEYWORDS_RE = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)

META_FIELDS_CORE = [
    "doc_type",
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "abstract",
    "keywords",
    "research_topics",
]
EXTRA_MD_FIELDS = [
    "peer_reviewed",
    "open_access",
    "license",
    "open_access_status",
]
META_FIELDS = META_FIELDS_CORE + EXTRA_MD_FIELDS

_DEF_META_TEMPLATE = {
    **{k: None for k in META_FIELDS_CORE},
    "doc_type": "scientific paper",
    "authors": [],
    "keywords": [],
    "research_topics": [],
    # new fields
    "peer_reviewed": None,
    "open_access": None,
    "license": None,
    "open_access_status": None,
}

# ───────────────────────── basic utilities ───────────────────────── #


def latin1_scrub(txt: str) -> str:
    return txt.translate(_TRANSLATE_LAT1)


def normalize_ws(txt: str) -> str:
    txt = _WS_RE.sub("\n", txt)
    return re.sub(r"[ \t]{2,}", " ", txt).strip()


def pct_ascii_letters(txt: str) -> float:
    letters = sum(ch.isascii() and ch.isalpha() for ch in txt)
    return letters / max(1, len(txt))


def needs_ocr(txt: str) -> bool:
    return (not txt.strip()) or ("\x00" in txt) or (pct_ascii_letters(txt) < 0.15)


# ──────────────────────── utility: ASCII-NUL scrubber ────────────────────── #
def scrub_nuls(obj: Any) -> Any:
    """Recursively remove ASCII-NUL characters from any str inside *obj*."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, list):
        return [scrub_nuls(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_nuls(v) for k, v in obj.items()}
    return obj


# ───────────── Docling → Unstructured → PyPDF2 cascade ───────────── #


def _make_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    return DocumentConverter({InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def elements_from_unstructured(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    from unstructured.partition.pdf import partition_pdf

    els = partition_pdf(str(pdf), strategy="hi_res")
    pages: Dict[int, List[Dict[str, str]]] = {}
    for el in els:
        pn = getattr(el.metadata, "page_number", 1)
        pages.setdefault(pn, []).append(
            {"type": el.category or "paragraph", "text": normalize_ws(str(el))}
        )
    out = []
    for pn, items in sorted(pages.items()):
        out.append(
            {
                "section": f"Page {pn}",
                "page_number": pn,
                "text": "\n".join(i["text"] for i in items),
                "elements": items,
            }
        )
    return out


def elements_from_pypdf(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(pdf, "rb") as fh:
        for pn, pg in enumerate(PyPDF2.PdfReader(fh).pages, 1):
            try:
                raw = pg.extract_text() or ""
            except Exception:
                raw = ""
            raw = normalize_ws(raw)
            paragraphs = [p for p in re.split(r"\n{2,}", raw) if p.strip()]
            els = [{"type": "paragraph", "text": p} for p in paragraphs]
            out.append(
                {
                    "section": f"Page {pn}",
                    "page_number": pn,
                    "text": "\n".join(paragraphs),
                    "elements": els,
                }
            )
    return out


# ────────────────── Docling element-level extractor ────────────────── #
def elements_from_docling(doc) -> List[Dict[str, Any]]:
    pages: Dict[int, List[Dict[str, str]]] = {}
    for item, _lvl in doc.iterate_items():
        pg  = getattr(item.prov[0], "page_no", 1)
        lbl = (getattr(item, "label", "") or "").upper()
        if lbl in ("TITLE", "SECTION_HEADER", "HEADER"):
            typ = "heading"
        elif lbl == "LIST_ITEM":
            typ = "list_item"
        elif lbl == "TABLE":
            typ = "table"
        else:
            typ = "paragraph"
        pages.setdefault(pg, []).append({"type": typ, "text": str(getattr(item, "text", item)).strip()})
    out = []
    for pn in sorted(pages):
        text = "\n".join(el["text"] for el in pages[pn])
        out.append({"section": f"Page {pn}", "page_number": pn, "text": text, "elements": pages[pn]})
    return out


# ───────────── Markdown serialiser that keeps lists / tables ──────────── #
def render_sections_as_markdown(sections: List[Dict[str, Any]]) -> str:
    md: List[str] = []
    for sec in sections:
        md.append(f"# {sec.get('section','(Untitled Section)')}")
        if "page_start" in sec and "page_end" in sec:
            md.append(f"*Pages {sec['page_start']}–{sec['page_end']}*")
        for el in sec.get("elements", []):
            typ = (el.get("type") or "paragraph").lower()
            txt = el.get("text","").rstrip()
            if typ == "list_item":
                md.append(f"- {txt}")
            elif typ == "table":
                md.append(txt)
            else:
                md.append(txt)
        md.append("")
    return "\n".join(md).strip()


# ───────────── group blocks into logical sections ───────────── #


def group_sections_by_headings(
    page_secs: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    logical: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    last_page = 1
    for psec in page_secs:
        pn = psec.get("page_number") or last_page
        last_page = pn
        for el in psec.get("elements", []):
            kind = (el.get("type") or "").lower()
            text = el.get("text", "").strip()
            if kind in _HEADING_TYPES and text:
                if current:
                    # 🆕 Materialise .text before we push it
                    current["text"] = "\n".join(
                        e["text"] for e in current["elements"]
                    )
                    logical.append(current)
                current = {
                    "section": text,
                    "page_start": pn,
                    "page_end": pn,
                    "elements": [el.copy()],
                }
            else:
                if not current:
                    current = {
                        "section": "(untitled)",
                        "page_start": pn,
                        "page_end": pn,
                        "elements": [],
                    }
                current["elements"].append(el.copy())
                current["page_end"] = pn
    # final pending section
    if current:
        # 🆕 Materialise .text before we push it
        current["text"] = "\n".join(e["text"] for e in current["elements"])
        logical.append(current)
    return logical


# ─────────────── bibliographic helpers (unchanged) ─────────────── #


def _authors(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(a).strip() for a in val if a]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())


def extract_bib_from_filename(pdf: pathlib.Path) -> Dict[str, Any]:
    stem = pdf.stem
    m = re.search(r"\b(19|20)\d{2}\b", stem)
    year = int(m.group(0)) if m else None
    if m:
        author = stem[: m.start()].strip()
        title = stem[m.end() :].strip(" -_")
    else:
        parts = stem.split(" ", 1)
        author = parts[0]
        title = parts[1] if len(parts) == 2 else None
    return {"authors": [author] if author else [], "year": year, "title": title}


def extract_bib_from_header(txt: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if m := _DOI_RE.search(txt):
        meta["doi"] = m.group(0)
    if m := _JOURNAL_RE.search(txt):
        meta["journal"] = " ".join(m.group(0).split())
    if m := _ABSTRACT_RE.search(txt):
        meta["abstract"] = " ".join(m.group(1).split())
    if m := _KEYWORDS_RE.search(txt):
        kws = [k.strip(" ;.,") for k in re.split(r"[;,]", m.group(1)) if k.strip()]
        meta["keywords"] = kws
        meta["research_topics"] = kws
    return meta


def merge_meta(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged = _DEF_META_TEMPLATE.copy()
    for src in sources:
        for k in META_FIELDS:
            v = src.get(k)
            if v not in (None, "", [], {}):
                merged[k] = v
    merged["authors"] = _authors(merged["authors"])
    merged["keywords"] = merged["keywords"] or []
    merged["research_topics"] = merged["research_topics"] or []
    return merged


# ─────────────────────── LLM metadata helpers ─────────────────────── #


def _extract_first_n_words(txt: str, n: int = 3000) -> str:
    return " ".join(txt.split()[:n])


def generate_metadata(
    client: OpenAI, text: str, model: str = GPT_MODEL_META
) -> Dict[str, Any]:
    """Call OpenAI API once to retrieve all metadata fields."""
    prompt = dedent(
        f"""\
        Extract the following metadata from this scientific paper and return exactly one JSON object with keys:
          • doc_type (e.g. "Scientific Paper, Blog Article, Interview, etc.")
          • title
          • authors (array of strings)
          • year (integer)
          • journal (string or null)
          • DOI (string or null)
          • abstract (string or null)
          • keywords (array of strings)
          • research_topics (array of strings)
          • peer_reviewed (boolean or null)
          • open_access (boolean or null)
          • license (string or null)
          • open_access_status (string or null)

        If a field is not present, set it to null or an empty array. Here is the paper's full text:

        {text}
    """
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts structured metadata from scientific papers.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    m = re.search(r"{[\s\S]*}", content)
    if m:
        content = m.group(0)
    return json.loads(content)


# ──────────────────────── extraction main step ─────────────────────── #


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
            page_secs = elements_from_docling(bundle.document)
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
            page_secs = elements_from_unstructured(pdf)
        except Exception as exc:
            log.warning("unstructured failed on %s → %s", pdf.name, exc)

    # –– 3. PyPDF last resort
    if not page_secs and not docling_only:
        log.info("PyPDF2 fallback on %s", pdf.name)
        page_secs = elements_from_pypdf(pdf)

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
        group_sections_by_headings(page_secs) if keep_markup else page_secs
    )

    header_txt = " ".join(s["text"] for s in page_secs[:2])[:8000]
    heuristic_meta = merge_meta(
        bundle.document.metadata.model_dump()
        if "bundle" in locals() and hasattr(bundle.document, "metadata")
        else {},
        extract_bib_from_filename(pdf),
        extract_bib_from_header(header_txt),
    )

    # single GPT call for **all** metadata fields -----------------------
    llm_meta: Dict[str, Any] = {}
    if enrich_llm and OPENAI_API_KEY:
        try:
            oa = OpenAI(api_key=OPENAI_API_KEY)
            llm_meta = generate_metadata(
                oa, _extract_first_n_words(" ".join(s["text"] for s in logical_secs))
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
    md_text = render_sections_as_markdown(logical_secs) if keep_markup else "\n".join(
        "# " + s.get("section", "(No title)") + "\n\n" + s.get("text", "") for s in logical_secs
    )
    txt_path.write_text(md_text, "utf-8")

    if stats is not None:
        stats["processed"] += 1

    return json_path


# ─────────────────────────── chunk & embed ─────────────────────────── #


def concat_tokens(
    sections: Sequence[Dict[str, Any]]
) -> tuple[List[str], List[int]]:
    tokens, page_map = [], []
    for s in sections:
        page = s.get("page_number") or 1
        words = s.get("text", "").split()
        tokens.extend(words)
        page_map.extend([page] * len(words))
    return tokens, page_map


def sliding_chunks(
    tokens: List[str], page_map: List[int], *, window: int, overlap: int
) -> List[Dict[str, Any]]:
    step = max(1, window - overlap)
    out, i = [], 0
    SENT_END_RE = re.compile(r"[.!?]$")
    while i < len(tokens):
        start, end = i, min(len(tokens), i + window)
        while (
            end < len(tokens)
            and not SENT_END_RE.search(tokens[end - 1])
            and end - start < window + 256
        ):
            end += 1
        out.append(
            {
                "chunk_index": len(out),
                "token_start": start,
                "token_end": end - 1,
                "page_start": page_map[start],
                "page_end": page_map[end - 1],
                "text": " ".join(tokens[start:end]),
            }
        )
        if end == len(tokens):
            break
        i += step
    return out


# ─────────────────────────── database helpers ──────────────────────── #


def init_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        sys.exit("⛔  SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def upsert_document(sb, meta: Dict[str, Any]) -> str:
    meta = scrub_nuls(meta)     # ← NUL-safe
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
    if getattr(res, "error", None):
        raise RuntimeError(res.error)
    return res.data[0]["id"]


def insert_chunks(
    sb, doc_id: str, chunks: Sequence[Dict[str, Any]], src_json: pathlib.Path
):
    ts = datetime.utcnow().isoformat()
    rows = [
        {
            "document_id": doc_id,
            **ch,
            "metadata": {},
            "chunking_strategy": "token_window",
            "source_file": str(src_json),
            "created_at": ts,
            "text": scrub_nuls(ch["text"]),
        }
        for ch in chunks
    ]
    for i in range(0, len(rows), 500):
        sb.table("research_chunks").insert(rows[i : i + 500]).execute()


def embed_and_store(sb, oa: OpenAI, chunks: Sequence[Dict[str, Any]]):
    for ch in tqdm(chunks, desc="Embedding", leave=False):
        vec = (
            oa.embeddings.create(model=EMBED_MODEL, input=ch["text"])
            .data[0]
            .embedding
        )
        sb.table("research_chunks").update({"embedding": vec}).match(
            {"document_id": ch["document_id"], "chunk_index": ch["chunk_index"]}
        ).execute()


# ─────────────────────── Supabase helper (skip duplicates) ───────────────── #
def get_processed_pdfs(sb) -> set[str]:
    """Return absolute PDF paths already present in `research_documents`."""
    try:
        rows = sb.table("research_documents").select("source_pdf").execute().data or []
        return {r["source_pdf"] for r in rows if r.get("source_pdf")}
    except Exception:
        return set()


# ───────────────────────────── orchestrator ────────────────────────── #


def process_one(pdf: pathlib.Path, args, conv, sb, oa, stats: Counter):
    log.info("⇢ %s", pdf.name)
    # ── 1. Extract (+clean/structure) ───────────────────────────────
    if RUN_EXTRACT:
        json_path = extract_pdf(
            pdf,
            TXT_DIR,
            JSON_DIR,
            conv,
            overwrite=args.overwrite,
            ocr_lang=args.ocr_lang,
            keep_markup=not args.no_markup,
            docling_only=args.docling_only,
            stats=stats,
            enrich_llm=RUN_LLM_ENRICH,
        )
    else:
        json_path = JSON_DIR / f"{pdf.stem}.json"
        if not json_path.exists():
            log.error("✖ %s skipped: JSON artefact missing and RUN_EXTRACT is False", pdf.name)
            stats["failed"] += 1
            return
    obj = json.loads(json_path.read_text("utf-8"))

    do_chunk = RUN_CHUNK and not args.skip_chunk
    if do_chunk:
        tokens, page_map = concat_tokens(obj["sections"])
        chunks = sliding_chunks(tokens, page_map,
                                window=args.window,
                                overlap=args.overlap)
    else:
        tokens, page_map, chunks = [], [], []

    if RUN_DB and not args.skip_db and sb:
        meta_row = {k: obj.get(k) for k in META_FIELDS}  # core fields
        meta_row["source_pdf"] = obj.get("source_pdf")  # ensure source_pdf is included
        doc_id = upsert_document(sb, meta_row)
        if chunks:
            insert_chunks(sb, doc_id, chunks, json_path)
            for ch in chunks:
                ch["document_id"] = doc_id
            if chunks and RUN_EMBED and not args.skip_embed and oa:
                embed_and_store(sb, oa, chunks)
    log.info("✓ done %s (%d chunks)", pdf.name, len(chunks))


def gather_pdfs(src: pathlib.Path):
    if src.is_file() and src.suffix.lower() == ".pdf":
        yield src.resolve()
    else:
        yield from (p for p in src.rglob("*.pdf") if p.is_file())


def cli():
    ap = argparse.ArgumentParser("Integrated PDF→Vector pipeline")
    ap.add_argument(
        "src",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("documents/research/Global"),
        help="PDF file or directory (default: documents/research/Global)",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if artefacts exist")
    ap.add_argument("--skip-chunk", action="store_true")
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--ocr-lang", default="eng")
    ap.add_argument("--no-markup", action="store_true", help="skip element capture")
    ap.add_argument("--docling-only", action="store_true", help="fail if Docling cannot extract")
    ap.add_argument("--max", type=int, default=0, help="process at most N PDFs")
    ap.add_argument("--selection", choices=["sequential","random"],
                    default="sequential",
                    help="choose first/--max or a random sample")
    args = ap.parse_args()

    if not OPENAI_API_KEY and not args.skip_embed:
        sys.exit("⛔  OPENAI_API_KEY required for embeddings")

    stats = Counter(processed=0, ocr_pages=0, failed=0)
    sb = init_supabase() if not args.skip_db else None
    oa = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    conv = _make_converter()

    TXT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    pdf_list = list(gather_pdfs(args.src))

    # ── skip PDFs already loaded into DB ──────────────────────────────────
    if sb:
        done = get_processed_pdfs(sb)
        pdf_list = [p for p in pdf_list if str(p.resolve()) not in done or args.overwrite]

    # ── selection / cap ───────────────────────────────────────────────────
    if args.max > 0:
        if args.selection == "random" and len(pdf_list) > args.max:
            pdf_list = random.sample(pdf_list, args.max)
        else:
            pdf_list = pdf_list[: args.max]

    for pdf in tqdm(pdf_list, desc="Files"):
        try:
            process_one(pdf, args, conv, sb, oa, stats)
        except Exception as exc:
            log.exception("Failed on %s → %s", pdf.name, exc)
            stats["failed"] += 1
    
    log.info("Done. processed=%s  ocr_pages=%s  failed=%s", stats["processed"], stats["ocr_pages"], stats["failed"])


if __name__ == "__main__":
    cli()
