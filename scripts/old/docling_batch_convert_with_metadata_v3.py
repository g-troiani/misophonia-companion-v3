#!/usr/bin/env python3
# scripts/docling_batch_convert_with_metadata_v3.py
# -----------------------------------------------------------------------------
"""
Convert every PDF in documents/research/Global →
  • plain UTF-8 TXT  (documents/research/txt/)
  • rich JSON        (documents/research/json/)

Features
--------
* Unicode-aware PDF text extraction (Docling ➜ unstructured ➜ PyPDF2)
* OCR fallback (pdfplumber + Tesseract) for scans / font-obfuscated PDFs
* "Mojibake" repair for common Windows-1252 bytes
* Hierarchical markup ("elements") per page
* Idempotent: re-runs touch only new / missing outputs
"""
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pathlib
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List
import textwrap

import PyPDF2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
# from docling.doc_items import TextItem, SectionHeaderItem, ListItem

# ───────────────────────── logging ───────────────────────── #

_CONSOLE = Console(width=120)  # fixed width → no "vertical" wrapping

def init_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), "INFO"),
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True)],
    )

log = logging.getLogger(__name__)

# ─────────────── Docling converter helper ─────────────── #

def make_converter() -> DocumentConverter:
    pdf_opts           = PdfPipelineOptions()
    pdf_opts.do_ocr    = True           # safe default: will OCR if needed
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )

# ───────────────────────── path helpers ───────────────────────── #

def iter_pdfs(root: pathlib.Path) -> Iterable[pathlib.Path]:
    yield from (p for p in root.rglob("*.pdf") if p.is_file() and not p.name.startswith("."))

def txt_path_for(src: pathlib.Path, txt_dir: pathlib.Path) -> pathlib.Path:
    return txt_dir / f"{src.stem}.txt"

def json_path_for(src: pathlib.Path, json_dir: pathlib.Path) -> pathlib.Path:
    return json_dir / f"{src.stem}.json"

# ──────────── mojibake detection & fixes ──────────── #

_LATIN1_REPLACEMENTS = {
    0x82: '‚',  0x83: 'ƒ',  0x84: '„',  0x85: '…',  0x86: '†',  0x87: '‡',
    0x88: 'ˆ',  0x89: '‰',  0x8A: 'Š',  0x8B: '‹',  0x8C: 'Œ',  0x8E: 'Ž',
    0x91: '‘',  0x92: '’',  0x93: '"',  0x94: '"',  0x95: '•',  0x96: '–',
    0x97: '—',  0x98: '˜',  0x99: '™',  0x9A: 'š',  0x9B: '›',  0x9C: 'œ',
    0x9E: 'ž',  0x9F: 'Ÿ',
}
_LATIN1_TRANS = str.maketrans(_LATIN1_REPLACEMENTS)

def latin1_scrub(txt: str) -> str:
    return txt.translate(_LATIN1_TRANS)

def pct_ascii_letters(txt: str) -> float:
    letters = sum(ch.isascii() and ch.isalpha() for ch in txt)
    return letters / max(1, len(txt))

def needs_ocr(txt: str) -> bool:
    return (not txt.strip()) or ("\x00" in txt) or (pct_ascii_letters(txt) < 0.15)

# ────────── OCR helpers (lazy import) ────────── #

_pytesseract = _pdfplumber = _PIL_Image = None

def _lazy_import_ocr() -> None:
    global _pytesseract, _pdfplumber, _PIL_Image
    if _pytesseract is None:
        _pytesseract = importlib.import_module("pytesseract")
        _pdfplumber  = importlib.import_module("pdfplumber")
        _PIL_Image   = importlib.import_module("PIL.Image")

def ocr_page(pdf: pathlib.Path, page_no: int, lang: str = "eng") -> str:
    _lazy_import_ocr()
    with _pdfplumber.open(str(pdf)) as doc:
        pil = doc.pages[page_no - 1].to_image(resolution=300).original
    return _pytesseract.image_to_string(pil, lang=lang)

# ─────────── utility ─────────── #

_ws_re = re.compile(r"[ \t]+\n")
def normalize_whitespace(txt: str) -> str:
    txt = _ws_re.sub("\n", txt)
    return re.sub(r"[ \t]{2,}", " ", txt).strip()

# ────────── per-page extraction helpers ────────── #

def elements_from_unstructured(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    from unstructured.partition.pdf import partition_pdf
    els = partition_pdf(str(pdf), strategy="hi_res")
    pages: Dict[int, List[Dict[str, str]]] = {}
    for el in els:
        pn = getattr(el.metadata, "page_number", 1)
        pages.setdefault(pn, []).append(
            {"type": el.category or "paragraph", "text": normalize_whitespace(str(el))}
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

def elements_from_docling(doc) -> list[dict]:
    """
    Docling ≥2.2 version – one pass over `doc.iterate_items()` and
    bucket everything per page_no.
    """
    pages: dict[int, list[dict]] = {}

    for item, _level in doc.iterate_items():
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

        pages.setdefault(pg, []).append(
            {"type": typ, "text": getattr(item, "text", str(item)).strip()}
        )

    out = []
    for pn in sorted(pages):
        text = "\n".join(el["text"] for el in pages[pn])
        out.append(
            {
                "section": f"Page {pn}",
                "page_number": pn,
                "text": text,
                "elements": pages[pn],
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
            raw = normalize_whitespace(raw)
            paragraphs = [p for p in re.split(r"\n{2,}", raw) if p.strip()]
            els = [{"type": "paragraph", "text": p} for p in paragraphs]
            out.append(
                {
                    "section": f"Page {pn}",
                    "page_number": pn,
                    "text": "\n".join(p for p in paragraphs),
                    "elements": els,
                }
            )
    return out

# ─────────────── bibliographic helpers (unchanged) ─────────────── #
META_FIELDS = [
    "title", "authors", "year", "journal", "doi",
    "abstract", "keywords", "research_topics",
]
DOI_RE       = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
JOURNAL_RE   = re.compile(
    r"(Journal|Revista|Proceedings|Annals|Neuroscience|Psychiatry|Psychology|Nature|Science)[^\n]{0,120}",
    re.I)
ABSTRACT_RE  = re.compile(r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:\n[A-Z][^\n]{0,60}\n|\Z)", re.S)
KEYWORDS_RE  = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)

def _authors(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(a).strip() for a in val if a]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())

def extract_bib_from_filename(pdf: pathlib.Path) -> Dict[str, Any]:
    stem = pdf.stem
    m    = re.search(r"\b(19|20)\d{2}\b", stem)
    year = int(m.group(0)) if m else None
    if m:
        author = stem[:m.start()].strip()
        title  = stem[m.end():].strip(" -_")
    else:
        parts  = stem.split(" ", 1)
        author = parts[0]
        title  = parts[1] if len(parts) == 2 else None
    return {"authors": [author] if author else [], "year": year, "title": title}

def extract_bib_from_header(txt: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if (m := DOI_RE.search(txt)):      meta["doi"]     = m.group(0)
    if (m := JOURNAL_RE.search(txt)):  meta["journal"] = " ".join(m.group(0).split())
    if (m := ABSTRACT_RE.search(txt)): meta["abstract"] = " ".join(m.group(1).split())
    if (m := KEYWORDS_RE.search(txt)):
        kws = [k.strip(" ;.,") for k in re.split(r"[;,]", m.group(1)) if k.strip()]
        meta["keywords"] = kws
        meta["research_topics"] = kws
    return meta

def merge_metadata(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"doc_type": "scientific paper"}
    for f in META_FIELDS: merged[f] = None
    for src in sources:
        for k in META_FIELDS:
            v = src.get(k)
            if v not in (None, "", [], {}):
                merged[k] = v
    merged["authors"]         = _authors(merged["authors"])
    merged["keywords"]        = merged["keywords"] or []
    merged["research_topics"] = merged["research_topics"] or []
    return merged

# ────────────── markdown renderer for .txt ────────────── #
def render_sections_as_markdown(sections: List[Dict[str, Any]]) -> str:
    """
    Turn each logical section into:
      # Section Title
      <elements...>
    exactly in the same order as JSON.
    """
    md: List[str] = []
    for sec in sections:
        # 1) heading (be safe if no 'section' key)
        md.append(f"# {sec.get('section', '(Untitled Section)')}")
        # optional page span if present
        if 'page_start' in sec and 'page_end' in sec:
            md.append(f"*Pages {sec['page_start']}–{sec['page_end']}*")

        # 2) body (safe get text)
        for el in sec.get("elements", []):
            typ = (el.get("type") or "paragraph").lower()
            txt = el.get("text", "").rstrip()
            if typ == "list_item":
                md.append(f"- {txt}")
            elif typ == "table":
                md.append(txt)
            else:
                md.append(txt)
        md.append("")  # blank line
    return "\n".join(md).strip()

# ───────── logical-section grouper ───────── #

_HEADING_TYPES = {"title", "heading", "header", "subtitle", "subheading"}

def group_sections_by_headings(page_secs: list[dict]) -> list[dict]:
    """
    Merge per-page sections into logical sections at each heading.
    - page_secs: list of {"page_number":…, "elements":[…]} entries
    Returns list of {
      section: heading text or "(untitled)",
      page_start, page_end,
      elements: [...]
    } in order.
    """
    logical: list[dict] = []
    current: dict | None = None
    last_page = 1

    for psec in page_secs:
        # get page number, default to last_page
        pn = psec.get("page_number") or last_page
        last_page = pn

        for el in psec.get("elements", []):
            kind = (el.get("type") or "").lower()
            text = el.get("text", "").strip()

            if kind in _HEADING_TYPES and text:
                # start a brand-new logical section
                if current:
                    logical.append(current)
                current = {
                    "section": text,
                    "page_start": pn,
                    "page_end": pn,
                    "elements": [el.copy()],
                }
            else:
                # non-heading or before any heading:
                if not current:
                    # first content without any heading
                    current = {
                        "section": "(untitled)",
                        "page_start": pn,
                        "page_end": pn,
                        "elements": [],
                    }
                current["elements"].append(el.copy())
                current["page_end"] = pn

    # if we never created any logical section, make a fallback
    if not logical and current:
        logical.append(current)

    return logical

# ───────────────────── single-file pipeline ───────────────────── #

def process_one(
    conv: DocumentConverter,
    pdf: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    overwrite: bool,
    ocr_lang: str,
    stats: Counter,
    keep_markup: bool = True,
    args: argparse.Namespace = None,
) -> bool:
    txt_path  = txt_path_for(pdf, txt_dir)
    json_path = json_path_for(pdf, json_dir)

    if (not overwrite) and txt_path.exists() and json_path.exists():
        return False

    # 1 ─ Docling  (always first, most trusted)
    sections: List[Dict[str, Any]] = []
    logical_secs: List[Dict[str, Any]] = []  # Initialize for all paths
    bundle   = None
    try:
        bundle = conv.convert(str(pdf))

        # ◀️  Fast-path disabled: fall back to per-block extraction so
        #    heading elements (title, heading, subtitle …) are preserved.
        if keep_markup:
            sections = elements_from_docling(bundle.document)
            # For Docling path, compute logical sections here
            logical_secs = group_sections_by_headings(sections)
        else:
            # plain-text fallback inside Docling (no markup wanted)
            full = bundle.document.export_to_text(page_break_marker="\f")
            sections = [{
                "section": "Full document",
                "page_number": 1,
                "text": full,
                "elements": [{"type": "paragraph", "text": full}],
            }]
            # For this case, sections are already logical
            logical_secs = sections

    except Exception as e:
        log.debug("Docling failed on %s: %s", pdf.name, e)

    # ── DEBUG summary: which extractor produced what? ────────────────────
    if log.isEnabledFor(logging.DEBUG):
        heading_types = {"title", "heading", "header", "subtitle", "subheading"}
        h_count = sum(
            1 for sec in sections
            for el in sec.get("elements", [])
            if (el.get("type") or "").lower() in heading_types
        )
        log.debug(
            "Extractor summary for %-40s  sections=%2d  total_elems=%4d  headings=%2d",
            pdf.name[:40],
            len(sections),
            sum(len(sec.get('elements', [])) for sec in sections),
            h_count
        )

    # 2 ─ unstructured  (only if Docling produced *nothing*)
    if not sections and keep_markup and not args.docling_only:
        try:
            page_sections = elements_from_unstructured(pdf)
            logical_secs = group_sections_by_headings(page_sections)
            sections = page_sections            # <- keep per-page view
        except Exception as e:
            log.debug("unstructured failed on %s: %s", pdf.name, e)

    # 3 ─ PyPDF fallback (last resort, unless --docling-only)
    if not sections and not args.docling_only:
        page_sections = elements_from_pypdf(pdf)
        logical_secs = group_sections_by_headings(page_sections)
        sections = page_sections            # <- keep per-page view

    if not sections:
        raise RuntimeError("no text extracted")

    # 4 ─ Latin-1 scrub + OCR
    for sec in sections:
        # safe get text, then scrub
        sec["text"] = latin1_scrub(sec.get("text", ""))
        for el in sec.get("elements", []):
            # safe get element text, then scrub
            el["text"] = latin1_scrub(el.get("text", ""))
        pn = sec.get("page_number") or sec.get("page_start")
        if pn is None:            # multi-page logical chunk
            continue
        if needs_ocr(sec["text"]):
            try:
                ocr_txt = normalize_whitespace(ocr_page(pdf, pn, ocr_lang))
                if ocr_txt:
                    sec["text"]     = ocr_txt
                    sec["elements"] = [
                        {"type": "paragraph", "text": p}
                        for p in re.split(r"\n{2,}", ocr_txt) if p.strip()
                    ]
                    stats["ocr_pages"] += 1
            except Exception as e:
                log.debug("OCR failed on %s p.%s: %s", pdf.name, pn, e)

    # 5 ─ metadata merge
    bundle_meta: Dict[str, Any] = {}
    if bundle and hasattr(bundle, "document"):
        meta_obj = getattr(bundle.document, "metadata", None)
        if meta_obj is not None:
            if hasattr(meta_obj, "model_dump"):
                bundle_meta = meta_obj.model_dump()
            elif hasattr(meta_obj, "dict"):
                bundle_meta = meta_obj.dict()

    header_meta   = extract_bib_from_header(" ".join(sec["text"] for sec in sections[:2])[:12000])
    filename_meta = extract_bib_from_filename(pdf)
    meta          = merge_metadata(bundle_meta, filename_meta, header_meta)

    payload = {
        **meta,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_pdf": str(pdf),
        "sections":   logical_secs,            # already computed for each path
    }

    # ───── write outputs ───── #
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Markdown serialisation ---
    # If Docling can natively export Markdown, use that (always has headings),
    # otherwise fall back to our per-section renderer.
    if keep_markup and bundle and hasattr(bundle.document, "export_to_markdown"):
        try:
            txt_md = bundle.document.export_to_markdown()
        except Exception:
            txt_md = render_sections_as_markdown(sections)
    else:
        txt_md = render_sections_as_markdown(sections)

    txt_path.write_text(txt_md, encoding="utf-8")

    # ── DEBUG: log element-type distribution right after extraction ──────
    if log.isEnabledFor(logging.DEBUG) and sections:
        for si, sec in enumerate(sections, 1):
            dist = Counter((el.get("type") or "paragraph").lower()
                           for el in sec.get("elements", []))
            log.debug(
                "§%-2d %-20s p%-3s  element_types=%s",
                si, sec["section"][:20], sec.get("page_number"), dict(dist)
            )

    stats["processed"] += 1
    return True

# ─────────────── batch driver ─────────────── #

def run(
    src_root: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    overwrite: bool,
    limit: int | None,
    ocr_lang: str = "eng",
    keep_markup: bool = True,
    args: argparse.Namespace = None,
) -> None:
    conv   = make_converter()
    pdfs   = list(iter_pdfs(src_root))
    if limit: pdfs = pdfs[:limit]

    stats = Counter()
    log.info("Starting – %s PDFs", len(pdfs))

    with tqdm(total=len(pdfs), unit="file", desc="PDFs") as bar:
        for pdf in pdfs:
            bar.set_postfix_str(pdf.name)
            try:
                process_one(conv, pdf, txt_dir, json_dir, overwrite, ocr_lang, stats, keep_markup, args)
            except Exception as e:
                stats["failed"] += 1
                log.error("‼️  %s failed: %s", pdf.name, e)
            bar.update()

    log.info("Done. processed=%s  ocr_pages=%s  failed=%s",
             stats["processed"], stats["ocr_pages"], stats["failed"])

# ─────────────── CLI ─────────────── #

def cli() -> None:
    ap = argparse.ArgumentParser(description="Batch PDF → TXT & JSON (Docling+OCR)")
    ap.add_argument("--src",      default="documents/research/Global")
    ap.add_argument("--txt-dir",  default="documents/research/txt")
    ap.add_argument("--json-dir", default="documents/research/json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log",      choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    ap.add_argument("--max",      type=int, default=0, help="process at most N PDFs")
    ap.add_argument("--ocr-lang", default="eng")
    ap.add_argument("--no-markup", action="store_true", help="skip element capture")
    ap.add_argument("--docling-only", action="store_true",
                    help="fail if Docling cannot extract (no fall-back extractors)")
    args = ap.parse_args()

    init_logging(args.log)
    run(
        pathlib.Path(args.src),
        pathlib.Path(args.txt_dir),
        pathlib.Path(args.json_dir),
        args.overwrite,
        None if args.max == 0 else args.max,
        args.ocr_lang,
        keep_markup=not args.no_markup,
        args=args,
    )

if __name__ == "__main__":
    cli()
