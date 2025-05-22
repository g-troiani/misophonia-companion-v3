#!/usr/bin/env python3
# scripts/docling_batch_convert_with_metadata_v2.py
# -----------------------------------------------------------------------------
"""
Convert every PDF in documents/research/Global →
  • plain UTF-8 TXT  (documents/research/txt/)
  • rich JSON        (documents/research/json/)

Features
---------
* Unicode-aware PDF text extraction (Docling ➜ unstructured ➜ PyPDF2)
* OCR fallback (pdfplumber + Tesseract) for scans / font-obfuscated PDFs
* "Mojibake" repair for common Windows-1252 bytes
* **Logical section detection** based on real headings, with page-span tracking
* Idempotent: re-runs touch only new / missing outputs

‼️  This version forces **UTF-8** encoding for std-streams and any
    implicit text-mode file openings, so you don't need to launch
    Python with `-X utf8` or set `PYTHONIOENCODING=utf-8`.
"""
# -----------------------------------------------------------------------------
from __future__ import annotations

# ──────────────────── UTF-8 bootstrap  (must be first!) ────────────────────
import os, sys, builtins, locale

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Best-effort locale tweak (ignored if locale isn't available on host)
try:
    locale.setlocale(locale.LC_CTYPE, "en_US.UTF-8")
except locale.Error:
    pass

# Re-configure std-streams – Python ≥ 3.7
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")

# Monkey-patch builtins.open so any **text-mode** call that forgets an
# explicit `encoding=` will still default to UTF-8 instead of the locale.
_builtin_open = builtins.open
def _open_utf8(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _builtin_open(*args, **kwargs)
builtins.open = _open_utf8
# ────────────────────────────────────────────────────────────────────────────

import argparse
import importlib
import json
import logging
import pathlib
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List

import PyPDF2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

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
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True  # safe default: will OCR if needed
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


# ───────────────────────── path helpers ───────────────────────── #


def iter_pdfs(root: pathlib.Path) -> Iterable[pathlib.Path]:
    yield from (
        p for p in root.rglob("*.pdf") if p.is_file() and not p.name.startswith(".")
    )


def txt_path_for(src: pathlib.Path, txt_dir: pathlib.Path) -> pathlib.Path:
    return txt_dir / f"{src.stem}.txt"


def json_path_for(src: pathlib.Path, json_dir: pathlib.Path) -> pathlib.Path:
    return json_dir / f"{src.stem}.json"


# ──────────── mojibake detection & fixes ──────────── #

_LATIN1_REPLACEMENTS = {
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
        _pdfplumber = importlib.import_module("pdfplumber")
        _PIL_Image = importlib.import_module("PIL.Image")


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


# ─────────── heading-level helper ─────────── #
def heading_level(block: Dict[str, Any], median_font: float) -> int:
    """
    Return 0 for body text, 1-for-Title (H1), 2-for-Section (H2), 3-for-Subsection (H3).
    Heuristics based on relative font size and/or explicit metadata from unstructured.
    """
    fs = block.get("font_size") or 0.0
    cat = (block.get("category") or "").lower()
    kind = (block.get("kind") or "").lower()

    # explicit tags from the extractors always win
    if kind in {"title", "heading", "header"} or cat in {"title", "section header"}:
        if fs >= median_font * 1.45:   # ≈ +45 %
            return 1                   # H1
        if fs >= median_font * 1.25:   # ≈ +25 %
            return 2                   # H2
        return 3

    # purely font-based fallback
    if fs >= median_font * 1.45:   # ≈ +45 %
        return 1                   # H1
    if fs >= median_font * 1.25:   # ≈ +25 %
        return 2                   # H2
    if fs >= median_font * 1.10:   # ≈ +10 %
        return 3                   # H3
    return 0


# ────────── per-page extraction helpers ────────── #


def elements_from_unstructured(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    from unstructured.partition.pdf import partition_pdf

    # ---- MODIFIED STRATEGY TEST ----
    # current_strategy = "hi_res"
    current_strategy = "fast"  # TRY THIS for text-based PDFs
    log.info(f"DEBUG: elements_from_unstructured - Using strategy: '{current_strategy}'")
    # ---- END MODIFIED STRATEGY TEST ----

    els_raw = partition_pdf(str(pdf), strategy=current_strategy)
    
    # ---- ADDED LOGGING FOR RAW UNSTRUCTURED ELEMENTS ----
    log.info(f"DEBUG: elements_from_unstructured - partition_pdf (strategy='{current_strategy}') returned {len(els_raw)} raw elements.")
    if els_raw:
        for el_idx, raw_el in enumerate(els_raw[:3]):  # Log first 3 raw elements
            metadata_dict = {}
            if hasattr(raw_el, 'metadata'):
                # Common metadata attributes from unstructured
                for meta_attr in ['coordinates', 'filename', 'filetype', 'page_number', 'link_texts', 'link_urls', 'links', 'font_size', 'text_color', 'emphasized_text_contents', 'emphasized_text_tags', 'parent_id', 'category_depth', 'detection_class_prob']:
                    if hasattr(raw_el.metadata, meta_attr):
                        attr_val = getattr(raw_el.metadata, meta_attr)
                        if isinstance(attr_val, str) and len(attr_val) > 70:
                             metadata_dict[meta_attr] = attr_val[:70] + "..."
                        elif isinstance(attr_val, list) and len(attr_val) > 0:
                            metadata_dict[meta_attr] = f"[List of {len(attr_val)}, first: {str(attr_val[0])[:50]}...]"
                        elif isinstance(attr_val, list):
                             metadata_dict[meta_attr] = "[]"
                        else:
                             metadata_dict[meta_attr] = attr_val
            
            log.info(f"DEBUG: elements_from_unstructured - Raw Unstructured Element {el_idx}: category='{raw_el.category}', text='{str(raw_el)[:70]}...', metadata='{metadata_dict}'")
    # ---- END ADDED LOGGING ----

    pages: Dict[int, List[Dict[str, Any]]] = {}
    for el in els_raw:  # Changed from 'els' to 'els_raw' to match partition_pdf output
        pn = getattr(el.metadata, "page_number", 1)
        pages.setdefault(pn, []).append(
            {
                "type": el.category or "paragraph",
                "text": normalize_whitespace(str(el)),
                "font_size": getattr(el.metadata, "font_size", None),
                "bold": getattr(el.metadata, "bold", False),
                "kind": el.category,
                "category": el.category,
            }
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


def elements_from_docling(doc) -> List[Dict[str, Any]]:
    pages_attr = getattr(doc, "pages", None)
    if not pages_attr:
        log.info("DEBUG: elements_from_docling - No 'pages' attribute found on doc object.")
        return []

    out: List[Dict[str, Any]] = []
    for idx, pg_val in enumerate(pages_attr, 1):  # Renamed pg to pg_val for clarity
        # ─ handle new API where pg == int ───────────────────────────────
        page_obj = pg_val
        if isinstance(pg_val, int):
            for meth in ("get_page", "page"):
                if hasattr(doc, meth):
                    try:
                        page_obj = getattr(doc, meth)(pg_val)
                        break
                    except Exception:
                        pass
        if page_obj is None:
            log.info(f"DEBUG: elements_from_docling - Page object for index {idx} is None. Skipping.")
            continue

        # ─ page number from metadata ───────────────────────────────────
        meta = getattr(page_obj, "metadata", None)
        pn = idx
        if meta is not None:
            if hasattr(meta, "model_dump"):
                pn = meta.model_dump().get("page_number", pn)
            elif hasattr(meta, "dict"):
                pn = meta.dict().get("page_number", pn)

        # ─ extract blocks (Docling ≥ 2.1) or fallback to plain text ────
        try:
            blocks = page_obj.export_to_blocks()
            # ---- ADDED LOGGING FOR RAW DOCLING BLOCKS ----
            if idx == 1:  # Log only for the first page to keep logs manageable
                log.info(f"DEBUG: elements_from_docling - Page {pn} - Extracted {len(blocks)} raw blocks via export_to_blocks().")
                for b_idx, raw_b in enumerate(blocks[:2]):  # Log first 2 raw blocks
                    log.info(f"DEBUG: elements_from_docling - Page {pn} - Raw Docling Block {b_idx} type: {type(raw_b)}")
                    # Try to print all attributes of the raw_b object
                    try:
                        block_dir = dir(raw_b)
                        log.info(f"DEBUG: elements_from_docling - Page {pn} - Raw Docling Block {b_idx} dir(): {block_dir}")
                        # Log some common attributes if they exist
                        block_data_to_log = {}
                        for attr_name in ['text', 'font', 'size', 'flags', 'color', 'font_size', 'font_name', 'kind', 'category', 'bbox', 'type', 'lines', 'spans']:
                            if hasattr(raw_b, attr_name):
                                val = getattr(raw_b, attr_name)
                                if isinstance(val, str) and len(val) > 100:
                                    block_data_to_log[attr_name] = val[:100] + "..."
                                elif isinstance(val, list) and len(val) > 3:
                                    block_data_to_log[attr_name] = f"List of {len(val)}, first 3: {val[:3]}"
                                else:
                                    block_data_to_log[attr_name] = val
                        log.info(f"DEBUG: elements_from_docling - Page {pn} - Raw Docling Block {b_idx} data: {block_data_to_log}")
                        # If it has spans (common in PyMuPDF detailed output)
                        if hasattr(raw_b, 'lines'):
                            lines = getattr(raw_b, 'lines', [])
                            if lines and hasattr(lines[0], 'spans'):
                                spans = getattr(lines[0], 'spans', [])
                                if spans:
                                    span_data = spans[0].__dict__ if hasattr(spans[0], '__dict__') else spans[0]
                                    log.info(f"DEBUG: elements_from_docling - Page {pn} - Block {b_idx}, First Line, First Span data: {span_data}")

                    except Exception as e_detail:
                        log.info(f"DEBUG: elements_from_docling - Page {pn} - Error getting details for raw block {b_idx}: {e_detail}")
                    # ---- END ADDED LOGGING ----

            els = [
                {
                    "type": getattr(b, "kind", "paragraph") or "paragraph",
                    "text": normalize_whitespace(getattr(b, "text", "")),
                    "font_size": getattr(b, "font_size", None),
                    "bold": getattr(b, "bold", False),
                    "kind": getattr(b, "kind", None) if hasattr(b, "kind") else None,
                    "category": getattr(b, "category", None),
                }
                for b in blocks
            ]
            txt = "\n".join(e["text"] for e in els)
        except Exception as e_docling_blocks:
            log.error(f"DEBUG: elements_from_docling - Page {pn} - Error in export_to_blocks or processing: {e_docling_blocks}")
            raw = page_obj.export_to_text() or ""
            parts = [p for p in re.split(r"\n{2,}", raw) if p.strip()]
            els = [
                {
                    "type": "paragraph",
                    "text": normalize_whitespace(p),
                    "font_size": None,
                    "bold": False,
                    "kind": "paragraph",
                    "category": None,
                }
                for p in parts
            ]
            txt = "\n".join(p for p in parts)

        out.append(
            {
                "section": f"Page {pn}",
                "page_number": pn,
                "text": txt,
                "elements": els,
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
            els = [
                {
                    "type": "paragraph",
                    "text": p,
                    "font_size": None,
                    "bold": False,
                    "kind": "paragraph",
                    "category": None,
                }
                for p in paragraphs
            ]
            out.append(
                {
                    "section": f"Page {pn}",
                    "page_number": pn,
                    "text": "\n".join(p for p in paragraphs),
                    "elements": els,
                }
            )
    return out


# ───────────── bibliographic helpers (unchanged) ───────────── #
META_FIELDS = [
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "abstract",
    "keywords",
    "research_topics",
]
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
JOURNAL_RE = re.compile(
    r"(Journal|Revista|Proceedings|Annals|Neuroscience|Psychiatry|Psychology|Nature|Science)[^\n]{0,120}",
    re.I,
)
ABSTRACT_RE = re.compile(
    r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:\n[A-Z][^\n]{0,60}\n|\Z)", re.S
)
KEYWORDS_RE = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)


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
    if m := DOI_RE.search(txt):
        meta["doi"] = m.group(0)
    if m := JOURNAL_RE.search(txt):
        meta["journal"] = " ".join(m.group(0).split())
    if m := ABSTRACT_RE.search(txt):
        meta["abstract"] = " ".join(m.group(1).split())
    if m := KEYWORDS_RE.search(txt):
        kws = [k.strip(" ;.,") for k in re.split(r"[;,]", m.group(1)) if k.strip()]
        meta["keywords"] = kws
        meta["research_topics"] = kws
    return meta


def merge_metadata(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"doc_type": "scientific paper"}
    for f in META_FIELDS:
        merged[f] = None
    for src in sources:
        for k in META_FIELDS:
            v = src.get(k)
            if v not in (None, "", [], {}):
                merged[k] = v
    merged["authors"] = _authors(merged["authors"])
    merged["keywords"] = merged["keywords"] or []
    merged["research_topics"] = merged["research_topics"] or []
    return merged


# ───────── markdown renderer for .txt ────────── #
def render_sections_as_markdown(sections: List[Dict[str, Any]]) -> str:
    """Serialize section list into GitHub-flavoured Markdown with proper heading levels."""
    md_lines: List[str] = []
    for sec in sections:
        if "level" in sec:
            lvl = max(1, min(int(sec["level"]), 6))   # clamp 1-6
            heading = sec.get("section", "")
            if heading:
                md_lines.append(f'{"#"*lvl} {heading}')
                md_lines.append("")                   # blank line after heading
        md_lines.append(sec.get("text", ""))
        md_lines.append("")  # blank line between sections
    return "\n".join(md_lines).strip()


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
    args: argparse.Namespace | None = None,
) -> bool:
    txt_path = txt_path_for(pdf, txt_dir)
    json_path = json_path_for(pdf, json_dir)

    if (not overwrite) and txt_path.exists() and json_path.exists():
        return False

    # 1 ─ Docling  (always first, most trusted)
    sections: List[Dict[str, Any]] = []
    bundle = None
    
    # --- START: TEMPORARY BLOCK TO FORCE UNSTRUCTURED ---
    log.info(f"DEBUG: PDF '{pdf.name}' - === FORCING UNSTRUCTURED EXTRACTION TEST ===")
    try:
        # This call directly attempts to use 'unstructured'
        sections = elements_from_unstructured(pdf)
        log.info(f"DEBUG: PDF '{pdf.name}' - elements_from_unstructured produced {len(sections)} initial page-sections.")
        if sections and sections[0].get("elements"):
            el_0_0 = sections[0]["elements"][0]
            el_text_preview = el_0_0.get("text", "")[:70]
            log.info(f"DEBUG: PDF '{pdf.name}' - First element from UNSTRUCTURED: font_size={el_0_0.get('font_size')}, kind='{el_0_0.get('kind')}', cat='{el_0_0.get('category')}', text_preview='{el_text_preview}...'")
        elif sections:
            log.info(f"DEBUG: PDF '{pdf.name}' - elements_from_unstructured produced page-sections, but first one has no 'elements' key or it's empty.")
        else:
            log.info(f"DEBUG: PDF '{pdf.name}' - elements_from_unstructured returned empty sections.")
    except Exception as e:
        log.error(f"DEBUG: PDF '{pdf.name}' - elements_from_unstructured FAILED: {e}")
    # --- END: TEMPORARY BLOCK TO FORCE UNSTRUCTURED ---

    # Only try Docling if unstructured didn't work
    if not sections:
        try:
            bundle = conv.convert(str(pdf))

            if keep_markup:
                log.info(f"DEBUG: PDF '{pdf.name}' - Attempting Docling element-based extraction as fallback.")
                sections = elements_from_docling(bundle.document)
            else:
                # plain-text fallback inside Docling
                full = bundle.document.export_to_text(page_break_marker="\f")
                sections = [
                    {
                        "section": "Full document",
                        "page_number": 1,
                        "text": full,
                        "elements": [],
                    }
                ]
        except Exception as e:
            log.debug("Docling failed on %s: %s", pdf.name, e)

    # PyPDF fallback (last resort)
    if not sections and not getattr(args, "docling_only", False):
        log.info(f"DEBUG: PDF '{pdf.name}' - Both unstructured and Docling failed or returned no sections. Using PyPDF2 fallback.")
        sections = elements_from_pypdf(pdf)
        log.info(f"DEBUG: PDF '{pdf.name}' - elements_from_pypdf produced {len(sections)} sections.")

    if not sections:
        raise RuntimeError(f"No text extracted by any method for PDF {pdf.name}")

    # Add detailed logging before collapsing to logical sections
    if not sections:  # If sections is still empty after all attempts
        log.error(f"DEBUG: PDF '{pdf.name}' - `sections` list is COMPLETELY EMPTY after all extraction attempts. Raising original error or a new one.")
        # The script would likely raise "no text extracted" shortly after if this is the case.
    else:
        log.info(f"DEBUG: PDF '{pdf.name}' - Before collapse_to_logical_sections. Initial `sections` count: {len(sections)}")
        for i, page_sec in enumerate(sections):
            elements = page_sec.get("elements")
            sec_text = page_sec.get("text", "")  # Get the text of the page section itself
            log.info(f"DEBUG: PDF '{pdf.name}' - Page section {i} (title: '{page_sec.get('section', 'N/A')}'): Has elements? {'Yes' if elements else 'No'}. Text preview (page level): '{sec_text[:100]}...'")
            if elements:
                log.info(f"DEBUG: PDF '{pdf.name}' - Page section {i} has {len(elements)} elements.")
                if len(elements) > 0 and isinstance(elements[0], dict):
                    log.info(f"DEBUG: PDF '{pdf.name}' - First element of page_section {i} (first 50 chars text): { {k: v for k, v in elements[0].items() if k != 'text'} | {'text': elements[0].get('text', '')[:50]} }")
                elif len(elements) > 0:
                    log.info(f"DEBUG: PDF '{pdf.name}' - First element of page_section {i} is not a dict: {elements[0]}")

    # Original line follows:
    if any(sec.get("elements") for sec in sections):
        log.info(f"DEBUG: PDF '{pdf.name}' - Condition `any(sec.get('elements') for sec in sections)` is TRUE. Calling collapse_to_logical_sections.")
        sections = collapse_to_logical_sections(sections)
    else:
        log.info(f"DEBUG: PDF '{pdf.name}' - Condition `any(sec.get('elements') for sec in sections)` is FALSE. SKIPPING collapse_to_logical_sections.")
        # If this path is taken, and the 'sections' are just page-based without levels,
        # render_sections_as_markdown will not create #, ## etc. unless levels are already there.
        # The H1 promotion in logical_sections is also skipped.
        # We might need to add a 'level' to these sections if they are to be rendered with headings.
        # For now, let's see the log. If sections exist but have no elements,
        # render_sections_as_markdown will just output their text.

    # 4 ─ Latin-1 scrub + OCR (page-based OCR if needed)
    for sec in sections:
        sec["text"] = latin1_scrub(sec["text"])
        for el in sec.get("elements", []):
            el["text"] = latin1_scrub(el["text"])

        if needs_ocr(sec["text"]):
            pn_for_ocr = sec.get("page_number") or sec.get("page_start") or 1
            try:
                ocr_txt = normalize_whitespace(ocr_page(pdf, pn_for_ocr, ocr_lang))
                if ocr_txt:
                    sec["text"] = ocr_txt
                    stats["ocr_pages"] += 1
            except Exception as e:
                log.debug("OCR failed on %s p.%s: %s", pdf.name, pn_for_ocr, e)

    # 5 ─ metadata merge
    bundle_meta: Dict[str, Any] = {}
    if bundle and hasattr(bundle, "document"):
        meta_obj = getattr(bundle.document, "metadata", None)
        if meta_obj is not None:
            if hasattr(meta_obj, "model_dump"):
                bundle_meta = meta_obj.model_dump()
            elif hasattr(meta_obj, "dict"):
                bundle_meta = meta_obj.dict()

    header_meta = extract_bib_from_header(
        " ".join(sec["text"] for sec in sections[:2])[:12000]
    )
    filename_meta = extract_bib_from_filename(pdf)
    meta = merge_metadata(bundle_meta, filename_meta, header_meta)

    payload = {
        **meta,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_pdf": str(pdf),
        "sections": sections,
    }

    # ───── write outputs ───── #
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown serialisation
    txt_md = render_sections_as_markdown(sections)
    txt_path.write_text(txt_md, encoding="utf-8")

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
    args: argparse.Namespace | None = None,
) -> None:
    conv = make_converter()
    pdfs = list(iter_pdfs(src_root))
    if limit:
        pdfs = pdfs[:limit]

    stats = Counter()
    log.info("Starting – %s PDFs", len(pdfs))

    with tqdm(total=len(pdfs), unit="file", desc="PDFs") as bar:
        for pdf in pdfs:
            bar.set_postfix_str(pdf.name)
            try:
                process_one(
                    conv, pdf, txt_dir, json_dir, overwrite, ocr_lang, stats, keep_markup, args
                )
            except Exception as e:
                stats["failed"] += 1
                log.error("‼️  %s failed: %s", pdf.name, e)
            bar.update()

    log.info(
        "Done. processed=%s  ocr_pages=%s  failed=%s",
        stats["processed"],
        stats["ocr_pages"],
        stats["failed"],
    )


# ─────────────── CLI ─────────────── #


def cli() -> None:
    ap = argparse.ArgumentParser(description="Batch PDF → TXT & JSON (Docling+OCR)")
    ap.add_argument("--src", default="documents/research/Global")
    ap.add_argument("--txt-dir", default="documents/research/txt")
    ap.add_argument("--json-dir", default="documents/research/json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    ap.add_argument("--max", type=int, default=0, help="process at most N PDFs")
    ap.add_argument("--ocr-lang", default="eng")
    ap.add_argument(
        "--no-markup", action="store_true", help="skip element capture / markdown"
    )
    ap.add_argument(
        "--docling-only",
        action="store_true",
        help="fail if Docling cannot extract (no fall-back extractors)",
    )
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


# ─────────────────── logical-section helpers ─────────────────── #


def logical_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse a flat list of (page, text, …) blocks into logical sections while
    *keeping* heading level and page span.
    """
    # ── 1. compute a robust font-size median ────────────────────────────
    sizes = sorted(b["font_size"] for b in blocks if b.get("font_size"))
    median = sizes[len(sizes)//2] if sizes else 12.0  # Default to 12.0 if no font sizes found
    log.info(f"DEBUG: `logical_sections` - Received {len(blocks)} blocks. Median font size: {median:.2f}")

    sections, cur = [], None
    if not blocks:
        log.info("DEBUG: `logical_sections` - Received an empty list of blocks. Returning empty sections.")
        return []  # Explicitly return empty if no blocks

    for i, b in enumerate(blocks):
        # Ensure block 'text' is a string, provide default for get('text', '') if missing
        block_text_content = b.get("text", "")
        if not isinstance(block_text_content, str):  # Defensive check
            log.warning(f"DEBUG: Block {i} text is not a string: {type(block_text_content)}. Converting to empty string.")
            block_text_content = ""

        lvl = heading_level(b, median)
        txt = block_text_content.strip()  # Use the sanitized text

        # Update logging to be more concise, showing first 5 blocks and any headings up to 10
        if i < 5 or (lvl > 0 and i < 10):
            log.info(f"DEBUG: `logical_sections` - Block {i}: lvl={lvl}, font_size={b.get('font_size')}, kind='{b.get('kind')}', text_preview='{txt[:70]}...'")

        if lvl:  # It's a heading
            if cur:  # Close previous
                sections.append(cur)
            cur = {
                "section": txt or "Untitled",
                "level": lvl,
                "page_start": b["page"],
                "page_end": b["page"],
                "text_parts": [],
            }
            continue  # No body text for heading

        # body
        if cur is None:  # File starts w/ body
            cur = {
                "section": "Untitled",
                "level": 2,
                "page_start": b["page"],
                "page_end": b["page"],
                "text_parts": [],
            }
        cur["text_parts"].append(txt)
        cur["page_end"] = b["page"]

    if cur:
        sections.append(cur)

    # ── 3. flatten text_parts and return ────────────────────────────────
    for s in sections:
        s["text"] = "\n\n".join(s.pop("text_parts")).strip()
    
    # Replace verbose section summary logging with more concise version
    if sections:
        log.info(f"DEBUG: `logical_sections` - Created {len(sections)} sections before H1 promotion.")
        s0_text_preview = sections[0].get("text", "")[:70]
        log.info(f"DEBUG: `logical_sections` - First section: level={sections[0].get('level')}, title='{sections[0].get('section')}', text_preview='{s0_text_preview}...'")
    else:
        log.info("DEBUG: `logical_sections` - No sections were created.")

    # H1 promotion logging - keep logic the same, update logging
    if sections and not any(sec.get("level") == 1 for sec in sections):
        if isinstance(sections[0], dict):
            sections[0]["level"] = 1
            log.info(f"DEBUG: `logical_sections` - Promoted first section to H1: '{sections[0].get('section')}'")
        else:
            log.warning("DEBUG: `logical_sections` - Could not promote first section to H1 (not a dict).")
    elif not sections:
        log.info("DEBUG: `logical_sections` - No sections exist, skipping H1 promotion.")
    
    return sections


def collapse_to_logical_sections(page_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert page-based *sections* (with .elements) to logical sections.
    """
    blocks: List[Dict[str, Any]] = []
    for sec_idx, sec in enumerate(page_sections):
        pn = sec.get("page_number", 1)
        elements = sec.get("elements", [])
        if not elements:
            # Skip this verbose log to reduce output
            continue
        for el in elements:
            blocks.append(
                {
                    "page": pn,
                    "text": el.get("text", ""),
                    "font_size": el.get("font_size"),
                    "bold": el.get("bold", False),
                    "kind": el.get("kind") or el.get("type"),
                    "category": el.get("category"),
                }
            )
    
    log.info(f"DEBUG: `collapse_to_logical_sections` created {len(blocks)} blocks in total to send to logical_sections.")
    if blocks:
        b0_text_preview = blocks[0].get("text", "")[:70]
        log.info(f"DEBUG: `collapse_to_logical_sections` - First block example: page={blocks[0].get('page')}, font_size={blocks[0].get('font_size')}, kind='{blocks[0].get('kind')}', text_preview='{b0_text_preview}...'")
    else:
        log.info("DEBUG: `collapse_to_logical_sections` - No blocks were created.")
    
    return logical_sections(blocks)


if __name__ == "__main__":
    cli()
