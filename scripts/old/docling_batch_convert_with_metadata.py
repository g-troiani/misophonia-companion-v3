#!/usr/bin/env python3
# scripts/docling_batch_convert_with_metadata.py
"""
Convert every PDF under  documents/research/Global
→ flat TXT (documents/research/txt/) **and** rich JSON
(documents/research/json/) that carries per‑page text plus
bibliographic metadata (title, authors, year, journal, DOI,
abstract, keywords, research_topics).

The script is idempotent: if both TXT and JSON already
exist it skips the file; if one is missing it creates just that one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List

import PyPDF2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from rich.logging import RichHandler
from tqdm import tqdm

# ───────── optional unstructured fallback ───────── #

try:
    from unstructured.partition.pdf import partition_pdf

    HAVE_UNSTRUCTURED = True
except Exception:  # pragma: no cover
    HAVE_UNSTRUCTURED = False

# ───────────────────────── logging ───────────────────────── #


def init_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), "INFO"),
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler()],
    )


log = logging.getLogger(__name__)

# ─────────────── Docling converter helper ─────────────── #


def make_converter() -> DocumentConverter:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True  # scan‑safe
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


# ───────────────────────── path helpers ───────────────────────── #


def iter_pdfs(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file() and not p.name.startswith("."):
            yield p


def txt_path_for(src: pathlib.Path, txt_dir: pathlib.Path) -> pathlib.Path:
    return txt_dir / f"{src.stem}.txt"


def json_path_for(src: pathlib.Path, json_dir: pathlib.Path) -> pathlib.Path:
    return json_dir / f"{src.stem}.json"


# ─────────────── bibliographic helpers ─────────────── #

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


def _authors(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(a).strip() for a in val if a]
    return re.split(r"\s*,\s*|\s+and\s+", str(val).strip())


def extract_bib_from_filename(pdf: pathlib.Path) -> Dict[str, Any]:
    """
    Infer author/year/title from filenames like “Smith 2019 Some Paper.pdf”.
    """
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


# ───────── header‑text regex extraction (journal / DOI …) ───────── #


DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
JOURNAL_RE = re.compile(
    r"(Journal|Journals|Revista|Proceedings|Annals|Neuroscience|Psychiatry|Psychology|Nature|Science)[^\n]{0,120}",
    re.I,
)
ABSTRACT_RE = re.compile(r"(?<=\bAbstract\b[:\s])(.{50,2000}?)(?:\n[A-Z][^\n]{0,60}\n|\Z)", re.S)
KEYWORDS_RE = re.compile(r"\bKey\s*words?\b[:\s]*(.+)", re.I)


def extract_bib_from_header(header_txt: str) -> Dict[str, Any]:
    """
    Very tolerant regexes on the first pages' raw text.
    """
    meta: Dict[str, Any] = {}

    doi = DOI_RE.search(header_txt)
    if doi:
        meta["doi"] = doi.group(1)

    jour = JOURNAL_RE.search(header_txt)
    if jour:
        meta["journal"] = " ".join(jour.group(0).split())

    abst = ABSTRACT_RE.search(header_txt)
    if abst:
        meta["abstract"] = " ".join(abst.group(1).split())

    kws = KEYWORDS_RE.search(header_txt)
    if kws:
        meta["keywords"] = [k.strip(" ;.,") for k in re.split(r"[;,]", kws.group(1)) if k.strip()]

    # derive research_topics from keywords
    if meta.get("keywords"):
        meta["research_topics"] = meta["keywords"]

    return meta


def merge_metadata(*sources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep, ordered merge of metadata sources according to META_FIELDS precedence.
    """
    merged: Dict[str, Any] = {"doc_type": "scientific paper"}
    for field in META_FIELDS:
        merged[field] = None

    for src in sources:  # earlier dicts have lower priority
        for k in META_FIELDS:
            v = src.get(k)
            if v not in (None, "", [], {}):
                merged[k] = v

    # canonical types / defaults
    merged["authors"] = _authors(merged["authors"])
    merged["keywords"] = merged["keywords"] or []
    merged["research_topics"] = merged["research_topics"] or []
    return merged


# ──────────── per‑page text extraction helpers ──────────── #


def sections_from_docling(doc) -> List[Dict[str, Any]]:
    """
    First choice: iterate doc.pages (reliable).  Fallback: split on \f.
    """
    secs: List[Dict[str, Any]] = []
    try:
        pages = getattr(doc, "pages", None)
        if pages:  # new Docling API
            for idx, page in enumerate(pages, 1):
                try:
                    txt = page.export_to_text()
                except Exception:  # pragma: no cover
                    txt = ""
                if txt and txt.strip():
                    pn = getattr(page, "metadata", {}).get("page_number", idx)
                    secs.append(
                        {
                            "section": f"Page {pn}",
                            "page_number": pn,
                            "text": txt.strip(),
                        }
                    )
        if secs:
            return secs
    except Exception as e:  # pragma: no cover
        log.debug("Docling page iteration failed: %s", e)

    # ---- fallback to page_break marker ---- #
    try:
        full = doc.export_to_text(page_break_marker="\f")
        if "\f" in full:
            for idx, txt in enumerate(full.split("\f"), 1):
                t = txt.strip()
                if t:
                    secs.append({"section": f"Page {idx}", "page_number": idx, "text": t})
        elif full.strip():
            secs.append({"section": "Page 1", "page_number": 1, "text": full.strip()})
    except Exception as e:
        log.debug("Docling export_to_text failed: %s", e)
    return secs


def sections_from_unstructured(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    elements = partition_pdf(str(pdf), strategy="hi_res")
    sections: List[Dict[str, Any]] = []
    buf, cur = "", None
    for el in elements:
        page = getattr(el.metadata, "page_number", None)
        if page is None:
            continue
        if cur is None:
            cur = page
        if page != cur:
            sections.append(
                {"section": f"Page {cur}", "page_number": cur, "text": buf.strip()}
            )
            buf, cur = "", page
        buf += " " + str(el)
    if buf.strip():
        sections.append(
            {"section": f"Page {cur}", "page_number": cur, "text": buf.strip()}
        )
    return sections


def sections_from_pypdf(pdf: pathlib.Path) -> List[Dict[str, Any]]:
    secs: List[Dict[str, Any]] = []
    with open(pdf, "rb") as f:
        for i, page in enumerate(PyPDF2.PdfReader(f).pages, 1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            secs.append({"section": f"Page {i}", "page_number": i, "text": txt.strip()})
    return secs


# ─────────────── single‑file processing pipeline ─────────────── #


def process_one(
    conv: DocumentConverter,
    pdf: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    overwrite: bool,
) -> bool:
    txt_path = txt_path_for(pdf, txt_dir)
    json_path = json_path_for(pdf, json_dir)

    has_txt, has_json = txt_path.exists(), json_path.exists()
    if has_txt and has_json and not overwrite:
        log.debug("✓ Skip %s (TXT+JSON present)", pdf.name)
        return False

    # ───── Docling first ───── #
    doc_meta: Dict[str, Any] = {}
    sections: List[Dict[str, Any]] = []
    try:
        bundle = conv.convert(str(pdf))
        doc = bundle.document
        meta_obj = getattr(doc, "metadata", None)
        if meta_obj is not None:
            for meth in ("model_dump", "dict", "to_dict"):
                if hasattr(meta_obj, meth):
                    doc_meta = getattr(meta_obj, meth)()
                    break
        sections = sections_from_docling(doc)
    except Exception as e:
        log.debug("Docling failed on %s (%s) – trying fallbacks", pdf.name, e)

    # ───── fallbacks if Docling text inadequate ───── #
    if not sections or all(len(s["text"]) < 40 for s in sections):
        if HAVE_UNSTRUCTURED:
            try:
                sections = sections_from_unstructured(pdf)
            except Exception as e:
                log.debug("unstructured failed on %s (%s)", pdf.name, e)
        if not sections:
            sections = sections_from_pypdf(pdf)

    if not sections:
        raise RuntimeError("no text extracted")

    # ───── compile metadata ───── #
    first_pages_text = " ".join(s["text"] for s in sections[:2])[:12000]
    header_meta = extract_bib_from_header(first_pages_text)
    filename_meta = extract_bib_from_filename(pdf)
    meta = merge_metadata(doc_meta, filename_meta, header_meta)

    # ───── assemble JSON payload ───── #
    payload = {
        **meta,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_pdf": str(pdf),
        "sections": sections,
    }

    # ───── write outputs ───── #
    if overwrite or not has_json:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if overwrite or not has_txt:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n\n".join(s["text"] for s in sections), encoding="utf-8")

    return True


# ─────────────── batch driver ─────────────── #


def run(
    src_root: pathlib.Path,
    txt_dir: pathlib.Path,
    json_dir: pathlib.Path,
    overwrite: bool,
    limit: int | None,
) -> None:
    conv = make_converter()
    pdfs = list(iter_pdfs(src_root))
    if limit:
        pdfs = pdfs[:limit]

    processed = skipped = failed = 0
    log.info("Starting – %s PDFs (src=%s)", len(pdfs), src_root)

    with tqdm(total=len(pdfs), unit="file", desc="Processing") as bar:
        for pdf in pdfs:
            bar.set_postfix_str(pdf.name)
            try:
                changed = process_one(conv, pdf, txt_dir, json_dir, overwrite)
                if changed:
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                log.error("‼️  %s failed: %s", pdf.name, e)
            bar.update()

    log.info(
        "Done. processed=%s  skipped=%s  failed=%s  (TXT → %s , JSON → %s)",
        processed,
        skipped,
        failed,
        txt_dir,
        json_dir,
    )


# ─────────────── CLI ─────────────── #


def cli() -> None:
    p = argparse.ArgumentParser(
        description="Convert research PDFs to TXT and JSON with metadata"
    )
    p.add_argument("--src", default="documents/research/Global", help="PDF folder")
    p.add_argument("--txt-dir", default="documents/research/txt", help="TXT output dir")
    p.add_argument("--json-dir", default="documents/research/json", help="JSON output dir")
    p.add_argument("--overwrite", action="store_true", help="Force re‑create outputs")
    p.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    p.add_argument("--max", type=int, default=0, help="Process at most N PDFs")
    args = p.parse_args()

    init_logging(args.log)
    run(
        pathlib.Path(args.src),
        pathlib.Path(args.txt_dir),
        pathlib.Path(args.json_dir),
        args.overwrite,
        None if args.max == 0 else args.max,
    )


if __name__ == "__main__":
    cli()
