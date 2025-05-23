#!/usr/bin/env python3
"""
One-command orchestrator that chains the six stand-alone stages.

Usage
-----
    python scripts/pipeline_modular.py [PDF | DIR] [--selection sequential|random] [--cap N]
"""
from __future__ import annotations
import argparse, logging, pathlib, random
from collections import Counter
from rich.console import Console
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
def _make_converter():
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    return DocumentConverter({InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

# ─── stage modules ──────────────────────────────────────────────────────────
from stages import extract_clean, llm_enrich, chunk_text, db_upsert, embed_vectors

# ─── toggles (flip at will) ─────────────────────────────────────────────────
RUN_EXTRACT    = True
RUN_LLM_ENRICH = True
RUN_CHUNK      = True
RUN_DB         = True      # ← turn off if you only want local JSON
RUN_EMBED      = True      # ← requires OpenAI + Supabase creds
# ────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("pipeline-modular")
Console().rule("[bold cyan]Misophonia PDF → Vector pipeline (modular)")

def main(src:pathlib.Path, selection:str="sequential", cap:int=0, stage:int=0, overwrite:bool=False) -> None:
    pdfs = [src] if src.is_file() else sorted(src.rglob("*.pdf"))
    if cap:
        pdfs = random.sample(pdfs, cap) if selection == "random" else pdfs[:cap]

    stats = Counter()
    conv = _make_converter()
    for pdf in tqdm(pdfs, desc="Papers"):
        try:
            # ─── Stage 1-2  Extract + Clean ────────────────────────────────
            json_doc = extract_clean.run_one(pdf, conv=conv, overwrite=overwrite) if RUN_EXTRACT else \
                       extract_clean.json_path_for(pdf)
            # ─── Stage 4  LLM metadata enrich ─────────────────────────────
            if RUN_LLM_ENRICH:
                llm_enrich.enrich(json_doc)
            # ─── Stage 5  Chunk ------------------------------------------------
            chunks = chunk_text.chunk(json_doc) if RUN_CHUNK else None
            # ─── Stage 6  DB upsert + (optional) Stage 7 embed  ───────────────
            if RUN_DB:
                db_upsert.upsert(json_doc, chunks, do_embed=RUN_EMBED, overwrite=overwrite)
            stats["ok"] += 1
        except Exception as exc:          # noqa: BLE001
            log.exception("✖ %s failed → %s", pdf.name, exc)
            stats["fail"] += 1

    Console().rule("[green]Finished")
    log.info("Processed=%s   Failed=%s", stats["ok"], stats["fail"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s — %(levelname)s — %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("src", type=pathlib.Path,
                   default=pathlib.Path("documents/research/Global"), nargs="?")
    p.add_argument("--selection", choices=["sequential", "random"],
                   default="sequential")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args.src, args.selection, args.cap, args.stage, args.overwrite) 