#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# File: scripts/topic_filter_and_title.py
# -----------------------------------------------------------------------------
"""
Loop over every TXT file in **documents/research/txt/**, send a fixed-length
excerpt (~MAX_WORDS words ≈ two paragraphs) to OpenAI and ask:

    • Is the paper about *TOPIC*?
    • What’s the paper title?

The script writes **two helper files in the same */scripts/* folder** (the
“script library”, as requested):

  1.  not_about_<topic>.txt    — one TXT filename per line (safe to delete)
  2.  rename_map_<topic>.tsv   — “current_filename<TAB>inferred_title”

Change *TOPIC*, *MAX_WORDS* or *MODEL* below as needed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# ────────────────────────── user-tunable settings ───────────────────────────

TOPIC: str = "Misophonia"            # ← change for other topics
MAX_WORDS: int = 300                 # ≈ two paragraphs
MODEL: str = "gpt-4.1-mini-2025-04-14"
RATE_LIMIT_SLEEP: float = 1.2        # s between API calls to stay polite

# ───────────────────────────── paths & client ───────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent            # /scripts/
REPO_ROOT  = SCRIPT_DIR.parent                          # project root
TXT_DIR    = REPO_ROOT / "documents" / "research" / "txt"

OUT_NO_TOPIC = SCRIPT_DIR / f"not_about_{TOPIC.lower()}.txt"
OUT_RENAME   = SCRIPT_DIR / f"rename_map_{TOPIC.lower()}.tsv"

load_dotenv()  # picks up OPENAI_API_KEY from .env or environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────── helper functions ──────────────────────────────

def read_excerpt(txt_path: Path, max_words: int = MAX_WORDS) -> str:
    """
    Return the first *max_words* words of the TXT file.
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    return " ".join(words[:max_words])

def query_llm(content: str) -> Dict[str, Any]:
    """
    Ask GPT-4 whether the paper is about *TOPIC* and get its title.
    Returns a dict like:  { "relevant": true/false, "title": "…" }
    """
    system_msg = (
        "You are a scholarly assistant. You will receive an excerpt from a "
        f"scientific paper. Decide whether the paper is about the topic "
        f"'{TOPIC}'. Respond **ONLY** with valid JSON containing two keys:\n"
        '  "relevant": true or false\n'
        '  "title":    full paper title if present, else ""'
    )
    user_msg = f"Excerpt:\n\"\"\"\n{content}\n\"\"\""

    chat = client.chat.completions.create(
        model=MODEL,
        temperature=0,            # deterministic
        top_p=1,
        max_tokens=256,
        stream=False,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(chat.choices[0].message.content)

# ──────────────────────────────── main loop ─────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not TXT_DIR.exists():
        logging.error(f"TXT source folder not found: {TXT_DIR}")
        return

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        logging.error(f"No TXT files in {TXT_DIR}")
        return

    not_about: List[str] = []
    rename_rows: List[str] = []

    for fp in txt_files:
        logging.info(f"→ {fp.name}")
        excerpt = read_excerpt(fp)

        # default values in case of any exception
        relevant = False
        title = ""

        try:
            resp = query_llm(excerpt)
            relevant = bool(resp.get("relevant"))
            title    = (resp.get("title") or "").strip()
        except Exception as e:
            logging.warning(f"  ⚠️  LLM error ({e}); treating as NOT relevant")

        if not relevant:
            not_about.append(fp.name)
        else:
            rename_rows.append(f"{fp.name}\t{title}")

        time.sleep(RATE_LIMIT_SLEEP)   # simple client-side rate-limit

    # ───────────────────────────── write outputs ────────────────────────────
    if not_about:
        OUT_NO_TOPIC.write_text("\n".join(not_about) + "\n", encoding="utf-8")
        logging.info(f"✍️  {len(not_about)} non-topic files → {OUT_NO_TOPIC.name}")

    if rename_rows:
        OUT_RENAME.write_text("\n".join(rename_rows) + "\n", encoding="utf-8")
        logging.info(f"✍️  {len(rename_rows)} rename rows → {OUT_RENAME.name}")

    if not not_about and not rename_rows:
        logging.info("✅ No files processed (empty dataset?).")
    else:
        logging.info("✅ Finished.")

# ───────────────────────── entry point ──────────────────────────────────────

if __name__ == "__main__":
    main()
