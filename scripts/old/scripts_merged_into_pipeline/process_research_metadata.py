#!/usr/bin/env python3
################################################################################
# File: scripts/process_research_metadata.py
################################################################################
"""
Research Metadata Processor

This script:
1. Iterates over text files in documents/research/txt
2. For each file:
   - Takes first 3000 words from the text
   - Makes an OpenAI API call with a specific prompt
   - Updates the corresponding JSON file in documents/research/json
   - Records processed files to avoid reprocessing on subsequent runs

Required environment variables:
------------------------------
OPENAI_API_KEY

Usage:
-----
python process_research_metadata.py [--batch-size N] [--force] [--model MODEL]
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ─────────────────────────── Configuration ─────────────────────────────── #

load_dotenv()

OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TXT_DIR = ROOT_DIR / "documents" / "research" / "txt"
JSON_DIR = ROOT_DIR / "documents" / "research" / "json"
PROCESSED_FILE = ROOT_DIR / "scripts" / "processed_files.json"

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────── Helpers ────────────────────────────────── #

def load_processed_files() -> Set[str]:
    """Load the set of already processed files."""
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE, "r") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                log.warning("Error decoding processed files list, starting fresh")
    return set()

def save_processed_files(processed: Set[str]) -> None:
    """Save the set of processed files."""
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(processed), f, indent=2)

def extract_first_n_words(text: str, n: int = 3000) -> str:
    """Extract first n words from text."""
    words = text.split()
    return " ".join(words[:n])

def get_corresponding_json_path(txt_path: Path) -> Path:
    """Get the path to the corresponding JSON file."""
    return JSON_DIR / f"{txt_path.stem}.json"

def generate_metadata(client: OpenAI, text: str, model: str) -> Dict[str, Any]:
    """Call OpenAI API to generate metadata from text."""
    prompt = f"""
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

If a field is not present, set it to null or an empty array. Here is the paper's full text:

{text}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured metadata from scientific papers."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON from response (in case there's additional text)
        json_match = re.search(r'({[\s\S]*})', content)
        if json_match:
            content = json_match.group(1)
            
        return json.loads(content)
    except Exception as e:
        log.error(f"Error calling OpenAI API: {e}")
        raise

def update_json_file(json_path: Path, metadata: Dict[str, Any]) -> None:
    """Update JSON file with metadata from API response."""
    try:
        # Create if not exists
        if not json_path.exists():
            json_data = {
                "doc_type": "scientific paper",
                "title": "",
                "authors": [],
                "year": None,
                "journal": None,
                "doi": None,
                "abstract": None,
                "keywords": [],
                "research_topics": [],
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source_pdf": "",
                "sections": []
            }
        else:
            # Read existing JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        # Update with new metadata
        for key, value in metadata.items():
            if key in json_data and value not in (None, [], ""):
                json_data[key] = value

        # Write updated JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        log.error(f"Error updating JSON file {json_path}: {e}")
        return False

# ────────────────────────────── Main Process ────────────────────────────── #

def process_files(client: OpenAI, batch_size: int, force: bool, model: str) -> Dict[str, Any]:
    """Process text files and update JSON files with metadata."""
    results = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "file_results": []
    }
    
    # Get all text files
    txt_files = list(TXT_DIR.glob("*.txt"))
    log.info(f"Found {len(txt_files)} text files in {TXT_DIR}")
    
    # Load set of processed files
    processed_files = set() if force else load_processed_files()
    
    # Filter unprocessed files or limit to batch size
    if not force:
        txt_files = [f for f in txt_files if f.name not in processed_files]
    
    log.info(f"Found {len(txt_files)} unprocessed files")
    if batch_size > 0:
        txt_files = txt_files[:batch_size]
        log.info(f"Processing batch of {len(txt_files)} files")
    
    # Process each file
    for txt_path in tqdm(txt_files, desc="Processing files"):
        # Log the current file being processed
        log.info(f"Processing file: {txt_path.name}")
        
        file_result = {
            "file": txt_path.name,
            "success": False,
            "error": None
        }
        
        try:
            # Read text file
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Extract first 3000 words
            truncated_text = extract_first_n_words(text)
            
            # Generate metadata - log that we're calling the API
            log.info(f"Calling OpenAI API for: {txt_path.name}")
            metadata = generate_metadata(client, truncated_text, model)
            
            # Get corresponding JSON path
            json_path = get_corresponding_json_path(txt_path)
            
            # Update JSON file
            log.info(f"Updating JSON file for: {txt_path.name}")
            if update_json_file(json_path, metadata):
                file_result["success"] = True
                processed_files.add(txt_path.name)
                results["processed"] += 1
                log.info(f"Successfully processed: {txt_path.name}")
            else:
                file_result["error"] = "Failed to update JSON file"
                results["errors"] += 1
                log.error(f"Failed to update JSON for: {txt_path.name}")
                
        except Exception as e:
            file_result["error"] = str(e)
            results["errors"] += 1
            log.error(f"Error processing {txt_path.name}: {e}")
            
        results["file_results"].append(file_result)
    
    # Save processed files
    save_processed_files(processed_files)
    
    return results

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process research files and extract metadata using OpenAI API")
    parser.add_argument("--batch-size", type=int, default=0, help="Number of files to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Process all files even if previously processed")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set. Please set it in your environment variables.")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Create directories if they don't exist
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process files
    log.info(f"Starting metadata extraction with model: {args.model}")
    results = process_files(client, args.batch_size, args.force, args.model)
    
    # Output results
    log.info(f"Processing complete: {results['processed']} processed, {results['skipped']} skipped, {results['errors']} errors")
    
    # Save report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT_DIR / f"metadata_extraction_report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()