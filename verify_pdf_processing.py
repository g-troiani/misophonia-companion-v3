#!/usr/bin/env python

import json
import pathlib
import sys
from scripts.old.pipeline_integrated import _make_converter, extract_pdf, concat_tokens, TXT_DIR, JSON_DIR

def verify_processing():
    """Verify that sections now contain the text field and tokens are generated properly."""
    if len(sys.argv) < 2:
        print("Usage: python verify_pdf_processing.py <path/to/sample.pdf>")
        return 1
    
    sample = pathlib.Path(sys.argv[1])
    if not sample.exists() or sample.suffix.lower() != '.pdf':
        print(f"Error: {sample} is not a valid PDF file")
        return 1
    
    print(f"Processing {sample}...")
    conv = _make_converter()
    
    # Extract PDF
    try:
        json_path = extract_pdf(sample, TXT_DIR, JSON_DIR, conv, overwrite=True)
        print(f"Successfully extracted to {json_path}")
    except Exception as e:
        print(f"Extraction error: {e}")
        return 1
    
    # Load JSON and verify text field
    obj = json.loads(json_path.read_text())
    
    # Check sections have text field
    section_count = len(obj["sections"])
    sections_with_text = sum(1 for s in obj["sections"] if "text" in s)
    
    print(f"\nFound {section_count} sections, {sections_with_text} with text field")
    
    # Check token generation
    tokens, page_map = concat_tokens(obj["sections"])
    print(f"Generated {len(tokens)} tokens from sections\n")
    
    # Print first few sections for inspection
    for i, section in enumerate(obj["sections"][:3]):
        print(f"Section {i+1}: {section.get('section', '(unnamed)')}")
        print(f"  Page: {section.get('page_start')}-{section.get('page_end')}")
        text = section.get('text', '')
        print(f"  Text length: {len(text)} chars")
        print(f"  Text preview: {text[:100]}...\n")
    
    if len(tokens) == 0:
        print("ERROR: No tokens generated - fix might not be working!")
        return 1
        
    print("SUCCESS: Text fields are present and tokens are generated!")
    return 0

if __name__ == "__main__":
    sys.exit(verify_processing()) 