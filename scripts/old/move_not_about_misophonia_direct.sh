#!/bin/bash

# Script to move files listed in not_about_misophonia.txt to documents/research/not_about_misophonia

# Source directory where the files are located
SRC_DIR="documents/research/Global"
# Destination directory
DEST_DIR="documents/research/not_about_misophonia"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Read the not_about_misophonia.txt file line by line
while IFS= read -r filename || [[ -n "$filename" ]]; do
  # Skip empty lines
  if [[ -z "$filename" ]]; then
    continue
  fi
  
  # Remove .txt extension if it exists and add .pdf extension
  base_name="${filename%.txt}"
  pdf_name="${base_name}.pdf"
  
  # Check if the PDF file exists
  if [[ -f "$SRC_DIR/$pdf_name" ]]; then
    echo "Moving: $SRC_DIR/$pdf_name -> $DEST_DIR/$pdf_name"
    mv "$SRC_DIR/$pdf_name" "$DEST_DIR/$pdf_name"
  else
    echo "File not found: $SRC_DIR/$pdf_name"
  fi
done < scripts/not_about_misophonia.txt

echo "Moving completed." 