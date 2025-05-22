#!/bin/bash

# Script to rename files based on rename_map_misophonia.tsv mapping

# Function to convert to title case (first letter of each word capitalized)
to_title_case() {
  echo "$1" | awk '{
    for(i=1;i<=NF;i++) {
      $i=toupper(substr($i,1,1)) tolower(substr($i,2))
    }
    print
  }'
}

# Directory where the files are located
DIR="documents/research/Global"

# Read the TSV file line by line
while IFS=$'\t' read -r old_name new_name || [[ -n "$old_name" ]]; do
  # Skip empty lines
  if [[ -z "$old_name" ]]; then
    continue
  fi
  
  # Remove .txt extension from old_name if it exists
  old_name="${old_name%.txt}"
  
  # Convert new_name to title case
  new_name=$(to_title_case "$new_name")
  
  # Check if the old file exists
  if [[ -f "$DIR/$old_name.pdf" ]]; then
    echo "Renaming: $old_name.pdf -> $new_name.pdf"
    mv "$DIR/$old_name.pdf" "$DIR/$new_name.pdf"
  else
    echo "File not found: $DIR/$old_name.pdf"
  fi
done < scripts/rename_map_misophonia.tsv

echo "Renaming completed." 