#!/bin/bash

# Script to move files listed in not_about_misophonia.txt to documents/research/not_about_misophonia

# Source directory where the files are located
SRC_DIR="documents/research/Global"
# Destination directory
DEST_DIR="documents/research/not_about_misophonia"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Read the not_about_misophonia.txt file line by line
while IFS= read -r old_name || [[ -n "$old_name" ]]; do
  # Skip empty lines
  if [[ -z "$old_name" ]]; then
    continue
  fi
  
  # Remove .txt extension if it exists
  old_name="${old_name%.txt}"
  
  # Find the new name in the mapping file
  # Use grep to find the line in the mapping that starts with the old name
  mapping_line=$(grep -F -m 1 "${old_name}" scripts/rename_map_misophonia.tsv)
  
  if [[ -n "$mapping_line" ]]; then
    # Extract the new name from the mapping
    new_name=$(echo "$mapping_line" | awk -F'\t' '{print $2}')
    
    # Convert new_name to title case (first letter of each word capitalized)
    title_case_name=$(echo "$new_name" | awk '{
      for(i=1;i<=NF;i++) {
        $i=toupper(substr($i,1,1)) tolower(substr($i,2))
      }
      print
    }')
    
    # Check if the renamed file exists
    if [[ -f "$SRC_DIR/$title_case_name.pdf" ]]; then
      echo "Moving: $SRC_DIR/$title_case_name.pdf -> $DEST_DIR/$title_case_name.pdf"
      mv "$SRC_DIR/$title_case_name.pdf" "$DEST_DIR/$title_case_name.pdf"
    else
      echo "File not found: $SRC_DIR/$title_case_name.pdf"
    fi
  else
    echo "No mapping found for: $old_name"
  fi
done < scripts/not_about_misophonia.txt

echo "Moving completed." 