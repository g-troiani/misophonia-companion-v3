import os
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

def calculate_file_hash(filepath, algorithm='sha256', buffer_size=65536):
    """Calculate a hash for a file to identify duplicates."""
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        buffer = f.read(buffer_size)
        while buffer:
            hash_obj.update(buffer)
            buffer = f.read(buffer_size)
    
    return hash_obj.hexdigest()

def find_duplicates(directory):
    """Find duplicate files in the specified directory."""
    files_by_hash = defaultdict(list)
    duplicate_sets = []
    
    # Get all files in the directory
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: '{directory}' is not a valid directory")
        return duplicate_sets
    
    print(f"Scanning directory: {directory}")
    
    # Calculate hashes for all files
    all_files = list(target_dir.glob('*'))
    total_files = len(all_files)
    
    for i, file_path in enumerate(all_files):
        if file_path.is_file():
            try:
                file_hash = calculate_file_hash(file_path)
                files_by_hash[file_hash].append(file_path)
                print(f"Processed file {i+1}/{total_files}: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Identify duplicate sets (files with the same hash)
    for file_hash, paths in files_by_hash.items():
        if len(paths) > 1:
            duplicate_sets.append(paths)
    
    return duplicate_sets

def delete_duplicates(duplicate_sets, interactive=True):
    """Delete duplicate files, keeping only one copy of each."""
    total_deleted = 0
    total_size_saved = 0
    
    for duplicate_set in duplicate_sets:
        # Sort by name for consistent results
        duplicate_set.sort(key=lambda p: str(p))
        
        # Keep the first file, show options for the rest
        keep_file = duplicate_set[0]
        print(f"\nDuplicate set ({len(duplicate_set)} files):")
        print(f"  Keeping: {keep_file}")
        
        for i, dup_file in enumerate(duplicate_set[1:], 1):
            size = dup_file.stat().st_size
            
            if interactive:
                response = input(f"  Delete duplicate #{i}: {dup_file}? (y/n/a=all/q=quit): ").lower()
                
                if response == 'q':
                    print("Operation aborted.")
                    return total_deleted, total_size_saved
                    
                if response == 'a':
                    interactive = False
                    response = 'y'
            else:
                response = 'y'
                print(f"  Deleting duplicate #{i}: {dup_file}")
            
            if response == 'y':
                try:
                    dup_file.unlink()
                    total_deleted += 1
                    total_size_saved += size
                    print(f"  Deleted: {dup_file}")
                except Exception as e:
                    print(f"  Error deleting {dup_file}: {e}")
    
    return total_deleted, total_size_saved

def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def main():
    parser = argparse.ArgumentParser(description="Find and remove duplicate files")
    parser.add_argument('--directory', '-d', default='documents/research/Global',
                        help="Directory to scan for duplicates (default: documents/research/Global)")
    parser.add_argument('--delete', '-r', action='store_true',
                        help="Delete duplicate files")
    parser.add_argument('--auto', '-a', action='store_true',
                        help="Automatically delete all duplicates without prompting")
    
    args = parser.parse_args()
    
    # Find duplicates
    duplicate_sets = find_duplicates(args.directory)
    
    # Print summary of duplicates found
    if not duplicate_sets:
        print("\nNo duplicate files found.")
        return
    
    total_duplicates = sum(len(dups) - 1 for dups in duplicate_sets)
    print(f"\nFound {len(duplicate_sets)} sets of duplicate files ({total_duplicates} redundant files)")
    
    # Display details about each duplicate set
    for i, dups in enumerate(duplicate_sets, 1):
        size = dups[0].stat().st_size
        size_str = format_size(size)
        print(f"\nDuplicate Set #{i} - {len(dups)} files, {size_str} each:")
        for path in dups:
            print(f"  {path}")
    
    # Delete duplicates if requested
    if args.delete or args.auto:
        deleted, size_saved = delete_duplicates(duplicate_sets, not args.auto)
        print(f"\nSummary: Deleted {deleted} duplicate files, saving {format_size(size_saved)}")
    else:
        print("\nTo delete duplicates, run again with --delete or --auto flag")

if __name__ == "__main__":
    main()