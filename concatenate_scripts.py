import os
import sys
import datetime
import re

# --- Configuration Constants ---

# Define allowed file extensions and specific filenames
ALLOWED_EXTENSIONS = [
    '.js', '.jsx', '.html', '.css', '.py', '.md', 
    '.json', '.toml', '.yaml', '.yml', '.gitignore'
]
ALLOWED_FILENAMES = [
    'eslint.config.js', 
    'vite.config.js',
    'requirements.txt' 
]

# Variables for files and directories to exclude
OUTPUT_FILENAME_TEMPLATE = 'concatenated_scripts_part{}.txt'
# Dynamically get the name of the script file itself
SCRIPT_FILENAME = os.path.basename(sys.argv[0]) 

EXCLUDED_FILES = [
    'package-lock.json',
    'concatenated_scripts_part1.txt',
    'concatenated_scripts_part2.txt',
    'concatenated_scripts_part3.txt',
    SCRIPT_FILENAME, # Exclude the script file itself
    '.env', # Exclude environment variable files
    'rename_map_misophonia.tsv', # Data mapping file
    'not_about_misophonia.txt', # Content filtering data file
    '.DS_Store', # macOS system file
]

# Expanded list of exclusions for virtual environments and node modules
EXCLUDED_DIRS = [
    '__pycache__',
    '.git',
    'node_modules',       # Node modules
    'dist',               # Vite build output
    '.netlify',           # Netlify directory
    'venv',               # Common Python virtual env name
    '.venv',              # Another common virtual env name
    'env',                # Another common virtual env name
    'virtualenv',         # Another virtual env name
    'misophonia-companion', # Specific virtual env folder
    'misophonia_companion', # Alternative naming
    'misophonia_env',     # Potential virtual env name
    'misophonia-env',     # Potential virtual env name
    'documents',          # Documents library
    'docs',               # Alternative name for documents
    'documents_library',  # Alternative naming
    'document_library',   # Alternative naming
    'cache',              # Cache directories (scripts/cache)
    'artefacts',          # Generated artifacts (scripts/artefacts)
    'artifacts',          # Alternative spelling
    'reports',            # Report files (embedding reports, etc.)
]

# Path-based exclusions - these are specific paths we want to exclude
EXCLUDED_PATHS = [
    'scripts/old',        # Version history and deprecated files - not used by current pipeline stages
    'scripts/old/templates', # Old template files
    'scripts/old/scripts_merged_into_pipeline', # Files that have been integrated elsewhere
]

# Essential documentation files that contain architectural information
ESSENTIAL_DOCS = [
    'documents/development/technical-architecture.md',  # System architecture overview
]

# Additional patterns to identify virtual environments
VENV_PATTERNS = [
    'venv', 'virtualenv', 'env', 'python3', 'python'
]

# --- Helper Functions ---

def is_venv_or_node_modules(path):
    """
    More robust check for virtual environments and node_modules.
    Returns True if the path appears to be a virtual environment or node_modules.
    """
    path_lower = path.lower()
    path_parts = os.path.normpath(path).split(os.sep)
    
    # Check for node_modules
    if 'node_modules' in path_parts:
        return True
    
    # Check for common virtual environment patterns
    for pattern in VENV_PATTERNS:
        if any(part.startswith(pattern) for part in path_parts):
            # Additional check - does it contain typical venv structures?
            if os.path.exists(os.path.join(path, 'bin', 'activate')) or \
               os.path.exists(os.path.join(path, 'Scripts', 'activate.bat')) or \
               os.path.exists(os.path.join(path, 'lib', 'python')):
                return True
    
    # Check for the specific misophonia-companion venv
    if 'misophonia-companion' in path_parts or 'misophonia_companion' in path_parts:
        return True
    
    return False

def get_comment_style(filename):
    """Gets the appropriate comment style based on file extension."""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # JavaScript family - uses //
    if ext in ['.js', '.jsx', '.ts', '.tsx']:
        return ('// ', '') 
        
    # CSS uses /* ... */ block comments
    elif ext in ['.css']:
        return ('/* ', ' */')
        
    # Python, Shell, YAML, etc. - uses #
    elif ext in ['.py', '.sh', '.yaml', '.yml', '.toml', '.gitignore', '.r', '.pl', '.rb']:
        return ('# ', '')
        
    # HTML family - uses <!-- ... -->
    elif ext in ['.html', '.xml', '.vue', '.svg']:
        return ('<!-- ', ' -->')
        
    # SQL - uses --
    elif ext == '.sql':
        return ('-- ', '')
        
    # Markdown - can use HTML comments
    elif ext == '.md':
        return ('<!-- ', ' -->')
        
    # Special files
    elif filename.lower() == 'requirements.txt':
        return ('# ', '')
        
    # JSON doesn't support comments
    elif ext == '.json':
        return None
        
    # Default for unknown types
    else:
        print(f"[WARN] Unknown file type '{ext}' for header comment. Using '# '.")
        return ('# ', '')

def should_process_file(file_path, filename):
    """Checks if a file should be processed based on exclusions and allowed types."""
    # Check if path contains node_modules or virtual environment
    if is_venv_or_node_modules(file_path):
        print(f"[DEBUG] Skipping file in node_modules or venv: {file_path}")
        return False
    
    # Include essential documentation files regardless of other exclusions
    relative_path = os.path.relpath(file_path, os.getcwd()).replace('\\', '/')
    if any(relative_path == doc_path or relative_path.endswith(doc_path) for doc_path in ESSENTIAL_DOCS):
        return True
    
    # Always include files in the Stages directory
    if 'stages' in file_path.lower().split(os.sep):
        _, ext = os.path.splitext(filename)
        if ext.lower() in ALLOWED_EXTENSIONS:
            return True
    
    # Check absolute exclusions first
    if filename in EXCLUDED_FILES:
        # print(f"[DEBUG] Skipping explicitly excluded file: {filename}")
        return False
        
    # Check if it's an allowed specific filename
    if filename in ALLOWED_FILENAMES:
        return True
        
    # Check if it has an allowed extension
    _, ext = os.path.splitext(filename)
    if ext.lower() in ALLOWED_EXTENSIONS:
        return True
        
    # print(f"[DEBUG] Skipping file with disallowed type or name: {filename}")
    return False

def create_file_header(file_path, relative_path):
    """
    Creates a properly formatted header for the file based on its type.
    Returns the header text using the appropriate comment style.
    """
    filename = os.path.basename(file_path)
    comment_style = get_comment_style(filename)
    
    if comment_style is None:  # No comments supported (e.g., JSON)
        return None
    
    comment_start, comment_end = comment_style
    header_content = f"File: {relative_path}"
    
    # For multi-line block comments (CSS, HTML, etc.)
    if comment_end:
        header = f"{comment_start}\n{header_content}\n{comment_end}"
    else:  # Line comments (JS, Python, etc.)
        header = f"{comment_start}{header_content}"
    
    return header

def check_for_existing_header(content, relative_path):
    """
    Checks if the file already has a header about its path.
    Returns the content with ALL existing headers removed.
    """
    # Common header patterns with capture groups
    header_patterns = [
        r'^\s*(//\s*File:.*?)\n',        # JavaScript style
        r'^\s*(#\s*File:.*?)\n',         # Python style
        r'^\s*(/\*\s*File:.*?\*/)',       # CSS style
        r'^\s*(<!--\s*File:.*?-->)',      # HTML style
        r'^\s*(--\s*File:.*?)\n',        # SQL style
    ]
    
    # Check for and remove any header pattern at the beginning of the file
    clean_content = content
    
    # First, try looking for headers at the very beginning
    for pattern in header_patterns:
        clean_content = re.sub(f'^{pattern}\\s*', '', clean_content, flags=re.MULTILINE|re.DOTALL)
    
    # Look for multiple header blocks with separating lines
    clean_content = re.sub(r'^#{80}\s*\n^#\s*File:.*?\n^#{80}\s*\n\s*', '', clean_content, flags=re.MULTILINE|re.DOTALL)
    
    # Check if we have the file path in a header anywhere in the first 10 lines
    first_lines = content.split('\n')[:10]
    first_block = '\n'.join(first_lines)
    
    file_path_pattern = re.escape(relative_path)
    has_header = re.search(file_path_pattern, first_block) is not None
    
    return has_header, clean_content

def prepend_header_if_needed(content, header, relative_path):
    """
    Prepends a header to the content if no suitable header exists.
    Returns the content with a header.
    """
    if header is None:
        return content
    
    # Check if content already has a header and clean up any duplicates
    has_header, clean_content = check_for_existing_header(content, relative_path)
    
    # If it already has a header, just return the cleaned content
    if has_header:
        return clean_content
    
    # Add the header to the cleaned content
    return f"{header}\n\n{clean_content}"

def generate_directory_structure(root_dir='.'):
    """Generates a text representation of the directory structure."""
    print("[DEBUG] Generating directory structure...")
    structure = ["# Directory Structure", "#" * 80]
    processed_paths = set() 
    abs_root = os.path.abspath(root_dir)
    abs_excluded_dirs = {os.path.join(abs_root, d) for d in EXCLUDED_DIRS}


    def add_directory(path, prefix=""):
        real_path = os.path.realpath(path)
        if real_path in processed_paths:
            structure.append(f"{prefix}[WARN] Symlink loop or duplicate processing: {path}")
            return
        processed_paths.add(real_path)

        # Additional check for node_modules and virtual environments
        if is_venv_or_node_modules(real_path):
            return
            
        # Check if the current directory is in an excluded path
        if is_path_excluded(real_path, abs_root):
            return
            
        # Check if the current directory itself is excluded
        if any(os.path.basename(real_path) == d for d in EXCLUDED_DIRS):
            return
        
        # Check if path is *under* an excluded dir (needed for topdown=False or initial call)
        is_under_excluded = any(real_path.startswith(excluded + os.path.sep) or real_path == excluded for excluded in abs_excluded_dirs)
        if is_under_excluded:
            return

        try:
            items = sorted(os.listdir(path))
        except OSError as e:
            print(f"[WARN] Could not list directory {path}: {e}")
            structure.append(f"{prefix}[ERROR] Cannot access directory: {e}")
            return

        entries = []
        for item in items:
             item_path = os.path.join(path, item)
             item_real_path = os.path.realpath(item_path)
             
             is_dir = os.path.isdir(item_path)
             is_file = os.path.isfile(item_path)

             # Skip node_modules and virtual environments
             if is_venv_or_node_modules(item_path):
                 continue

             # Check directory exclusions
             if is_dir and item not in EXCLUDED_DIRS and item_real_path not in abs_excluded_dirs:
                  is_under = any(item_real_path.startswith(excluded + os.path.sep) or item_real_path == excluded for excluded in abs_excluded_dirs)
                  if not is_under:
                      entries.append((item, True)) # Mark as directory
             # Check file exclusions
             elif is_file and item not in EXCLUDED_FILES:
                 entries.append((item, False)) # Mark as file
        
        for i, (entry_name, is_dir_entry) in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            
            if is_dir_entry:
                 structure.append(f"{prefix}{connector}{entry_name}/")
                 child_prefix = prefix + ("    " if is_last else "│   ")
                 add_directory(os.path.join(path, entry_name), child_prefix)
            else:
                 structure.append(f"{prefix}{connector}{entry_name}")

    add_directory(os.path.abspath(root_dir))
    print("[DEBUG] Directory structure generation complete.")
    return "\n".join(structure)


def is_path_excluded(path, root_dir):
    """
    Checks if the given path is in an excluded path.
    """
    rel_path = os.path.relpath(path, root_dir)
    for excluded_path in EXCLUDED_PATHS:
        # Check if rel_path is or starts with the excluded path
        if rel_path == excluded_path or rel_path.startswith(excluded_path + os.sep):
            return True
    return False

def collect_file_contents(root_dir='.'):
    """
    Collects contents of all files to be processed, returning a list of file blocks
    where each block contains the file path and content.
    """
    print(f"[DEBUG] Starting content collection process. Root: {root_dir}")
    abs_root = os.path.abspath(root_dir)
    abs_excluded_dirs = {os.path.join(abs_root, d) for d in EXCLUDED_DIRS}
    
    file_blocks = []
    
    # --- Walk Directory and Process Files ---
    print(f"[DEBUG] Walking directory tree from: {abs_root}")
    processed_files_count = 0
    skipped_files_count = 0
    skipped_venv_count = 0
    skipped_node_modules_count = 0

    for root, dirs, files in os.walk(abs_root, topdown=True):
        # Skip this directory and its subdirectories if it's a virtual env or node_modules
        if is_venv_or_node_modules(root):
            print(f"[DEBUG] Skipping virtual environment or node_modules directory: {root}")
            if 'node_modules' in root:
                skipped_node_modules_count += 1
            else:
                skipped_venv_count += 1
            dirs[:] = []  # Skip all subdirectories
            continue
            
        # Skip if the current directory is in an excluded path
        if is_path_excluded(root, abs_root):
            print(f"[DEBUG] Skipping excluded path directory: {root}")
            dirs[:] = []  # Skip all subdirectories
            continue

        # Filter excluded directories *before* recursion
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not is_venv_or_node_modules(os.path.join(root, d))]
        
        files.sort()

        relative_root = os.path.relpath(root, abs_root)
        if relative_root == '.': relative_root = '' 

        # Safety check: ensure current root isn't inside an excluded dir
        is_in_excluded_dir = any(root.startswith(excluded + os.path.sep) or root == excluded for excluded in abs_excluded_dirs)
        if is_in_excluded_dir: 
            continue

        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.normpath(os.path.join(relative_root, file))

            # 1. Check if file should be processed at all (type, name, exclusion)
            if not should_process_file(file_path, file):
                skipped_files_count += 1
                continue

            # 2. Read content for concatenation
            print(f"[DEBUG] Processing file for concatenation: {relative_file_path}")
            processed_files_count += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                # 3. Create and add a properly formatted header
                header = create_file_header(file_path, relative_file_path)
                content_with_header = prepend_header_if_needed(content, header, relative_file_path)
                
                # 4. Create the block for the concatenated output
                block_content = []
                block_content.append("#" * 80)
                block_content.append(f"# File: {relative_file_path}")
                block_content.append("#" * 80 + "\n")
                block_content.append(content_with_header) 
                block_content.append("\n\n" + "="*80 + "\n\n")  # Separator
                
                file_blocks.append({
                    'path': relative_file_path,
                    'content': "\n".join(block_content),
                    'size': len("\n".join(block_content))
                })

            except Exception as e:
                print(f"[WARN] Error reading {file_path} for concatenation: {e}. Skipping content.")
                # Add error note as a block
                block_content = []
                block_content.append("#" * 80)
                block_content.append(f"# File: {relative_file_path}")
                block_content.append("#" * 80 + "\n")
                block_content.append(f"[ERROR: Could not read file content due to: {e}]\n\n")
                block_content.append("="*80 + "\n\n")
                
                file_blocks.append({
                    'path': relative_file_path,
                    'content': "\n".join(block_content),
                    'size': len("\n".join(block_content))
                })

    print(f"[INFO] Successfully processed {processed_files_count} files")
    print(f"[INFO] Skipped {skipped_files_count} files (excluded types/names)")
    print(f"[INFO] Skipped {skipped_venv_count} virtual environment directories")
    print(f"[INFO] Skipped {skipped_node_modules_count} node_modules directories")
    return file_blocks, processed_files_count, skipped_files_count


def distribute_files_across_parts(file_blocks, num_parts=3):
    """
    Distributes file blocks across multiple parts ensuring roughly equal size
    and that no file is split across parts.
    """
    # Calculate total size
    total_size = sum(block['size'] for block in file_blocks)
    target_size_per_part = total_size / num_parts
    
    print(f"[DEBUG] Total content size: {total_size} bytes")
    print(f"[DEBUG] Target size per part: {target_size_per_part} bytes")
    
    # Sort files by size (largest first) to help balance distribution
    file_blocks.sort(key=lambda x: x['size'], reverse=True)
    
    # Initialize parts
    parts = [[] for _ in range(num_parts)]
    part_sizes = [0] * num_parts
    
    # Greedy algorithm to distribute files
    for block in file_blocks:
        # Find the part with the smallest current size
        smallest_part_idx = part_sizes.index(min(part_sizes))
        
        # Add the file to that part
        parts[smallest_part_idx].append(block)
        part_sizes[smallest_part_idx] += block['size']
    
    # Print the distribution results
    for i, size in enumerate(part_sizes):
        print(f"[INFO] Part {i+1} size: {size} bytes ({len(parts[i])} files)")
    
    return parts


def write_parts_to_files(parts, root_dir='.'):
    """Writes each part to a separate file."""
    abs_root = os.path.abspath(root_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i, part in enumerate(parts, 1):
        output_file = OUTPUT_FILENAME_TEMPLATE.format(i)
        output_path = os.path.join(abs_root, output_file)
        
        # Generate directory structure and header for each part
        directory_structure = generate_directory_structure(abs_root)
        
        # Create content
        all_content = []
        
        # Add header
        concatenated_header = (
            f"# Concatenated Project Code - Part {i} of {len(parts)}\n"
            f"# Generated: {timestamp}\n"
            f"# Root Directory: {abs_root}\n"
            f"{'='*80}\n"
        )
        all_content.append(concatenated_header)
        
        # Add directory structure to first part only
        if i == 1:
            all_content.append(directory_structure)
            all_content.append("\n\n" + "="*80 + "\n\n")
        
        # Add file contents for this part
        for block in part:
            all_content.append(block['content'])
        
        # Write the file
        print(f"[DEBUG] Writing part {i} to: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_content))
            print(f"[INFO] Successfully created {output_path} with {len(part)} files")
        except Exception as e:
            print(f"[ERROR] Critical error writing output file {output_path}: {e}")


# --- Main Function ---
def split_concatenated_scripts(num_parts=3, root_dir='.'):
    """
    Collects file contents, splits them into multiple parts with similar sizes,
    and writes each part to a separate file.
    """
    # 1. Collect all file contents
    file_blocks, processed_count, skipped_count = collect_file_contents(root_dir)
    
    # 2. Distribute files across parts
    parts = distribute_files_across_parts(file_blocks, num_parts)
    
    # 3. Write each part to a file
    write_parts_to_files(parts, root_dir)
    
    print(f"[INFO] Successfully split {processed_count} files into {num_parts} parts")
    print(f"[INFO] Files created: {', '.join([OUTPUT_FILENAME_TEMPLATE.format(i+1) for i in range(num_parts)])}")


# --- Main Execution ---
if __name__ == '__main__':
    split_concatenated_scripts(num_parts=3, root_dir='.') 