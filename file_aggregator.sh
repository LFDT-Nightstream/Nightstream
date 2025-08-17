#!/usr/bin/env bash

# Usage: ./file_aggregator.sh "<directory1,directory2,...>" <output_file> [exclude_paths...]

# Usage Examples:
#   1. Basic usage with single directory:
#      ./file_aggregator.sh "src" combined.txt "node_modules"
#
#   2. Multiple directories and exclusions:
#      ./file_aggregator.sh "src,tests,docs" output.txt "node_modules,.git,*.tmp"
#
#   3. Complex exclusion patterns:
#      ./file_aggregator.sh "." mega_output.txt "build,dist,*.log,temporary_*"

if [ $# -lt 2 ]; then
  echo "Usage: $0 \"<directory1,directory2,...>\" <output_file> [exclude_paths...]"
  exit 1
fi

# Split the comma-separated directories into an array
IFS=',' read -ra dirs <<< "$1"
outfile="$2"
# Split the comma-separated exclusions into an array
IFS=',' read -ra excludes <<< "${3:-}"  # Use empty string if no excludions provided
shift 3  # Remove processed arguments

# Remove trailing slashes from all directories
for i in "${!dirs[@]}"; do
    dirs[$i]="${dirs[$i]%/}"
done

rm -f "$outfile"
touch "$outfile"

# Get the script's filename
script_name=$(basename "$0")

# Process each directory
for dir in "${dirs[@]}"; do
    echo "Processing directory: $dir"
    
    # Build the find command with exclusions
    find_cmd="find \"$dir\""
    # First exclude the script itself and the output file
    find_cmd="$find_cmd -name \"$script_name\" -prune -o -name \"$(basename "$outfile")\" -prune -o"
    # Then add user-specified exclusions
    for exclude in "${excludes[@]}"; do
        find_cmd="$find_cmd -name \"$exclude\" -prune -o"
    done
    find_cmd="$find_cmd -type f -print"

    # Execute the constructed find command
    eval $find_cmd | while read file; do
        echo "Processing: $file"
        # Strip the leading path portion to get a relative path
        relpath="${file#$dir/}"
        echo "# $relpath" >> "$outfile"
        cat "$file" >> "$outfile"
        echo >> "$outfile"
    done
done

# Show final file statistics
if [ -f "$outfile" ]; then
    final_size=$(wc -c < "$outfile")
    final_words=$(wc -w < "$outfile")
    # Approximate AI token count (1 token ≈ 4 characters for English text)
    final_ai_tokens=$((final_size / 4))
    echo
    echo "=== Final Output Statistics ==="
    echo "Output file: $outfile"
    echo "Total size: ${final_size} bytes"
    echo "Total words: ${final_words}"
    echo "Total AI tokens (est): ${final_ai_tokens}"
fi
