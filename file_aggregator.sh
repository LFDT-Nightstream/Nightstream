#!/usr/bin/env bash

# Usage: ./file_aggregator.sh [--no-comments] "<directory1,directory2,...>" <output_file> [exclude_paths...]

# Usage Examples:
#   1. Basic usage with single directory:
#      ./file_aggregator.sh "src" combined.txt "node_modules"
#
#   1b. Strip comments (Rust/JS/TS/C/CPP/Java/Go/Swift styles):
#      ./file_aggregator.sh --no-comments "crates,src" output.txt ".git,target"
#
#   2. Multiple directories and exclusions:
#      ./file_aggregator.sh "src,tests,docs" output.txt "node_modules,.git,*.tmp"
#
#   3. Complex exclusion patterns:
#      ./file_aggregator.sh "." mega_output.txt "build,dist,*.log,temporary_*"

if [ $# -lt 2 ]; then
  echo "Usage: $0 [--no-comments] \"<directory1,directory2,...>\" <output_file> [exclude_paths...]"
  exit 1
fi

# Flags and positional args (flags can be anywhere)
NO_COMMENTS=false
non_flag_args=()
for arg in "$@"; do
    case "$arg" in
        --no-comments|--strip-comments)
            NO_COMMENTS=true ;;
        *)
            non_flag_args+=("$arg") ;;
    esac
done

if [ ${#non_flag_args[@]} -lt 2 ]; then
  echo "Usage: $0 [--no-comments] \"<directory1,directory2,...>\" <output_file> [exclude_paths...]"
  exit 1
fi

DIRS_ARG="${non_flag_args[0]}"
outfile="${non_flag_args[1]}"
EXCLUDES_ARG="${non_flag_args[2]:-}"

# Split the comma-separated directories into an array
IFS=',' read -ra dirs <<< "$DIRS_ARG"
# Split the comma-separated exclusions into an array
IFS=',' read -ra excludes <<< "$EXCLUDES_ARG"  # Use empty string if no exclusions provided

# Stripper for C-style comments (handles // and nested /* */) used by .rs/.c/.cpp/.java/.js/.ts/.go/.swift/.css/.scss
strip_comments_cstyle() {
    awk '
    BEGIN { in_block = 0; sq = sprintf("%c", 39) }
    {
        line = $0
        output = ""
        i = 1
        len = length(line)
        in_s = 0; in_d = 0; in_bt = 0; escape = 0
        while (i <= len) {
            c = substr(line, i, 1)
            n2 = substr(line, i, 2)
            if (in_block) {
                if (n2 == "*/") {
                    in_block = 0
                    i += 2
                } else {
                    i++
                }
                continue
            }

            if (!in_s && !in_d && !in_bt) {
                if (n2 == "//") {
                    break
                } else if (n2 == "/*") {
                    in_block = 1
                    i += 2
                    continue
                } else if (c == "\"") {
                    in_d = 1; output = output c; i++; escape = 0; continue
                } else if (c == sq) {
                    in_s = 1; output = output c; i++; escape = 0; continue
                } else if (c == "`") {
                    in_bt = 1; output = output c; i++; escape = 0; continue
                } else {
                    output = output c; i++; continue
                }
            } else {
                if (escape) {
                    output = output c; escape = 0; i++; continue
                }
                if (c == "\\") {
                    output = output c; escape = 1; i++; continue
                }
                if (in_d && c == "\"") { in_d = 0; output = output c; i++; continue }
                if (in_s && c == sq) { in_s = 0; output = output c; i++; continue }
                if (in_bt && c == "`")   { in_bt = 0; output = output c; i++; continue }
                output = output c; i++; continue
            }
        }
        sub(/[ \t]+$/, "", output)
        if (output ~ /^[[:space:]]*$/) next
        print output
    }'
}

should_strip() {
    case "$1" in
        rs|c|h|hh|hpp|cpp|cc|cxx|java|js|jsx|ts|tsx|go|swift|kt|kts|scala|css|scss)
            return 0 ;;
        *)
            return 1 ;;
    esac
}

# Format number with comma separators (e.g., 5000 -> 5,000)
format_number() {
    echo "$1" | awk '{
        n = $0
        s = ""
        while (length(n) > 3) {
            s = "," substr(n, length(n)-2) s
            n = substr(n, 1, length(n)-3)
        }
        print n s
    }'
}

# Remove trailing slashes from all directories
for i in "${!dirs[@]}"; do
    dirs[$i]="${dirs[$i]%/}"
done

rm -f "$outfile"
touch "$outfile"

# Get the script's filename
script_name=$(basename "$0")

declare -a files_to_process=()

# Collect files from each directory (or file path), allowing duplicates for now
for dir in "${dirs[@]}"; do
    echo "Processing directory: $dir"
    [ -e "$dir" ] || { echo "Skipping missing: $dir"; continue; }

    # Build the find command with exclusions
    find_cmd="find \"$dir\""
    # First exclude the script itself and the output file
    find_cmd="$find_cmd -name \"$script_name\" -prune -o -name \"$(basename \"$outfile\")\" -prune -o"
    # Then add user-specified exclusions
    for exclude in "${excludes[@]}"; do
        [ -z "$exclude" ] && continue
        find_cmd="$find_cmd -name \"$exclude\" -prune -o"
    done
    find_cmd="$find_cmd -type f -print"

    # Append results to files_to_process (avoid subshell to preserve array)
    while IFS= read -r file; do
        files_to_process+=("$file")
    done < <(eval "$find_cmd")
done

# Deduplicate while preserving input order, then process
printf '%s\n' "${files_to_process[@]}" | awk '!seen[$0]++' | while IFS= read -r file; do
    [ -e "$file" ] || continue
    # Get size before processing this file
    size_before=$(wc -c < "$outfile")
    if $NO_COMMENTS; then
        ext="${file##*.}"
        if should_strip "$ext"; then
            echo "# $file" >> "$outfile"
            strip_comments_cstyle < "$file" >> "$outfile"
        else
            echo "# $file" >> "$outfile"
            cat "$file" >> "$outfile"
        fi
    else
        echo "# $file" >> "$outfile"
        cat "$file" >> "$outfile"
    fi
    echo >> "$outfile"
    # Calculate this file's token contribution
    size_after=$(wc -c < "$outfile")
    file_size=$((size_after - size_before))
    file_tokens=$((file_size / 4))
    echo "Processing: $file ($(format_number $file_tokens))"
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
    echo "Total size: $(format_number $final_size) bytes"
    echo "Total words: $(format_number $final_words)"
    echo "Total AI tokens (est): $(format_number $final_ai_tokens)"
fi
