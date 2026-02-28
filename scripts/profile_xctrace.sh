#!/usr/bin/env bash
set -eo pipefail

# Profile a Rust test using Apple Instruments (xctrace) and export symbolicated results
#
# Usage: ./scripts/profile_xctrace.sh <package> <test_file> <test_function> [--ignored] [--template <name>] [--features <csv>]
#
# Templates (run `xcrun xctrace list templates` for full list):
#   Time Profiler (default), Allocations, Leaks, File Activity, System Trace, etc.
#
# Examples:
#   ./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_128_bytes --ignored
#   ./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_128_bytes --ignored --template Allocations
#   ./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_128_bytes --ignored --template Leaks
#   ./scripts/profile_xctrace.sh neo-fold test_riscv_circuit_l2_transfer_compiled_trace_prove_verify test_note_spend_1in_1out_transfer_prove_verify --ignored --features poseidon-precompile --time-limit 20
#
# Output:
#   - profile-xctrace.trace     : Open with `open profile-xctrace.trace` for Instruments GUI
#   - profile-xctrace.txt       : Symbolicated text output for AI analysis
#
# Requirements:
#   - Xcode Command Line Tools (xcrun xctrace)
#   - jq (brew install jq)
#   - Python 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <package> <test_file> <test_function> [--ignored] [--template <name>] [--features <csv>] [--time-limit <N|Nms|Ns|Nm|Nh>]"
  echo ""
  echo "Templates: Time Profiler (default), Allocations, Leaks, File Activity, System Trace"
  echo "Run 'xcrun xctrace list templates' for full list"
  echo ""
  echo "Examples:"
  echo "  $0 neo-fold test_sha256_single_step test_sha256_preimage_128_bytes --ignored"
  echo "  $0 neo-fold test_sha256_single_step test_sha256_preimage_128_bytes --ignored --template Allocations"
  echo "  $0 neo-fold test_riscv_circuit_l2_transfer_compiled_trace_prove_verify test_note_spend_1in_1out_transfer_prove_verify --ignored --features poseidon-precompile --time-limit 20"
  exit 1
fi

PACKAGE="$1"
TEST_FILE="$2"
TEST_FUNCTION="$3"
shift 3

IGNORED_FLAG=""
TEMPLATE="Time Profiler"
TIME_LIMIT=""
FEATURES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --ignored)
      IGNORED_FLAG="--ignored"
      shift
      ;;
    --template)
      TEMPLATE="$2"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Normalize --time-limit: plain numbers default to seconds for xctrace.
if [ -n "$TIME_LIMIT" ]; then
  if [[ "$TIME_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    TIME_LIMIT="${TIME_LIMIT}s"
  elif [[ ! "$TIME_LIMIT" =~ ^[0-9]+([.][0-9]+)?(ms|s|m|h)$ ]]; then
    echo "❌ Invalid --time-limit '$TIME_LIMIT'. Use e.g. 20, 20s, 500ms, 2m."
    exit 1
  fi
fi

OUTPUT_FILE="$PROJECT_ROOT/profile-xctrace.txt"
TRACE_FILE="$PROJECT_ROOT/profile-xctrace.trace"

echo "🔍 XCtrace Profiling for AI analysis"
echo "   Package: $PACKAGE"
echo "   Test file: $TEST_FILE"
echo "   Test function: $TEST_FUNCTION"
echo "   Template: $TEMPLATE"
[ -n "$FEATURES" ] && echo "   Features: $FEATURES"
[ -n "$TIME_LIMIT" ] && echo "   Time limit: $TIME_LIMIT"
echo ""

# Check for jq
if ! command -v jq &> /dev/null; then
  echo "❌ jq not found. Install with: brew install jq"
  exit 1
fi

# Check for python3
if ! command -v python3 &> /dev/null; then
  echo "❌ python3 not found"
  exit 1
fi

# Build with profiling profile (debug symbols + optimizations)
echo "🔨 Building test with profiling profile..."
CARGO_FEATURES_ARGS=()
if [ -n "$FEATURES" ]; then
  CARGO_FEATURES_ARGS=(--features "$FEATURES")
fi

cargo build --profile profiling -p "$PACKAGE" --test "$TEST_FILE" "${CARGO_FEATURES_ARGS[@]}" 2>&1 | tail -3

# Find the test binary
echo "🔎 Finding test binary..."
TEST_BINARY=$(cargo test --profile profiling -p "$PACKAGE" --test "$TEST_FILE" "${CARGO_FEATURES_ARGS[@]}" \
  --no-run --message-format=json 2>/dev/null | \
  jq -r 'select(.executable != null) | .executable' | head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary"
  exit 1
fi

echo "   Binary: $TEST_BINARY"

# Generate dSYM for symbolication
echo "🔧 Generating dSYM bundle..."
DSYM_PATH="/tmp/$(basename "$TEST_BINARY").dSYM"
dsymutil "$TEST_BINARY" -o "$DSYM_PATH" 2>/dev/null

# Clean up old trace
rm -rf "$TRACE_FILE"

# Build test arguments
TEST_ARGS="$TEST_FUNCTION --nocapture"
if [ -n "$IGNORED_FLAG" ]; then
  TEST_ARGS="$TEST_FUNCTION --ignored --nocapture"
fi

# Record with xctrace, capture test output
echo "🚀 Recording with xctrace $TEMPLATE..."
TEST_OUTPUT_FILE="/tmp/xctrace-test-output-$$.txt"

TIME_LIMIT_ARG=""
if [ -n "$TIME_LIMIT" ]; then
  TIME_LIMIT_ARG="--time-limit $TIME_LIMIT"
fi

# xctrace may return non-zero even after producing a valid trace
# (e.g. target killed at time limit). Continue if output trace exists.
set +e
xcrun xctrace record --template "$TEMPLATE" \
  --output "$TRACE_FILE" \
  $TIME_LIMIT_ARG \
  --launch -- "$TEST_BINARY" $TEST_ARGS 2>&1 | tee "$TEST_OUTPUT_FILE"
RECORD_STATUS=${PIPESTATUS[0]}
set -e

if [ ! -d "$TRACE_FILE" ]; then
  echo "❌ xctrace did not produce trace output at $TRACE_FILE (status=$RECORD_STATUS)"
  exit 1
fi

if [ "$RECORD_STATUS" -ne 0 ]; then
  echo "⚠️  xctrace exited with status $RECORD_STATUS, but trace was produced; continuing."
fi

# Symbolicate the trace
echo "📝 Symbolicating trace..."
SYMBOLICATED_TRACE="$PROJECT_ROOT/profile-xctrace-sym.trace"
rm -rf "$SYMBOLICATED_TRACE"
xcrun xctrace symbolicate \
  --input "$TRACE_FILE" \
  --output "$SYMBOLICATED_TRACE" \
  --dsym "$DSYM_PATH" 2>&1 || true

# Export to XML
echo "📊 Exporting profile data..."
TEMP_XML="/tmp/xctrace-export-$$.xml"
xcrun xctrace export \
  --input "$SYMBOLICATED_TRACE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output "$TEMP_XML" 2>/dev/null || \
xcrun xctrace export \
  --input "$TRACE_FILE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output "$TEMP_XML" 2>/dev/null

echo "🔍 Parsing and symbolicating call stacks..."

# Python script to parse XML, count backtraces, symbolicate, and build call tree
python3 << 'PYTHON_SCRIPT' - "$TEMP_XML" "$TEST_BINARY" "$OUTPUT_FILE" "$PACKAGE" "$TEST_FILE" "$TEST_FUNCTION" "$TEST_OUTPUT_FILE"
import sys
import re
import subprocess
from collections import defaultdict
import xml.etree.ElementTree as ET

xml_file = sys.argv[1]
test_binary = sys.argv[2]
output_file = sys.argv[3]
package = sys.argv[4]
test_file = sys.argv[5]
test_function = sys.argv[6]
test_output_file = sys.argv[7]

# Read test output for timing info
timing_lines = []
try:
    with open(test_output_file, 'r') as f:
        for line in f:
            if any(kw in line.lower() for kw in ['duration', 'time', 'elapsed', 'ccs:', 'took']):
                timing_lines.append(line.strip())
except:
    pass

# Parse XML
print("   Parsing XML...")
with open(xml_file, 'r') as f:
    content = f.read()

# Extract load address for our binary
binary_name = test_binary.split('/')[-1]
load_addr_match = re.search(rf'name="{re.escape(binary_name)}"[^>]*load-addr="(0x[0-9a-f]+)"', content)
load_addr = load_addr_match.group(1) if load_addr_match else None
print(f"   Load address: {load_addr}")

# Parse all frames (id -> {name, addr, binary})
frames = {}
for m in re.finditer(r'<frame id="(\d+)" name="([^"]*)" addr="(0x[0-9a-f]+)"', content):
    frames[m.group(1)] = {'name': m.group(2), 'addr': m.group(3)}

# Parse all backtraces (id -> list of frame ids/refs)
backtraces = {}
# Find backtrace definitions with their frames
bt_pattern = re.compile(r'<backtrace id="(\d+)">(.*?)</backtrace>', re.DOTALL)
for m in bt_pattern.finditer(content):
    bt_id = m.group(1)
    bt_content = m.group(2)
    frame_ids = []
    # Get frame ids (either id= or ref=)
    for fm in re.finditer(r'<frame (?:id|ref)="(\d+)"', bt_content):
        frame_ids.append(fm.group(1))
    backtraces[bt_id] = frame_ids

# Count backtrace occurrences (both id= and ref=)
backtrace_counts = defaultdict(int)
for m in re.finditer(r'<backtrace (?:id|ref)="(\d+)"', content):
    backtrace_counts[m.group(1)] += 1

print(f"   Found {len(backtraces)} unique backtraces, {sum(backtrace_counts.values())} total samples")

# Collect all addresses that need symbolication
addrs_to_symbolicate = set()
for bt_id, frame_ids in backtraces.items():
    for fid in frame_ids:
        if fid in frames:
            addr = frames[fid]['addr']
            name = frames[fid]['name']
            # If name looks like an address, we need to symbolicate
            if name.startswith('0x'):
                addrs_to_symbolicate.add(addr)

print(f"   Symbolicating {len(addrs_to_symbolicate)} addresses with atos...")

# Symbolicate using atos
addr_to_symbol = {}
if load_addr and addrs_to_symbolicate:
    try:
        addr_list = list(addrs_to_symbolicate)
        cmd = ['atos', '-o', test_binary, '-l', load_addr] + addr_list
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        symbols = result.stdout.strip().split('\n')
        for addr, sym in zip(addr_list, symbols):
            addr_to_symbol[addr] = sym
    except Exception as e:
        print(f"   Warning: atos failed: {e}")

# Build symbol lookup for all frames
def get_frame_symbol(fid):
    if fid not in frames:
        return f"<unknown frame {fid}>"
    frame = frames[fid]
    addr = frame['addr']
    name = frame['name']
    if name.startswith('0x') and addr in addr_to_symbol:
        return addr_to_symbol[addr]
    return name

# Count self-time (top of stack) per function
self_time = defaultdict(int)
# Count inclusive time (anywhere in stack) per function
inclusive_time = defaultdict(int)

for bt_id, count in backtrace_counts.items():
    if bt_id not in backtraces:
        continue
    frame_ids = backtraces[bt_id]
    if not frame_ids:
        continue
    
    # Top of stack (first frame) = self time
    top_symbol = get_frame_symbol(frame_ids[0])
    self_time[top_symbol] += count
    
    # All frames = inclusive time
    seen = set()
    for fid in frame_ids:
        sym = get_frame_symbol(fid)
        if sym not in seen:
            inclusive_time[sym] += count
            seen.add(sym)

# Build call tree (aggregate by stack pattern)
# Group by thread
thread_trees = defaultdict(lambda: defaultdict(int))

# Find thread info per sample
thread_pattern = re.compile(r'<thread[^>]*fmt="([^"]*)"')
threads_found = set()
for m in thread_pattern.finditer(content):
    threads_found.add(m.group(1))

# Build call tree structure
call_tree = defaultdict(lambda: {'count': 0, 'children': defaultdict(lambda: {'count': 0, 'children': {}})})

for bt_id, count in backtrace_counts.items():
    if bt_id not in backtraces:
        continue
    frame_ids = backtraces[bt_id]
    if not frame_ids:
        continue
    
    # Reverse to get root -> leaf order
    symbols = [get_frame_symbol(fid) for fid in reversed(frame_ids)]
    
    # Build nested tree
    current = call_tree
    for sym in symbols:
        if sym not in current:
            current[sym] = {'count': 0, 'children': {}}
        current[sym]['count'] += count
        current = current[sym]['children']

# Generate output
total_samples = sum(backtrace_counts.values())

output_lines = []
output_lines.append("=" * 80)
output_lines.append("XCTRACE PROFILING REPORT FOR AI ANALYSIS")
output_lines.append("=" * 80)
output_lines.append(f"Package: {package}")
output_lines.append(f"Test file: {test_file}")
output_lines.append(f"Test function: {test_function}")
output_lines.append(f"Total samples: {total_samples}")
output_lines.append("Tool: xcrun xctrace (Time Profiler template)")
output_lines.append("=" * 80)
output_lines.append("")

if timing_lines:
    output_lines.append("=== TIMING FROM TEST OUTPUT ===")
    for line in timing_lines:
        output_lines.append(line)
    output_lines.append("")

output_lines.append("=== TOP HOTSPOTS BY SELF-TIME (samples at top of stack) ===")
output_lines.append("Functions where CPU was actually executing (not waiting on callees)")
output_lines.append("")

# Filter and sort by self-time
sorted_self = sorted(self_time.items(), key=lambda x: -x[1])
# Filter out system stuff
filtered_self = [(s, c) for s, c in sorted_self 
                 if not any(x in s for x in ['libsystem', 'dyld', 'libdispatch', '__psynch', 'semaphore', 'thread_start', '_pthread'])]

for sym, count in filtered_self[:30]:
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
    output_lines.append(f"{count:6d} ({pct:5.1f}%)  {sym}")

output_lines.append("")
output_lines.append("=== TOP HOTSPOTS BY INCLUSIVE TIME (anywhere in stack) ===")
output_lines.append("Functions that were on the stack (self + time in callees)")
output_lines.append("")

sorted_inclusive = sorted(inclusive_time.items(), key=lambda x: -x[1])
filtered_inclusive = [(s, c) for s, c in sorted_inclusive 
                      if not any(x in s for x in ['libsystem', 'dyld', 'libdispatch', '__psynch', 'semaphore', 'thread_start', '_pthread', 'start '])]

for sym, count in filtered_inclusive[:30]:
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
    output_lines.append(f"{count:6d} ({pct:5.1f}%)  {sym}")

output_lines.append("")
output_lines.append("=== CALL TREE (heaviest paths) ===")
output_lines.append("")

def print_tree(tree, indent=0, max_depth=15, min_samples=2, prefix=""):
    if indent > max_depth:
        return
    
    # Sort children by count
    items = sorted(tree.items(), key=lambda x: -x[1]['count'] if isinstance(x[1], dict) else 0)
    
    for i, (sym, data) in enumerate(items):
        if not isinstance(data, dict):
            continue
        count = data.get('count', 0)
        if count < min_samples:
            continue
        
        # Skip system functions at top level
        if indent == 0 and any(x in sym for x in ['dyld', 'libSystem']):
            continue
            
        pct = 100.0 * count / total_samples if total_samples > 0 else 0
        
        # Truncate long symbols
        display_sym = sym
        if len(display_sym) > 80:
            display_sym = display_sym[:77] + "..."
        
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "
        
        output_lines.append(f"{prefix}{connector}{count:4d} ({pct:4.1f}%)  {display_sym}")
        
        children = data.get('children', {})
        if children:
            print_tree(children, indent + 1, max_depth, min_samples, prefix + child_prefix)

print_tree(call_tree, min_samples=max(1, total_samples // 100))

# Write output
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

# Print summary to console
print("")
print("━" * 80)
print("📋 XCTRACE PROFILE REPORT")
print("━" * 80)
print("")
print("🔥 TOP HOTSPOTS BY SELF-TIME:")
for sym, count in filtered_self[:15]:
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
    # Clean up symbol for display
    clean_sym = re.sub(r'\s*\(in [^)]+\)', '', sym)
    clean_sym = re.sub(r'::h[a-f0-9]{16}', '', clean_sym)
    if len(clean_sym) > 70:
        clean_sym = clean_sym[:67] + "..."
    print(f"  {count:5d} ({pct:5.1f}%)  {clean_sym}")

print("")
print("📊 TOP BY INCLUSIVE TIME:")
for sym, count in filtered_inclusive[:10]:
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
    clean_sym = re.sub(r'\s*\(in [^)]+\)', '', sym)
    clean_sym = re.sub(r'::h[a-f0-9]{16}', '', clean_sym)
    if len(clean_sym) > 70:
        clean_sym = clean_sym[:67] + "..."
    print(f"  {count:5d} ({pct:5.1f}%)  {clean_sym}")

PYTHON_SCRIPT

# Cleanup
rm -f "$TEMP_XML" "$TEST_OUTPUT_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Full profile saved to: $OUTPUT_FILE"
echo "📂 Trace file (for Instruments): $TRACE_FILE"
echo "💡 Open in Instruments: open $TRACE_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
