#!/usr/bin/env bash
set -eo pipefail

# Profile a Rust test using Apple Instruments (xctrace) and export symbolicated results
#
# Usage: ./scripts/profile_xctrace.sh <package> <test_file> <test_function> [--ignored] [--template <name>] [--instrument <name>] [--features <csv>] [--time-limit <N|Nms|Ns|Nm|Nh>]
#
# Templates (run `xcrun xctrace list templates` for full list):
#   Time Profiler (default), Metal System Trace, Allocations, Leaks, etc.
#
# Output:
#   - profile-xctrace.trace     : Open with `open profile-xctrace.trace` for Instruments GUI
#   - profile-xctrace.txt       : Symbolicated CPU report, or capture summary for non-CPU templates
#
# Requirements:
#   - Full Xcode with its license accepted (xcrun xctrace)
#   - Metal Toolchain component when the target builds Metal shaders
#   - jq (brew install jq)
#   - Python 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Instruments and Metal are full-Xcode tools. Respect an explicit selection,
# otherwise avoid inheriting a system-wide CommandLineTools selection.
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
if [ ! -d "$DEVELOPER_DIR" ]; then
  echo "❌ Full Xcode was not found at '$DEVELOPER_DIR'."
  echo "   Set DEVELOPER_DIR to the Developer directory inside Xcode.app."
  exit 1
fi

if [ $# -lt 3 ]; then
  echo "Usage: $0 <package> <test_file> <test_function> [--ignored] [--template <name>] [--instrument <name>] [--features <csv>] [--time-limit <N|Nms|Ns|Nm|Nh>]"
  echo ""
  echo "Templates: Time Profiler (default), Metal System Trace, Allocations, Leaks, File Activity, System Trace"
  echo "Run 'xcrun xctrace list templates' for full list"
  echo ""
  echo "Examples:"
  echo "  $0 nightstream circuit_lifecycle poseidon_base_step_matches_lean_and_verifies --ignored"
  echo "  $0 nightstream circuit_lifecycle poseidon_base_step_matches_lean_and_verifies --ignored --template Allocations"
  echo "  $0 nightstream circuit_lifecycle poseidon_metal_recursive_lifecycle --ignored --features metal --template \"Metal System Trace\" --instrument \"Metal GPU Counters\" --time-limit 60s"
  echo "  $0 nightstream application_poseidon2 every_poseidon2_application_row_matches_the_saved_lean_reference --time-limit 20"
  exit 1
fi

PACKAGE="$1"
TEST_FILE="$2"
TEST_FUNCTION="$3"
shift 3

IGNORED_FLAG=""
TEMPLATE="Time Profiler"
TIME_LIMIT="5m"
FEATURES=""
INSTRUMENTS=()

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
    --instrument)
      INSTRUMENTS+=("$2")
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

if [[ "$TIME_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  TIME_LIMIT="${TIME_LIMIT}s"
elif [[ ! "$TIME_LIMIT" =~ ^[0-9]+([.][0-9]+)?(ms|s|m|h)$ ]]; then
  echo "❌ Invalid --time-limit '$TIME_LIMIT'. Use e.g. 20, 20s, 500ms, 2m."
  exit 1
fi

case "$TIME_LIMIT" in
  *ms) LIMIT_VALUE="${TIME_LIMIT%ms}"; LIMIT_FACTOR=1 ;;
  *s) LIMIT_VALUE="${TIME_LIMIT%s}"; LIMIT_FACTOR=1000 ;;
  *m) LIMIT_VALUE="${TIME_LIMIT%m}"; LIMIT_FACTOR=60000 ;;
  *h) LIMIT_VALUE="${TIME_LIMIT%h}"; LIMIT_FACTOR=3600000 ;;
esac
if ! awk -v value="$LIMIT_VALUE" -v factor="$LIMIT_FACTOR" 'BEGIN { exit !(value * factor <= 300000) }'; then
  echo "❌ --time-limit must not exceed the repository's 5-minute test cap."
  exit 1
fi

OUTPUT_FILE="$PROJECT_ROOT/profile-xctrace.txt"
TRACE_FILE="$PROJECT_ROOT/profile-xctrace.trace"

echo "🔍 XCtrace Profiling for AI analysis"
echo "   Package: $PACKAGE"
echo "   Test file: $TEST_FILE"
echo "   Test function: $TEST_FUNCTION"
echo "   Template: $TEMPLATE"
[ "${#INSTRUMENTS[@]}" -gt 0 ] && echo "   Extra instruments: ${INSTRUMENTS[*]}"
[ -n "$FEATURES" ] && echo "   Features: $FEATURES"
[ -n "$TIME_LIMIT" ] && echo "   Time limit: $TIME_LIMIT"
echo ""

if ! command -v jq &> /dev/null; then
  echo "❌ jq not found. Install with: brew install jq"
  exit 1
fi

if ! command -v python3 &> /dev/null; then
  echo "❌ python3 not found"
  exit 1
fi

if ! xcodebuild -license check &> /dev/null; then
  echo "❌ The Xcode license has not been accepted."
  echo "   Run: sudo xcodebuild -license"
  echo "   Then: sudo xcodebuild -runFirstLaunch"
  exit 1
fi
if ! xcrun --sdk macosx metal --version &> /dev/null; then
  echo "❌ The Xcode Metal Toolchain component is unavailable."
  echo "   Run: xcodebuild -downloadComponent MetalToolchain"
  exit 1
fi

echo "🔨 Building test with profiling profile..."
CARGO_FEATURES_ARGS=()
if [ -n "$FEATURES" ]; then
  CARGO_FEATURES_ARGS=(--features "$FEATURES")
fi

cargo build --profile profiling -p "$PACKAGE" --test "$TEST_FILE" "${CARGO_FEATURES_ARGS[@]}" 2>&1 | tail -3

echo "🔎 Finding test binary..."
TEST_BINARY=$(cargo build --profile profiling -p "$PACKAGE" --test "$TEST_FILE" "${CARGO_FEATURES_ARGS[@]}" \
  --message-format=json 2>/dev/null | \
  jq -r --arg test_name "$TEST_FILE" \
    'select(.executable != null and .target.name == $test_name and (.target.kind | index("test"))) | .executable' | \
  head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary"
  exit 1
fi

echo "   Binary: $TEST_BINARY"

echo "🔧 Generating dSYM bundle..."
DSYM_PATH="/tmp/$(basename "$TEST_BINARY").dSYM"
dsymutil "$TEST_BINARY" -o "$DSYM_PATH" 2>/dev/null

rm -rf "$TRACE_FILE"

TEST_ARGS="$TEST_FUNCTION --exact --nocapture"
if [ -n "$IGNORED_FLAG" ]; then
  TEST_ARGS="$TEST_FUNCTION --exact --ignored --nocapture"
fi

echo "🚀 Recording with xctrace $TEMPLATE..."
TEST_OUTPUT_FILE="/tmp/xctrace-test-output-$$.txt"

TIME_LIMIT_ARG="--time-limit $TIME_LIMIT"
INSTRUMENT_ARGS=()
for instrument in "${INSTRUMENTS[@]}"; do
  INSTRUMENT_ARGS+=(--instrument "$instrument")
done

# xctrace may return non-zero even after producing a valid trace
# (e.g. target killed at time limit). Continue if output trace exists.
set +e
xcrun xctrace record --template "$TEMPLATE" \
  "${INSTRUMENT_ARGS[@]}" \
  --output "$TRACE_FILE" \
  $TIME_LIMIT_ARG \
  --target-stdout - \
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

# Other templates do not expose the Time Profiler XML schema parsed below.
if [ "$TEMPLATE" != "Time Profiler" ]; then
  {
    echo "XCTRACE CAPTURE SUMMARY"
    echo "Package: $PACKAGE"
    echo "Test file: $TEST_FILE"
    echo "Test function: $TEST_FUNCTION"
    echo "Template: $TEMPLATE"
    [ "${#INSTRUMENTS[@]}" -gt 0 ] && echo "Extra instruments: ${INSTRUMENTS[*]}"
    echo "Time limit: $TIME_LIMIT"
    echo "Trace: $TRACE_FILE"
    echo ""
    echo "TEST OUTPUT"
    sed -n '1,240p' "$TEST_OUTPUT_FILE"
  } > "$OUTPUT_FILE"
  rm -f "$TEST_OUTPUT_FILE"
  echo ""
  echo "✅ Capture summary saved to: $OUTPUT_FILE"
  echo "📂 Trace file (for Instruments): $TRACE_FILE"
  echo "💡 Open in Instruments: open $TRACE_FILE"
  exit 0
fi

echo "📝 Symbolicating trace..."
SYMBOLICATED_TRACE="$PROJECT_ROOT/profile-xctrace-sym.trace"
rm -rf "$SYMBOLICATED_TRACE"
xcrun xctrace symbolicate \
  --input "$TRACE_FILE" \
  --output "$SYMBOLICATED_TRACE" \
  --dsym "$DSYM_PATH" 2>&1 || true

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

timing_lines = []
try:
    with open(test_output_file, 'r') as f:
        for line in f:
            if any(kw in line.lower() for kw in ['duration', 'time', 'elapsed', 'ccs:', 'took']):
                timing_lines.append(line.strip())
except:
    pass

print("   Parsing XML...")
with open(xml_file, 'r') as f:
    content = f.read()

binary_name = test_binary.split('/')[-1]
load_addr_match = re.search(rf'name="{re.escape(binary_name)}"[^>]*load-addr="(0x[0-9a-f]+)"', content)
load_addr = load_addr_match.group(1) if load_addr_match else None
print(f"   Load address: {load_addr}")

frames = {}
for m in re.finditer(r'<frame id="(\d+)" name="([^"]*)" addr="(0x[0-9a-f]+)"', content):
    frames[m.group(1)] = {'name': m.group(2), 'addr': m.group(3)}

backtraces = {}
bt_pattern = re.compile(r'<backtrace id="(\d+)">(.*?)</backtrace>', re.DOTALL)
for m in bt_pattern.finditer(content):
    bt_id = m.group(1)
    bt_content = m.group(2)
    frame_ids = []
    for fm in re.finditer(r'<frame (?:id|ref)="(\d+)"', bt_content):
        frame_ids.append(fm.group(1))
    backtraces[bt_id] = frame_ids

backtrace_counts = defaultdict(int)
for m in re.finditer(r'<backtrace (?:id|ref)="(\d+)"', content):
    backtrace_counts[m.group(1)] += 1

print(f"   Found {len(backtraces)} unique backtraces, {sum(backtrace_counts.values())} total samples")

addrs_to_symbolicate = set()
for bt_id, frame_ids in backtraces.items():
    for fid in frame_ids:
        if fid in frames:
            addr = frames[fid]['addr']
            name = frames[fid]['name']
            if name.startswith('0x'):
                addrs_to_symbolicate.add(addr)

print(f"   Symbolicating {len(addrs_to_symbolicate)} addresses with atos...")

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

def get_frame_symbol(fid):
    if fid not in frames:
        return f"<unknown frame {fid}>"
    frame = frames[fid]
    addr = frame['addr']
    name = frame['name']
    if name.startswith('0x') and addr in addr_to_symbol:
        return addr_to_symbol[addr]
    return name

self_time = defaultdict(int)
inclusive_time = defaultdict(int)

for bt_id, count in backtrace_counts.items():
    if bt_id not in backtraces:
        continue
    frame_ids = backtraces[bt_id]
    if not frame_ids:
        continue
    
    top_symbol = get_frame_symbol(frame_ids[0])
    self_time[top_symbol] += count
    
    seen = set()
    for fid in frame_ids:
        sym = get_frame_symbol(fid)
        if sym not in seen:
            inclusive_time[sym] += count
            seen.add(sym)

thread_trees = defaultdict(lambda: defaultdict(int))

thread_pattern = re.compile(r'<thread[^>]*fmt="([^"]*)"')
threads_found = set()
for m in thread_pattern.finditer(content):
    threads_found.add(m.group(1))

call_tree = defaultdict(lambda: {'count': 0, 'children': defaultdict(lambda: {'count': 0, 'children': {}})})

for bt_id, count in backtrace_counts.items():
    if bt_id not in backtraces:
        continue
    frame_ids = backtraces[bt_id]
    if not frame_ids:
        continue
    
    symbols = [get_frame_symbol(fid) for fid in reversed(frame_ids)]
    
    current = call_tree
    for sym in symbols:
        if sym not in current:
            current[sym] = {'count': 0, 'children': {}}
        current[sym]['count'] += count
        current = current[sym]['children']

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

sorted_self = sorted(self_time.items(), key=lambda x: -x[1])
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
    
    items = sorted(tree.items(), key=lambda x: -x[1]['count'] if isinstance(x[1], dict) else 0)
    
    for i, (sym, data) in enumerate(items):
        if not isinstance(data, dict):
            continue
        count = data.get('count', 0)
        if count < min_samples:
            continue
        
        if indent == 0 and any(x in sym for x in ['dyld', 'libSystem']):
            continue
            
        pct = 100.0 * count / total_samples if total_samples > 0 else 0
        
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

with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print("")
print("━" * 80)
print("📋 XCTRACE PROFILE REPORT")
print("━" * 80)
print("")
print("🔥 TOP HOTSPOTS BY SELF-TIME:")
for sym, count in filtered_self[:15]:
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
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

rm -f "$TEMP_XML" "$TEST_OUTPUT_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Full profile saved to: $OUTPUT_FILE"
echo "📂 Trace file (for Instruments): $TRACE_FILE"
echo "💡 Open in Instruments: open $TRACE_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
