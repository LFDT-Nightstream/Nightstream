#!/usr/bin/env bash
set -eo pipefail

# Profile a test and output text-based analysis for AI review
# Usage: ./scripts/profile_for_ai.sh <package> <test_file> <test_function> [--ignored] [--features <csv>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <package> <test_file> <test_function> [options]"
  echo ""
  echo "Options (order doesn't matter):"
  echo "  --ignored    Run ignored tests"
  echo "  --features   Cargo features (comma-separated)"
  echo "  <number>     Sample duration in seconds (default: 10)"
  echo ""
  echo "Examples:"
  echo "  $0 nightstream circuit_lifecycle poseidon_base_step_matches_lean_and_verifies --ignored"
  echo "  $0 nightstream circuit_lifecycle poseidon_base_step_matches_lean_and_verifies --ignored 20"
  echo "  $0 nightstream application_poseidon2 every_poseidon2_application_row_matches_the_saved_lean_reference 30"
  echo "  $0 nightstream circuit_lifecycle poseidon_metal_recursive_lifecycle --ignored --features metal"
  echo ""
  echo "Output: Writes profile to profile-output.txt for AI analysis"
  exit 1
fi

PACKAGE="$1"
TEST_FILE="$2"
TEST_FUNCTION="$3"

EXTRA_FLAGS=""
SAMPLE_DURATION=10
FEATURES=""

shift 3
while [ $# -gt 0 ]; do
  case "$1" in
    --ignored)
      EXTRA_FLAGS="--ignored"
      ;;
    --features)
      FEATURES="$2"
      shift
      ;;
    [0-9]*)
      SAMPLE_DURATION="$1"
      ;;
  esac
  shift
done

OUTPUT_FILE="$PROJECT_ROOT/profile-output.txt"
SAMPLE_INTERVAL=1

echo "🔍 Profiling for AI analysis"
echo "   Package: $PACKAGE"
echo "   Test file: $TEST_FILE"
echo "   Test function: $TEST_FUNCTION"
echo "   Extra flags: $EXTRA_FLAGS"
[ -n "$FEATURES" ] && echo "   Features: $FEATURES"
echo ""

if ! command -v jq &> /dev/null; then
  echo "❌ jq not found. Install with: brew install jq"
  exit 1
fi

# Ensure frame pointers for `sample(1)` while preserving any caller-provided RUSTFLAGS.
RUSTFLAGS_COMBINED="${RUSTFLAGS:-}"
if [ -n "$RUSTFLAGS_COMBINED" ]; then
  RUSTFLAGS_COMBINED="$RUSTFLAGS_COMBINED "
fi
RUSTFLAGS_COMBINED="${RUSTFLAGS_COMBINED}-C force-frame-pointers=yes"

echo "🔨 Building test with profiling profile..."
CARGO_FEATURES_ARGS=()
if [ -n "$FEATURES" ]; then
  CARGO_FEATURES_ARGS=(--features "$FEATURES")
fi

RUSTFLAGS="$RUSTFLAGS_COMBINED" cargo test --profile profiling \
  -p "$PACKAGE" --test "$TEST_FILE" "${CARGO_FEATURES_ARGS[@]}" --no-run 2>&1 | tail -5

echo "🔎 Finding test binary..."
TEST_BINARY=$(RUSTFLAGS="$RUSTFLAGS_COMBINED" cargo test --profile profiling -p "$PACKAGE" --test "$TEST_FILE" \
  "${CARGO_FEATURES_ARGS[@]}" \
  --no-run --message-format=json 2>/dev/null | \
  jq -r --arg test_name "$TEST_FILE" '
    select(.executable != null and .target != null and (.target.kind | index("test")) and .target.name == $test_name)
    | .executable
  ' | head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary for $PACKAGE::$TEST_FILE"
  exit 1
fi

echo "   Binary: $TEST_BINARY"
echo ""

TEST_ARGS="$TEST_FUNCTION --nocapture"
if [ "$EXTRA_FLAGS" = "--ignored" ]; then
  TEST_ARGS="$TEST_FUNCTION --ignored --nocapture"
fi

TEMP_TEST_OUTPUT="/tmp/profile-test-output-$$.txt"
TEMP_SAMPLE="/tmp/profile-sample-$$.txt"

echo "🚀 Starting test and sampling for ${SAMPLE_DURATION}s..."
perl -e 'alarm shift; exec @ARGV' 300 "$TEST_BINARY" $TEST_ARGS > "$TEMP_TEST_OUTPUT" 2>&1 &
TEST_PID=$!

sleep 0.3

if ! kill -0 $TEST_PID 2>/dev/null; then
  echo "⚠️  Test finished too quickly to sample."
  cat "$TEMP_TEST_OUTPUT"
  rm -f "$TEMP_TEST_OUTPUT"
  exit 0
fi

/usr/bin/sample $TEST_PID $SAMPLE_DURATION $SAMPLE_INTERVAL -file "$TEMP_SAMPLE" 2>&1

wait $TEST_PID 2>/dev/null || true

echo ""
cat "$TEMP_TEST_OUTPUT"

echo ""
echo "📊 Processing profile data..."

cat > "$OUTPUT_FILE" << EOF
================================================================================
PROFILING REPORT FOR AI ANALYSIS
================================================================================
Package: $PACKAGE
Test file: $TEST_FILE
Test function: $TEST_FUNCTION
Date: $(date)
Sample duration: ${SAMPLE_DURATION}s at ${SAMPLE_INTERVAL}ms intervals
================================================================================

=== TIMING FROM TEST OUTPUT ===
EOF

grep -Ei "(duration|timing|time|elapsed|took|CCS|setup|append|finish|verify|online|throughput|rows|vars|total)" "$TEMP_TEST_OUTPUT" >> "$OUTPUT_FILE" 2>/dev/null || echo "(no timing captured)" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

=== TOP HOTSPOTS (excluding system waits) ===
Functions sorted by self-time (samples at top of stack).
Filtered: removed thread waits, semaphores, kernel sleeps.

EOF

{
  sed -n '/Sort by top of stack/,/^$/p' "$TEMP_SAMPLE" 2>/dev/null | \
    grep -v -E "(psynch|semaphore|swtch_pri|mach_msg|kevent|workq|thread_start|_pthread|libsystem_kernel|libdispatch|Sort by)" | \
    grep -E '\s+[0-9]+$' | \
    head -25
} >> "$OUTPUT_FILE" 2>/dev/null || true

cat >> "$OUTPUT_FILE" << EOF

=== CALL TREE SUMMARY ===
Call graph for the test thread (first 150 lines):

EOF

{
  sed -n "/Thread.*${TEST_FUNCTION}/,/^[[:space:]]*[0-9]* Thread/p" "$TEMP_SAMPLE" 2>/dev/null | \
    head -150
} >> "$OUTPUT_FILE" 2>/dev/null || true

cat >> "$OUTPUT_FILE" << EOF

=== FULL SAMPLE OUTPUT ===
EOF

cat "$TEMP_SAMPLE" >> "$OUTPUT_FILE"

rm -f "$TEMP_SAMPLE" "$TEMP_TEST_OUTPUT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 QUICK SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔥 TOP HOTSPOTS (your code, excluding waits):"
{
  sed -n '/=== TOP HOTSPOTS/,/=== CALL TREE/p' "$OUTPUT_FILE" | \
    grep -E '\s+[0-9]+$' | head -15 | \
    while read -r line; do
      echo "$line" | sed 's/(in [^)]*)//' | sed 's/::h[a-f0-9]*\s/ /'
    done
} || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Full profile saved to: $OUTPUT_FILE"
echo "💡 Share with AI: cat profile-output.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
