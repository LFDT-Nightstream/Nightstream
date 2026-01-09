#!/usr/bin/env bash
set -eo pipefail
# Note: not using -u to avoid issues with empty variables in pipes

# Profile a test and output text-based analysis for AI review
# Usage: ./scripts/profile_for_ai.sh <package> <test_file> <test_function> [--ignored]
#
# Examples:
#   ./scripts/profile_for_ai.sh neo-fold test_sha256_single_step test_sha256_preimage_64_bytes
#   ./scripts/profile_for_ai.sh neo-fold test_starstream_tx_valid_optimized test_starstream_tx_valid_optimized
#   NEO_SPARTAN_BRIDGE_DEBUG_SHAPE=1 ./scripts/profile_for_ai.sh neo-spartan-bridge starstream_compression_smoke test_starstream_tx_export_spartan_phase1_smoke 400

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <package> <test_file> <test_function> [options]"
  echo ""
  echo "Options (order doesn't matter):"
  echo "  --ignored    Run ignored tests"
  echo "  <number>     Sample duration in seconds (default: 10)"
  echo ""
  echo "Examples:"
  echo "  $0 neo-fold test_sha256_single_step test_sha256_preimage_64_bytes --ignored"
  echo "  $0 neo-fold test_sha256_single_step test_sha256_preimage_64_bytes --ignored 20"
  echo "  $0 neo-fold test_starstream_tx_valid_optimized test_starstream_tx_valid_optimized 30"
  echo "  $0 neo-fold test_riscv_program_compiled_full_prove_verify test_riscv_program_compiled_full_prove_verify"
  echo "  NEO_SPARTAN_BRIDGE_DEBUG_SHAPE=1 $0 neo-spartan-bridge starstream_compression_smoke test_starstream_tx_export_spartan_phase1_smoke 400"
  echo ""
  echo "Output: Writes profile to profile-output.txt for AI analysis"
  exit 1
fi

PACKAGE="$1"
TEST_FILE="$2"
TEST_FUNCTION="$3"

# Parse remaining arguments
EXTRA_FLAGS=""
SAMPLE_DURATION=10  # Default 10 seconds

shift 3
while [ $# -gt 0 ]; do
  case "$1" in
    --ignored)
      EXTRA_FLAGS="--ignored"
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
echo ""

# Check for jq
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

# Build with profiling profile
echo "🔨 Building test with profiling profile..."
RUSTFLAGS="$RUSTFLAGS_COMBINED" cargo test --profile profiling \
  -p "$PACKAGE" --test "$TEST_FILE" --no-run 2>&1 | tail -5

# Find the test binary
echo "🔎 Finding test binary..."
TEST_BINARY=$(RUSTFLAGS="$RUSTFLAGS_COMBINED" cargo test --profile profiling -p "$PACKAGE" --test "$TEST_FILE" \
  --no-run --message-format=json 2>/dev/null | \
  jq -r 'select(.executable != null) | .executable' | head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary for $PACKAGE::$TEST_FILE"
  exit 1
fi

echo "   Binary: $TEST_BINARY"
echo ""

# Build the test command arguments
TEST_ARGS="$TEST_FUNCTION --nocapture"
if [ "$EXTRA_FLAGS" = "--ignored" ]; then
  TEST_ARGS="$TEST_FUNCTION --ignored --nocapture"
fi

# Temp files
TEMP_TEST_OUTPUT="/tmp/profile-test-output-$$.txt"
TEMP_SAMPLE="/tmp/profile-sample-$$.txt"

# Run the test in background and sample it
echo "🚀 Starting test and sampling for ${SAMPLE_DURATION}s..."
"$TEST_BINARY" $TEST_ARGS > "$TEMP_TEST_OUTPUT" 2>&1 &
TEST_PID=$!

# Wait a moment for the test to start
sleep 0.3

# Check if process is still running
if ! kill -0 $TEST_PID 2>/dev/null; then
  echo "⚠️  Test finished too quickly to sample."
  cat "$TEMP_TEST_OUTPUT"
  rm -f "$TEMP_TEST_OUTPUT"
  exit 0
fi

# Sample the process
/usr/bin/sample $TEST_PID $SAMPLE_DURATION $SAMPLE_INTERVAL -file "$TEMP_SAMPLE" 2>&1

# Wait for test to complete
wait $TEST_PID 2>/dev/null || true

# Show test output
echo ""
cat "$TEMP_TEST_OUTPUT"

echo ""
echo "📊 Processing profile data..."

# Create the output file with header
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

# Add timing info
grep -E "(duration|time|elapsed|took|CCS)" "$TEMP_TEST_OUTPUT" >> "$OUTPUT_FILE" 2>/dev/null || echo "(no timing captured)" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

=== TOP HOTSPOTS (excluding system waits) ===
Functions sorted by self-time (samples at top of stack).
Filtered: removed thread waits, semaphores, kernel sleeps.

EOF

# Extract and filter the "Sort by top of stack" section
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

# Extract the call tree for the test function thread
{
  sed -n "/Thread.*${TEST_FUNCTION}/,/^[[:space:]]*[0-9]* Thread/p" "$TEMP_SAMPLE" 2>/dev/null | \
    head -150
} >> "$OUTPUT_FILE" 2>/dev/null || true

cat >> "$OUTPUT_FILE" << EOF

=== FULL SAMPLE OUTPUT ===
EOF

cat "$TEMP_SAMPLE" >> "$OUTPUT_FILE"

# Clean up temp files
rm -f "$TEMP_SAMPLE" "$TEMP_TEST_OUTPUT"

# Display summary to console
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
      # Clean up function names for display
      echo "$line" | sed 's/(in [^)]*)//' | sed 's/::h[a-f0-9]*\s/ /'
    done
} || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Full profile saved to: $OUTPUT_FILE"
echo "💡 Share with AI: cat profile-output.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
