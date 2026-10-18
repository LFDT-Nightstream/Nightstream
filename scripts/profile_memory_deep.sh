#!/usr/bin/env bash
set -eo pipefail

# Deep memory profiling with multiple snapshots over time
# Captures heap state at multiple points during test execution
#
# Usage: ./scripts/profile_memory_deep.sh <package> <test_file> <test_function> [--ignored]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <package> <test_file> <test_function> [--ignored]"
  echo ""
  echo "This script captures multiple heap snapshots during test execution."
  echo "It shows how memory grows over time and which functions allocate the most."
  echo ""
  echo "Example:"
  echo "  $0 neo-fold test_sha256_single_step test_sha256_preimage_64_bytes --ignored"
  echo ""
  echo "Output: profile-memory-deep.txt"
  exit 1
fi

PACKAGE="$1"
TEST_FILE="$2"
TEST_FUNCTION="$3"
EXTRA_FLAGS="${4:-}"

OUTPUT_FILE="$PROJECT_ROOT/profile-memory-deep.txt"

echo "🔬 Deep Memory Profiling"
echo "   Package: $PACKAGE"
echo "   Test file: $TEST_FILE"
echo "   Test function: $TEST_FUNCTION"
echo ""

if ! command -v jq &> /dev/null; then
  echo "❌ jq not found. Install with: brew install jq"
  exit 1
fi

echo "🔨 Building test..."
RUSTFLAGS="-C force-frame-pointers=yes" cargo test --profile profiling \
  -p "$PACKAGE" --test "$TEST_FILE" --no-run 2>&1 | tail -3

TEST_BINARY=$(cargo test --profile profiling -p "$PACKAGE" --test "$TEST_FILE" \
  --no-run --message-format=json 2>/dev/null | \
  jq -r 'select(.executable != null) | .executable' | head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary"
  exit 1
fi

echo "   Binary: $TEST_BINARY"
echo ""

TEST_ARGS="$TEST_FUNCTION --nocapture"
if [ "$EXTRA_FLAGS" = "--ignored" ]; then
  TEST_ARGS="$TEST_FUNCTION --ignored --nocapture"
fi

TEMP_OUTPUT="/tmp/profile-deep-output-$$.txt"

echo "🚀 Starting test with memory monitoring..."
echo ""

MallocStackLogging=lite "$TEST_BINARY" $TEST_ARGS > "$TEMP_OUTPUT" 2>&1 &
TEST_PID=$!

echo "   Process ID: $TEST_PID"

cat > "$OUTPUT_FILE" << EOF
================================================================================
DEEP MEMORY PROFILING REPORT
================================================================================
Package: $PACKAGE
Test file: $TEST_FILE
Test function: $TEST_FUNCTION
Date: $(date)
================================================================================

=== MEMORY GROWTH TIMELINE ===
Snapshots taken every 0.5 seconds during test execution.

EOF

echo "📊 Capturing memory snapshots over time..."
echo ""

snapshot_num=0
while kill -0 $TEST_PID 2>/dev/null; do
  snapshot_num=$((snapshot_num + 1))
  
  MEM_KB=$(ps -o rss= -p $TEST_PID 2>/dev/null || echo "0")
  MEM_GB=$(awk "BEGIN {printf \"%.2f\", $MEM_KB/1024/1024}")
  TIMESTAMP=$(awk "BEGIN {printf \"%.1f\", $snapshot_num * 0.5}")
  
  echo "   ${TIMESTAMP}s: ${MEM_GB}GB"
  echo "Snapshot $snapshot_num at ${TIMESTAMP}s: ${MEM_GB}GB" >> "$OUTPUT_FILE"
  
  MEM_MB=$(awk "BEGIN {printf \"%.0f\", $MEM_KB/1024}")
  if [ "$MEM_MB" -gt 100 ] 2>/dev/null; then
    HEAP_FILE="/tmp/heap-snapshot-$snapshot_num-$$.txt"
    heap $TEST_PID -s -guessNonObjects > "$HEAP_FILE" 2>/dev/null || true
    
    if [ -s "$HEAP_FILE" ]; then
      cat >> "$OUTPUT_FILE" << EOF

--- Heap snapshot at ${TIMESTAMP}s (${MEM_GB}GB) ---
EOF
      sed -n '/COUNT.*BYTES/,/^$/p' "$HEAP_FILE" | head -40 >> "$OUTPUT_FILE"
      echo "   📸 Heap snapshot captured at ${TIMESTAMP}s"
    fi
    rm -f "$HEAP_FILE"
  fi
  
  sleep 0.5
done

echo ""
echo "Test completed after $snapshot_num snapshots"

cat >> "$OUTPUT_FILE" << EOF

=== TEST OUTPUT ===
EOF
cat "$TEMP_OUTPUT" >> "$OUTPUT_FILE" 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 DEEP MEMORY PROFILE COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 TEST OUTPUT:"
cat "$TEMP_OUTPUT" 2>/dev/null
echo ""

if grep -q "Heap snapshot at" "$OUTPUT_FILE" 2>/dev/null; then
  echo "🧠 PEAK MEMORY ALLOCATIONS (from latest heap snapshot):"
  echo ""
  LAST_SNAPSHOT_LINE=$(grep -n "Heap snapshot at" "$OUTPUT_FILE" | tail -1 | cut -d: -f1)
  if [ -n "$LAST_SNAPSHOT_LINE" ]; then
    tail -n +"$LAST_SNAPSHOT_LINE" "$OUTPUT_FILE" | grep -E '^\s+[0-9]+\s+[0-9]+' | head -10 | \
      while read -r line; do
        count=$(echo "$line" | awk '{print $1}')
        bytes=$(echo "$line" | awk '{print $2}')
        rest=$(echo "$line" | awk '{$1=$2=$3=""; print $0}' | sed 's/^[[:space:]]*//')
        
        if [ -n "$bytes" ] && [ "$bytes" -gt 100000 ] 2>/dev/null; then
          if [ "$bytes" -gt 1000000000 ]; then
            size=$(awk "BEGIN {printf \"%.1fGB\", $bytes/1000000000}")
          elif [ "$bytes" -gt 1000000 ]; then
            size=$(awk "BEGIN {printf \"%.1fMB\", $bytes/1000000}")
          else
            size=$(awk "BEGIN {printf \"%.0fKB\", $bytes/1000}")
          fi
          func=$(echo "$rest" | sed 's/[[:space:]]*C++.*$//' | sed 's/[[:space:]]*C[[:space:]].*$//' | sed 's/::h[a-f0-9]*$//' | \
                 sed 's/\$LT\$/</g' | sed 's/\$GT\$/>/g' | sed 's/\$u20\$/ /g' | \
                 sed 's/\$C\$/, /g' | sed 's/malloc in //g' | sed 's/calloc in //g' | sed 's/realloc in //g')
          printf "   %6s x %-10s %s\n" "$count" "$size" "$func"
        fi
      done
  fi
  echo ""
else
  echo "⚠️  No heap snapshots captured (memory stayed below 100MB threshold)"
  echo "   The test may have run too quickly or uses minimal memory"
  echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Full report saved to: $OUTPUT_FILE"
echo "💡 View with: cat profile-memory-deep.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rm -f "$TEMP_OUTPUT"
