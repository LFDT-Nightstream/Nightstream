#!/usr/bin/env bash
set -euo pipefail

# Profile script for Poseidon2 single-step tests using samply
# Usage: ./scripts/profile_poseidon2_test.sh [batch_size]
#   batch_size: 1, 10, 20, 30, or 40 (default: 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BATCH_SIZE="${1:-10}"
TEST_FILE="test_poseidon2_single_step"
TEST_FUNCTION="test_poseidon2_ic_batch_size_${BATCH_SIZE}"

echo "🎯 Profiling Poseidon2 test with samply"
echo "   Test file: $TEST_FILE"
echo "   Test function: $TEST_FUNCTION"
echo "   Batch size: $BATCH_SIZE"
echo ""

# Check if samply is installed
if ! command -v samply &> /dev/null; then
  echo "❌ samply not found. Installing..."
  cargo install samply
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo "❌ jq not found. Please install it:"
  echo "   brew install jq"
  exit 1
fi

# Build test with profiling profile (has debug symbols)
echo "🔨 Building test with profiling profile (optimized + debug symbols)..."
cargo test --profile profiling --package neo-fold \
  --test "$TEST_FILE" --no-run

# Find the test binary
TEST_BINARY=$(cargo test --profile profiling \
  --package neo-fold --test "$TEST_FILE" \
  --no-run --message-format=json 2>/dev/null | \
  jq -r 'select(.executable != null) | .executable' | head -1)

if [ -z "$TEST_BINARY" ]; then
  echo "❌ Could not find test binary"
  exit 1
fi

PROFILE_FILE="$PROJECT_ROOT/samply-poseidon2-batch-${BATCH_SIZE}.json"

echo "🎯 Running samply profiler..."
echo "   Binary: $TEST_BINARY"
echo "   Output: $PROFILE_FILE"
echo ""

# Record to file, running only the specific test function
samply record --save-only -o "$PROFILE_FILE" \
  "$TEST_BINARY" "$TEST_FUNCTION" --exact --nocapture

echo ""
echo "✅ Profile saved to: $PROFILE_FILE"
echo "🌐 Opening in browser..."

# Load the profile (opens in default browser)
samply load "$PROFILE_FILE"

echo ""
echo "💡 Tips:"
echo "   - The profile opens in Firefox Profiler UI (works in any browser)"
echo "   - Use the flame chart to find hot spots"
echo "   - Filter by searching for function names"
echo ""
echo "🎉 Profiling complete!"

