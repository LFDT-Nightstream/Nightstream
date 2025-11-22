#!/usr/bin/env bash
set -euo pipefail

# General-purpose Rust test profiler
# Usage: ./scripts/profile_any_test.sh <package> <test_name> [mode]
#   e.g., ./scripts/profile_any_test.sh neo-fold test_starstream_tx_valid_optimized flamegraph

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <package> <test_name> [mode]"
  echo ""
  echo "Examples:"
  echo "  $0 neo-fold test_starstream_tx_valid_optimized flamegraph"
  echo "  $0 neo-ajtai basic_commit_tests instruments"
  echo "  $0 neo-ccs ccs_property_tests samply"
  echo ""
  echo "Modes: flamegraph (default), instruments, samply, chrome"
  exit 1
fi

PACKAGE="$1"
TEST_NAME="$2"
PROFILE_MODE="${3:-flamegraph}"

echo "🔍 Profiling test:"
echo "   Package: $PACKAGE"
echo "   Test: $TEST_NAME"
echo "   Mode: $PROFILE_MODE"
echo ""

case "$PROFILE_MODE" in
  flamegraph)
    echo "📈 Generating flamegraph..."
    
    if ! command -v cargo-flamegraph &> /dev/null; then
      echo "❌ cargo-flamegraph not found. Installing..."
      cargo install flamegraph
    fi
    
    echo "🔨 Building test with profiling profile (optimized + debug symbols)..."
    RUSTFLAGS="-C force-frame-pointers=yes" cargo test --profile profiling \
      --package "$PACKAGE" --test "$TEST_NAME" --no-run
    
    TEST_BINARY=$(cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary for $PACKAGE::$TEST_NAME"
      echo "Make sure the package and test name are correct."
      exit 1
    fi
    
    OUTPUT_FILE="$PROJECT_ROOT/flamegraph-${PACKAGE}-${TEST_NAME}.svg"
    
    echo "🔥 Running flamegraph profiler..."
    sudo flamegraph \
      --output="$OUTPUT_FILE" \
      --root \
      -- "$TEST_BINARY" --nocapture
    
    echo "✅ Flamegraph saved to: $OUTPUT_FILE"
    open "$OUTPUT_FILE" 2>/dev/null || true
    ;;
    
  instruments)
    echo "🎼 Using macOS Instruments..."
    
    cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" --no-run
    
    TEST_BINARY=$(cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary"
      exit 1
    fi
    
    TRACE_FILE="$PROJECT_ROOT/profile-${PACKAGE}-${TEST_NAME}.trace"
    
    echo "📊 Recording profile..."
    xcrun xctrace record \
      --template 'Time Profiler' \
      --output "$TRACE_FILE" \
      --launch -- "$TEST_BINARY" --nocapture
    
    echo "✅ Trace saved to: $TRACE_FILE"
    open "$TRACE_FILE"
    ;;
    
  samply)
    echo "🎯 Using samply profiler..."
    
    if ! command -v samply &> /dev/null; then
      echo "❌ samply not found. Installing..."
      cargo install samply
    fi
    
    cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" --no-run
    
    TEST_BINARY=$(cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary"
      exit 1
    fi
    
    PROFILE_FILE="$PROJECT_ROOT/samply-${PACKAGE}-${TEST_NAME}.json"
    
    echo "🎯 Running samply profiler..."
    # Record to file
    samply record --save-only -o "$PROFILE_FILE" "$TEST_BINARY" --nocapture
    
    echo "✅ Profile saved to: $PROFILE_FILE"
    echo "🌐 Opening in browser..."
    
    # Load the profile (opens in default browser)
    samply load "$PROFILE_FILE"
    
    echo ""
    echo "💡 If the profile opened in Safari, you can:"
    echo "   1. Copy the localhost URL and paste it in Chrome, or"
    echo "   2. Just use it in Safari - it works the same!"
    ;;
    
  bench)
    echo "📊 Running as benchmark (time measurement only)..."
    echo ""
    
    for i in {1..5}; do
      echo "Run $i/5..."
      cargo test --profile profiling --package "$PACKAGE" --test "$TEST_NAME" -- --nocapture
    done
    ;;
    
  *)
    echo "❌ Unknown mode: $PROFILE_MODE"
    echo "Available modes: flamegraph, instruments, samply, bench"
    exit 1
    ;;
esac

echo ""
echo "🎉 Profiling complete!"

