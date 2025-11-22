#!/usr/bin/env bash
set -euo pipefail

# Profile script for neo-fold starstream test
# Usage: ./scripts/profile_starstream_test.sh [flamegraph|instruments|chrome|samply]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TEST_NAME="test_starstream_tx_valid_optimized"
TEST_PATH="crates/neo-fold/starstream-tests"
PROFILE_MODE="${1:-flamegraph}"

echo "🔍 Profiling test: $TEST_NAME"
echo "📊 Mode: $PROFILE_MODE"
echo ""

case "$PROFILE_MODE" in
  flamegraph)
    echo "📈 Generating flamegraph..."
    
    # Check if cargo-flamegraph is installed
    if ! command -v cargo-flamegraph &> /dev/null; then
      echo "❌ cargo-flamegraph not found. Installing..."
      cargo install flamegraph
    fi
    
    # Build the test binary with profiling profile
    echo "🔨 Building test with profiling profile (optimized + debug symbols)..."
    RUSTFLAGS="-C force-frame-pointers=yes" cargo test --profile profiling \
      --package neo-fold --test "$TEST_NAME" --no-run
    
    # Find the test binary
    TEST_BINARY=$(cargo test --profile profiling --package neo-fold --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary"
      exit 1
    fi
    
    echo "🔥 Running flamegraph profiler..."
    echo "Test binary: $TEST_BINARY"
    
    # Run with sudo on macOS (required for DTrace)
    sudo flamegraph \
      --output="$PROJECT_ROOT/flamegraph-starstream.svg" \
      --root \
      -- "$TEST_BINARY" --nocapture
    
    echo "✅ Flamegraph saved to: flamegraph-starstream.svg"
    echo "💡 Open it in your browser!"
    open flamegraph-starstream.svg 2>/dev/null || true
    ;;
    
  instruments)
    echo "🎼 Using macOS Instruments (Time Profiler)..."
    
    # Build test with profiling profile
    echo "🔨 Building test with profiling profile (optimized + debug symbols)..."
    cargo test --profile profiling --package neo-fold --test "$TEST_NAME" --no-run
    
    # Find the test binary
    TEST_BINARY=$(cargo test --profile profiling --package neo-fold --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary"
      exit 1
    fi
    
    TRACE_FILE="$PROJECT_ROOT/starstream-profile.trace"
    
    echo "📊 Recording profile..."
    xcrun xctrace record \
      --template 'Time Profiler' \
      --output "$TRACE_FILE" \
      --launch -- "$TEST_BINARY" --nocapture
    
    echo "✅ Trace saved to: $TRACE_FILE"
    echo "💡 Opening in Instruments..."
    open "$TRACE_FILE"
    ;;
    
  chrome)
    echo "🌐 Generating Chrome tracing format..."
    
    # Check if tracing-chrome is in Cargo.toml
    if ! grep -q "tracing-chrome" "$PROJECT_ROOT/crates/neo-fold/Cargo.toml"; then
      echo "⚠️  tracing-chrome not found in Cargo.toml"
      echo "You need to add the following dependencies to crates/neo-fold/Cargo.toml:"
      echo ""
      echo "[dev-dependencies]"
      echo "tracing = \"0.1\""
      echo "tracing-subscriber = { version = \"0.3\", features = [\"env-filter\"] }"
      echo "tracing-chrome = \"0.7\""
      echo ""
      echo "And instrument your test code with #[tracing::instrument] or manual spans."
      exit 1
    fi
    
    echo "🔨 Running test with chrome tracing..."
    cargo test --release --package neo-fold --test "$TEST_NAME" -- --nocapture
    
    # Find the trace file
    TRACE_FILE=$(find "$PROJECT_ROOT" -name "trace-*.json" -type f -mmin -1 | head -1)
    
    if [ -n "$TRACE_FILE" ]; then
      echo "✅ Trace saved to: $TRACE_FILE"
      echo "💡 Open chrome://tracing in Chrome and load the trace file"
      echo "   or run: open -a 'Google Chrome' '$TRACE_FILE'"
    else
      echo "⚠️  No trace file found. Make sure tracing is properly configured."
    fi
    ;;
    
  samply)
    echo "🎯 Using samply profiler..."
    
    # Check if samply is installed
    if ! command -v samply &> /dev/null; then
      echo "❌ samply not found. Installing..."
      cargo install samply
    fi
    
    # Build test with profiling profile (has debug symbols)
    echo "🔨 Building test with profiling profile (optimized + debug symbols)..."
    cargo test --profile profiling --package neo-fold \
      --test "$TEST_NAME" --no-run
    
    # Find the test binary
    TEST_BINARY=$(cargo test --profile profiling \
      --package neo-fold --test "$TEST_NAME" \
      --no-run --message-format=json 2>/dev/null | \
      jq -r 'select(.executable != null) | .executable' | head -1)
    
    if [ -z "$TEST_BINARY" ]; then
      echo "❌ Could not find test binary"
      exit 1
    fi
    
    PROFILE_FILE="$PROJECT_ROOT/samply-starstream-profile.json"
    
    echo "🎯 Running samply profiler..."
    echo "Binary: $TEST_BINARY"
    
    # Record to file
    samply record --save-only -o "$PROFILE_FILE" "$TEST_BINARY" --nocapture
    
    echo "✅ Profile saved to: $PROFILE_FILE"
    echo "🌐 Opening in browser..."
    
    # Load the profile (opens in default browser, but works in any browser)
    samply load "$PROFILE_FILE"
    
    echo ""
    echo "💡 If the profile opened in Safari, you can:"
    echo "   1. Copy the localhost URL and paste it in Chrome, or"
    echo "   2. Just use it in Safari - it works the same!"
    ;;
    
  criterion)
    echo "📊 Using Criterion benchmarks..."
    echo ""
    echo "To use Criterion, you need to:"
    echo "1. Add criterion to Cargo.toml dev-dependencies"
    echo "2. Create a benches/ directory with benchmark files"
    echo "3. Run with: cargo bench --package neo-fold"
    echo ""
    echo "Would you like a criterion benchmark template? (y/n)"
    ;;
    
  *)
    echo "❌ Unknown profiling mode: $PROFILE_MODE"
    echo ""
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Available modes:"
    echo "  flamegraph   - Generate interactive SVG flamegraph (default)"
    echo "  instruments  - Use macOS Instruments Time Profiler (GUI)"
    echo "  chrome       - Generate Chrome tracing JSON (requires tracing setup)"
    echo "  samply       - Use samply profiler (opens Firefox Profiler)"
    echo ""
    echo "Examples:"
    echo "  $0 flamegraph"
    echo "  $0 instruments"
    echo "  $0 samply"
    exit 1
    ;;
esac

echo ""
echo "🎉 Profiling complete!"

