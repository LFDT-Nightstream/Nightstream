#!/usr/bin/env bash
# Check which profiling tools are available

echo "🔍 Checking profiling tools availability..."
echo ""

check_tool() {
    local name=$1
    local cmd=$2
    local install_hint=$3
    
    if command -v "$cmd" &> /dev/null; then
        version=$($cmd --version 2>&1 | head -1 || echo "unknown version")
        echo "✅ $name - installed ($version)"
    else
        echo "❌ $name - not installed"
        if [ -n "$install_hint" ]; then
            echo "   Install: $install_hint"
        fi
    fi
}

echo "Core Tools:"
check_tool "Cargo" "cargo" "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
check_tool "jq (JSON parser)" "jq" "brew install jq"
echo ""

echo "Profiling Tools:"
check_tool "cargo-flamegraph" "cargo-flamegraph" "cargo install flamegraph"
check_tool "samply" "samply" "cargo install samply"
check_tool "Instruments (xctrace)" "xctrace" "xcode-select --install"
echo ""

echo "Optional Tools:"
check_tool "cargo-criterion" "cargo-criterion" "cargo install cargo-criterion"
check_tool "hyperfine (benchmarking)" "hyperfine" "brew install hyperfine"
check_tool "perf (Linux)" "perf" "N/A on macOS"
echo ""

# Check if DTrace is available (needed for flamegraph on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS-specific:"
    if sudo -n dtrace -V &> /dev/null; then
        echo "✅ DTrace - available (with sudo)"
    else
        echo "⚠️  DTrace - requires sudo password for flamegraph"
    fi
fi

echo ""
echo "📚 Quick Start:"
echo "   ./scripts/profile_starstream_test.sh flamegraph    # Generate interactive CPU profile"
echo "   ./scripts/profile_starstream_test.sh instruments   # Use macOS Instruments"
echo "   ./scripts/profile_starstream_test.sh samply        # Modern web-based profiler"
echo ""
echo "See scripts/README.md for detailed documentation"

