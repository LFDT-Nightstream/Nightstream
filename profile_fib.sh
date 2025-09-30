#!/bin/bash
# Profile the fib_folding_nivc example with samply and open in Chrome

set -e

# Number of steps (default 100)
STEPS=${1:-100}

echo "🔥 Profiling fib_folding_nivc with $STEPS steps..."
echo "   Using profiling profile (optimized + debug symbols)"
echo ""

# Set browser to Chrome for samply
export BROWSER="open -a 'Google Chrome'"

# Run samply with NEO_DETERMINISTIC flag using profiling profile
# The profiling profile has thin LTO and more codegen units for better symbols
NEO_DETERMINISTIC=1 samply record cargo run --profile profiling --example fib_folding_nivc -p neo -- "$STEPS"

echo ""
echo "✅ Done! The profiler should have opened in Chrome."
echo "   If not, it should be at http://127.0.0.1:3000"
echo ""
echo "📝 Note: This uses the 'profiling' profile (~5% slower than release, but with full debug symbols)"