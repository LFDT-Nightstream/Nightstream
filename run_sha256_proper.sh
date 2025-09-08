#!/bin/bash

# SHA-256 Proper Benchmark Runner
# Runs the arkworks-based SHA-256 preimage proof benchmark

set -e

# Default message if none provided
MESSAGE="${1:-hello from neo}"

echo "🔒 SHA-256 Proper Preimage Proof Benchmark"
echo "Message: \"$MESSAGE\""
echo ""

# Change to benchmarks directory
cd benchmarks

# Run the arkworks-enabled SHA-256 benchmark
echo "🔧 Running with arkworks features enabled..."
RUST_BACKTRACE=1 cargo run --release --features arkworks --bin sha256_ark_preimage_benchmark -- "$MESSAGE"
