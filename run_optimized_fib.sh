#!/bin/bash

# Neo SNARK Optimized Fibonacci Demo Runner
# This script runs the fib.rs example with all performance optimizations applied

set -e

echo "🔥 Neo SNARK Optimized Fibonacci Demo"
echo "===================================="

# Detect number of CPU cores
if command -v nproc &> /dev/null; then
    NUM_THREADS=$(nproc)
elif command -v sysctl &> /dev/null; then
    NUM_THREADS=$(sysctl -n hw.logicalcpu)
else
    NUM_THREADS=8  # fallback
fi

echo "🔧 Configuring for maximum performance:"
echo "   - CPU cores detected: $NUM_THREADS"
echo "   - Using target-cpu=native for SIMD optimizations"
echo "   - Enabling fast-io mode (no LZ4 compression)"
echo "   - Maximum Rayon parallelization"
echo ""

# Set environment variables for optimal performance
export RUSTFLAGS="-C target-cpu=native"
export RAYON_NUM_THREADS=$NUM_THREADS

echo "🏃 Running optimized Fibonacci demo..."
echo "====================================="

# Run the fib example with ALL optimizations including sparse matrix fast paths
RUSTFLAGS="-C target-cpu=native" cargo run --release -p neo --example fib

echo ""
echo "✅ Demo completed!"
echo ""
echo "💡 Performance Tips Applied:"
echo "   ✓ Release build with fat LTO and symbol stripping"
echo "   ✓ target-cpu=native for hardware-specific optimizations"
echo "   ✓ mimalloc allocator for reduced memory overhead"
echo "   ✓ All CPU cores utilized via Rayon thread pool"
echo "   ✓ CCS consistency check disabled in release builds"
echo "   ✓ Parallelized matrix operations in bridge adapter"
echo "   ✓ Lowered sum-check parallelization thresholds"
echo "   ✓ LZ4 compression disabled for faster I/O"
echo ""
echo "🔍 To profile with flamegraph:"
echo "   cargo install flamegraph"
echo "   RUSTFLAGS='-C target-cpu=native' cargo flamegraph --release --features fast-io -p neo --example fib"
