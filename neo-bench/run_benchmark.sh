#!/bin/bash

# Nova vs Neo Fibonacci Benchmark Runner
# This script runs a comprehensive comparison between Nova and Neo implementations

echo "🚀 Starting Nova vs Neo Fibonacci Benchmark..."
echo "📊 Testing from 8 steps (2^3) to 64 steps (2^6)"
echo ""

cargo run --release -p neo-bench --features with-nova -- --nova-fibo --spartan2-fibo
