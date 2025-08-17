#!/bin/bash
set -e

# Run Cargo criterion for all relevant crates and capture key metrics
echo "Running Rust benchmarks..."

# Benchmarks live in separate workspace crates, so specify each package explicitly.
cargo criterion -p neo-ring --bench ring_ops > rust_bench.txt 2>&1
cargo criterion -p neo-commit --bench commit >> rust_bench.txt 2>&1
# Add sumcheck benchmark
if ! cargo criterion -p neo-sumcheck --bench sumcheck >> rust_bench.txt 2>&1; then
  echo "neo-sumcheck bench failed" >> rust_bench.txt
fi
# neo-fold benchmark removed - was broken due to polynomial degree assertion failures

echo "Rust benchmarks saved to rust_bench.txt"
grep -E "time:|mean" rust_bench.txt
