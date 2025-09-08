# Neo Benchmarks

This project contains benchmarks for the Neo SNARK system. It is located inside the halo3 project directory but is **not part of the workspace** to allow independent benchmarking.

## Available Benchmarks

### 1. Fibonacci Benchmark (`fib_benchmark`)
- Tests Fibonacci sequence CCS construction and proving
- Equivalent to SP1 loop implementation
- Benchmarks different sequence lengths

### 2. Criterion Benchmark Suite
- Comprehensive benchmarking using Criterion.rs
- Fibonacci prove/verify performance analysis
- CCS construction benchmarks
- Statistical analysis with HTML reports

## Usage

### Running Binary Benchmarks

```bash
# Fibonacci benchmark
cargo run --release --bin fib_benchmark

# Custom Fibonacci sizes
N="100,1000,5000" cargo run --release --bin fib_benchmark
```

### Running Criterion Benchmarks

```bash
# Run all criterion benchmarks
cargo bench

# Run specific benchmark group
cargo bench fibonacci_prove
cargo bench sha256_prove
cargo bench ccs_construction

# Generate HTML reports
cargo bench --features html_reports
```

## Output

### CSV Format (Binary Benchmarks)
The binary benchmarks output results in CSV format suitable for analysis:

```
n,prover time (ms),proof size (bytes)
100,245,1024
1000,1250,1024
10000,12500,1024
```

### Criterion Reports
Criterion benchmarks generate detailed reports in `target/criterion/` including:
- Performance statistics
- Regression analysis
- HTML reports (with `html_reports` feature)

## Project Structure

```
benchmarks/
├── Cargo.toml          # Independent cargo project
├── README.md           # This file
├── src/
│   ├── main.rs         # Main entry point with usage info
│   └── bin/
│       ├── fib_benchmark.rs      # Fibonacci benchmark
│       └── sha256_benchmark.rs   # SHA-256 benchmark
├── benches/
│   └── neo_benchmarks.rs         # Criterion benchmark suite
└── examples/           # Future benchmark examples
```

## Dependencies

This project uses local path dependencies to reference Neo crates:

- `neo` - Main Neo facade crate
- `neo-ccs` - CCS structures and utilities  
- `neo-math` - Mathematical primitives

Additional dependencies:
- `criterion` - Benchmarking framework
- `sha2` - SHA-256 implementation for binding tests
- `clap` - Command line argument parsing

## Notes

- **Release Mode**: Always use `--release` for accurate benchmarks  
- **Power-of-Two Padding**: All benchmarks include automatic padding for proper Boolean hypercube evaluation
- **Memory Usage**: Large benchmark sizes may require significant memory
- **CSV Output**: Fibonacci benchmark produces CSV results for analysis
- **Statistical Rigor**: Criterion benchmarks provide confidence intervals and outlier detection

## Customization

You can customize benchmarks by:

1. **Environment Variables**: Use `N="sizes"` for custom Fibonacci lengths
2. **Command Line Arguments**: Pass custom messages to SHA-256 benchmark
3. **Code Modification**: Edit benchmark parameters in the source files
4. **New Benchmarks**: Add new binary benchmarks in `src/bin/`

## Performance Tips

- Use `--release` flag for all benchmarks
- Ensure adequate memory for large benchmark sizes
- Run benchmarks on a quiet system for accurate results
- Use Criterion for statistical analysis of performance variations
