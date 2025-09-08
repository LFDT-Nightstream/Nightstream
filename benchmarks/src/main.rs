//! Neo Benchmarks
//!
//! This project contains benchmarks for the Neo SNARK system.
//! 
//! Available benchmarks:
//! - fib_benchmark: Comprehensive Neo performance testing with Fibonacci sequence CCS
//!   * Tests CCS construction, proving, verification
//!   * Multiple problem sizes with statistical analysis  
//!   * Power-of-two padding and constraint satisfaction checking
//!
//! - sha256_ark_preimage_benchmark: SHA-256 preimage proof using arkworks (experimental)
//!   * Requires arkworks feature: --features arkworks
//!   * Attempts to build proper cryptographic SHA-256 circuit
//!   * Currently has field compatibility issues (see README)
//!
//! Usage:
//!   cargo run --release --bin fib_benchmark
//!   cargo run --release --features arkworks --bin sha256_ark_preimage_benchmark -- "hello world"
//!   N="100,1000,10000" cargo run --bin fib_benchmark  # Custom sizes

fn main() {
    println!("Neo Benchmarks");
    println!("==============");
    println!();
    println!("Available benchmarks:");
    println!("  cargo run --release --bin fib_benchmark");
    println!("  cargo run --release --features arkworks --bin sha256_ark_preimage_benchmark -- \"your message\"");
    println!();
    println!("Criterion benchmarks:");
    println!("  cargo bench");
    println!();
    println!("ℹ️  Neo is optimized for small prime fields like Goldilocks");
    println!("   The Fibonacci benchmark provides comprehensive performance analysis:");
    println!("   • CCS construction and R1CS→CCS conversion");  
    println!("   • Proving performance across different witness sizes");
    println!("   • Verification timing and proof size measurement");
    println!("   • Power-of-two constraint system padding");
    println!();
    println!("⚠️  The SHA-256 benchmark is experimental and currently has compilation issues");
    println!("   due to arkworks 0.3 API compatibility problems with Goldilocks field.");
    println!();
    println!("📊 For detailed statistical analysis, run: cargo bench");
}