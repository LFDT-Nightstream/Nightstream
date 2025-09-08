//! Criterion-based benchmarks for Neo SNARK system
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use neo::{prove, verify, NeoParams, ProveInput, F};
use neo_ccs::{CcsStructure, r1cs_to_ccs, Mat, check_ccs_rowwise_zero};
use std::hint::black_box;
use std::time::Duration;
use benchmarks::util::{pad_ccs_rows_to_pow2, triplets_to_dense};
use p3_field::PrimeCharacteristicRing;

/// Fibonacci CCS builder with proper padding
fn fibonacci_ccs(n: usize) -> (CcsStructure<F>, Vec<F>) {
    let rows = n + 1;
    let cols = n + 2;

    let mut a_trips: Vec<(usize, usize, F)> = Vec::with_capacity(3 * (n - 1) + 2);
    let mut b_trips: Vec<(usize, usize, F)> = Vec::with_capacity(rows);
    let c_trips: Vec<(usize, usize, F)> = Vec::new();

    // Seed constraints
    a_trips.push((0, 1, F::ONE));
    b_trips.push((0, 0, F::ONE));
    a_trips.push((1, 2, F::ONE));
    a_trips.push((1, 0, -F::ONE));
    b_trips.push((1, 0, F::ONE));

    // Recurrence rows
    for i in 0..(n - 1) {
        let r = 2 + i;
        a_trips.push((r, (i + 3),  F::ONE));
        a_trips.push((r, (i + 2), -F::ONE));
        a_trips.push((r, (i + 1), -F::ONE));
        b_trips.push((r, 0, F::ONE));
    }

    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));

    // Convert to CCS and pad to power-of-2 for proper evaluation
    let ccs = pad_ccs_rows_to_pow2(r1cs_to_ccs(a, b, c));

    // Generate witness
    let mut witness: Vec<F> = Vec::with_capacity(n + 2);
    witness.push(F::ONE);  // constant 1
    witness.push(F::ZERO); // z0
    witness.push(F::ONE);  // z1
    for k in 2..=n {
        let next = witness[k] + witness[k - 1];
        witness.push(next);
    }

    (ccs, witness)
}

// SHA-256 binding benchmarks removed - keeping only proper in-circuit SHA-256

fn bench_fibonacci_prove(c: &mut Criterion) {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    let mut group = c.benchmark_group("fibonacci_prove");
    // Optimize settings for heavy operations to avoid 15+ minute benches
    group.sample_size(10).measurement_time(Duration::from_secs(12));
    
    for n in [10, 50, 100, 500].iter() {
        let (ccs, witness) = fibonacci_ccs(*n);
        
        // Preflight check - fail fast with clear error if CCS is not satisfied
        check_ccs_rowwise_zero(&ccs, &[], &witness)
            .expect("Fibonacci CCS constraint satisfaction check failed");
        
        group.bench_with_input(BenchmarkId::new("prove", n), n, |b, _n| {
            b.iter(|| {
                let proof = prove(ProveInput {
                    params: &params,
                    ccs: &ccs,
                    public_input: &[],
                    witness: &witness,
                }).unwrap();
                black_box(proof)
            })
        });
    }
    group.finish();
}

fn bench_fibonacci_verify(c: &mut Criterion) {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    let mut group = c.benchmark_group("fibonacci_verify");
    group.sample_size(50); // Verification is faster, can afford more samples
    
    for n in [10, 50, 100, 500].iter() {
        let (ccs, witness) = fibonacci_ccs(*n);
        
        // Preflight check
        check_ccs_rowwise_zero(&ccs, &[], &witness)
            .expect("Fibonacci CCS constraint satisfaction check failed");
            
        let proof = prove(ProveInput {
            params: &params,
            ccs: &ccs,
            public_input: &[],
            witness: &witness,
        }).unwrap();
        
        group.bench_with_input(BenchmarkId::new("verify", n), n, |b, _n| {
            b.iter(|| {
                let result = verify(&ccs, &[], &proof).unwrap();
                black_box(result)
            })
        });
    }
    group.finish();
}

// SHA-256 benchmarks removed - only proper in-circuit version available via arkworks feature

fn bench_ccs_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ccs_construction");
    // CCS construction is fast, can use more samples
    group.sample_size(100);
    
    for n in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("fibonacci", n), n, |b, &n| {
            b.iter(|| {
                let (ccs, witness) = fibonacci_ccs(n);
                black_box((ccs, witness))
            })
        });
    }
    
    // SHA-256 CCS construction benchmarks removed - only proper in-circuit version available
    
    group.finish();
}

criterion_group!(
    benches,
    bench_fibonacci_prove,
    bench_fibonacci_verify,
    bench_ccs_construction
);
criterion_main!(benches);
