//! Criterion-based performance benchmarks for Spartan2 integration
//! 
//! These benchmarks provide detailed performance analysis comparing
//! NARK mode vs SNARK mode across different problem sizes.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neo_arithmetize::fibonacci_ccs;
use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_orchestrator::{prove, verify};
use neo_modint::ModInt;
use neo_ring::RingElement;
use neo_decomp::decomp_b;
use p3_field::PrimeCharacteristicRing;
use std::time::Duration;

/// Helper function to create Fibonacci test case
fn create_fibonacci_benchmark_case(length: usize) -> (neo_ccs::CcsStructure, CcsInstance, CcsWitness) {
    // Create CCS structure
    let ccs = fibonacci_ccs(length);
    
    // Generate Fibonacci witness
    let mut z: Vec<ExtF> = vec![ExtF::ZERO; length];
    z[0] = embed_base_to_ext(F::ZERO);
    z[1] = embed_base_to_ext(F::ONE);
    for i in 2..length {
        z[i] = z[i - 1] + z[i - 2];
    }
    let witness = CcsWitness { z: z.clone() };
    
    // Create commitment
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
    let z_base: Vec<F> = z.iter().map(|e| e.to_array()[0]).collect();
    let decomp_mat = decomp_b(&z_base, committer.params().b, committer.params().d);
    let z_packed: Vec<RingElement<ModInt>> = AjtaiCommitter::pack_decomp(&decomp_mat, &committer.params());
    let (commitment, _, _, _) = committer.commit(&z_packed, &mut vec![]).unwrap();
    
    let instance = CcsInstance {
        commitment,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    
    (ccs, instance, witness)
}

/// Benchmark proof generation across different Fibonacci lengths
fn bench_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");
    
    // Test different Fibonacci lengths
    let lengths = vec![4, 6, 8, 10];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        // Verify the test case is valid
        assert!(check_satisfiability(&ccs, &instance, &witness), 
               "Benchmark case should be satisfiable");
        
        group.throughput(Throughput::Elements(length as u64));
        
        #[cfg(feature = "snark_mode")]
        group.bench_with_input(
            BenchmarkId::new("snark_mode", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = prove(&ccs, &instance, &witness);
                    assert!(result.is_ok(), "SNARK proof should succeed");
                    result.unwrap()
                });
            },
        );
        
        #[cfg(not(feature = "snark_mode"))]
        group.bench_with_input(
            BenchmarkId::new("nark_mode", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = prove(&ccs, &instance, &witness);
                    assert!(result.is_ok(), "NARK proof should succeed");
                    result.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark proof verification across different proof sizes
fn bench_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_verification");
    
    let lengths = vec![4, 6, 8, 10];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        // Pre-generate the proof
        let (proof, _) = prove(&ccs, &instance, &witness).unwrap();
        
        group.throughput(Throughput::Elements(length as u64));
        
        #[cfg(feature = "snark_mode")]
        group.bench_with_input(
            BenchmarkId::new("snark_verify", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = verify(&ccs, &proof);
                    assert!(result, "SNARK verification should succeed");
                    result
                });
            },
        );
        
        #[cfg(not(feature = "snark_mode"))]
        group.bench_with_input(
            BenchmarkId::new("nark_verify", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = verify(&ccs, &proof);
                    assert!(result, "NARK verification should succeed");
                    result
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark end-to-end proof generation and verification
fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.measurement_time(Duration::from_secs(10));
    
    let lengths = vec![4, 6, 8];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        group.throughput(Throughput::Elements(length as u64));
        
        #[cfg(feature = "snark_mode")]
        group.bench_with_input(
            BenchmarkId::new("snark_end_to_end", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let (proof, _) = prove(&ccs, &instance, &witness).unwrap();
                    let verified = verify(&ccs, &proof);
                    assert!(verified, "End-to-end SNARK should succeed");
                    verified
                });
            },
        );
        
        #[cfg(not(feature = "snark_mode"))]
        group.bench_with_input(
            BenchmarkId::new("nark_end_to_end", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let (proof, _) = prove(&ccs, &instance, &witness).unwrap();
                    let verified = verify(&ccs, &proof);
                    assert!(verified, "End-to-end NARK should succeed");
                    verified
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark commitment operations
fn bench_commitment_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("commitment_operations");
    
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
    let sizes = vec![4, 8, 16, 32];
    
    for size in sizes {
        // Create test data
        let test_data: Vec<RingElement<ModInt>> = (0..size)
            .map(|i| RingElement::from_scalar(
                ModInt::from_u64(i as u64 + 1),
                SECURE_PARAMS.n
            ))
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("ajtai_commit", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = committer.commit(&test_data, &mut vec![]);
                    assert!(result.is_ok(), "Commitment should succeed");
                    result.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark CCS satisfiability checking
fn bench_satisfiability_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("satisfiability_check");
    
    let lengths = vec![4, 6, 8, 10, 12];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        group.throughput(Throughput::Elements(length as u64));
        
        group.bench_with_input(
            BenchmarkId::new("ccs_satisfiability", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = check_satisfiability(&ccs, &instance, &witness);
                    assert!(result, "Satisfiability check should pass");
                    result
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark proof size analysis
fn bench_proof_size_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_size_analysis");
    
    let lengths = vec![4, 6, 8, 10];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        group.throughput(Throughput::Elements(length as u64));
        
        #[cfg(feature = "snark_mode")]
        group.bench_with_input(
            BenchmarkId::new("snark_proof_size", length),
            &length,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    let mut total_size = 0;
                    
                    for _ in 0..iters {
                        let (proof, metrics) = prove(&ccs, &instance, &witness).unwrap();
                        total_size += metrics.proof_bytes;
                        
                        // Verify to ensure proof is valid
                        assert!(verify(&ccs, &proof), "Proof should verify");
                    }
                    
                    // Print size information
                    if iters > 0 {
                        let avg_size = total_size / iters as usize;
                        println!("SNARK proof average size for length {}: {} bytes", length, avg_size);
                    }
                    
                    start.elapsed()
                });
            },
        );
        
        #[cfg(not(feature = "snark_mode"))]
        group.bench_with_input(
            BenchmarkId::new("nark_proof_size", length),
            &length,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    let mut total_size = 0;
                    
                    for _ in 0..iters {
                        let (proof, metrics) = prove(&ccs, &instance, &witness).unwrap();
                        total_size += metrics.proof_bytes;
                        
                        // Verify to ensure proof is valid
                        assert!(verify(&ccs, &proof), "Proof should verify");
                    }
                    
                    // Print size information
                    if iters > 0 {
                        let avg_size = total_size / iters as usize;
                        println!("NARK proof average size for length {}: {} bytes", length, avg_size);
                    }
                    
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(all(feature = "snark_mode", test))]
/// Benchmark CCS to R1CS conversion (SNARK mode only)
fn bench_ccs_to_r1cs_conversion(c: &mut Criterion) {
    use neo_ccs::convert_ccs_to_r1cs_full;
    
    let mut group = c.benchmark_group("ccs_to_r1cs_conversion");
    
    let lengths = vec![4, 6, 8, 10];
    
    for length in lengths {
        let (ccs, instance, witness) = create_fibonacci_benchmark_case(length);
        
        group.throughput(Throughput::Elements(length as u64));
        
        group.bench_with_input(
            BenchmarkId::new("conversion", length),
            &length,
            |b, _| {
                b.iter(|| {
                    let result = convert_ccs_to_r1cs_full(&ccs, &instance, &witness);
                    assert!(result.is_ok(), "CCS to R1CS conversion should succeed");
                    result.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

// Configure benchmark groups
criterion_group!(
    benches,
    bench_proof_generation,
    bench_proof_verification,
    bench_end_to_end,
    bench_commitment_operations,
    bench_satisfiability_check,
    bench_proof_size_analysis,
);

#[cfg(all(feature = "snark_mode", test))]
criterion_group!(
    snark_benches,
    bench_ccs_to_r1cs_conversion,
);

#[cfg(feature = "snark_mode")]
criterion_main!(benches, snark_benches);

#[cfg(not(feature = "snark_mode"))]
criterion_main!(benches);
