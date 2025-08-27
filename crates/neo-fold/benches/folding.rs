use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use neo_ccs::{mv_poly, verifier_ccs, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_fold::FoldState;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

fn create_test_structure(size: usize) -> CcsStructure {
    if size <= 1 {
        // Minimal structure for small sizes
        let mat = RowMajorMatrix::<F>::new(vec![F::ONE], 1);
        return CcsStructure::new(vec![mat], mv_poly(|vars: &[ExtF]| vars[0], 1));
    }
    
    // Create a structure with `size` variables and constraints
    // Use simple identity matrices for efficiency
    let matrices = (0..size).map(|i| {
        let mut data = vec![F::ZERO; size * size];
        data[i * size + i] = F::ONE; // Diagonal matrix
        RowMajorMatrix::<F>::new(data, size)
    }).collect();
    
    // Sum of all variables polynomial
    let poly = mv_poly(|vars: &[ExtF]| vars.iter().copied().sum(), size);
    CcsStructure::new(matrices, poly)
}

fn create_test_instances(size: usize) -> ((CcsInstance, CcsWitness), (CcsInstance, CcsWitness)) {
    let instance1 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ZERO,
    };
    
    let witness1 = CcsWitness {
        z: (0..size).map(|i| embed_base_to_ext(F::from_u64((i as u64 + 1) % 100))).collect(),
    };
    
    let instance2 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ZERO,
    };
    
    let witness2 = CcsWitness {
        z: (0..size).map(|i| embed_base_to_ext(F::from_u64((i as u64 + 2) % 100))).collect(),
    };
    
    ((instance1, witness1), (instance2, witness2))
}

fn bench_single_fold(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_fold");
    
    for size in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::new("variables", size), size, |b, &size| {
            let structure = create_test_structure(size);
            let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
            let (inst1, inst2) = create_test_instances(size);
            
            b.iter(|| {
                let mut state = FoldState::new(structure.clone());
                let _proof = state.generate_proof(inst1.clone(), inst2.clone(), &committer);
            });
        });
    }
    group.finish();
}

fn bench_fold_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_verification");
    
    for size in [1, 4, 8, 16].iter() {
        group.bench_with_input(BenchmarkId::new("variables", size), size, |b, &size| {
            let structure = create_test_structure(size);
            let mut state = FoldState::new(structure);
            let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
            let (inst1, inst2) = create_test_instances(size);
            
            // Pre-generate proof for verification benchmark
            let proof = state.generate_proof(inst1, inst2, &committer);
            
            b.iter(|| {
                let verifier_state = FoldState::new(state.structure.clone());
                let _result = verifier_state.verify(&proof.transcript, &committer);
            });
        });
    }
    group.finish();
}

fn bench_recursive_ivc(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_ivc");
    group.sample_size(10); // Fewer samples for longer benchmarks
    
    for depth in [1, 2, 3, 5].iter() {
        group.bench_with_input(BenchmarkId::new("depth", depth), depth, |b, &depth| {
            let structure = create_test_structure(4); // Fixed size for IVC
            let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
            
            b.iter(|| {
                let mut state = FoldState::new(structure.clone());
                let instance = CcsInstance {
                    commitment: vec![],
                    public_input: vec![],
                    u: F::ONE,
                    e: F::ZERO,
                };
                let witness = CcsWitness {
                    z: vec![
                        embed_base_to_ext(F::from_u64(1)),
                        embed_base_to_ext(F::from_u64(2)),
                        embed_base_to_ext(F::from_u64(3)),
                        embed_base_to_ext(F::from_u64(4)),
                    ],
                };
                state.ccs_instance = Some((instance, witness));
                let _result = state.recursive_ivc(depth, &committer);
            });
        });
    }
    group.finish();
}

fn bench_proof_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_compression");
    
    for transcript_size in [32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("bytes", transcript_size), transcript_size, |b, &size| {
            let state = FoldState::new(create_test_structure(4));
            let transcript = (0..size).map(|i| (i % 256) as u8).collect::<Vec<_>>();
            
            b.iter(|| {
                let (_commit, _proof) = state.compress_proof(&transcript);
            });
        });
    }
    group.finish();
}

fn bench_verifier_ccs(c: &mut Criterion) {
    c.bench_function("verifier_ccs_fold", |b| {
        let structure = verifier_ccs();
        let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
        
        // Create valid instances for the verifier CCS (4-element witness: [a, b, a*b, a+b])
        let instance1 = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness1 = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)), // 2*3
                embed_base_to_ext(F::from_u64(5)), // 2+3
            ],
        };
        
        let instance2 = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness2 = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(4)),
                embed_base_to_ext(F::from_u64(1)),
                embed_base_to_ext(F::from_u64(4)), // 4*1
                embed_base_to_ext(F::from_u64(5)), // 4+1
            ],
        };
        
        b.iter(|| {
            let mut state = FoldState::new(structure.clone());
            let _proof = state.generate_proof((instance1.clone(), witness1.clone()), 
                                             (instance2.clone(), witness2.clone()), 
                                             &committer);
        });
    });
}

fn bench_extractor(c: &mut Criterion) {
    let mut group = c.benchmark_group("extractor");
    
    for transcript_size in [50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("bytes", transcript_size), transcript_size, |b, &size| {
            let transcript = (0..size).map(|i| (i % 256) as u8).collect::<Vec<_>>();
            let proof = neo_fold::Proof { transcript };
            
            b.iter(|| {
                let _witness = neo_fold::extractor(&proof);
            });
        });
    }
    group.finish();
}

fn bench_full_folding_workflow(c: &mut Criterion) {
    c.bench_function("full_workflow", |b| {
        let structure = create_test_structure(8);
        let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
        let (inst1, inst2) = create_test_instances(8);
        
        b.iter(|| {
            // Complete workflow: fold -> verify -> compress -> extract
            let mut state = FoldState::new(structure.clone());
            let proof = state.generate_proof(inst1.clone(), inst2.clone(), &committer);
            
            let verifier_state = FoldState::new(structure.clone());
            let verify_result = verifier_state.verify(&proof.transcript, &committer);
            assert!(verify_result);
            
            let (_commit, _compressed_proof) = verifier_state.compress_proof(&proof.transcript);
            let _extracted_witness = neo_fold::extractor(&proof);
        });
    });
}

criterion_group!(
    benches,
    bench_single_fold,
    bench_fold_verification,
    bench_recursive_ivc,
    bench_proof_compression,
    bench_verifier_ccs,
    bench_extractor,
    bench_full_folding_workflow
);
criterion_main!(benches);
