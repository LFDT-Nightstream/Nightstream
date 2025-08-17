use criterion::{criterion_group, criterion_main, Criterion, black_box};
use neo_sumcheck::{
    multilinear_sumcheck_prover, ExtF, F, FriOracle, Polynomial
};
use p3_field::PrimeCharacteristicRing;
use rand::Rng;

fn bench_multilinear_sumcheck_n1024(c: &mut Criterion) {
    let mut rng = rand::rng();
    let n = 1024;
    
    // Create random multilinear evaluation
    let evals: Vec<ExtF> = (0..n)
        .map(|_| ExtF::new_complex(F::from_u64(rng.random()), F::from_u64(rng.random())))
        .collect();
    let claim = evals.iter().copied().sum::<ExtF>();
    
    c.bench_function("multilinear_sumcheck_N1024", |bencher| {
        bencher.iter(|| {
            let mut evals_copy = evals.clone();
            let mut transcript = Vec::new();
            let mut oracle = FriOracle::new(vec![Polynomial::new(vec![ExtF::ZERO, ExtF::ONE])], &mut transcript);
            
            // Run prover
            let result = multilinear_sumcheck_prover(
                &mut evals_copy,
                claim,
                &mut oracle,
                &mut transcript,
            );
            
            // Ensure computation actually happens
            black_box(result)
        })
    });
}

criterion_group!(benches, bench_multilinear_sumcheck_n1024);
criterion_main!(benches);
