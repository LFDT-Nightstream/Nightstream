use criterion::{criterion_group, criterion_main, Criterion};
use neo_commit::{AjtaiCommitter, NeoParams, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::F;
use p3_field::PrimeCharacteristicRing;
use rand::Rng;

const SMALL_PARAMS: NeoParams = NeoParams {
    q: (1u64 << 61) - 1,
    n: 8,
    k: 4,
    d: 8,
    b: 2,
    e_bound: 4,
    norm_bound: 16,
    sigma: 3.2,
    beta: 3,
    max_blind_norm: (1u64 << 61) - 1,
};

fn bench_commit(c: &mut Criterion) {
    let params = if std::env::var("NEO_BENCH_SECURE").is_ok() {
        SECURE_PARAMS
    } else {
        SMALL_PARAMS
    }; // small parameters for fast benches
    let committer = AjtaiCommitter::setup_unchecked(params);
    let mut rng = rand::rng();
    let z: Vec<F> = (0..params.n)
        .map(|_| F::from_u64(rng.random_range(0..params.b)))
        .collect();
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    c.bench_function("commit_small", |b| {
        b.iter(|| {
            let mut t = Vec::new();
            committer.commit(&w, &mut t)
        })
    });
}

criterion_group!(benches, bench_commit);
criterion_main!(benches);
