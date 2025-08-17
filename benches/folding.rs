use criterion::{criterion_group, criterion_main, Criterion};
use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, NeoParams, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{ExtF, F};
use neo_fold::{pi_ccs, pi_rlc, FoldState};
use neo_ring::RingElement;
use neo_modint::ModInt;
use p3_field::PrimeField64;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
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

fn large_commit_bench(c: &mut Criterion) {
    let mut rng = rand::rng();
    let params = if std::env::var("NEO_BENCH_SECURE").is_ok() {
        SECURE_PARAMS
    } else {
        SMALL_PARAMS
    };
    let committer = AjtaiCommitter::setup_unchecked(params);
    let z: Vec<F> = (0..params.n).map(|_| F::from_u64(rng.random())).collect();
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    c.bench_function("commit_small", |b| {
        b.iter(|| {
            let mut t = Vec::new();
            committer.commit(&w, &mut t)
        })
    });
}

fn large_fold_bench(c: &mut Criterion) {
    let params = if std::env::var("NEO_BENCH_SECURE").is_ok() {
        SECURE_PARAMS
    } else {
        SMALL_PARAMS
    };
    let num_constraints = 8; // small power of 2
    let witness_size = params.n;

    // Create simple matrices like in our working test - all constraints are identical
    // Each constraint checks the same pattern across all witness elements
    // M_0: picks element 0 for all constraints
    let mut a_data = vec![F::ZERO; num_constraints * witness_size];
    for i in 0..num_constraints {
        a_data[i * witness_size] = F::ONE; // All pick element 0
    }
    let a = RowMajorMatrix::new(a_data, witness_size);

    // M_1: picks element 1 for all constraints
    let mut b_data = vec![F::ZERO; num_constraints * witness_size];
    for i in 0..num_constraints {
        b_data[i * witness_size + 1] = F::ONE; // All pick element 1
    }
    let b = RowMajorMatrix::new(b_data, witness_size);

    // M_2: picks element 2 for all constraints
    let mut c_data = vec![F::ZERO; num_constraints * witness_size];
    for i in 0..num_constraints {
        c_data[i * witness_size + 2] = F::ONE; // All pick element 2
    }
    let mat_c = RowMajorMatrix::new(c_data, witness_size);

    let mats = vec![a, b, mat_c];

    // Linear constraint: inputs[0] + inputs[1] - inputs[2] = 0
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] + inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );
    let structure = CcsStructure::new(mats, f);

    // Create witnesses that satisfy z[0] + z[1] - z[2] = 0
    let mut rng = rand::rng();

    let a1 = F::from_u64(rng.random_range(1..10));
    let b1 = F::from_u64(rng.random_range(1..10));
    let c1 = a1 + b1;
    let mut z1 = vec![F::ZERO; witness_size];
    z1[0] = a1;
    z1[1] = b1;
    z1[2] = c1;

    let a2 = F::from_u64(rng.random_range(1..10));
    let b2 = F::from_u64(rng.random_range(1..10));
    let c2 = a2 + b2;
    let mut z2 = vec![F::ZERO; witness_size];
    z2[0] = a2;
    z2[1] = b2;
    z2[2] = c2;

    let witness1 = CcsWitness { z: z1.clone() };
    let witness2 = CcsWitness { z: z2.clone() };

    let committer = AjtaiCommitter::setup_unchecked(params);

    let z1_mat = decomp_b(&z1, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&z1_mat, &params);
    let mut t1 = Vec::new();
    let (commit1, _, _, _) = committer.commit(&w1, &mut t1);
    let instance1 = CcsInstance {
        commitment: commit1,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    let z2_mat = decomp_b(&z2, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&z2_mat, &params);
    let mut t2 = Vec::new();
    let (commit2, _, _, _) = committer.commit(&w2, &mut t2);
    let instance2 = CcsInstance {
        commitment: commit2,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    c.bench_function("fold_small", |b| {
        b.iter(|| {
            let mut fold_state = FoldState::new(structure.clone());
            let mut transcript = Vec::new();
            fold_state.ccs_instance = Some((instance1.clone(), witness1.clone()));
            pi_ccs(&mut fold_state, &committer, &mut transcript);

            fold_state.ccs_instance = Some((instance2.clone(), witness2.clone()));
            pi_ccs(&mut fold_state, &committer, &mut transcript);

            let rho = F::from_u64(rng.random_range(1..10));
            let rho_rot = RingElement::from_scalar(ModInt::from_u64(rho.as_canonical_u64()), params.n);
            pi_rlc(&mut fold_state, rho_rot, &committer, &mut transcript);
        })
    });
}

criterion_group!(benches, large_commit_bench, large_fold_bench);
criterion_main!(benches);
