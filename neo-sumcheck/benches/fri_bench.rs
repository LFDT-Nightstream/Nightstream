use criterion::{criterion_group, criterion_main, Criterion};
use neo_sumcheck::{ExtF, FriOracle, PolyOracle, Polynomial, from_base};
use neo_fields::{random_extf};
use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;
use rand::{rng, Rng};

use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

type Val = Goldilocks;
type Challenge = Val;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type FriPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

fn p3_pcs() -> (FriPcs, Challenger) {
    let mut rng = rng();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 40,
        proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = FriPcs::new(Dft::default(), val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    (pcs, challenger)
}

fn to_p3(e: ExtF) -> Val {
    let [r, _] = e.to_array();
    r
}

fn bench_fri(c: &mut Criterion, log_deg: usize) {
    let mut rng = rng();
    let degree = (1 << log_deg) - 1;
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());

    let mut neo_oracle = FriOracle::new_with_blinds(vec![poly.clone()], vec![ExtF::ZERO]);
    c.bench_function(&format!("neo_commit_{}", log_deg), |b| b.iter(|| neo_oracle.commit()));
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    c.bench_function(&format!("neo_open_{}", log_deg), |b| b.iter(|| neo_oracle.open_at_point(&point)));
    let neo_comms = neo_oracle.commit();
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    c.bench_function(&format!("neo_verify_{}", log_deg), |b| b.iter(|| neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs)));

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain.clone().into_iter().map(|x| {
        let x_ext = from_base(x);
        to_p3(poly.eval(x_ext))
    }).collect();
    let matrix = RowMajorMatrix::new(p3_vals.clone(), 1);
    let (p3_comm, prover_data) = <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix.clone())]);
    c.bench_function(&format!("p3_commit_{}", log_deg), |b| b.iter(|| <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix.clone())])));
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    c.bench_function(&format!("p3_open_{}", log_deg), |b| b.iter(|| <FriPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&prover_data, vec![vec![p3_point]])], &mut p_challenger.clone())));
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&prover_data, vec![vec![p3_point]])], &mut p_challenger);
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    c.bench_function(&format!("p3_verify_{}", log_deg), |b| b.iter(|| <FriPcs as Pcs<Challenge, Challenger>>::verify(&pcs, vec![(p3_comm.clone(), vec![(domain.clone(), vec![(p3_point, opened[0][0][0].clone())])])], &proof, &mut v_challenger.clone())));
}

fn fri_bench(c: &mut Criterion) {
    for log_deg in 2..=8 {
        bench_fri(c, log_deg);
    }
}

criterion_group!(benches, fri_bench);
criterion_main!(benches);
