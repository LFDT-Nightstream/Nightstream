#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_fold::sumcheck::RoundOracle;
use neo_fold::paper_exact_engine::oracle::PaperExactOracle;
use neo_fold::paper_exact_engine as refimpl;
use neo_ccs::{CcsStructure, McsWitness, Mat, SparsePoly, Term};
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    // f(y0) = y0
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    CcsStructure::new(vec![m0], f).unwrap()
}

fn make_digits_matrix(val: F, d: usize, m: usize) -> Mat<F> {
    Mat::from_row_major(d, m, vec![val; d * m])
}

#[test]
fn round0_sum_matches_hypercube_sum_k1() {
    // Small instance: n=2 (ell_n=1), m=2, t=1
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    // One MCS witness with all-ones digits
    let z = make_digits_matrix(F::ONE, D, m);
    let mcs_w = [McsWitness { w: vec![], Z: z }];
    let me_w: [Mat<F>; 0] = [];

    // Challenges sized to the round dimensions
    let ell_n = 1usize; // since n=2
    let ell_d = 1usize; // use 1 Ajtai bit for the sum-check domain in this test
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha:  vec![K::from(F::from_u64(3)); ell_d],
        beta_a: vec![K::from(F::from_u64(5)); ell_d],
        beta_r: vec![K::from(F::from_u64(7)); ell_n],
        gamma:  K::from(F::from_u64(11)),
    };

    // Degree bound not used by this check; set a small placeholder
    let d_sc = 4usize;

    // Build oracle (no ME inputs → Eval block gated off)
    let mut oracle = PaperExactOracle::<'_, F>::new(
        &s, &params, &mcs_w, &me_w, ch.clone(), ell_d, ell_n, d_sc, None,
    );

    // Left: g0(0) + g0(1)
    let g0 = oracle.evals_at(&[K::ZERO, K::ONE]);
    let lhs = g0[0] + g0[1];

    // Right: literal ∑_{X∈{0,1}^{ell_d+ell_n}} Q(X)
    let rhs = refimpl::sum_q_over_hypercube_paper_exact(
        &s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, None,
    );

    assert_eq!(lhs, rhs, "round-0 sum must equal hypercube sum (k=1)");
}

#[test]
fn round0_sum_matches_hypercube_sum_k2_with_eval() {
    // Small instance with one MCS and one ME witness to enable Eval block
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    let z0 = make_digits_matrix(F::from_u64(2), D, m);
    let z1 = make_digits_matrix(F::from_u64(3), D, m);
    let mcs_w = [McsWitness { w: vec![], Z: z0 }];
    let me_w = [z1];

    let ell_n = 1usize; // n=2
    let ell_d = 1usize; // keep Ajtai domain tiny
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha:  vec![K::from(F::from_u64(13)); ell_d],
        beta_a: vec![K::from(F::from_u64(17)); ell_d],
        beta_r: vec![K::from(F::from_u64(19)); ell_n],
        gamma:  K::from(F::from_u64(23)),
    };

    let r_inputs = vec![K::from(F::from_u64(29)); ell_n];
    let d_sc = 4usize;

    let mut oracle = PaperExactOracle::<'_, F>::new(
        &s, &params, &mcs_w, &me_w, ch.clone(), ell_d, ell_n, d_sc, Some(&r_inputs),
    );

    // Left: round-0 sum
    let g0 = oracle.evals_at(&[K::ZERO, K::ONE]);
    let lhs = g0[0] + g0[1];

    // Right: brute-force hypercube sum with Eval block active (r_inputs provided)
    let rhs = refimpl::sum_q_over_hypercube_paper_exact(
        &s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, Some(&r_inputs),
    );

    assert_eq!(lhs, rhs, "round-0 sum must equal hypercube sum (k=2 with Eval)");
}

