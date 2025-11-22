#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_reductions::sumcheck::RoundOracle;
use neo_reductions::paper_exact_engine::oracle::PaperExactOracle;
use neo_reductions::paper_exact_engine as refimpl;
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

fn tiny_ccs_perm(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    // Simple non-identity M_1: swap rows/cols for n=2; generalizes to reverse permutation
    let mut data = vec![F::ZERO; n * n];
    for r in 0..n { data[r * n + (n - 1 - r)] = F::ONE; }
    let m0 = Mat::from_row_major(n, n, data);
    // f(y0) = y0
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    CcsStructure::new(vec![m0], f).unwrap()
}

fn tiny_ccs_custom_m1(m1: Mat<F>) -> CcsStructure<F> {
    assert_eq!(m1.rows(), m1.cols(), "M1 must be square");
    let _n = m1.rows();
    // f(y0) = y0
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    CcsStructure::new(vec![m1], f).unwrap()
}

fn make_digits_matrix(val: F, d: usize, m: usize) -> Mat<F> {
    Mat::from_row_major(d, m, vec![val; d * m])
}

#[test]
fn round0_sum_matches_hypercube_sum_k1() {
    // Small instance: n=2 (ell_n=1), m=2, t=1
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    // One MCS witness with all-ones digits
    let z = make_digits_matrix(F::ONE, D, m);
    let mcs_w = [McsWitness { w: vec![], Z: z }];
    let me_w: [Mat<F>; 0] = [];

    // Challenges sized to the round dimensions
    let ell_n = 1usize; // since n=2
    let ell_d = 1usize; // use 1 Ajtai bit for the sum-check domain in this test
    let ch = neo_reductions::Challenges {
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
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    let z0 = make_digits_matrix(F::from_u64(2), D, m);
    let z1 = make_digits_matrix(F::from_u64(3), D, m);
    let mcs_w = [McsWitness { w: vec![], Z: z0 }];
    let me_w = [z1];

    let ell_n = 1usize; // n=2
    let ell_d = 1usize; // keep Ajtai domain tiny
    let ch = neo_reductions::Challenges {
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

#[test]
#[ignore] // Requires separate NC implementation that was removed when optimized_engine became paper-exact
fn nc_sum_engine_matches_paper_nc_when_m1_not_identity() {
    // Construct a minimal CCS where M_1 ≠ I to expose NC drift
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_perm(n, m);

    // One MCS witness with constant digits (ensures y differs from a single table lookup)
    let Z = make_digits_matrix(F::ONE, D, m);
    let mcs_w = [McsWitness { w: vec![], Z }];
    let me_w: [Mat<F>; 0] = [];

    // Challenges (tiny domains): ell_n = 1, ell_d = 1
    let ell_n = 1usize;
    let ell_d = 1usize;
    let ch = neo_reductions::Challenges {
        alpha:  vec![K::from(F::from_u64(3)); ell_d],
        beta_a: vec![K::from(F::from_u64(5)); ell_d],
        beta_r: vec![K::from(F::from_u64(7)); ell_n],
        gamma:  K::from(F::from_u64(11)),
    };

    // Paper-exact total sum over the hypercube (k=1 ⇒ no Eval)
    let paper_total = refimpl::sum_q_over_hypercube_paper_exact(
        &s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, None,
    );

    // Manually compute F(β_r) = Σ_row χ_{β_r}(row) · (M_1 · z)[row]
    // where z[c] = Σ_{rho=0..D-1} b^rho · Z[rho,c]
    let bF = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; D];
    for i in 1..D { pow_b[i] = pow_b[i - 1] * bF; }
    let z_vec: Vec<F> = (0..m)
        .map(|c| {
            let mut acc = F::ZERO;
            for rho in 0..D { acc += mcs_w[0].Z[(rho, c)] * pow_b[rho]; }
            acc
        })
        .collect();

    // χ_{β_r}
    let chi_beta_r = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    let mut f_beta = K::ZERO;
    for row in 0..n {
        // (M_1 · z)[row]
        let mut dot = F::ZERO;
        for c in 0..m { dot += s.matrices[0][(row, c)] * z_vec[c]; }
        f_beta += chi_beta_r[row] * K::from(dot);
    }

    // Paper NC component extracted as paper_total - F_beta
    let _nc_paper = paper_total - f_beta;

    // Engine NC: since optimized_engine now uses paper-exact code, this test is obsolete
    // Stub out the comparison
    panic!("This test requires a separate NC implementation that was removed when optimized_engine became paper-exact");
}

#[test]
#[ignore] // Requires separate NC implementation that was removed when optimized_engine became paper-exact
fn nc_sum_engine_vs_paper_drift_with_custom_m1_and_Z() {
    // Design M1 and Z so that engine NC uses a dot-product giving large non-range values,
    // while paper NC (table lookup) sees different entries; this should expose drift if any.
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);

    // Custom M1: first row selects second column, second row sums both columns
    // M1 = [ [0,1], [1,1] ]
    let m1 = Mat::from_row_major(n, m, vec![F::ZERO, F::ONE, F::ONE, F::ONE]);
    let s = tiny_ccs_custom_m1(m1);

    // Z with distinct larger values to avoid accidental zeros in range product
    // Z rows (Ajtai): rho=0: [100, 17], rho=1: [9, 23], remaining rows: zeros
    let mut z_data = vec![F::ZERO; D * m];
    z_data[0] = F::from_u64(100);
    z_data[1] = F::from_u64(17);
    z_data[m] = F::from_u64(9);
    z_data[m + 1] = F::from_u64(23);
    let Z = Mat::from_row_major(D, m, z_data);
    let mcs_w = [McsWitness { w: vec![], Z }];
    let me_w: [Mat<F>; 0] = [];

    let ell_n = 1usize; // n=2
    let ell_d = 1usize; // keep Ajtai domain tiny
    let ch = neo_reductions::Challenges {
        alpha:  vec![K::from(F::from_u64(13)); ell_d],
        beta_a: vec![K::from(F::from_u64(17)); ell_d],
        beta_r: vec![K::from(F::from_u64(19)); ell_n],
        gamma:  K::from(F::from_u64(23)),
    };

    // Paper-exact total (k=1)
    let paper_total = refimpl::sum_q_over_hypercube_paper_exact(
        &s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, None,
    );

    // Compute F_beta directly
    let bF = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; D];
    for i in 1..D { pow_b[i] = pow_b[i - 1] * bF; }
    let z_vec: Vec<F> = (0..m)
        .map(|c| {
            let mut acc = F::ZERO;
            for rho in 0..D { acc += mcs_w[0].Z[(rho, c)] * pow_b[rho]; }
            acc
        })
        .collect();
    let chi_beta_r = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    let mut f_beta = K::ZERO;
    for row in 0..n {
        let mut dot = F::ZERO;
        for c in 0..m { dot += s.matrices[0][(row, c)] * z_vec[c]; }
        f_beta += chi_beta_r[row] * K::from(dot);
    }

    let _nc_paper = paper_total - f_beta;
    
    // Engine NC: since optimized_engine now uses paper-exact code, this test is obsolete
    // Stub out the comparison
    panic!("This test requires a separate NC implementation that was removed when optimized_engine became paper-exact");
}

