//! Row-phase per-block sum-check invariant tests
//!
//! These tests isolate each block (F, NC, Eval) during row rounds (ell_n=2)
//! and run the sum-check driver. If any block has a pairing/endianness mismatch
//! between how it evaluates vs how it folds, the engine's round-1
//! p(0)+p(1)=running_sum check will fail in that test, pinpointing the culprit.

use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold::pi_ccs::oracle::GenericCcsOracle;
use neo_fold::pi_ccs::precompute::MlePartials;
use neo_fold::pi_ccs::sparse_matrix::Csr;
use neo_fold::sumcheck::{RoundOracle, run_sumcheck, run_sumcheck_skip_eval_at_one};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn f64_(x: i64) -> F {
    if x >= 0 { F::from_u64(x as u64) } else { F::ZERO - F::from_u64((-x) as u64) }
}
fn k64_(x: i64) -> K { K::from(f64_(x)) }

fn sample_initial_sum(oracle: &mut GenericCcsOracle<'_, F>, xs: &[K]) -> K {
    // For round 0, define initial_sum as s(0)+s(1) based on oracle evaluations
    // This mirrors how sum-check derives the claimed initial sum in many flows.
    let ys = oracle.evals_at(xs);
    // locate indices for 0 and 1
    let i0 = xs.iter().position(|&x| x == K::ZERO).unwrap();
    let i1 = xs.iter().position(|&x| x == K::ONE).unwrap();
    ys[i0] + ys[i1]
}

fn base_ccs_identity(n: usize, t: usize) -> CcsStructure<F> {
    // Build CCS with M_0=I_n and t-1 additional identity matrices (not used directly here)
    let mut mats = Vec::with_capacity(t);
    for _ in 0..t { mats.push(Mat::<F>::identity(n)); }
    // f(m1,...) = m1 (identity on first argument). Degree 1.
    let f = SparsePoly::new(
        t,
        vec![Term { coeff: F::ONE, exps: {
            let mut v = vec![0; t]; v[0] = 1; v
        }}]
    );
    CcsStructure { matrices: mats, f, n, m: 1 }
}

fn dummy_csr(n: usize, m: usize) -> Csr<F> {
    Csr { rows: n, cols: m, indptr: vec![0; n+1], indices: vec![], data: vec![] }
}

#[test]
fn row_invariant_f_block_only() {
    // ell_n=2 (n=4), ell_d=0 (Ajtai skipped). Only F contributes.
    let n = 4usize; // 2 row rounds
    let s = base_ccs_identity(n, 1);

    // Row weights (β_r gate) over 4 rows
    let w_beta_r_partial = vec![k64_(3), k64_(5), k64_(7), k64_(11)];

    // MLE partials for F: s_per_j[0] over 4 rows
    let s_per_j = vec![k64_(2), k64_(4), k64_(6), k64_(8)];

    // Build oracle with only F active
    let mut oracle = GenericCcsOracle {
        s: &s,
        partials_first_inst: MlePartials { s_per_j: vec![s_per_j] },
        // Ajtai gates length 1 (ell_d=0)
        w_beta_a_partial: vec![K::ONE],
        w_alpha_a_partial: vec![K::ONE],
        w_beta_r_partial: w_beta_r_partial.clone(),
        w_beta_r_full: w_beta_r_partial,
        w_eval_r_partial: vec![K::ZERO; n],
        z_witnesses: vec![],
        gamma: K::ONE,
        k_total: 1,
        b: 2,
        ell_d: 0,
        ell_n: 2,
        d_sc: 3,
        round_idx: 0,
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial: vec![K::ZERO; n],
        row_chals: vec![],
        csr_m1: &dummy_csr(n, 1),
        csrs: &[],
        eval_ajtai_partial: None,
        me_offset: 0,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc: None,
    };

    let mut tr = Poseidon2Transcript::new(b"sumcheck-row-f-only");
    let sample_xs: Vec<K> = (0..=3u64).map(|u| K::from(F::from_u64(u))).collect();
    let initial_sum = sample_initial_sum(&mut oracle, &sample_xs);
    let out = run_sumcheck(&mut tr, &mut oracle, initial_sum, &sample_xs)
        .expect("sumcheck should hold for F-only row phase");
    assert_eq!(out.rounds.len(), 2, "should have two row rounds only");
}

#[test]
fn row_invariant_eval_block_only() {
    // ell_n=2 (n=4), ell_d=0. Only Eval contributes.
    let n = 4usize;
    let s = base_ccs_identity(n, 1);

    let w_eval_r_partial = vec![k64_(1), k64_(2), k64_(3), k64_(4)];
    let eval_row_partial = vec![k64_(9), k64_(10), k64_(11), k64_(12)];

    let mut oracle = GenericCcsOracle {
        s: &s,
        partials_first_inst: MlePartials { s_per_j: vec![vec![K::ZERO; n]] },
        w_beta_a_partial: vec![K::ONE],
        w_alpha_a_partial: vec![K::ONE],
        w_beta_r_partial: vec![K::ZERO; n],
        w_beta_r_full: vec![K::ZERO; n],
        w_eval_r_partial,
        z_witnesses: vec![],
        gamma: K::ONE,
        k_total: 1,
        b: 2,
        ell_d: 0,
        ell_n: 2,
        d_sc: 3,
        round_idx: 0,
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial,
        row_chals: vec![],
        csr_m1: &dummy_csr(n, 1),
        csrs: &[],
        eval_ajtai_partial: None,
        me_offset: 0,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc: None,
    };

    let mut tr = Poseidon2Transcript::new(b"sumcheck-row-eval-only");
    let sample_xs: Vec<K> = (0..=3u64).map(|u| K::from(F::from_u64(u))).collect();
    let initial_sum = sample_initial_sum(&mut oracle, &sample_xs);
    let _out = run_sumcheck(&mut tr, &mut oracle, initial_sum, &sample_xs)
        .expect("sumcheck should hold for Eval-only row phase");
}

#[test]
fn row_invariant_nc_block_only() {
    // ell_n=2 (n=4), ell_d matches D (Ajtai rows). Only NC contributes via nc_y_matrices.
    let n = 4usize;
    let s = base_ccs_identity(n, 1);

    // One instance, D rows, 4 columns (rows). Fill simple ramp values per row index.
    let d = neo_math::D; // Ajtai rows
    let mut y_mat: Vec<Vec<K>> = Vec::with_capacity(d);
    for _rho in 0..d {
        // row values over the 4 row positions
        y_mat.push(vec![k64_(1), k64_(2), k64_(3), k64_(4)]);
    }

    let nc_y_matrices = vec![y_mat];
    let nc_row_gamma_pows = vec![K::ONE];

    // Compute ell_d from D (pad to power-of-two domain for Ajtai bits)
    let ell_d = d.next_power_of_two().trailing_zeros() as usize;

    // β_a equality weights: use proper HalfTableEq over a fixed β_a point
    use neo_fold::pi_ccs::eq_weights::{HalfTableEq, RowWeight};
    let beta_a = vec![K::ZERO; ell_d];
    let w_beta_a = HalfTableEq::new(&beta_a);
    let w_beta_a_partial: Vec<K> = (0..(1usize << ell_d)).map(|i| w_beta_a.w(i)).collect();

    let mut oracle = GenericCcsOracle {
        s: &s,
        partials_first_inst: MlePartials { s_per_j: vec![vec![K::ZERO; n]] },
        w_beta_a_partial,
        w_alpha_a_partial: vec![K::ONE; 1 << ell_d],
        w_beta_r_partial: vec![k64_(5), k64_(6), k64_(7), k64_(8)],
        w_beta_r_full: vec![k64_(5), k64_(6), k64_(7), k64_(8)],
        w_eval_r_partial: vec![K::ZERO; n],
        z_witnesses: vec![],
        gamma: K::ONE,
        k_total: 1,
        b: 2,
        ell_d,
        ell_n: 2,
        d_sc: 3,
        round_idx: 0,
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial: vec![K::ZERO; n],
        row_chals: vec![],
        csr_m1: &dummy_csr(n, 1),
        csrs: &[],
        eval_ajtai_partial: None,
        me_offset: 0,
        nc_y_matrices,
        nc_row_gamma_pows,
        nc: None,
    };

    let mut tr = Poseidon2Transcript::new(b"sumcheck-row-nc-only");
    let sample_xs_full: Vec<K> = (0..=3u64).map(|u| K::from(F::from_u64(u))).collect();
    let initial_sum = sample_initial_sum(&mut oracle, &sample_xs_full);
    let _out = run_sumcheck_skip_eval_at_one(&mut tr, &mut oracle, initial_sum, &sample_xs_full)
        .expect("sumcheck(skip-1) should hold for NC-only row phase");
}
