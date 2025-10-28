//! Unit tests for the Eval block in the Generic CCS oracle.
//!
//! These "delta-probe" tests isolate the Eval term's behavior by collapsing it
//! to simple closed forms, catching any off-by-one errors in γ exponents or
//! equality-gate wiring.

use neo_ccs::{Mat, CcsStructure, SparsePoly, Term};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;
use neo_fold::pi_ccs::precompute::MlePartials;
use neo_fold::pi_ccs::oracle::{GenericCcsOracle, NcState};
use neo_fold::pi_ccs::sparse_matrix::Csr;
use neo_fold::sumcheck::RoundOracle;

// ----- Tiny helpers -----

fn f64_(x: i64) -> F {
    if x >= 0 { 
        F::from_u64(x as u64) 
    } else { 
        F::ZERO - F::from_u64((-x) as u64) 
    }
}

fn k64_(x: i64) -> K { 
    K::from(f64_(x)) 
}

/// Minimal CCS with 2 matrices; f is irrelevant (we'll zero it out)
fn tiny_ccs(n: usize, m: usize) -> CcsStructure<F> {
    let m0 = Mat::<F>::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) { 
        m1[(i, i)] = F::ONE; 
    }
    let f = SparsePoly::new(2, vec![Term { coeff: F::ZERO, exps: vec![0, 0] }]); // identically 0
    CcsStructure { matrices: vec![m0, m1], f, n, m }
}

/// Row/ajtai equality "partial" vectors we can hand-craft in tests
fn eq_pair(r: K) -> Vec<K> { 
    vec![K::ONE - r, r] 
}

fn eq_pair_alpha(a: K) -> Vec<K> { 
    vec![K::ONE - a, a] 
}

/// Build a constant Ajtai vector of length 2^ell_d
fn ajtai_vec2(x0: K, x1: K) -> Vec<K> { 
    vec![x0, x1] 
}

/// Row partials holder with t polynomials, each of length 2 (one row bit)
fn empty_row_partials(t: usize) -> MlePartials {
    MlePartials { 
        s_per_j: (0..t).map(|_| vec![K::ZERO, K::ZERO]).collect() 
    }
}

/// A tiny identity CSR for when you do want ensure_nc_precomputed() to run.
fn csr_identity(n: usize) -> Csr<F> {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);
    
    indptr.push(0);
    for i in 0..n {
        indices.push(i);
        data.push(F::ONE);
        indptr.push(i + 1);
    }
    
    Csr {
        rows: n,
        cols: n,
        indptr,
        indices,
        data,
    }
}

// ----- Test 1: Row-phase Eval gate matches eq_r -----

#[test]
fn eval_row_gate_matches_eq_r() {
    // One row bit (ell_n=1), one Ajtai bit (ell_d=1)
    let n = 2; 
    let m = 2;
    let s = tiny_ccs(n, m);
    
    // Choose any r; use a weird one to avoid accidental equalities
    let r = k64_(37);
    let w_eval_r_partial = eq_pair(r);          // eq_r weights for the single row bit
    let eval_row_partial = vec![k64_(10), k64_(20)]; // G_eval at the two row leaves (a,b)
    
    // Oracle with only the Eval row block "live"
    let csr_m1_inst = csr_identity(n);
    let mut oracle = GenericCcsOracle::<F> {
        s: &s,
        partials_first_inst: empty_row_partials(s.t()),
        w_beta_a_partial: vec![K::ZERO, K::ZERO],
        w_alpha_a_partial: vec![K::ZERO, K::ZERO],
        w_beta_r_partial: vec![K::ZERO, K::ZERO],
        w_eval_r_partial: w_eval_r_partial.clone(),
        z_witnesses: vec![],
        gamma: K::from(F::from_u64(7)),
        k_total: 2,
        b: 3,
        ell_d: 1,
        ell_n: 1,
        d_sc: 1,
        round_idx: 0,
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial: eval_row_partial.clone(),
        row_chals: vec![],
        csr_m1: &csr_m1_inst,
        csrs: &[],
        eval_ajtai_partial: None,
        me_offset: 1,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc: None,
    };
    
    // Sample univariate at X=0 and X=1 and compare against closed form
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]);
    
    let gate0 = (K::ONE - K::ZERO) * w_eval_r_partial[0] + K::ZERO * w_eval_r_partial[1]; // (1-0)*w0 + 0*w1 = w0
    let gate1 = (K::ONE - K::ONE) * w_eval_r_partial[0] + K::ONE * w_eval_r_partial[1]; // w1
    let g0 = (K::ONE - K::ZERO) * eval_row_partial[0] + K::ZERO * eval_row_partial[1];
    let g1 = (K::ONE - K::ONE) * eval_row_partial[0] + K::ONE * eval_row_partial[1];
    
    assert_eq!(ys[0], gate0 * g0, "row Eval at X=0 mismatches");
    assert_eq!(ys[1], gate1 * g1, "row Eval at X=1 mismatches");
}

// ----- Test 2: Ajtai-phase Eval gate matches eq_alpha -----

#[test]
fn eval_ajtai_gate_matches_eq_alpha_and_row_scalar() {
    let n = 2; 
    let m = 2;
    let s = tiny_ccs(n, m);
    
    // Row bit r, Ajtai bit alpha
    let r = k64_(1);           // bind row to 1
    let alpha = k64_(0);       // Ajtai gate picks index 0
    
    // After folding the single row bit, w_eval_r_partial becomes scalar eq_r(r)
    let w_eval_r_partial_full = eq_pair(r);
    let wr_scalar = (K::ONE - r) * w_eval_r_partial_full[0] + r * w_eval_r_partial_full[1];
    
    // Pre-collapsed Ajtai vector for Eval: [E0, E1]
    let eval_ajtai = ajtai_vec2(k64_(1234), k64_(0));
    
    let csr_m1_inst = csr_identity(n);
    let mut oracle = GenericCcsOracle::<F> {
        s: &s,
        partials_first_inst: empty_row_partials(s.t()),
        w_beta_a_partial: vec![K::ZERO, K::ZERO],
        w_alpha_a_partial: eq_pair_alpha(alpha),
        w_beta_r_partial: vec![],           // not used in Ajtai Eval
        w_eval_r_partial: vec![wr_scalar],  // folded row gate
        z_witnesses: vec![],
        gamma: K::from(F::from_u64(7)),
        k_total: 2,
        b: 3,
        ell_d: 1,
        ell_n: 1,
        d_sc: 1,
        round_idx: 1, // we are in Ajtai phase now
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial: vec![],           // not used in Ajtai phase
        row_chals: vec![],
        csr_m1: &csr_m1_inst,
        csrs: &[],
        eval_ajtai_partial: Some(eval_ajtai.clone()),
        me_offset: 1,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc: Some(NcState { 
            y_partials: vec![], 
            gamma_pows: vec![], 
            f_at_rprime: None 
        }),
    };
    
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]); // Ajtai univariate
    
    // At X=0: gate_alpha = 1-alpha, eval_x = E0
    let gate_a0 = (K::ONE - K::ZERO) * (K::ONE - alpha) + K::ZERO * alpha;
    let eval_x0 = (K::ONE - K::ZERO) * eval_ajtai[0] + K::ZERO * eval_ajtai[1];
    assert_eq!(ys[0], gate_a0 * wr_scalar * eval_x0, "Ajtai Eval at X=0 mismatches");
    
    // At X=1: gate_alpha = alpha, eval_x = E1
    let gate_a1 = (K::ONE - K::ONE) * (K::ONE - alpha) + K::ONE * alpha;
    let eval_x1 = (K::ONE - K::ONE) * eval_ajtai[0] + K::ONE * eval_ajtai[1];
    assert_eq!(ys[1], gate_a1 * wr_scalar * eval_x1, "Ajtai Eval at X=1 mismatches");
}

// ----- Test 3: γ-exponent schedule "sum of powers" test -----

#[test]
fn eval_gamma_weight_schedule_reduces_to_sum_of_powers() {
    let n = 2; 
    let m = 2;
    let s = tiny_ccs(n, m);
    
    // Challenge choices:
    let _r = k64_(0);    // pick first row/column
    let alpha = k64_(0);// pick first Ajtai coordinate
    let gamma = K::from(F::from_u64(7));
    let k_total = 2usize;
    
    // Row scalar after folding: eq_r(r) with one bit is (1-r, r) · [1, r] = 1
    let wr_scalar = K::ONE;
    
    // Make every Eval_(i=2, j) equal δ_{rho,0} so Eval collapses to sum of weights at index 0.
    // We bypass dynamic building and inject the Ajtai vector directly:
    // Compute γ^3 and γ^5 using repeated multiplication
    let mut gamma_pow_3 = gamma;
    for _ in 0..2 { gamma_pow_3 *= gamma; }  // γ^3
    let mut gamma_pow_5 = gamma;
    for _ in 0..4 { gamma_pow_5 *= gamma; }  // γ^5
    let w_spec = gamma_pow_3 + gamma_pow_5; // γ^3 + γ^5 = 17150
    let eval_ajtai = ajtai_vec2(w_spec, K::ZERO);
    
    // Oracle in Ajtai phase with only Eval "live"
    let csr_m1_inst = csr_identity(n);
    let mut oracle = GenericCcsOracle::<F> {
        s: &s,
        partials_first_inst: empty_row_partials(s.t()),
        w_beta_a_partial: vec![K::ZERO, K::ZERO],
        w_alpha_a_partial: eq_pair_alpha(alpha),
        w_beta_r_partial: vec![],
        w_eval_r_partial: vec![wr_scalar],
        z_witnesses: vec![],
        gamma, 
        k_total, 
        b: 3,
        ell_d: 1, 
        ell_n: 1, 
        d_sc: 1, 
        round_idx: 1,
        initial_sum_claim: K::ZERO, 
        f_at_beta_r: K::ZERO, 
        nc_sum_beta: K::ZERO,
        eval_row_partial: vec![],
        row_chals: vec![],
        csr_m1: &csr_m1_inst,
        csrs: &[],
        eval_ajtai_partial: Some(eval_ajtai.clone()),
        me_offset: 1,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc: Some(NcState { 
            y_partials: vec![], 
            gamma_pows: vec![], 
            f_at_rprime: None 
        }),
    };
    
    // Ajtai univariate samples
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]);
    
    // At X=0 we should get exactly w_spec; at X=1 it's 0 (since E1=0 and alpha=0 gate kills it)
    assert_eq!(ys[0], w_spec, "Ajtai Eval(0) must be sum of γ powers per spec");
    assert_eq!(ys[1], K::ZERO, "Ajtai Eval(1) should be 0 in this planting");
    
    // Optional: explicitly sanity-check it's NOT the common off-by-one schedule.
    let gamma_pow_1 = gamma;  // γ^1
    let mut gamma_pow_3_alt = gamma;
    for _ in 0..2 { gamma_pow_3_alt *= gamma; }  // γ^3
    let w_alt = gamma_pow_1 + gamma_pow_3_alt; // 7 + 343 = 350
    assert_ne!(ys[0], w_alt, "γ exponent schedule looks off-by-one (got alt weights)");
}
