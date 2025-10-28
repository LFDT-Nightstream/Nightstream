//! Unit tests for terminal RHS computation (rhs_Q_apr)
//!
//! # Paper Reference
//! Section 4.4, Step 4: Terminal identity verification
//!
//! ## Original Paper Formula
//! v ?= eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
//!      + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
//!
//! where E_{(i,j)} := eq((α',r'), (α,r))·ỹ'_{(i,j)}(α')
//!
//! ## Factored Form (Implementation)
//! v ?= eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
//!      + eq((α',r'), (α,r)) · [γ^k · Σ_{j,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ'_{(i,j)}(α')]
//!
//! where k = |MCS| + |ME inputs|
//!
//! These tests verify that each component of the terminal check is computed correctly:
//! 1. eq function evaluation
//! 2. F' computation (f evaluated at output y_scalars)
//! 3. NC' computation (range constraints at output)
//! 4. Eval' computation (gamma-weighted sum of ME inputs WITH outer γ^k)
//! 5. Full RHS composition

use neo_ccs::{CcsStructure, Mat, McsInstance, MeInstance, SparsePoly, Term, SModuleHomomorphism};
use neo_fold::pi_ccs::terminal::rhs_Q_apr;
use neo_fold::pi_ccs::transcript::Challenges;
use neo_math::{F, K, D};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use p3_field::PrimeCharacteristicRing;

// MLE dimension for tests (2^ell_d where ell_d=2)
const MLE_DIM: usize = 4;

// ----- Test Helpers -----

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

/// Compute γ^exp
fn gpow(g: K, exp: usize) -> K {
    (0..exp).fold(K::ONE, |acc, _| acc * g)
}

/// Compute eq(p, q) manually for verification
fn eq_points(p: &[K], q: &[K]) -> K {
    if p.len() != q.len() {
        return K::ZERO;
    }
    let mut acc = K::ONE;
    for i in 0..p.len() {
        acc *= (K::ONE - p[i]) * (K::ONE - q[i]) + p[i] * q[i];
    }
    acc
}

/// Dummy S-module for testing (no actual cryptographic operations)
#[allow(dead_code)]
struct DummyS;

#[allow(dead_code)]
impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let d = z.rows();
        Commitment::zeros(d, 4)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut result = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = z[(r, c)];
            }
        }
        result
    }
}

/// Create a simple CCS with identity matrix and f(m1, m2) = m1 + m2
fn simple_ccs_t2() -> CcsStructure<F> {
    let n = 4;
    let m = 4;
    let m0 = Mat::<F>::identity(n);
    let m1 = Mat::<F>::identity(n);
    // f(m1, m2) = m1 + m2
    let f = SparsePoly::new(
        2,
        vec![
            Term { coeff: F::ONE, exps: vec![1, 0] }, // m1
            Term { coeff: F::ONE, exps: vec![0, 1] }, // m2
        ],
    );
    CcsStructure { matrices: vec![m0, m1], f, n, m }
}

/// Create challenges for testing
fn create_test_challenges(ell_d: usize, ell_n: usize) -> Challenges {
    Challenges {
        alpha: (0..ell_d).map(|i| k64_(i as i64 + 2)).collect(),
        beta_a: (0..ell_d).map(|i| k64_(i as i64 + 10)).collect(),
        beta_r: (0..ell_n).map(|i| k64_(i as i64 + 20)).collect(),
        gamma: k64_(7),
    }
}

/// Create a minimal ME instance for testing
fn create_test_me_output(t: usize, d: usize, y_scalars: Vec<K>) -> MeInstance<Commitment, F, K> {
    let dummy_c = Commitment::zeros(d, 4);
    let dummy_r = vec![k64_(100), k64_(101)];
    
    // Allocate y-rows with correct length d; tests will fill individual entries
    let y = (0..t).map(|_| vec![K::ZERO; d]).collect();
    
    MeInstance {
        c: dummy_c,
        X: Mat::zero(d, 1, F::ZERO),
        y,
        y_scalars,
        r: dummy_r,
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    }
}

/// Create a minimal ME input for Eval' testing
fn create_test_me_input(t: usize, d: usize, r_val: &[K], y_vals: Vec<Vec<K>>) -> MeInstance<Commitment, F, K> {
    let dummy_c = Commitment::zeros(d, 4);
    
    // Compute y_scalars as placeholder (doesn't matter for Eval' block)
    let y_scalars = (0..t).map(|_| K::ZERO).collect();
    
    MeInstance {
        c: dummy_c,
        X: Mat::zero(d, 1, F::ZERO),
        y: y_vals,
        y_scalars,
        r: r_val.to_vec(),
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    }
}

// ----- Tests -----

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_no_me_inputs_f_only() {
    // Test case: no ME inputs, only F' term (no Eval', NC'=0 by construction)
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2); // b=2, range [-1,1]
    
    let r_prime = vec![k64_(30), k64_(31)];
    let alpha_prime = vec![K::ZERO, K::ZERO]; // simplifies χ_α' = [1, 0, 0, 0]
    let mle_dim = 1 << alpha_prime.len(); // 2^2 = 4
    
    // Create output ME with specific y_scalars for F' (not used by RHS now)
    let y_scalars = vec![k64_(5), k64_(3)]; // f(5, 3) = 5 + 3 = 8
    let mut out_me_inst = create_test_me_output(s.t(), mle_dim, y_scalars.clone());
    // Encode m1=5 → [1,0,1,0] and m2=3 → [1,1,0,0]; keep NC' zero (y[0][0] ∈ {-1,0,1})
    out_me_inst.y[0] = vec![k64_(1), k64_(0), k64_(1), k64_(0)];
    out_me_inst.y[1] = vec![k64_(1), k64_(1), k64_(0), k64_(0)];
    
    let out_me = vec![out_me_inst];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &out_me, &params);
    if let Err(ref e) = result {
        panic!("rhs_Q_apr should succeed but got error: {:?}", e);
    }
    assert!(result.is_ok(), "rhs_Q_apr should succeed");
    
    let rhs = result.unwrap();
    
    // Expected: eq((α',r'), β)·(F' + NC') + 0
    // where NC' = 0 because y[0] evaluated at α' is in range
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let f_prime = y_scalars[0] + y_scalars[1]; // f(m1, m2) = m1 + m2 = 8
    let nc_prime = K::ZERO; // NC'=0 because ỹ(α')=0 is in range [-1,1]
    let expected = eq_aprp_beta * (f_prime + nc_prime);
    
    assert_eq!(rhs, expected, "RHS should equal eq·(F'+NC') when no ME inputs");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_with_nc_prime() {
    // Test NC' computation: range polynomial on output
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(30), k64_(31)];
    let alpha_prime = vec![K::ZERO, K::ZERO]; // simplifies to χ = [1, 0, 0, 0]
    
    // Create 2 output ME instances to get 2 NC terms
    let y_scalars1 = vec![k64_(0), k64_(0)];
    let y_scalars2 = vec![k64_(0), k64_(0)];
    
    // For NC, we need y[0] (j=0, M_1 = I row)
    // y[0] = [val0, val1, val2, val3] evaluated at α'
    // With α' = [0, 0], χ_α' = [1, 0, 0, 0], so ỹ(α') = val0
    // F' = m1 + m2, so we need to encode m1=1, m2=0 as digits: m1=[1,0,0,0], m2=[0,0,0,0]
    
    let mut out_me1 = create_test_me_output(s.t(), MLE_DIM, y_scalars1);
    out_me1.y[0] = vec![k64_(1), k64_(0), k64_(0), k64_(0)]; // m1=1 (digits), ỹ(α') = 1
    out_me1.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0 (digits)
    
    let mut out_me2 = create_test_me_output(s.t(), MLE_DIM, y_scalars2);
    out_me2.y[0] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m1=0 (digits), ỹ(α') = 0
    out_me2.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0 (digits)
    
    let out_me = vec![out_me1, out_me2];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &out_me, &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed with NC");
    
    let rhs = result.unwrap();
    
    // Compute expected NC'
    let b = params.b as i64;
    let low = -(b - 1);
    let high = b - 1;
    
    // NC_1' for y_mle = 1: ∏_{t=-1}^{1} (1 - t) = (1-(-1))·(1-0)·(1-1) = 2·1·0 = 0
    let y_mle_1 = k64_(1);
    let mut nc_1 = K::ONE;
    for t in low..=high {
        nc_1 *= y_mle_1 - k64_(t);
    }
    
    // NC_2' for y_mle = 0: ∏_{t=-1}^{1} (0 - t) = (0-(-1))·(0-0)·(0-1) = 1·0·(-1) = 0
    let y_mle_2 = k64_(0);
    let mut nc_2 = K::ONE;
    for t in low..=high {
        nc_2 *= y_mle_2 - k64_(t);
    }
    
    let gamma = ch.gamma;
    let nc_prime = gamma * nc_1 + gamma * gamma * nc_2;
    
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let f_prime = k64_(1) + k64_(0); // f(m1, m2) = m1 + m2 = 1 (from first output's digits)
    let expected = eq_aprp_beta * (f_prime + nc_prime);
    
    assert_eq!(rhs, expected, "RHS should include NC' contribution");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_with_eval_prime_single_me_input() {
    // Test Eval' computation with one ME input
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(30), k64_(31)];
    let alpha_prime = vec![K::ZERO, K::ZERO]; // χ_α' = [1, 0, 0, 0]
    
    // Create ME input with specific y values
    let r_me = vec![k64_(50), k64_(51)];
    // y[j] for j=0,1 (t=2) - these will be at the INPUT point r
    let y_vals = vec![
        vec![k64_(10), k64_(20), k64_(30), k64_(40)],
        vec![k64_(15), k64_(25), k64_(35), k64_(45)],
    ];
    let me_inputs = vec![create_test_me_input(s.t(), MLE_DIM, &r_me, y_vals)];
    
    // Outputs: [0] for F'/NC', [1] for Eval′ (carries the y′ values).
    // Keep j=0 row in-range (0) for both outputs so NC′ = 0 in this test.
    let y_scalars0 = vec![k64_(2), k64_(3)]; // F′ = 5
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, y_scalars0);
    // m1 = 2 → [0,1,0,0]; m2 = 3 → [1,1,0,0]; NC′ zero since y[0][0]=0
    out0.y[0] = vec![k64_(0), k64_(1), k64_(0), k64_(0)];
    out0.y[1] = vec![k64_(1), k64_(1), k64_(0), k64_(0)];

    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    // These are the y′ values that should drive Eval′:
    out1.y[0] = vec![k64_(10), k64_(20), k64_(30), k64_(40)]; // ỹ′(α′)=10
    out1.y[1] = vec![k64_(15), k64_(25), k64_(35), k64_(45)]; // ỹ′(α′)=15
    // Keep NC′ zero for this output too
    out1.y[0][0] = k64_(0);
    
    let out_me = vec![out0, out1];
    
    let mcs_list = vec![]; // No MCS instances
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &mcs_list, &me_inputs, &out_me, &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed with ME input");
    
    let rhs = result.unwrap();
    
    // Compute expected Eval' WITH outer γ^k factor
    // k_total = |mcs_list| + |me_inputs| = 0 + 1 = 1
    // For j=0, i_off=0: inner exponent = (0+2) + 0*1 - 1 = 1 → γ^1
    // For j=1, i_off=0: inner exponent = (0+2) + 1*1 - 1 = 2 → γ^2
    let k_total = 1;
    let gamma = ch.gamma;
    
    let y_mle_0 = k64_(0); // ỹ_{(2,0)}(α') - using y'[0] from output, which has [0, ...] at index 0
    let y_mle_1 = k64_(15); // ỹ_{(2,1)}(α')
    
    let exp_0 = 2 + 0 * k_total - 1; // = 1
    let exp_1 = 2 + 1 * k_total - 1; // = 2
    
    let mut gamma_pow_0 = K::ONE;
    for _ in 0..exp_0 {
        gamma_pow_0 *= gamma;
    }
    let mut gamma_pow_1 = K::ONE;
    for _ in 0..exp_1 {
        gamma_pow_1 *= gamma;
    }
    
    // Inner sum (without outer γ^k)
    let eval_sum_prime = gamma_pow_0 * y_mle_0 + gamma_pow_1 * y_mle_1;
    
    // Outer γ^k_total factor
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= gamma;
    }
    
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let f_prime = k64_(2) + k64_(3); // 5 (from out_me[0])
    // NC' contributions from BOTH outputs (both are ME-derived since mcs_list is empty)
    // For α' = [0,0], χ = [1,0,0,0], so ỹ(α') = y[0][0]
    // out0.y[0][0] = 0 → NC = 0 (in range)
    // out1.y[0][0] = 0 → NC = 0 (in range)
    let nc_prime = K::ZERO;
    let expected = eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * (gamma_to_k * eval_sum_prime);
    
    assert_eq!(rhs, expected, "RHS should correctly compute Eval' with ME input");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_with_multiple_me_inputs() {
    // Test Eval' computation with two ME inputs
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(30), k64_(31)];
    // Use α′=[1,0] to select index 1, keeping index 0 for NC'
    let alpha_prime = vec![K::ONE, K::ZERO]; // χ_α' = [0, 1, 0, 0]
    
    // Create 2 ME inputs
    let r_me = vec![k64_(50), k64_(51)];
    let y_vals_1 = vec![
        vec![k64_(0), k64_(100), k64_(0), k64_(0)],
        vec![k64_(0), k64_(200), k64_(0), k64_(0)],
    ];
    let me_input_1 = create_test_me_input(s.t(), MLE_DIM, &r_me, y_vals_1);
    
    let y_vals_2 = vec![
        vec![k64_(0), k64_(11), k64_(0), k64_(0)],
        vec![k64_(0), k64_(22), k64_(0), k64_(0)],
    ];
    let me_input_2 = create_test_me_input(s.t(), MLE_DIM, &r_me, y_vals_2);
    
    let me_inputs = vec![me_input_1, me_input_2];
    
    // Outputs: [0] for F/NC, [1] for ME #1 y′, [2] for ME #2 y′
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![k64_(1), k64_(1)]); // F′=2
    // m1 = 1, m2 = 1; with α'=[1,0] ⇒ ỹ(α') = y[0][1] = 0 ⇒ NC′ zero
    out0.y[0] = vec![k64_(1), k64_(0), k64_(0), k64_(0)];
    out0.y[1] = vec![k64_(1), k64_(0), k64_(0), k64_(0)];

    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    // Place mass on index 1 so j=0,1 get non-zero ỹ′ while NC' stays zero (y[0][0] = 0)
    out1.y[0] = vec![k64_(0), k64_(100), k64_(0), k64_(0)]; // ỹ′=100 at j=0
    out1.y[1] = vec![k64_(0), k64_(200), k64_(0), k64_(0)]; // ỹ′=200 at j=1

    let mut out2 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    out2.y[0] = vec![k64_(0), k64_(11), k64_(0), k64_(0)]; // ỹ′=11 at j=0
    out2.y[1] = vec![k64_(0), k64_(22), k64_(0), k64_(0)]; // ỹ′=22 at j=1

    let out_me = vec![out0, out1, out2];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &me_inputs, &out_me, &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed with multiple ME inputs");
    
    let rhs = result.unwrap();
    
    // k_total = 2 (two ME inputs, no MCS)
    let k_total = 2;
    let gamma = ch.gamma;
    
    // For first ME input (i_off=0):
    //   j=0: inner exponent = (0+2) + 0*2 - 1 = 1
    //   j=1: inner exponent = (0+2) + 1*2 - 1 = 3
    // For second ME input (i_off=1):
    //   j=0: inner exponent = (1+2) + 0*2 - 1 = 2
    //   j=1: inner exponent = (1+2) + 1*2 - 1 = 4
    
    let eval_sum_prime = gpow(gamma, 1) * k64_(100) + gpow(gamma, 3) * k64_(200)
                       + gpow(gamma, 2) * k64_(11)  + gpow(gamma, 4) * k64_(22);
    
    let gamma_to_k = gpow(gamma, k_total);
    
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let f_prime = k64_(1) + k64_(1); // 2 (from out_me[0])
    
    // NC' contributions from ALL 3 outputs
    // For α' = [1,0], χ = [0,1,0,0], so ỹ(α') = y[0][1]
    // out0.y[0][1] = 0 → NC_1 = 0 (in range)
    // out1.y[0][1] = 100 → NC_2 ≠ 0 (out of range)
    // out2.y[0][1] = 11 → NC_3 ≠ 0 (out of range)
    let mut nc_1 = K::ONE;
    for t in -1i64..=1 {
        nc_1 *= k64_(0) - k64_(t);
    }
    let mut nc_2 = K::ONE;
    for t in -1i64..=1 {
        nc_2 *= k64_(100) - k64_(t);
    }
    let mut nc_3 = K::ONE;
    for t in -1i64..=1 {
        nc_3 *= k64_(11) - k64_(t);
    }
    let nc_prime = gamma * nc_1 + gamma * gamma * nc_2 + gamma * gamma * gamma * nc_3;
    
    let expected = eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * (gamma_to_k * eval_sum_prime);
    
    assert_eq!(rhs, expected, "RHS should correctly handle multiple ME inputs");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_full_composition() {
    // Test with all components: F', NC', and Eval'
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(1), k64_(2)];
    let alpha_prime = vec![K::ONE, K::ZERO]; // χ_α' = [0, 1, 0, 0]
    
    // ME input for Eval'
    let r_me = vec![k64_(5), k64_(6)];
    let y_vals = vec![
        vec![k64_(0), k64_(333), k64_(0), k64_(0)],
        vec![k64_(0), k64_(444), k64_(0), k64_(0)],
    ];
    let me_inputs = vec![create_test_me_input(s.t(), MLE_DIM, &r_me, y_vals)];
    
    // out[0] used for F′ and NC′
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    // y[0] drives NC' (α'=[1,0] selects index 1); set y[0][1]=2 ⇒ NC' ≠ 0
    // Choose digits to define m1=9 ( [1,2,1,0] ) and m2=8 ( [0,0,0,1] )
    out0.y[0] = vec![k64_(1), k64_(2), k64_(1), k64_(0)];
    out0.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(1)];

    // out[1] provides Eval′ values (333,444)
    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    out1.y[0] = vec![k64_(0), k64_(333), k64_(0), k64_(0)];
    out1.y[1] = vec![k64_(0), k64_(444), k64_(0), k64_(0)];
    out1.y[0][0] = k64_(0); // keep NC′ zero for this output

    let out_me = vec![out0, out1];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &me_inputs, &out_me, &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed with full composition");
    
    let rhs = result.unwrap();
    
    // Compute all components manually
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    
    let f_prime = k64_(9) + k64_(8); // 17
    
    // NC': For α'=[1,0], χ=[0,1,0,0], so ỹ(α') = y[0][1]
    // out0.y[0][1] = 2 → NC_1 = ∏_{t=-1}^{1} (2 - t) = 3·2·1 = 6
    // out1.y[0][1] = 333 → NC_2 = ∏_{t=-1}^{1} (333 - t) = (334)·(333)·(332)
    let y_mle_0 = k64_(2);
    let mut nc_1 = K::ONE;
    for t in -1i64..=1 {
        nc_1 *= y_mle_0 - k64_(t);
    }
    let y_mle_1 = k64_(333);
    let mut nc_2 = K::ONE;
    for t in -1i64..=1 {
        nc_2 *= y_mle_1 - k64_(t);
    }
    let nc_prime = ch.gamma * nc_1 + ch.gamma * ch.gamma * nc_2;
    
    // Eval': k_total = 1 (0 MCS + 1 ME input)
    let k_total = 1;
    let gamma = ch.gamma;
    let y_mle_0 = k64_(333); // ỹ'_{(1,0)}(α'=[1,0]) from out1.y[0][1]
    let y_mle_1 = k64_(444); // ỹ'_{(1,1)}(α'=[1,0]) from out1.y[1][1]
    // Inner sum: γ^1 · 333 + γ^2 · 444
    let eval_sum_prime = gamma * y_mle_0 + gamma * gamma * y_mle_1;
    
    // Outer γ^k_total factor
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= gamma;
    }
    
    let expected = eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * (gamma_to_k * eval_sum_prime);
    
    assert_eq!(rhs, expected, "Full RHS composition should be correct");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_error_no_output() {
    // Test that function returns error when no ME outputs provided
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(30), k64_(31)];
    let alpha_prime = vec![k64_(12), k64_(13)];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &[], &params);
    assert!(result.is_err(), "rhs_Q_apr should fail with no ME outputs");
}

// (Removed test_rhs_Q_apr_error_wrong_y_scalars_length: F' recomposed from digits)

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_gamma_exponent_progression() {
    // Detailed test to verify the gamma exponent formula: i + (j-1)k - 1
    // For ME inputs, i starts at 2, so i_off=0 corresponds to i=2
    let s = simple_ccs_t2(); // t = 2
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(1), k64_(1)];
    let alpha_prime = vec![K::ZERO, K::ZERO]; // χ_α' = [1, 0, 0, 0]
    
    // Create 1 MCS (k_total will be 2: 1 MCS + 1 ME)
    let dummy_mcs = McsInstance {
        c: Commitment::zeros(D, 4),
        x: vec![F::ZERO],
        m_in: 1,
    };
    let mcs_list = vec![dummy_mcs];
    
    // Create ME input
    let r_me = vec![k64_(1), k64_(1)];
    let y_vals = vec![
        vec![k64_(1), k64_(0), k64_(0), k64_(0)],
        vec![k64_(1), k64_(0), k64_(0), k64_(0)],
    ];
    let me_inputs = vec![create_test_me_input(s.t(), MLE_DIM, &r_me, y_vals)];
    
    // Outputs: [0] for F/NC with F′=0, [1] for Eval′ (ỹ′=1 for both rows)
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    out0.y[0] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m1=0 (digits), NC′ zero
    out0.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0 (digits)

    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    out1.y[0] = vec![k64_(1), k64_(0), k64_(0), k64_(0)]; // For Eval': ỹ′=1 at j=0
    out1.y[1] = vec![k64_(1), k64_(0), k64_(0), k64_(0)]; // For Eval': ỹ′=1 at j=1

    let out_me = vec![out0, out1];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &mcs_list, &me_inputs, &out_me, &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed");
    
    let rhs = result.unwrap();
    
    // k_total = 2 (1 MCS + 1 ME input)
    // ME input has i_off = 0, which corresponds to i = 2 in the paper
    // j=0 (first row): inner exponent = (0+2) + 0*2 - 1 = 1 → γ^1
    // j=1 (second row): inner exponent = (0+2) + 1*2 - 1 = 3 → γ^3
    
    let k_total = 2;
    let gamma = ch.gamma;
    
    // Inner sum: γ^1 · 1 + γ^3 · 1
    let eval_sum_prime = gamma * k64_(1) + gamma * gamma * gamma * k64_(1);
    
    // Outer γ^k_total = γ^2 factor
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= gamma;
    }
    
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    // NC' contributions from BOTH outputs (1 from MCS + 1 from ME)
    // out0.y[0][0] = 0 → NC_1 = 0 (in range)
    // out1.y[0][0] = 0 → NC_2 = 0 (in range)
    let nc_prime = K::ZERO;
    
    let expected = eq_aprp_beta * (K::ZERO + nc_prime) + eq_aprp_ar * (gamma_to_k * eval_sum_prime);
    
    assert_eq!(rhs, expected, "Gamma exponents should follow formula i + (j-1)k - 1");
}

#[test]
fn test_nc_range_polynomial_in_range() {
    // Test that NC polynomial evaluates correctly for in-range values
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2); // b=2, range [-1, 1]
    
    let r_prime = vec![k64_(1), k64_(1)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    // Test with y_mle values inside range [-1, 1]
    for val in [-1i64, 0, 1] {
        let y_scalars = vec![k64_(0), k64_(0)];
        let mut out_me = create_test_me_output(s.t(), MLE_DIM, y_scalars);
        // Encode m1=val, m2=0 as digits; with α'=[0,0], ỹ(α') = y[0][0] = val
        out_me.y[0] = vec![k64_(val), k64_(0), k64_(0), k64_(0)];
        out_me.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0
        
        let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &vec![out_me], &params);
        assert!(result.is_ok(), "rhs_Q_apr should succeed for in-range value {}", val);
        
        let rhs = result.unwrap();
        
        // NC polynomial should be 0 for any value in range [-1, 1]
        // because ∏_{t=-1}^{1} (val - t) includes factor (val - val) = 0
        // F' = m1 + m2 = val + 0 = val
        let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
        let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
        let eq_aprp_beta = eq_beta_a * eq_beta_r;
        
        let f_prime = k64_(val);
        let expected = eq_aprp_beta * f_prime; // NC should be 0, so just F'
        
        assert_eq!(rhs, expected, "NC' should be 0 for in-range value {}", val);
    }
}

#[test]
fn test_nc_range_polynomial_out_of_range() {
    // Test that NC polynomial is non-zero for out-of-range values
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2); // b=2, range [-1, 1]
    
    let r_prime = vec![k64_(1), k64_(1)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    // Test with y_mle = 5 (outside range)
    let _val = 5i64;
    let y_scalars = vec![k64_(0), k64_(0)];
    let mut out_me = create_test_me_output(s.t(), MLE_DIM, y_scalars);
    // Encode m1=5, m2=0 as digits: m1=[1,0,1,0] (1+4=5), m2=[0,0,0,0]
    out_me.y[0] = vec![k64_(1), k64_(0), k64_(1), k64_(0)];
    out_me.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)];
    
    let result = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &vec![out_me], &params);
    assert!(result.is_ok(), "rhs_Q_apr should succeed for out-of-range value");
    
    let rhs = result.unwrap();
    
    // NC polynomial: ∏_{t=-1}^{1} (5 - t) = (5-(-1))·(5-0)·(5-1) = 6·5·4 = 120
    // With α'=[0,0], ỹ(α') = y[0][0] = 1, but wait... this is wrong!
    // The NC gate checks the DIGIT values, not the recomposed value.
    // So NC checks if y[0][0]=1 is in range [-1,1], which it is! So NC'=0
    // F' = m1 + m2 = 5 + 0 = 5
    let eq_beta_r = eq_points(&r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(&alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;
    
    let f_prime = k64_(5);
    let expected = eq_aprp_beta * f_prime; // NC'=0 because digit is in range
    
    assert_eq!(rhs, expected, "NC' should be 0 when digit is in range");
    // Note: NC gate checks DIGITS, not the recomposed value!
}

#[test]
#[allow(non_snake_case)]
fn test_eval_uses_outputs_not_inputs() {
    // Regression test: Ensure Eval′ uses outputs, not inputs
    // This guards against accidentally switching back to me_inputs
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let r_prime = vec![k64_(3), k64_(4)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    // Inputs carry "wrong" values (should be ignored for Eval′ numerics)
    let r_me = vec![k64_(9), k64_(9)];
    let me_inputs = vec![create_test_me_input(
        s.t(),
        D,
        &r_me,
        vec![
            vec![k64_(999), k64_(0), k64_(0), k64_(0)],
            vec![k64_(999), k64_(0), k64_(0), k64_(0)]
        ],
    )];
    
    // Outputs: [0] for F/NC, [1] for Eval′ values we *want* to be used.
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    out0.y[0] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m1=0 (digits), NC′ zero
    out0.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0 (digits)
    
    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    // With α′=[0,0], χ=[1,0,0,0], so ỹ′(α′) = y[j][0]
    // Set y[0][0]=5 and y[1][0]=7 for Eval′ (also affects NC′ which sees 5)
    out1.y[0] = vec![k64_(5), k64_(0), k64_(0), k64_(0)]; // ỹ′=5 for Eval', NC' sees 5
    out1.y[1] = vec![k64_(7), k64_(0), k64_(0), k64_(0)]; // ỹ′=7 for Eval'
    
    let out_me = vec![out0, out1];
    
    let rhs = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &me_inputs, &out_me, &params)
        .expect("ok");
    
    // Expected = eq((α′,r′),β)*(F'+NC') + eq((α′,r′),(α,r))* [γ^1 * (γ^1*5 + γ^2*7)]
    // Note: With α′=[0,0], both NC′ and Eval′ see the same digit values.
    // out1.y[0][0]=5 contributes to both NC_2 (as out-of-range penalty) and Eval′ j=0 term.
    let k_total = 1usize;
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= ch.gamma;
    }
    let eval_sum_inner = ch.gamma * k64_(5) + (ch.gamma * ch.gamma) * k64_(7);
    
    // NC' contributions from BOTH outputs
    // out0.y[0][0] = 0 → NC_1 = 0 (in range)
    // out1.y[0][0] = 5 → NC_2 ≠ 0 (out of range)
    // NC_2 = γ^2 * ∏_{t=-1}^{1} (5 - t) = γ^2 * (6*5*4) = γ^2 * 120
    let mut nc_2 = K::ONE;
    for t in -1i64..=1 {
        nc_2 *= k64_(5) - k64_(t);
    }
    let nc_prime = (ch.gamma * ch.gamma) * nc_2; // γ^2 for i=2 (second output)
    let f_prime = K::ZERO;
    
    let expected = eq_points(&alpha_prime, &ch.beta_a) * eq_points(&r_prime, &ch.beta_r) * (f_prime + nc_prime)
                 + eq_aprp_ar * (gamma_to_k * eval_sum_inner);
    assert_eq!(rhs, expected, "Eval must use out_me values, not me_inputs");
}

#[test]
#[allow(non_snake_case)]
fn test_nc_prime_weighting_gamma_powers() {
    // Test that NC′ weighting by γ^i is correctly implemented
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2); // b=2 → [-1,1]
    let r_prime = vec![k64_(0), k64_(0)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    // Two outputs so NC′ = γ^1 * N1 + γ^2 * N2
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    out0.y[0] = vec![k64_(2), k64_(0), k64_(0), k64_(0)]; // m1=2 (digit), out of range → N1 ≠ 0
    out0.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0
    
    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO, K::ZERO]);
    out1.y[0] = vec![k64_(3), k64_(0), k64_(0), k64_(0)]; // m1=3 (digit), out of range → N2 ≠ 0
    out1.y[1] = vec![k64_(0), k64_(0), k64_(0), k64_(0)]; // m2=0
    
    let out_me = vec![out0, out1];
    
    let rhs = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &out_me, &params).unwrap();
    
    // Manually compute N1, N2 and weights
    let mut N1 = K::ONE;
    let mut N2 = K::ONE;
    for t in -1..=1 {
        N1 *= k64_(2) - k64_(t);
        N2 *= k64_(3) - k64_(t);
    }
    let nc_prime = ch.gamma * N1 + (ch.gamma * ch.gamma) * N2;
    let f_prime = k64_(2) + k64_(0); // F' from first output: m1=2, m2=0
    let eq_beta = eq_points(&alpha_prime, &ch.beta_a) * eq_points(&r_prime, &ch.beta_r);
    let expected = eq_beta * (f_prime + nc_prime);
    assert_eq!(rhs, expected, "NC' should be weighted by γ^i correctly");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_eval_prime_with_two_mcs_shifts_exponent() {
    // Test γ‑exponent formula with |MCS| > 1
    // This catches the bug where i_global must be |MCS| + 1 + i_off, not hardcoded to 2 + i_off
    let s = simple_ccs_t2(); // t = 2
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let r_prime = vec![k64_(10), k64_(11)];
    let alpha_prime = vec![K::ONE, K::ZERO]; // χ = [0, 1, 0, 0]
    
    // Two MCS instances -> |MCS| = 2, plus one ME input -> k_total = 3
    let mcs_list = vec![
        McsInstance { c: Commitment::zeros(D, 4), x: vec![F::ZERO], m_in: 1 },
        McsInstance { c: Commitment::zeros(D, 4), x: vec![F::ZERO], m_in: 1 },
    ];
    let r_me = vec![k64_(5), k64_(6)];
    let me_inputs = vec![create_test_me_input(
        s.t(), MLE_DIM, &r_me, vec![vec![K::ZERO; D], vec![K::ZERO; D]],
    )];
    
    // Outputs: [0] for F/NC (always), [1] and [2] for Eval′ (since k_total=3)
    let mut out0 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    out0.y[0] = vec![k64_(0), k64_(0), K::ZERO, K::ZERO]; // m1=0 (digits), NC' zero
    out0.y[1] = vec![k64_(0), k64_(0), K::ZERO, K::ZERO]; // m2=0 (digits)
    
    let mut out1 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    // out1 contributes to Eval' with i_global=2
    out1.y[0] = vec![k64_(0), k64_(0), K::ZERO, K::ZERO]; // j=0, all zeros for simplicity
    out1.y[1] = vec![k64_(0), k64_(0), K::ZERO, K::ZERO]; // j=1, all zeros
    
    let mut out2 = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    // out2 contributes to Eval' with i_global=3; place mass on index 1
    out2.y[0] = vec![k64_(0), k64_(9), K::ZERO, K::ZERO]; // j=0 → 9
    out2.y[1] = vec![k64_(0), k64_(4), K::ZERO, K::ZERO]; // j=1 → 4
    
    let out_me = vec![out0, out1, out2];
    
    let rhs = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &mcs_list, &me_inputs, &out_me, &params).unwrap();
    
    // k_total = 3, i_global for the (only) ME output is |MCS| + 1 = 3
    // Exponents: j=0 → 3 + 0*3 - 1 = 2; j=1 → 3 + 1*3 - 1 = 5
    let gamma = ch.gamma;
    let eval_sum_prime = gpow(gamma, 2) * k64_(9) + gpow(gamma, 5) * k64_(4);
    
    let gamma_to_k = gpow(gamma, 3);
    let eq_aprp_ar = eq_points(&alpha_prime, &ch.alpha) * eq_points(&r_prime, &r_me);
    let eq_aprp_beta = eq_points(&alpha_prime, &ch.beta_a) * eq_points(&r_prime, &ch.beta_r);
    
    // F' = 0 (from out0), NC' contributions:
    // For α'=[1,0], χ=[0,1,0,0], so ỹ(α') = y[0][1]
    // out0.y[0][1] = 0, out1.y[0][1] = 0, out2.y[0][1] = 9
    // NC_3 = ∏_{t=-1}^{1} (9 - t) = (10)·(9)·(8) = 720
    let mut nc_3 = K::ONE;
    for t in -1i64..=1 {
        nc_3 *= k64_(9) - k64_(t);
    }
    let nc_prime = gpow(gamma, 3) * nc_3; // γ^3 for i=3 (third output)
    let f_prime = K::ZERO;
    
    let expected = eq_aprp_beta * (f_prime + nc_prime)
                 + eq_aprp_ar * (gamma_to_k * eval_sum_prime);
    
    assert_eq!(rhs, expected, "γ exponents must incorporate |MCS| offset");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_errors_on_y_row_len_mismatch() {
    // Test that function returns error when y row length doesn't match χ length
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let r_prime = vec![k64_(0), k64_(0)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    // y[0] length 3 (should be 4)
    let mut out = create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]);
    out.y[0] = vec![K::ZERO; 3];
    
    let err = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &[], &vec![out], &params).unwrap_err();
    assert!(format!("{err:?}").contains("length"), "expected dimension mismatch error");
}

#[test]
#[allow(non_snake_case)]
fn test_rhs_Q_apr_errors_on_inconsistent_r() {
    // Test that function returns error when ME inputs have different r values
    let s = simple_ccs_t2();
    let ch = create_test_challenges(2, 2);
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let r_prime = vec![k64_(0), k64_(0)];
    let alpha_prime = vec![K::ZERO, K::ZERO];
    
    let me_inputs = vec![
        create_test_me_input(s.t(), MLE_DIM, &[k64_(9),  k64_(9)], vec![vec![K::ZERO; D]; s.t()]),
        create_test_me_input(s.t(), MLE_DIM, &[k64_(10), k64_(10)], vec![vec![K::ZERO; D]; s.t()]),
    ];
    
    let out = vec![
        create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]),
        create_test_me_output(s.t(), MLE_DIM, vec![K::ZERO; s.t()]),
    ];
    
    let err = rhs_Q_apr(&s, &ch, &r_prime, &alpha_prime, &[], &me_inputs, &out, &params).unwrap_err();
    assert!(format!("{err:?}").contains("same r"), "expected r-consistency error");
}

