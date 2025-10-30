//! Oracle Ajtai-phase semantics test: ensure Option B excludes NC in Ajtai rounds
//!
//! This test constructs a minimal GenericCcsOracle already at the Ajtai phase
//! with F(r') = 0 and no Eval contribution. We set NC Ajtai partials to non-zero
//! values. Under the intended Option B semantics, the Ajtai-phase evaluation
//! should exclude NC and therefore return zero. The current implementation
//! incorrectly includes NC again in the Ajtai phase, so this test will fail
//! until that is fixed.

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold::pi_ccs::oracle::{GenericCcsOracle, NcState};
use neo_fold::pi_ccs::precompute::MlePartials;
use neo_fold::pi_ccs::sparse_matrix::Csr;
use neo_fold::sumcheck::RoundOracle;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

#[test]
#[ignore = "Paper-faithful mode: Ajtai phase includes NC/Eval; this Option B test is disabled"]
fn oracle_ajtai_phase_excludes_nc_option_b() {
    // Minimal CCS: n=1, m=1, f ≡ 0 (so F(r') = 0 always)
    let n = 1usize;
    let m = 1usize;
    let m0 = Mat::<F>::zero(n, m, F::ZERO);
    let f_zero = SparsePoly::new(1, vec![]); // f ≡ 0
    let s = CcsStructure { matrices: vec![m0.clone()], f: f_zero, n, m };

    // CSR for M_1 (identity-zero here). Not used because we pre-fill NC state.
    let csr_m1 = Csr { rows: n, cols: m, indptr: vec![0, 0], indices: vec![], data: vec![] };
    let csrs = vec![csr_m1.clone()];

    // Ajtai-only phase: ell_n = 0 (no row rounds), ell_d = 2 ⇒ 4 Ajtai points
    let ell_n = 0usize;
    let ell_d = 2usize;
    let d_sc = 6usize;

    // Equality gates: set Ajtai gates to 1s so gating doesn't mask contributions
    let w_beta_a_partial = vec![K::ONE; 1 << ell_d];
    let w_alpha_a_partial = vec![K::ONE; 1 << ell_d];
    let w_beta_r_partial = vec![];      // wr_scalar = 1 when empty
    let w_eval_r_partial = vec![];      // no eval gate in rows
    let eval_row_partial = vec![];      // no Eval contribution

    // NC Ajtai partials: non-in-range values ensure NC ≠ 0 if included
    // With b=2 (range {-1,0,1}), values like 2,3,4,5 yield non-zero range polynomials
    let y_partials = vec![vec![
        K::from(F::from_u64(2)),
        K::from(F::from_u64(3)),
        K::from(F::from_u64(4)),
        K::from(F::from_u64(5)),
    ]];
    let gamma_pows = vec![K::ONE];
    let nc = Some(NcState { y_partials, gamma_pows, f_at_rprime: Some(K::ZERO) });

    // Build oracle already positioned at Ajtai phase (round_idx >= ell_n)
    let mut oracle = GenericCcsOracle {
        s: &s,
        partials_first_inst: MlePartials { s_per_j: vec![vec![K::ZERO; 1]] },
        w_beta_a_partial,
        w_alpha_a_partial,
        w_beta_r_partial: w_beta_r_partial.clone(),
        w_eval_r_partial,
        z_witnesses: vec![],
        gamma: K::ONE,
        k_total: 1,
        b: 2,
        ell_d,
        ell_n,
        d_sc,
        // Start at Ajtai phase since there are zero row rounds
        round_idx: 0,
        initial_sum_claim: K::ZERO,
        f_at_beta_r: K::ZERO,
        nc_sum_beta: K::ZERO,
        eval_row_partial,
        row_chals: vec![],
        csr_m1: &csr_m1,
        csrs: &csrs,
        eval_ajtai_partial: None,
        me_offset: 0,
        nc_y_matrices: vec![],
        nc_row_gamma_pows: vec![],
        nc_state: nc,
    };

    // Ajtai phase begins immediately (ell_n=0). Evaluate at X=0 and X=1.
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]);

    // Option B semantics: Ajtai phase excludes NC; with F=0 and no Eval, result must be 0
    // This will FAIL under the current implementation (reveals double-count of NC).
    assert_eq!(ys[0], K::ZERO, "Ajtai phase must exclude NC (X=0)");
    assert_eq!(ys[1], K::ZERO, "Ajtai phase must exclude NC (X=1)");
}
