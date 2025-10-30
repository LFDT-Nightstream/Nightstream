//! Focused unit tests for row-pair mapping and NC nonlinearity.
//!
//! These tests are small and precise to help localize issues:
//! - Row pair mapping across rounds (formula vs. baseline dynamic-stride layout)
//! - NC nonlinearity sanity: product( interpolate(y) ) != interpolate( product(y) )
//! - Ajtai NC exact-univariate vs public hypercube (paper-faithful)
//! - Oracle regression: oracle Ajtai round must equal exact univariate & hypercube

use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

// ---------- helpers ----------

/// Compute (j0, j1) for the k-th folded pair at row round `round_idx`.
/// This mirrors the mapping used in the oracle for NC row evaluation.
fn row_pair_indices(round_idx: usize, k: usize) -> (usize, usize) {
    let stride = 1usize << round_idx;              // 2^i
    let j0 = (k & (stride - 1)) + ((k >> round_idx) << (round_idx + 1));
    (j0, j0 + stride)
}

/// Compute dims **without** calling extension-policy (to avoid policy failures in unit tests).
fn test_dims(n_rows: usize, b_base: u32, s_max_deg: usize) -> (usize, usize, usize, usize) {
    let d_pad = neo_math::D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    let n_pad = n_rows.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;
    let ell = ell_d + ell_n;
    // replica of context.rs formula, but local
    let d_sc = core::cmp::max(
        s_max_deg + 1,                // deg(f)+1
        core::cmp::max(2, 2 * (b_base as usize) + 2),
    );
    (ell_d, ell_n, ell, d_sc)
}

// ---------- tests ----------

#[test]
fn row_pair_mapping_round0_and_round1() {
    // n = 4 rows → ell_n = 2; full row domain indices = {0,1,2,3}
    let full_rows = 4usize;

    // Round 0: stride=1 → pairs should be (0,1), (2,3)
    let round0_pairs: Vec<(usize, usize)> = (0..(full_rows >> 1))
        .map(|k| row_pair_indices(0, k))
        .collect();
    assert_eq!(round0_pairs, vec![(0, 1), (2, 3)]);

    // Baseline dynamic-stride enumeration for round 0
    let stride0 = 1usize;
    let block0 = 2 * stride0;
    let groups0 = full_rows / block0; // 4 / 2 = 2 groups
    let mut baseline0 = Vec::new();
    for g in 0..groups0 {
        let base = g * block0;
        for off in 0..stride0 {
            let j0 = base + off;
            let j1 = base + off + stride0;
            baseline0.push((j0, j1));
        }
    }
    assert_eq!(round0_pairs, baseline0);

    // Round 1: stride=2 → (0,2) in the compressed-view
    let round1_pairs: Vec<(usize, usize)> = (0..(full_rows >> 2))
        .map(|k| row_pair_indices(1, k))
        .collect();
    assert_eq!(round1_pairs, vec![(0, 2)]);

    // Baseline dynamic-stride view for round 1:
    let stride1 = 1usize << 1; // 2
    let block1 = 2 * stride1;  // 4
    let groups1 = full_rows / block1; // 1
    let mut baseline1 = Vec::new();
    for g in 0..groups1 {
        let base = g * block1; // 0
        for off in 0..stride1 {
            let j0 = base + off;           // 0, 1
            let j1 = base + off + stride1; // 2, 3
            baseline1.push((j0, j1));
        }
    }
    assert_eq!(round1_pairs[0], baseline1[0]);
}

/// Range polynomial N_b(x) = ∏_{t=-(b-1)}^{b-1} (x - t) is nonlinear; therefore:
/// N( (1-X)*y0 + X*y1 ) != (1-X)*N(y0) + X*N(y1) as polynomials in X, in general.
#[test]
fn nc_nonlinearity_interpolation_vs_product() {
    // Use b=2 (range [-1, 1]) and pick values away from roots {-1,0,1}
    // so the check is meaningful: choose y0=2, y1=4 and X=2.
    let b: i64 = 2;
    let y0 = K::from(F::from_u64(2));
    let y1 = K::from(F::from_u64(4));
    let x  = K::from(F::from_u64(2));

    let nb = |y: K| {
        let mut acc = K::ONE;
        for t in (-(b - 1))..=(b - 1) { acc *= y - K::from(F::from_i64(t)); }
        acc
    };

    let yi = (K::ONE - x) * y0 + x * y1;
    let lhs = nb(yi);
    let rhs = (K::ONE - x) * nb(y0) + x * nb(y1);

    assert!(lhs != rhs, "NC nonlinearity violated: lhs == rhs for chosen inputs");
}

// === Ajtai NC univariate vs hypercube (paper-faithful, no oracle) ===
// This never hits the oracle; it only checks that the exact Ajtai branch sum equals the
// public hypercube NC sum when β_r = r'.

use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, SModuleHomomorphism};
use neo_fold::pi_ccs::precompute;
use neo_fold::pi_ccs::eq_weights::{HalfTableEq, RowWeight};
use neo_fold::pi_ccs::nc_constraints::compute_nc_hypercube_sum;
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[test]
fn ajtai_nc_univariate_equals_hypercube_b3() {
    // All-zero R1CS is fine; we need a valid structure with M1 = I after identity-first.
    let rows = 4usize;
    let cols = 4usize;
    let a = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let b = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let c = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let s_raw = r1cs_to_ccs(a, b, c);
    let s = s_raw.ensure_identity_first().expect("identity-first");

    // Params: keep s=2; set b=3 locally; we won't call extension policy.
    let base_params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let params = NeoParams { b: 3, ..base_params };
    assert_eq!(params.b, 3);

    // Witness with non-trivial base-3 balanced digits (still within [-2,2]):
    // z_full = [5, 0, 0, 0] gives digits like (-1, 2, 0, ...)
    let z_full = vec![F::from_u64(5), F::ZERO, F::ZERO, F::ZERO];
    let m_in = 1usize;

    // Decompose Z in row-major (d × m)
    let d = neo_math::D;
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * cols];
    for col in 0..cols { for rho in 0..d { row_major[rho * cols + col] = z_digits[col * d + rho]; } }
    let z_mat = Mat::from_row_major(d, cols, row_major);

    let cmt = Commitment::zeros(d, 4);
    let mcs_instance = McsInstance { c: cmt.clone(), x: vec![z_full[0]], m_in };
    let mcs_witness  = McsWitness  { w: z_full[m_in..].to_vec(), Z: z_mat };

    // Manual dims (no policy):
    let (ell_d, ell_n, ell, _d_sc) = test_dims(s.n, params.b, s.max_degree() as usize);

    // Transcript just for deterministic challenges
    let mut tr = Poseidon2Transcript::new(b"test/ajtai-nc-univariate-b3");
    neo_fold::pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs_instance.clone()], ell, 8, 0).unwrap();
    neo_fold::pi_ccs::transcript::bind_me_inputs(&mut tr, &[]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // Precompute Ajtai full rows (k=1)
    let y_full = precompute::precompute_nc_full_rows(&s, &[mcs_witness.clone()], &[], ell_n).expect("y_full");
    assert_eq!(y_full.len(), 1);

    // Exact Ajtai branch sums at r' = 0^ell_n
    let w_beta_a = HalfTableEq::new(&ch.beta_a);
    let mut s0 = K::ZERO;
    let mut s1 = K::ZERO;
    let low  = -((params.b as i64) - 1);
    let high =  (params.b as i64) - 1;

    for rho in 0..D {
        let eq_xa = w_beta_a.w(rho);
        let y_val = y_full[0][rho][0];  // select row 0 (r' = 0^ell_n)
        let mut range = K::ONE;
        for t in low..=high { range *= y_val - K::from(F::from_i64(t)); }
        let term = ch.gamma * range;
        if (rho & 1) == 0 { s0 += eq_xa * term; } else { s1 += eq_xa * term; }
    }

    let branch_sum = s0 + s1;

    // Public hypercube NC sum with β_r = r' (zero vector)
    let beta_r_zero = vec![K::ZERO; ell_n];
    let nc_hcube = compute_nc_hypercube_sum(
        &s, &[mcs_witness.clone()], &[], &ch.beta_a, &beta_r_zero, ch.gamma, &params, ell_d, ell_n,
    );

    assert_eq!(branch_sum, nc_hcube, "Ajtai exact branch sum must match public hypercube NC sum");
}

// === Oracle regression: compare oracle Ajtai univariate to exact & hypercube ===

use neo_fold::pi_ccs::oracle::GenericCcsOracle;
use neo_fold::pi_ccs::sparse_matrix::to_csr;
use neo_fold::pi_ccs::precompute::{build_mle_partials_first_inst, compute_initial_sum_components, precompute_beta_block, precompute_eval_row_partial};
use neo_fold::sumcheck::RoundOracle;

struct DummyS;
impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment { Commitment::zeros(z.rows(), 4) }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows(); let cols = m_in.min(z.cols());
        let mut out = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows { for c in 0..cols { out[(r,c)] = z[(r,c)]; } }
        out
    }
}

#[test]
fn regression_ajtai_nc_sumcheck_round0_vs_exact_and_hypercube_b3() {
    // Structure: all-zero R1CS → identity-first CCS
    let rows = 4usize;
    let cols = 4usize;
    let a = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let b = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let c = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let s_raw = r1cs_to_ccs(a, b, c);
    let s = s_raw.ensure_identity_first().expect("identity-first");

    // Params: keep s=2; set b=3; no policy checks in this test.
    let base_params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let params = NeoParams { b: 3, ..base_params };

    // Honest non-trivial witness
    let z_full = vec![F::from_u64(5), F::ZERO, F::ZERO, F::ZERO];
    let m_in = 1usize;
    let d = neo_math::D;
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * cols];
    for col in 0..cols { for rho in 0..d { row_major[rho * cols + col] = z_digits[col * d + rho]; } }
    let z_mat = Mat::from_row_major(d, cols, row_major);

    let cmt = Commitment::zeros(d, 4);
    let mcs_inst = McsInstance { c: cmt.clone(), x: vec![z_full[0]], m_in };
    let mcs_wit  = McsWitness  { w: z_full[m_in..].to_vec(), Z: z_mat };

    // Dims (manual)
    let (ell_d, ell_n, ell, d_sc) = test_dims(s.n, params.b, s.max_degree() as usize);

    // Transcript & challenges
    let mut tr = Poseidon2Transcript::new(b"test/regress/ajtai-nc-invariant-b3");
    neo_fold::pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs_inst.clone()], ell, d_sc, 0).unwrap();
    neo_fold::pi_ccs::transcript::bind_me_inputs(&mut tr, &[]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // CSR & precompute
    let mats_csr: Vec<_> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    let l = DummyS;
    let wit_binding = [mcs_wit.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[mcs_inst.clone()], &wit_binding, &mats_csr, &l).unwrap();

    let partials_first = build_mle_partials_first_inst(&s, ell_n, &insts).unwrap();
    let beta_block = precompute_beta_block(&s, &params, &insts, &wit_binding, &[], &ch, ell_d, ell_n).unwrap();
    let eval_row_partial = precompute_eval_row_partial(&s, &[], &ch, 1, ell_n).unwrap();
    let nc_y = precompute::precompute_nc_full_rows(&s, &wit_binding, &[], ell_n).unwrap();

    // Initial sum for k=1: in a valid decomposed witness, NC hypercube sum == 0; F(β_r) may be 0 here (zero A/B/C).
    let initial_sum = compute_initial_sum_components(&s, &[], &ch, 1, &beta_block).unwrap();

    // Equality weights (row & Ajtai)
    use neo_fold::pi_ccs::eq_weights::RowWeight;
    let w_beta_a = HalfTableEq::new(&ch.beta_a);
    let w_alpha_a = HalfTableEq::new(&ch.alpha);
    let w_beta_r = HalfTableEq::new(&ch.beta_r);

    let w_beta_a_partial = (0..(1usize << ell_d)).map(|i| w_beta_a.w(i)).collect::<Vec<_>>();
    let w_alpha_a_partial = (0..(1usize << ell_d)).map(|i| w_alpha_a.w(i)).collect::<Vec<_>>();
    let w_beta_r_partial = (0..(1usize << ell_n)).map(|i| w_beta_r.w(i)).collect::<Vec<_>>();
    let w_eval_r_partial = vec![K::ZERO; 1usize << ell_n]; // no ME inputs → Eval row gate 0

    // Oracle with **current** implementation under test
    let z_refs: Vec<&Mat<F>> = std::iter::once(&mcs_wit.Z).collect();
    let gamma_pows = vec![ch.gamma];
    let mut oracle = GenericCcsOracle::<F> {
        s: &s,
        partials_first_inst: partials_first,
        w_beta_a_partial: w_beta_a_partial.clone(),
        w_alpha_a_partial: w_alpha_a_partial.clone(),
        w_beta_r_partial: w_beta_r_partial.clone(),
        w_beta_r_full: w_beta_r_partial.clone(),
        w_eval_r_partial,
        z_witnesses: z_refs,
        gamma: ch.gamma,
        k_total: 1,
        b: params.b,
        ell_d, ell_n, d_sc,
        round_idx: 0,
        initial_sum_claim: initial_sum,
        f_at_beta_r: beta_block.f_at_beta_r,
        nc_sum_beta: K::ZERO,
        eval_row_partial: eval_row_partial.clone(),
        row_chals: Vec::new(),
        csr_m1: &mats_csr[0],
        csrs: &mats_csr,
        eval_ajtai_partial: None,
        me_offset: 1,                 // no ME inputs; outputs would begin at 1
        nc_y_matrices: nc_y.clone(),  // full Ajtai rows for NC
        nc_row_gamma_pows: gamma_pows,
        nc: None,
    };

    // Fold only the row rounds with r_i = 0 to get r' = 0^ell_n
    for _ in 0..ell_n { oracle.fold(K::ZERO); }
    assert_eq!(oracle.round_idx, ell_n);

    // First Ajtai round: query at X ∈ {0,1}
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]);
    assert_eq!(ys.len(), 2);
    let ys_sum = ys[0] + ys[1];

    // Exact Ajtai NC S0/S1 at r' using precomputed nc_y_matrices and β_a
    let w_beta_a_full = HalfTableEq::new(&ch.beta_a);
    let mut s0_exact = K::ZERO;
    let mut s1_exact = K::ZERO;
    let low  = -((params.b as i64) - 1);
    let high =  (params.b as i64) - 1;

    for rho in 0..D {
        let eq_xa = w_beta_a_full.w(rho);
        let y_val = nc_y[0][rho][0]; // r' row = 0
        let mut range = K::ONE;
        for t in low..=high { range *= y_val - K::from(F::from_i64(t)); }
        let term = ch.gamma * range;
        if (rho & 1) == 0 { s0_exact += eq_xa * term; } else { s1_exact += eq_xa * term; }
    }

    let exact_branch_sum = s0_exact + s1_exact;

    // Public hypercube sum with β_r == r' (0^ell_n)
    let beta_r_eq_rprime: Vec<K> = vec![K::ZERO; ell_n];
    let nc_hcube = compute_nc_hypercube_sum(
        &s, &[mcs_wit.clone()], &[], &ch.beta_a, &beta_r_eq_rprime, ch.gamma, &params, ell_d, ell_n,
    );

    // The oracle **must** match exact Ajtai branch sum AND the public hypercube sum.
    // If the oracle still does "per-pair Ajtai gating", this assert will FAIL.
    assert_eq!(ys_sum, exact_branch_sum,
        "Oracle Ajtai NC s(0)+s(1) = {:?} != exact Ajtai branch sum {:?}", ys_sum, exact_branch_sum);
    assert_eq!(exact_branch_sum, nc_hcube,
        "Exact Ajtai branch sum {:?} != public hypercube sum {:?}", exact_branch_sum, nc_hcube);

    // Optional: check the sum-check invariant against the (prover-claimed) initial sum.
    assert_eq!(ys_sum, initial_sum,
        "Ajtai round 0 invariant broken: s(0)+s(1)={:?}, expected {:?}", ys_sum, initial_sum);
}
