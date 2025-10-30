//! Focused unit tests for row-pair mapping and NC nonlinearity.
//!
//! These tests are small and precise to help localize issues:
//! - Row pair mapping across rounds (formula vs. baseline dynamic-stride layout)
//! - NC nonlinearity sanity: product( interpolate(y) ) != interpolate( product(y) )

use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

/// Compute (j0, j1) for the k-th folded pair at row round `round_idx`.
/// This mirrors the mapping used in the oracle for NC row evaluation.
fn row_pair_indices(round_idx: usize, k: usize) -> (usize, usize) {
    let stride = 1usize << round_idx;              // 2^i
    let j0 = (k & (stride - 1)) + ((k >> round_idx) << (round_idx + 1));
    (j0, j0 + stride)
}

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

    // Round 1: stride=2 → we have only one folded pair covering (0,2) (and the
    // complementary (1,3) is aggregated under a different off in the baseline view)
    let round1_pairs: Vec<(usize, usize)> = (0..(full_rows >> 2))
        .map(|k| row_pair_indices(1, k))
        .collect();
    assert_eq!(round1_pairs, vec![(0, 2)]);

    // Baseline dynamic-stride enumeration for round 1
    let stride1 = 1usize << 1; // 2
    let block1 = 2 * stride1;  // 4
    let groups1 = full_rows / block1; // 1
    let mut baseline1 = Vec::new();
    for g in 0..groups1 {
        let base = g * block1; // 0
        for off in 0..stride1 {
            let j0 = base + off;         // 0, 1
            let j1 = base + off + stride1; // 2, 3
            baseline1.push((j0, j1));
        }
    }
    // The folded pair mapping enumerates k over the compressed domain; it should
    // match the first (off=0) pair in the baseline view for round 1.
    assert_eq!(round1_pairs[0], baseline1[0]);
}

/// Range polynomial N_b(x) = ∏_{t=-(b-1)}^{b-1} (x - t) is nonlinear; therefore:
/// N( (1-X)*y0 + X*y1 ) != (1-X)*N(y0) + X*N(y1) as polynomials in X, in general.
#[test]
fn nc_nonlinearity_interpolation_vs_product() {
    // Use b=2 (range [-1, 1]) and pick values away from the roots {-1,0,1}
    // so the check is meaningful: choose y0=2, y1=4 and X=2.
    let b: i64 = 2;
    let y0 = K::from(F::from_u64(2));
    let y1 = K::from(F::from_u64(4));
    let x = K::from(F::from_u64(2));

    // N(y) = ∏_{t=-(b-1)}^{b-1} (y - t)
    let nb = |y: K| {
        let mut acc = K::ONE;
        for t in (-(b - 1))..=(b - 1) {
            acc *= y - K::from(F::from_i64(t));
        }
        acc
    };

    let yi = (K::ONE - x) * y0 + x * y1;
    let lhs = nb(yi);
    let rhs = (K::ONE - x) * nb(y0) + x * nb(y1);

    // They should differ for generic choices (here they do).
    assert!(lhs != rhs, "NC nonlinearity violated: lhs == rhs for chosen inputs");
}

// === Ajtai NC univariate regression: roots at b=2 must vanish ===
// Paper-faithful check at the first Ajtai round: for r'=0^ell_n and b=2,
// if the Ajtai MLE ỹ'(α') lands in {−1,0,1}, then each branch value S0,S1 must be 0 and
// S0+S1 must equal the public hypercube NC sum with β_r=r'.
use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, SModuleHomomorphism};
use neo_fold::pi_ccs::{context, precompute};
use neo_fold::pi_ccs::eq_weights::HalfTableEq;
use neo_fold::pi_ccs::eq_weights::RowWeight;
use neo_fold::pi_ccs::nc_constraints::compute_nc_hypercube_sum;
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[test]
fn ajtai_nc_univariate_roots_b2_are_zero() {
    // Build a tiny CCS via R1CS→CCS with a non-trivial honest witness
    // Matrices A,B,C all zero is acceptable; witness z_full = [1,0,0,0]
    let rows = 4usize;
    let cols = 4usize;
    let a = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let b = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let c = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let s_raw = r1cs_to_ccs(a, b, c);
    let s = s_raw.ensure_identity_first().expect("identity-first");

    // Params with b=3 (range [-2,2]) to catch per-pair Ajtai gating bugs
    // For b=3, degree bound d_sc = 2*3+2 = 8; with ell≈8, we need lambda≤122 for s=2
    let params = NeoParams::goldilocks_for_circuit(8, 8, 6); // Conservative lambda≈116
    let params = NeoParams { b: 3, ..params }; // Override to b=3
    assert_eq!(params.b, 3, "expected base b=3 for this test");

    // Non-trivial honest witness z_full
    let z_full = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO];
    let m_in = 1usize; // one public input position

    // Decompose Z in row-major (d × m)
    let d = neo_math::D;
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * cols];
    for col in 0..cols { for row in 0..d { row_major[row * cols + col] = z_digits[col * d + row]; } }
    let z_mat = Mat::from_row_major(d, cols, row_major);

    // Minimal S-module behavior
    let cmt = Commitment::zeros(d, 4);
    let mcs_instance = McsInstance { c: cmt.clone(), x: vec![z_full[0]], m_in };
    let mcs_witness  = McsWitness  { w: z_full[m_in..].to_vec(), Z: z_mat };

    // Dimensions and transcript-challenges
    let dims = context::build_dims_and_policy(&params, &s).expect("dims");
    let (ell_d, ell_n) = (dims.ell_d, dims.ell_n);
    let mut tr = Poseidon2Transcript::new(b"test/ajtai-nc-univariate-b2");
    neo_fold::pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs_instance.clone()], dims.ell, dims.d_sc, 0).unwrap();
    neo_fold::pi_ccs::transcript::bind_me_inputs(&mut tr, &[]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, dims.ell).unwrap();

    // Precompute full NC y rows; expect k=1
    let y_full = precompute::precompute_nc_full_rows(&s, &[mcs_witness.clone()], &[], ell_n).expect("y_full");
    assert_eq!(y_full.len(), 1);

    // Compute exact Ajtai univariate samples for the first Ajtai bit at r' = 0^ell_n
    // Fix β_a from sampled challenges
    let w_beta_a = HalfTableEq::new(&ch.beta_a);

    // gamma powers per instance (i starts at 1; we have one instance)
    let gamma_pows = vec![ch.gamma];
    let low  = -((params.b as i64) - 1);
    let high =  (params.b as i64) - 1;

    // Branch accumulators S0 (current Ajtai bit=0), S1 (bit=1)
    // NOTE: y_full[i][rho][row] is NOT padded to 2^ell_d, only has D entries for rho
    let mut s0 = K::ZERO;
    let mut s1 = K::ZERO;
    for rho in 0..D {  // Iterate over actual Ajtai dimension D, not padded d_tot
        let eq_xa = w_beta_a.w(rho);
        // At r'=0^ell_n, χ_{r'} selects row index 0
        let mut sum_over_i = K::ZERO;
        for (i, y_mat) in y_full.iter().enumerate() {
            let y_val = y_mat[rho][0];
            let mut range = K::ONE;
            for t in low..=high { range *= y_val - K::from(F::from_i64(t)); }
            sum_over_i += gamma_pows[i] * range;
        }
        if (rho & 1) == 0 { s0 += eq_xa * sum_over_i; } else { s1 += eq_xa * sum_over_i; }
    }

    // For honest witness with b=3, the NC hypercube sum is 0 (digits in [-2,2])
    // but per-pair Ajtai gating bugs would produce non-zero branch sums
    let branch_sum = s0 + s1;

    // Cross-check with public hypercube NC sum at β_r = r' = 0^ell_n
    let beta_r_zero = vec![K::ZERO; ell_n];
    let nc_hcube = compute_nc_hypercube_sum(
        &s, &[mcs_witness.clone()], &[], &ch.beta_a, &beta_r_zero, ch.gamma, &params, ell_d, ell_n,
    );
    
    // Both should be zero for honest witness, and must match
    assert_eq!(branch_sum, K::ZERO, "Ajtai NC branch sum must be 0 for honest witness");
    assert_eq!(nc_hcube, K::ZERO, "Hypercube NC sum must be 0 for honest witness");
    assert_eq!(nc_hcube, branch_sum, "Hypercube NC sum must equal Ajtai univariate s(0)+s(1)");
}

use neo_fold::pi_ccs::oracle::GenericCcsOracle;
use neo_fold::pi_ccs::sparse_matrix::to_csr;
use neo_fold::pi_ccs::precompute::{build_mle_partials_first_inst, compute_initial_sum_components, precompute_beta_block, precompute_eval_row_partial};
use neo_fold::sumcheck::RoundOracle;

// Minimal S-module helper used for instance preparation
struct DummyS;
impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        Commitment::zeros(z.rows(), 4)
    }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut out = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows { for c in 0..cols { out[(r,c)] = z[(r,c)]; } }
        out
    }
}

#[test]
fn regression_ajtai_nc_sumcheck_round0_invariant_b2() {
    // Build minimal CCS (all-zero R1CS) with non-trivial honest witness
    let rows = 4usize;
    let cols = 4usize;
    let a = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let b = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let c = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);
    let s_raw = r1cs_to_ccs(a, b, c);
    let s = s_raw.ensure_identity_first().expect("identity-first");

    // b=3 params to catch per-pair Ajtai gating bugs
    // For b=3, degree bound d_sc = 2*3+2 = 8; with ell≈8, we need lambda≤122 for s=2
    let params = NeoParams::goldilocks_for_circuit(8, 8, 6); // Conservative lambda≈116
    let params = NeoParams { b: 3, ..params }; // Override to b=3
    assert_eq!(params.b, 3, "expected b=3");

    // Non-trivial honest witness z
    let z_full = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO];
    let m_in = 1usize;
    let d = neo_math::D;
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * cols];
    for col in 0..cols { for rho in 0..d { row_major[rho * cols + col] = z_digits[col * d + rho]; } }
    let z_mat = Mat::from_row_major(d, cols, row_major);
    let cmt = Commitment::zeros(d, 4);
    let mcs_inst = McsInstance { c: cmt.clone(), x: vec![z_full[0]], m_in };
    let mcs_wit  = McsWitness  { w: z_full[m_in..].to_vec(), Z: z_mat };

    // Transcript + challenges
    let dims = context::build_dims_and_policy(&params, &s).expect("dims");
    let (ell_d, ell_n, ell, d_sc) = (dims.ell_d, dims.ell_n, dims.ell, dims.d_sc);
    let mut tr = Poseidon2Transcript::new(b"test/regress/ajtai-nc-invariant-b2");
    neo_fold::pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs_inst.clone()], ell, d_sc, 0).unwrap();
    neo_fold::pi_ccs::transcript::bind_me_inputs(&mut tr, &[]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // CSR matrices and instance prep
    let mats_csr: Vec<_> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    let l = DummyS;
    let wit_binding = [mcs_wit.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[mcs_inst.clone()], &wit_binding, &mats_csr, &l).unwrap();

    // Partials for F, beta-block, eval row partials, NC y matrices
    let partials_first = build_mle_partials_first_inst(&s, ell_n, &insts).unwrap();
    let beta_block = precompute_beta_block(&s, &params, &insts, &[mcs_wit.clone()], &[], &ch, ell_d, ell_n).unwrap();
    let eval_row_partial = precompute_eval_row_partial(&s, &[], &ch, 1, ell_n).unwrap();
    let nc_y = precompute::precompute_nc_full_rows(&s, &[mcs_wit.clone()], &[], ell_n).unwrap();

    // Initial sum for k=1 should be 0 on zero-state
    let initial_sum = compute_initial_sum_components(&s, &[], &ch, 1, &beta_block).unwrap();
    assert_eq!(initial_sum, K::ZERO, "zero-state expects initial_sum=0");

    // Equality weights
    use neo_fold::pi_ccs::eq_weights::RowWeight;
    let w_beta_a = HalfTableEq::new(&ch.beta_a);
    let w_alpha_a = HalfTableEq::new(&ch.alpha);
    let w_beta_r = HalfTableEq::new(&ch.beta_r);
    let w_beta_a_partial = (0..(1usize << ell_d)).map(|i| w_beta_a.w(i)).collect::<Vec<_>>();
    let w_alpha_a_partial = (0..(1usize << ell_d)).map(|i| w_alpha_a.w(i)).collect::<Vec<_>>();
    let w_beta_r_partial = (0..(1usize << ell_n)).map(|i| w_beta_r.w(i)).collect::<Vec<_>>();
    let w_eval_r_partial = vec![K::ZERO; 1usize << ell_n];

    // Oracle
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
        me_offset: 1,
        nc_y_matrices: nc_y,
        nc_row_gamma_pows: gamma_pows,
        nc: None,
    };

    // Fold only the row rounds with r_i = 0
    for _ in 0..ell_n { oracle.fold(K::ZERO); }
    assert_eq!(oracle.round_idx, ell_n);

    // First Ajtai round: query at X ∈ {0,1}
    let ys = oracle.evals_at(&[K::ZERO, K::ONE]);
    assert_eq!(ys.len(), 2);
    let ys_sum = ys[0] + ys[1];

    // Compute exact Ajtai NC S0/S1 at r' using precomputed nc_y_matrices and β_a
    // NOTE: nc_y_matrices[i][rho][row] has only D entries for rho, not padded to 2^ell_d
    let w_beta_a_full = HalfTableEq::new(&ch.beta_a);
    let mut s0_exact = K::ZERO;
    let mut s1_exact = K::ZERO;
    let low  = -((params.b as i64) - 1);
    let high =  (params.b as i64) - 1;

    // One instance → γ^1
    for rho in 0..d {  // Iterate over actual Ajtai dimension D, not padded d_tot
        let eq_xa = w_beta_a_full.w(rho);
        // r' was produced by folding the row rounds; at this point rows are fully folded.
        // We folded with r_i=0, so row index 0 is selected:
        let y_val = oracle.nc_y_matrices[0][rho][0];
        let mut range = K::ONE;
        for t in low..=high { range *= y_val - K::from(F::from_i64(t)); }
        let term = ch.gamma * range;
        if (rho & 1) == 0 { s0_exact += eq_xa * term; } else { s1_exact += eq_xa * term; }
    }

    let exact_branch_sum = s0_exact + s1_exact;
    assert_eq!(ys_sum, exact_branch_sum, 
        "Oracle Ajtai NC output {:?} != exact Ajtai branch sum {:?}", ys_sum, exact_branch_sum);

    // Cross-check the public hypercube sum when β_r = r'
    let beta_r_eq_rprime: Vec<K> = vec![K::ZERO; ell_n]; // r' = 0^ell_n from folds
    let nc_hcube = compute_nc_hypercube_sum(
        &s, &[mcs_wit.clone()], &[], &ch.beta_a, &beta_r_eq_rprime, ch.gamma, &params, ell_d, ell_n,
    );
    assert_eq!(exact_branch_sum, nc_hcube, 
        "Exact Ajtai branch sum {:?} != public hypercube sum {:?}", exact_branch_sum, nc_hcube);

    // Also check the invariant: oracle output should match initial_sum
    assert_eq!(ys_sum, initial_sum, 
        "Ajtai round 0 invariant broken: s(0)+s(1)={:?}, expect {:?}", ys_sum, initial_sum);
}
