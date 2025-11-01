//! # Π_CCS: CCS Reduction Protocol
//!
//! Implements the CCS reduction protocol from Section 4.4 of the Neo paper.
//!
//! ## Protocol Overview
//!
//! **Π_CCS** reduces MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k using sum-check.
//!
//! Key idea: Encode CCS constraints, norm constraints, and evaluation claims into
//! a single polynomial Q(X) and use the sum-check protocol to reduce the problem
//! from a sum over the hypercube {0,1}^{log(dn)} to a single evaluation claim.
//!
//! ## Module Organization
//!
//! The implementation follows the protocol's four-step structure:
//!
//! 1. **Challenge Generation** (`transcript` module)
//!    - Bind inputs to transcript
//!    - Sample α ∈ K^{log d}, β ∈ K^{log(dn)}, γ ∈ K
//!
//! 2. **Sum-Check Protocol** (`oracle`, `precompute` modules)
//!    - Build polynomial Q(X) with components F, NC, Eval
//!    - Execute sum-check to reduce T = sum over hypercube to v = Q(α', r')
//!
//! 3. **Output Construction** (`outputs` module)
//!    - Compute y'_{(i,j)} = Z_i·M_j^T·r̂' for all instances
//!    - Build k ME instances from MCS and ME inputs
//!
//! 4. **Terminal Verification** (`terminal` module)
//!    - Check v ?= Q(α', r') using public values only
//!
//! ## Relations
//!
//! **MCS(b,L)** - Matrix Constraint System:
//! - Public: commitment c, public inputs x
//! - Witness: private inputs w, matrix Z = Decomp_b(x||w)
//! - Constraints: c = L(Z), f(M̃_1z,...,M̃_tz) ∈ ZS_n (zero set)
//!
//! **ME(b,L)** - Matrix Evaluation:
//! - Public: commitment c, X = Decomp_b(x), random point r, evaluations {y_j}
//! - Witness: matrix Z
//! - Constraints: c = L(Z), X = L_x(Z), ||Z||_∞ < b, y_j = Z·M_j^T·r̂ for all j
//!
//! ## The Polynomial Q(X)
//!
//! The sum-check is performed over:
//!
//! ```text
//! Q(X) = eq(X,β)·(F(X) + Σ_{i∈[k]} γ^i·NC_i(X))
//!        + γ^k·Σ_{j∈[t],i∈[2,k]} γ^{i+(j-1)k-1}·Eval_{(i,j)}(X)
//! ```
//!
//! where:
//! - **F(X)**: CCS constraint polynomial = f(M̃_1z_1,...,M̃_tz_1)
//! - **NC_i(X)**: Norm constraint = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
//! - **Eval_{(i,j)}(X)**: Evaluation claim = eq(X,(α,r))·M̃_{(i,j)}(X)

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use neo_transcript::{Transcript, Poseidon2Transcript};
#[allow(unused_imports)]
use neo_transcript::labels as tr_labels;
use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, KExtensions};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use crate::sumcheck::{run_sumcheck, verify_sumcheck_rounds, SumcheckOutput};
use crate::pi_ccs::sparse_matrix::{Csr, to_csr};
use crate::pi_ccs::oracle::GenericCcsOracle;
use crate::pi_ccs::transcript::Challenges;
// keep transcript helpers internal; exported via transcript module when needed

// ============================================================================
// SUBMODULES (Paper-aligned refactoring)
// ============================================================================
// Core protocol modules following Section 4.4 structure
pub mod context;         // Dimensions and security policy
pub mod transcript;      // Fiat-Shamir binding and challenge sampling  
pub mod precompute;      // Q polynomial components (F, NC, Eval)
pub mod checks;          // Input/output validation
pub mod terminal;        // Verifier terminal check RHS
pub mod outputs;         // ME instance construction
pub mod sumcheck_driver; // Oracle construction (placeholder)
pub mod oracle;          // Sum-check oracle for Q polynomial
pub mod transcript_replay; // Transcript replay utilities for debugging/testing

// Utility modules
pub mod nc_core;         // Core NC polynomial evaluation functions
pub mod nc_constraints;  // Norm/decomposition constraints
pub mod sparse_matrix;   // CSR sparse matrix operations
pub mod eq_weights;      // Equality polynomial evaluation

// Re-export commonly used public items
pub use transcript_replay::{
    TranscriptTail,
    pi_ccs_derive_transcript_tail,
    pi_ccs_derive_transcript_tail_with_me_inputs,
    pi_ccs_derive_transcript_tail_with_me_inputs_and_label,
    pi_ccs_derive_transcript_tail_from_bound_transcript,
    pi_ccs_compute_terminal_claim_r1cs_or_ccs,
};

// ============================================================================
// PROOF STRUCTURE
// ============================================================================

/// Format extension field element compactly as "r + i·u" where r, i are decimal values.
/// 
/// Example outputs:
/// - Pure real: `12345678901234567890`
/// - Complex: `12345678901234567890 + 98765432109876543210·u`
/// 
/// This is much more compact than the default Debug format which shows:
/// `BinomialExtensionField { value: [12345..., 98765...], _phantom: PhantomData<...> }`
#[allow(dead_code)]
pub(crate) fn format_ext(x: K) -> String {
    use neo_math::KExtensions;
    let coeffs = x.as_coeffs();
    let real = coeffs[0].as_canonical_u64();
    let imag = coeffs[1].as_canonical_u64();
    if imag == 0 {
        format!("{}", real)
    } else {
        format!("{} + {}·u", real, imag)
    }
}

/// Format a slice of extension field elements compactly as [elem1, elem2, ...]
#[allow(dead_code)]
fn format_ext_vec(xs: &[K]) -> String {
    let formatted: Vec<String> = xs.iter().map(|&x| format_ext(x)).collect();
    format!("[{}]", formatted.join(", "))
}

/// Π_CCS proof containing the single sum-check over K
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sum-check protocol rounds (univariate polynomials as coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    /// Sum-check sampled challenges r_0..r_{ℓ-1} (carried for deterministic replay)
    pub sumcheck_challenges: Vec<K>,
    /// Sum-check final running sum Q(α', r')
    pub sumcheck_final: K,
    /// Extension policy binding digest  
    pub header_digest: [u8; 32],
    /// Sum-check initial claim s(0)+s(1) when the R1CS engine is used; None for generic CCS.
    pub sc_initial_sum: Option<K>,
    /// Public challenges (α, β=(β_a,β_r), γ) carried for deterministic replay
    pub challenges_public: Challenges,
}



// ===== MLE Folding DP Helpers =====


// ============================================================================
// CONSTRAINT EVALUATION FUNCTIONS
// ============================================================================
// NC constraints (compute_nc_hypercube_sum) are in pi_ccs::nc_constraints submodule
// (Test-only eval_range_decomp_constraints moved to tests-core/test_helpers.rs)

/// Evaluate tie constraint polynomials at point u.
///
/// **Paper Reference**: Definition 18 (ME relation)
///
/// Checks the constraint: ∀j ∈ [t], y_j = Z·M_j^T·r̂
///
/// Computes: Σ_j Σ_ρ (⟨Z_ρ,*, M_j^T χ_u⟩ - y_{j,ρ})
///
/// This verifies that the claimed y_j values are consistent with the witness Z
/// and the evaluation point u. Returns zero if all tie constraints are satisfied.
pub fn eval_tie_constraints(
    s: &CcsStructure<F>,
    Z: &neo_ccs::Mat<F>,
    claimed_y: &[Vec<K>], // y_j entries as produced in ME (length t, each length d)
    u: &[K],
) -> K {
    // REAL MULTILINEAR EXTENSION EVALUATION
    // Implements: Σ_j Σ_ρ (⟨Z_ρ,*, M_j^T χ_u⟩ - y_{j,ρ})
    
    // χ_u ∈ K^n
    let chi_u = neo_ccs::utils::tensor_point::<K>(u);

    let d = Z.rows();       // Ajtai dimension
    let m = Z.cols();       // number of columns in Z (== s.m)

    debug_assert_eq!(m, s.m, "Z.cols() must equal s.m");
    
    // If claimed_y is missing or has wrong shape, we conservatively treat it as zero.
    // (This will force the prover's Q(u) to carry the full ⟨Z, M_j^T χ_u⟩ mass, which
    // then must cancel at r when the real y_j are used.)
    let safe_y = |j: usize, rho: usize| -> K {
        if j < claimed_y.len() && rho < claimed_y[j].len() {
            claimed_y[j][rho]
        } else {
            K::ZERO
        }
    };

    let mut total = K::ZERO;

    // For each matrix M_j, build v_j(u) = M_j^T χ_u ∈ K^m, then compute Z * v_j(u) ∈ K^d
    for (j, mj) in s.matrices.iter().enumerate() {
        // v_j[c] = Σ_{row=0..n-1} M_j[row,c] * χ_u[row]
        let mut vj = vec![K::ZERO; s.m];
        for row in 0..s.n {
            let coeff = chi_u[row];
            let mj_row = mj.row(row);
            for c in 0..s.m {
                vj[c] += K::from(mj_row[c]) * coeff;
            }
        }
        // lhs = Z * v_j(u) as a length-d K-vector
        let z_ref = neo_ccs::MatRef::from_mat(Z);
        let lhs = neo_ccs::utils::mat_vec_mul_fk::<F, K>(z_ref.data, z_ref.rows, z_ref.cols, &vj);

        // Accumulate the residual (lhs - y_j)
        for rho in 0..d {
            total += lhs[rho] - safe_y(j, rho);
        }
    }

    total
}

// ============================================================================
// PROVER: Π_CCS (Section 4.4)
// ============================================================================

/// Prove Π_CCS reduction: MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k
///
/// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS
///
/// # Protocol Steps (Paper Section 4.4)
///
/// 1. **V → P**: Sample challenges α ∈ K^{log d}, β ∈ K^{log(dn)}, γ ∈ K
///
/// 2. **V ↔ P**: Sum-check protocol over Q(X)
///    - Define polynomials:
///      * F(X) = f(M̃_1z_1,...,M̃_tz_1)        [CCS constraints]
///      * NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)  [norm constraints]
///      * Eval_{(i,j)}(X) = eq(X,(α,r))·M̃_{(i,j)}(X)  [eval claims]
///    - Q(X) = eq(X,β)·(F + Σ_i γ^i·NC_i) + γ^k·Σ_{i,j} γ^{i+(j-1)k-1}·Eval_{i,j}
///    - Claimed sum: T = γ^k·Σ_{j,i} γ^{i+(j-1)k-1}·ỹ_{(i,j)}(α)
///    - Reduce to evaluation claim: v ?= Q(α', r')
///
/// 3. **P → V**: Send y'_{(i,j)} = Z_i·M_j^T·r̂' for all i ∈ [k], j ∈ [t]
///
/// 4. **V**: Verify v ?= Q(α', r') using:
///    - F' = f(m_1,...,m_t) where m_j = Σ_ℓ b^{ℓ-1}·y'_{(1,j),ℓ}
///    - N_i' = ∏_{j=-b+1}^{b-1} (ỹ'_{(i,1)}(α') - j)
///    - E_{(i,j)}' = eq((α',r'), (α,r))·ỹ'_{(i,j)}(α')
///    - Check: v = eq((α',r'), β)·(F' + Σ_i γ^i·N_i') + γ^k·Σ_{i,j} γ^{i+(j-1)k-1}·E_{(i,j)}'
///
/// # Arguments
/// - `tr`: Fiat-Shamir transcript for challenge generation
/// - `params`: Security parameters (b, λ, etc.)
/// - `s_in`: CCS structure (matrices M_1,...,M_t and polynomial f)
/// - `mcs_list`: Input MCS instances (at least 1)
/// - `witnesses`: Witnesses for MCS instances
/// - `me_inputs`: Input ME instances from previous fold (empty for k=1)
/// - `me_witnesses`: Witnesses for ME inputs
/// - `l`: Commitment scheme L: F^{d×m} → C
///
/// # Returns
/// - `Vec<MeInstance>`: k output ME instances (1 per input MCS/ME)
/// - `PiCcsProof`: Sum-check rounds and binding data
pub fn pi_ccs_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s_in: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[neo_ccs::Mat<F>],
    l: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // =========================================================================
    // STEP 0: SETUP
    // =========================================================================
    // Initialize transcript, validate inputs, compute dimensions
    
    tr.append_message(tr_labels::PI_CCS, b"");
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidStructure(format!("CCS structure invalid for identity-first: {:?}", e)))?;

    checks::validate_inputs(&s, mcs_list, witnesses, me_inputs, me_witnesses)?;

    let context::Dims { ell_d, ell_n, ell, d_sc } = context::build_dims_and_policy(params, &s)?;

    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs] Dimensions: ell_d={}, ell_n={}, ell={}, d_sc={}", ell_d, ell_n, ell, d_sc);

    #[cfg(feature = "debug-logs")]
    let csr_start = std::time::Instant::now();
    let mats_csr: Vec<Csr<F>> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    #[cfg(feature = "debug-logs")]
    {
        let total_nnz: usize = mats_csr.iter().map(|c| c.data.len()).sum();
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔥 CSR conversion completed: {:.2}ms ({} matrices, {} nnz total, {:.4}% density)",
                     csr_start.elapsed().as_secs_f64() * 1000.0, 
                     mats_csr.len(), total_nnz, 
                     (total_nnz as f64) / (s.n * s.m * s.matrices.len()) as f64 * 100.0);
        }
    }

    #[cfg(feature = "debug-logs")]
    let instance_prep_start = std::time::Instant::now();
    let insts = precompute::prepare_instances(&s, params, mcs_list, witnesses, &mats_csr, l)?;
    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] Instance preparation total: {:.2}ms ({} instances)", 
                     instance_prep_start.elapsed().as_secs_f64() * 1000.0, insts.len());
        }
    }

    // =========================================================================
    // STEP 1: TRANSCRIPT BINDING & CHALLENGE GENERATION
    // =========================================================================
    // Paper Section 4.4, Step 1: V sends α, β, γ
    // Fiat-Shamir: derive challenges from transcript containing all public inputs
    
    #[cfg(feature = "debug-logs")]
    let transcript_start = std::time::Instant::now();
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs][prove] About to bind header. Using s after ensure_identity_first");
        eprintln!("[pi-ccs][prove] s.n={}, s.m={}, s.t()={}", s.n, s.m, s.t());
    }
    transcript::bind_header_and_instances(tr, params, &s, mcs_list, ell, d_sc, 0)?;
    transcript::bind_me_inputs(tr, me_inputs)?;
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[Fold] Current transcript digest after binding: {:?}", tr.clone().digest32());
        if !me_inputs.is_empty() {
            eprintln!("[Fold] Using {} ME inputs:", me_inputs.len());
            for (i, me) in me_inputs.iter().enumerate() {
                eprintln!("  ME[{}] fold_digest: {:?}", i, &me.fold_digest[..4]);
                eprintln!("  ME[{}] r[0..2]: {:?}", i, &me.r[..me.r.len().min(2)]);
            }
        }
    }
    
    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] Transcript absorption: {:.2}ms", 
                     transcript_start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    #[cfg(feature = "debug-logs")]
    let chal_start = std::time::Instant::now();
    // Sample: α ∈ K^{log d}, β = (β_a || β_r) ∈ K^{log(dn)}, γ ∈ K
    let ch = transcript::sample_challenges(tr, ell_d, ell)?;
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[Fold] Sampled challenges:");
        eprintln!("  γ = {:?}", ch.gamma);
        if !ch.beta_a.is_empty() {
            eprintln!("  β_a[0] = {:?}", ch.beta_a[0]);
        }
        if !ch.beta_r.is_empty() {
            eprintln!("  β_r[0] = {:?}", ch.beta_r[0]);
        }
    }
    
    #[cfg(feature = "debug-logs")]
    {
        let digest = {
            let mut tmp = tr.clone();
            tmp.digest32()
        };
        eprintln!("[pi-ccs][prove] Transcript state after challenge sampling: {:?}", &digest[..4]);
        
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] Challenge sampling: {:.2}ms (α: {}, β: {}, γ: 1)",
                     chal_start.elapsed().as_secs_f64() * 1000.0, ell_d, ell);
        }
    }

    // =========================================================================
    // STEP 2: SUM-CHECK PROTOCOL
    // =========================================================================
    // Paper Section 4.4, Step 2: Perform SumCheck(T; Q)
    // where T = γ^k·Σ_{j,i} γ^{i+(j-1)k-1}·ỹ_{(i,j)}(α)
    
    #[cfg(feature = "debug-logs")]
    let mle_start = std::time::Instant::now();
    // Precompute multilinear extension partials for first instance (used in F polynomial)
    let partials_first_inst = precompute::build_mle_partials_first_inst(&s, ell_n, &insts)?;
    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] MLE partials setup: {:.2}ms",
                     mle_start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let k_total = mcs_list.len() + me_inputs.len();

    #[cfg(feature = "debug-logs")]
    let precomp_start = std::time::Instant::now();
    // Precompute components at β for efficiency:
    // - F(β_r): CCS polynomial evaluated at fixed β point
    // - NC sum: sum of norm constraints over hypercube (multiplied by eq weights)
    let beta_block = precompute::precompute_beta_block(&s, params, &insts, witnesses, me_witnesses, &ch, ell_d, ell_n)?;
    // Precompute Eval polynomial components at (α, r_input)
    let eval_row_partial = precompute::precompute_eval_row_partial(&s, me_witnesses, &ch, k_total, ell_n)?;
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs] eval_row_partial.len: {}", eval_row_partial.len());
        eprintln!("[pi-ccs] eval_row_partial[0..4]: {:?}", 
                 &eval_row_partial[0..4.min(eval_row_partial.len())]
                 .iter()
                 .map(|x| format_ext(*x))
                 .collect::<Vec<_>>());
    }
    // Precompute NC matrices for all k instances
    let nc_y_matrices = precompute::precompute_nc_full_rows(&s, witnesses, me_witnesses, ell_n)?;
    #[cfg(feature = "debug-logs")]
    {
        println!("🔧 [PRECOMPUTE] f_at_beta_r, nc_sum_beta, G_eval: {:.2}ms",
                 precomp_start.elapsed().as_secs_f64() * 1000.0);
        println!("🔧 [F_BETA] f_at_beta_r = {}", format_ext(beta_block.f_at_beta_r));
        println!("🔧 [NC_SUM] nc_sum_hypercube = {}", format_ext(beta_block.nc_sum_hypercube));
    }

    // Compute initial sum: F(β_r) + NC_sum + <G_eval, χ_r>
    let initial_sum = precompute::compute_initial_sum_components(
        &beta_block,
        me_inputs.first().map(|me| me.r.as_slice()),
        &eval_row_partial,
    )?;
    #[cfg(feature = "debug-logs")]
    {
        println!("🔧 [INITIAL_SUM] Components:");
        println!("   F(β_r)   = {}", format_ext(beta_block.f_at_beta_r));
        println!("   NC_sum   = {}", format_ext(beta_block.nc_sum_hypercube));
        println!("   TOTAL    = {}", format_ext(initial_sum));
    }

    // Bind initial sum to transcript (for verifier)
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

    // Optional: cross-check claimed initial sum vs. paper-exact hypercube sum
    if std::env::var("NEO_PAPER_CROSSCHECK").ok().as_deref() == Some("1") {
        let paper_sum = crate::paper_exact::sum_q_over_hypercube_paper_exact(
            &s, params, witnesses, me_witnesses, &ch, ell_d, ell_n,
            me_inputs.first().map(|me| me.r.as_slice()),
        );
        let fmt_k = |x: K| { let c = x.as_coeffs(); format!("{} + {}·u", c[0], c[1]) };
        eprintln!("🔬 [crosscheck] initial_sum(engine) = {}", fmt_k(initial_sum));
        eprintln!("🔬 [crosscheck] hypercube_sum(paper) = {}", fmt_k(paper_sum));

        // Per-block breakdown: F_beta and NC_sum
        let chi_beta_r = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
        let mut f_beta_paper = K::ZERO;
        {
            let mut m_vals = vec![K::ZERO; s.t()];
            for row in 0..s.n {
                for j in 0..s.t() { m_vals[j] = K::from(insts[0].mz[j][row]); }
                let f_row = s.f.eval_in_ext::<K>(&m_vals);
                f_beta_paper += chi_beta_r[row] * f_row;
            }
        }
        let f_beta_engine = beta_block.f_at_beta_r;
        let nc_engine = beta_block.nc_sum_hypercube;
        let nc_paper = paper_sum - f_beta_paper; // no Eval term for k=1

        eprintln!("🔬 [crosscheck] F_beta(engine) = {}", fmt_k(f_beta_engine));
        eprintln!("🔬 [crosscheck] F_beta(paper)  = {}", fmt_k(f_beta_paper));
        eprintln!("🔬 [crosscheck] NC_sum(engine) = {}", fmt_k(nc_engine));
        eprintln!("🔬 [crosscheck] NC_sum(paper)  = {}", fmt_k(nc_paper));

        // Optional NC sampling probe: compare direct dot vs precomputed y_matrices at a few (xa,xr)
        if std::env::var("NEO_PAPER_NC_PROBE").ok().as_deref() == Some("1") {
            let d = neo_math::D;
            let d_pad = 1usize << ell_d;
            let n_pad = 1usize << ell_n;
            let mut xa_samples = vec![0usize, 1usize];
            if d > 0 { xa_samples.push(d - 1); }
            if d < d_pad { xa_samples.push(d); }
            if d_pad > 0 { xa_samples.push(d_pad - 1); }
            xa_samples.retain(|&x| x < d_pad);
            xa_samples.sort(); xa_samples.dedup();

            let mut xr_samples = vec![0usize, 1usize];
            if s.n > 0 { xr_samples.push(s.n - 1); }
            if s.n < n_pad { xr_samples.push(s.n); }
            if n_pad > 0 { xr_samples.push(n_pad - 1); }
            xr_samples.retain(|&x| x < n_pad);
            xr_samples.sort(); xr_samples.dedup();

            let w_beta_a = neo_ccs::utils::tensor_point::<K>(&ch.beta_a);
            let w_beta_r = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
            let m1 = &s.matrices[0];

            eprintln!("[NC-probe] sampling (xa,xr) across instances (showing first 2)");
            for &xa in &xa_samples {
                for &xr in &xr_samples {
                    let eq_a = w_beta_a[xa];
                    let eq_r = w_beta_r[xr];
                    eprintln!("[NC-probe] xa={}, xr={}, eq_a={}, eq_r={}", xa, xr, format_ext(eq_a), format_ext(eq_r));

                    // Iterate first two instances (if available)
                    let mut idx = 0usize;
                    for Zi in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()) {
                        if idx >= 2 { break; }
                        // Direct dot: y = sum_c Z[xa,c] * M1[xr,c] (with zero padding)
                        let mut y_direct = K::ZERO;
                        if xa < Zi.rows() && xr < m1.rows() {
                            for c in 0..s.m {
                                if c < Zi.cols() && c < m1.cols() {
                                    y_direct += K::from(Zi[(xa, c)]) * K::from(m1[(xr, c)]);
                                }
                            }
                        }
                        // From precomputed y_matrices (Ajtai rows × rows)
                        let y_from_mat = if xa < nc_y_matrices[idx].len() && xr < nc_y_matrices[idx][xa].len() {
                            nc_y_matrices[idx][xa][xr]
                        } else { K::ZERO };

                        let nc_val = crate::pi_ccs::nc_core::range_product::<F>(y_direct, params.b);
                        let weighted = eq_a * eq_r * nc_val;
                        eprintln!(
                            "  i={} y(dot)={} y(mat)={} NC(y)={} eq·NC={}",
                            idx + 1, format_ext(y_direct), format_ext(y_from_mat), format_ext(nc_val), format_ext(weighted)
                        );
                        idx += 1;
                    }
                }
            }
        }

        // Panic-on-drift toggle: stop immediately on mismatch to reduce log noise
        if std::env::var("NEO_PAPER_PANIC_ON_DRIFT").ok().as_deref() == Some("1") {
            if initial_sum != paper_sum {
                panic!(
                    "[PAPER-DRIFT] initial_sum diverged: engine={} paper={}",
                    fmt_k(initial_sum), fmt_k(paper_sum)
                );
            }
            if f_beta_engine != f_beta_paper {
                panic!(
                    "[PAPER-DRIFT] F_beta mismatch: engine={} paper={}",
                    fmt_k(f_beta_engine), fmt_k(f_beta_paper)
                );
            }
            if nc_engine != nc_paper {
                panic!(
                    "[PAPER-DRIFT] NC_sum mismatch: engine={} paper={}",
                    fmt_k(nc_engine), fmt_k(nc_paper)
                );
            }
        }
    }

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs][prove] About to start sumcheck with initial_sum={}", format_ext(initial_sum));
        let digest = {
            let mut tmp = tr.clone();
            tmp.digest32()
        };
        eprintln!("[pi-ccs][prove] Transcript state before sumcheck: {:?}", &digest[..4]);
    }

    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔍 Sum-check starting: {} instances, {} rounds", insts.len(), ell);
        }
    }

    let sample_xs_generic: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();
    
    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs][prove] d_sc={}, sample_xs_generic.len()={}", d_sc, sample_xs_generic.len());

    // Use tensor_point for all multilinear extensions to ensure consistent ordering
    let w_beta_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_a);
    let w_alpha_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let w_beta_r_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    let d_n_a = 1usize << ell_d;
    let d_n_r = 1usize << ell_n;
    
    debug_assert_eq!(w_beta_a_partial.len(), d_n_a);
    debug_assert_eq!(w_alpha_a_partial.len(), d_n_a);
    debug_assert_eq!(w_beta_r_partial.len(), d_n_r);
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs] beta_a challenges: {:?}", ch.beta_a.iter().map(|&x| format_ext(x)).collect::<Vec<_>>());
        eprintln!("[pi-ccs] w_beta_a_partial[0..4]: {:?}", w_beta_a_partial[0..4].iter().map(|&x| format_ext(x)).collect::<Vec<_>>());
        // Compare with direct computation
        let beta_full = [&ch.beta_a[..], &ch.beta_r[..]].concat();
        let chi_beta = neo_ccs::utils::tensor_point::<K>(&beta_full);
        eprintln!("[pi-ccs] tensor_point(beta_a+beta_r)[0..4]: {:?}", chi_beta[0..4].iter().map(|&x| format_ext(x)).collect::<Vec<_>>());
    }

    let w_eval_r_partial = if let Some(ref r_inp_full) = me_inputs.first().map(|me| &me.r) {
        // Use tensor_point for consistency with how it's used elsewhere
        let w_eval_r = neo_ccs::utils::tensor_point::<K>(r_inp_full);
        debug_assert_eq!(w_eval_r.len(), d_n_r);
        #[cfg(feature = "debug-logs")]
        {
            eprintln!("[pi-ccs] Initialized w_eval_r_partial with ME input's r: {:?}", r_inp_full);
            eprintln!("[pi-ccs] w_eval_r_partial[0..4]: {:?}", &w_eval_r[0..4.min(w_eval_r.len())]);
        }
        w_eval_r
    } else {
        vec![K::ZERO; d_n_r]
    };

    // Debug checks to ensure Eval row gate uses χ_r and not χ_{β_r}
    #[cfg(debug_assertions)]
    {
        if w_eval_r_partial.len() == w_beta_r_partial.len() {
            let same = w_eval_r_partial.iter().zip(&w_beta_r_partial).all(|(a,b)| *a == *b);
            debug_assert!(
                !same,
                "Eval row gate (chi_r) must not equal beta_r gate (chi_beta_r). Check r vs beta_r initialization."
            );
        }
    }

    #[cfg(feature = "debug-logs")]
    {
        use crate::pi_ccs::format_ext;
        // Cross-check round-0 Eval identity: <G_eval, χ_r> vs <G_eval, χ_{β_r}>
        let dot_eval_r: K = eval_row_partial
            .iter()
            .zip(&w_eval_r_partial)
            .fold(K::ZERO, |acc, (&g, &w)| acc + g * w);
        let dot_eval_beta: K = eval_row_partial
            .iter()
            .zip(&w_beta_r_partial)
            .fold(K::ZERO, |acc, (&g, &w)| acc + g * w);
        eprintln!("[pi-ccs][debug] Eval<r>       = {}", format_ext(dot_eval_r));
        eprintln!("[pi-ccs][debug] Eval<beta_r> = {}", format_ext(dot_eval_beta));
    }

    let z_witness_refs: Vec<&Mat<F>> = witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()).collect();
    let me_offset = witnesses.len();

    // Build gamma_pows for k instances: [γ^1, γ^2, ..., γ^k]
    let mut nc_row_gamma_pows_vec: Vec<K> = Vec::with_capacity(k_total);
    {
        let mut gcur = ch.gamma;
        for _i in 0..k_total { 
            nc_row_gamma_pows_vec.push(gcur);
            #[cfg(feature = "debug-logs")]
            eprintln!("[γ-power] γ^{} = {}", _i+1, format_ext(gcur));
            gcur *= ch.gamma; 
        }
    }

    let mut oracle = GenericCcsOracle::new(
        &s,
        partials_first_inst,
        w_beta_a_partial.clone(),
        w_alpha_a_partial.clone(),
        w_beta_r_partial.clone(),
        w_eval_r_partial.clone(),
        eval_row_partial.clone(),
        z_witness_refs,
        &mats_csr[0],
        &mats_csr,
        nc_y_matrices,
        nc_row_gamma_pows_vec,
        ch.gamma,
        k_total,
        params.b,
        ell_d,
        ell_n,
        d_sc,
        me_offset,
        initial_sum,
        beta_block.f_at_beta_r,
        beta_block.nc_sum_hypercube,
    );

    // Execute sum-check protocol: reduces T = sum over {0,1}^{log(dn)} to v = Q(α', r')
    let SumcheckOutput { rounds, challenges: r, final_sum: running_sum_sc } =
        run_sumcheck(tr, &mut oracle, initial_sum, &sample_xs_generic)?;
    
    #[cfg(feature = "debug-logs")]
    {
        // NOTE: digest32() is mutating! Don't call it for debug logging
        eprintln!("[pi-ccs][prove] Completed run_sumcheck");
    }

    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] Sum-check rounds complete: {} rounds", ell);
        }
    }

    // Extract final random point (α', r') from sum-check
    if r.len() != ell { return Err(PiCcsError::SumcheckError("bad r length".into())); }
    let (r_prime, alpha_prime) = r.split_at(ell_n);

    // =========================================================================
    // STEP 3: BUILD OUTPUT ME INSTANCES
    // =========================================================================
    // Paper Section 4.4, Step 3: P sends y'_{(i,j)} = Z_i·M_j^T·r̂' for all i,j
    
    #[cfg(feature = "debug-logs")]
    {
        // NOTE: digest32() is mutating! Don't call it for debug logging
        eprintln!("[pi-ccs][prove] About to call build_me_outputs");
    }
    
    #[cfg(feature = "debug-logs")]
    let me_start = std::time::Instant::now();
    
    // Capture header digest explicitly right after sumcheck, before building outputs
    let fold_digest = tr.digest32();
    
    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs][prove] Captured header_digest after sumcheck: {:?}", &fold_digest[..4]);
    
    let out_me = outputs::build_me_outputs(tr, &s, params, &mats_csr, &insts, me_inputs, me_witnesses, r_prime, ell_d, fold_digest, l)?;
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs][prove] Completed build_me_outputs");
    }
    #[cfg(feature = "debug-logs")]
    {
        let dbg_timing = std::env::var("NEO_TIMING").ok().as_deref() == Some("1");
        if dbg_timing {
            println!("🔧 [TIMING] ME instance building ({} outputs): {:.2}ms", 
                     out_me.len(), me_start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    #[cfg(feature = "debug-logs")]
    {
        use crate::pi_ccs::terminal::rhs_Q_apr;
        let rhs_dbg = rhs_Q_apr(&s, &ch, r_prime, alpha_prime, mcs_list, me_inputs, &out_me, params)?;
        eprintln!("[pi-ccs][prove] running_sum_sc={}, rhs_dbg={}", 
                  format_ext(running_sum_sc), format_ext(rhs_dbg));
    }

    // Optional cross-check against paper-exact reference implementation for drift debugging
    if std::env::var("NEO_PAPER_CROSSCHECK").ok().as_deref() == Some("1") {
        // Compute engine RHS (paper-faithful terminal) and paper-exact RHS using literal definitions
        let me_inputs_r_opt: Option<&[K]> = me_inputs.first().map(|me| me.r.as_slice());

        let engine_rhs = crate::pi_ccs::terminal::rhs_Q_apr(
            &s, &ch, r_prime, alpha_prime, mcs_list, me_inputs, &out_me, params,
        )?;

        let paper_out = crate::paper_exact::build_me_outputs_paper_exact(
            &s, params, mcs_list, witnesses, me_inputs, me_witnesses, r_prime, ell_d, fold_digest, l,
        );
        let paper_rhs = crate::paper_exact::rhs_terminal_identity_paper_exact(
            &s, params, &ch, r_prime, alpha_prime, &paper_out, me_inputs_r_opt,
        );

        // Also compute true Q(α', r') from witnesses (paper LHS at ext point)
        let (paper_q_ext_lhs, _rhs_dummy) = crate::paper_exact::q_eval_at_ext_point_paper_exact(
            &s, params,
            &insts.iter().map(|i| neo_ccs::McsWitness{ w: vec![], Z: i.Z.clone() }).collect::<Vec<_>>(),
            me_witnesses, alpha_prime, r_prime, &ch,
        );

        let fmt_k = |x: K| {
            let c = x.as_coeffs();
            format!("{} + {}·u", c[0], c[1])
        };

        eprintln!("🔬 [crosscheck] running_sum_sc = {}", fmt_k(running_sum_sc));
        eprintln!("🔬 [crosscheck] engine_rhs     = {}", fmt_k(engine_rhs));
        eprintln!("🔬 [crosscheck] paper_rhs      = {}", fmt_k(paper_rhs));
        eprintln!("🔬 [crosscheck] paper_Q(α',r') = {}", fmt_k(paper_q_ext_lhs));

        // Quick sanity on outputs count/shape parity
        if paper_out.len() != out_me.len() {
            eprintln!("🔬 [crosscheck] outputs count differ: engine={}, paper={}", out_me.len(), paper_out.len());
        } else if let (Some(e0), Some(p0)) = (out_me.get(0), paper_out.get(0)) {
            let t_e = e0.y.len();
            let t_p = p0.y.len();
            eprintln!("🔬 [crosscheck] outputs t (engine,paper) = ({},{})", t_e, t_p);
            if t_e == t_p && t_e > 0 {
                eprintln!("🔬 [crosscheck] y[0] len (engine,paper) = ({},{})", e0.y[0].len(), p0.y[0].len());
            }
        }

        // Panic-on-drift toggle at terminal: any mismatch stops immediately
        if std::env::var("NEO_PAPER_PANIC_ON_DRIFT").ok().as_deref() == Some("1") {
            if engine_rhs != paper_rhs {
                panic!(
                    "[PAPER-DRIFT] terminal RHS mismatch: engine_rhs={} paper_rhs={}",
                    fmt_k(engine_rhs), fmt_k(paper_rhs)
                );
            }
            if running_sum_sc != engine_rhs {
                panic!(
                    "[PAPER-DRIFT] running_sum != Q(α',r'): running_sum={} rhs={}",
                    fmt_k(running_sum_sc), fmt_k(engine_rhs)
                );
            }
        }
    }
    #[cfg(not(feature = "debug-logs"))]
    let _ = (running_sum_sc, alpha_prime);

    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs][prove] initial_sum for proof = {}", format_ext(initial_sum));
    
    let proof = PiCcsProof { 
        sumcheck_rounds: rounds,
        sumcheck_challenges: r,
        sumcheck_final: running_sum_sc,
        header_digest: fold_digest,
        sc_initial_sum: Some(initial_sum),
        challenges_public: ch,
    };
    
    // Defensive check: ensure output structure matches inputs
    // (Active in debug builds or with strict-checks feature)
    checks::sanity_check_outputs_against_inputs(&s, mcs_list, me_inputs, &out_me)?;
    
    Ok((out_me, proof))
}

// ============================================================================
// VERIFIER: Π_CCS (Section 4.4)
// ============================================================================

/// Verify Π_CCS reduction: MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k
///
/// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS (verification)
///
/// # Verification Strategy
///
/// The verifier checks the proof without access to witnesses by:
/// 1. Replaying the transcript to derive challenges α, β, γ
/// 2. Verifying sum-check rounds to extract (α', r') and running_sum v
/// 3. Computing Q(α', r') from public values only
/// 4. Checking v ?= Q(α', r')
///
/// # Computing Q(α', r') from Public Data
///
/// **F'**: Recompose from output y'_{(1,j)} values:
/// - m_j = Σ_ℓ b^{ℓ-1}·y'_{(1,j),ℓ}  (Ajtai digit recomposition)
/// - F' = f(m_1,...,m_t)
///
/// **N_i'**: Norm constraint polynomial at α':
/// - N_i' = ∏_{j=-b+1}^{b-1} (ỹ'_{(i,1)}(α') - j)
///
/// **E_{(i,j)}'**: Evaluation claims from input ME (if k>1):
/// - E_{(i,j)}' = eq((α',r'), (α,r))·ỹ'_{(i,j)}(α')
///
/// # Arguments
/// - `tr`: Transcript (must match prover's state)
/// - `params`: Security parameters
/// - `s_in`: CCS structure
/// - `mcs_list`: Input MCS instances
/// - `me_inputs`: Input ME instances (empty for k=1)
/// - `out_me`: Output ME instances claimed by prover
/// - `proof`: Sum-check rounds and binding data
///
/// # Returns
/// `true` if proof is valid, `false` otherwise
pub fn pi_ccs_verify(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s_in: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    // =========================================================================
    // STEP 0: SETUP
    // =========================================================================
    // Initialize transcript, compute dimensions (must match prover)
    
    tr.append_message(tr_labels::PI_CCS, b"");
    // Ensure identity-first CCS (M_0 = I_n) for verifier
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidStructure(format!("CCS structure invalid for identity-first: {:?}", e)))?;
    
    // Compute same parameters as prover (Section 4.3)
    // Use shared helper to ensure prover/verifier dimensions match exactly
    let context::Dims { ell_d, ell_n, ell, d_sc } = context::build_dims_and_policy(params, &s)?;

    // Bind header and instances using same functions as prover
    transcript::bind_header_and_instances(tr, params, &s, mcs_list, ell, d_sc, 0)?;
    transcript::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges using same helper as prover
    let ch = transcript::sample_challenges(tr, ell_d, ell)?;
    let alpha_vec = ch.alpha;
    let beta_a = ch.beta_a;
    let beta_r = ch.beta_r;
    let gamma = ch.gamma;
    
    // Compute T from public ME inputs for debugging purposes
    // Note: The actual initial sum includes F(β_r) + NC_sum + T, not just T alone.
    // The sum-check protocol itself enforces correctness of the full initial sum.
    #[cfg(feature = "debug-logs")]
    let t_from_public_inputs: K = {
        if me_inputs.is_empty() {
            K::ZERO
        } else {
            // χ_α for the Ajtai dimension
            let chi_alpha: Vec<K> = neo_ccs::utils::tensor_point::<K>(&alpha_vec);

            // total number of instances (MCS + ME)
            let k_total = mcs_list.len() + me_inputs.len();

            // precompute γ^k
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= gamma;
            }

            // precompute γ^{i_abs-1} for i_abs = 2..k_total (i.e., over the ME inputs only)
            let mut gamma_pow_i_abs = Vec::with_capacity(me_inputs.len());
            {
                let mut g = gamma; // γ^1 for the first ME input (i_abs = 2)
                for _ in 0..me_inputs.len() {
                    gamma_pow_i_abs.push(g);
                    g *= gamma;
                }
            }

            let mut t_sum = K::ZERO;
            for j in 0..s.t() {
                for (i_off, me) in me_inputs.iter().enumerate() {
                    let y_vec = &me.y[j];
                    // ỹ_{(i,j)}(α) = ⟨y_{(i,j)}, χ_α⟩
                    let y_mle: K = y_vec
                        .iter()
                        .zip(&chi_alpha)
                        .map(|(&y, chi)| y * *chi)
                        .sum();

                    // weight γ^{(i_abs-1) + j⋅k_total}
                    let mut w = gamma_pow_i_abs[i_off];
                    for _ in 0..j {
                        w *= gamma_to_k;
                    }
                    t_sum += w * y_mle;
                }
            }
            t_sum
        }
    };
    
    #[cfg(feature = "debug-logs")]
    {
        let digest = {
            let mut tmp = tr.clone();
            tmp.digest32()
        };
        eprintln!("[pi-ccs][verify] Transcript state after challenge sampling: {:?}", &digest[..4]);
        eprintln!("[pi-ccs][verify] Sampled challenges: alpha[0]={:?}, beta_a[0]={:?}, gamma={:?}", 
            alpha_vec.get(0), beta_a.get(0), gamma);
        eprintln!("[pi-ccs][verify] T component from public ME inputs: {}", format_ext(t_from_public_inputs));
    }

    // Use the input ME's shared r (if k>1, must be the same across me_inputs)
    // For k=1 (initial fold), me_inputs is empty
    let _r_input_opt = if !me_inputs.is_empty() {
        let r_input = &me_inputs[0].r;
        if !me_inputs.iter().all(|m| m.r == *r_input) {
            return Err(PiCcsError::InvalidInput("ME inputs must share the same r".into()));
        }
        Some(r_input)
    } else {
        None
    };

    // =========================================================================
    // STEP 2: VERIFY SUM-CHECK ROUNDS
    // =========================================================================
    // Paper Section 4.4, Step 2: Verify sum-check protocol
    // Check each round's polynomial and extract final evaluation point (α', r')
    
    if proof.sumcheck_rounds.len() != ell {
        return Err(PiCcsError::SumcheckError(format!(
            "round count mismatch: have {}, expect {}", 
            proof.sumcheck_rounds.len(), ell
        )));
    }
    // Check sum-check rounds using shared helper (derives r and running_sum)
    let d_round = d_sc;
    
    // Use the prover-carried initial sum when present; else derive from round 0
    let claimed_initial = match proof.sc_initial_sum {
        Some(s) => s,
        None => {
            if let Some(round0) = proof.sumcheck_rounds.get(0) {
                use crate::sumcheck::poly_eval_k;
                poly_eval_k(round0, K::ZERO) + poly_eval_k(round0, K::ONE)
            } else {
                K::ZERO
            }
        }
    };
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs][verify] d_round={}, claimed_initial={}", d_round, format_ext(claimed_initial));
        eprintln!("[pi-ccs][verify] Prover's initial sum = {} (includes F + NC + T)", format_ext(claimed_initial));
        eprintln!("[pi-ccs][verify] T component alone = {} (from public ME inputs)", format_ext(t_from_public_inputs));
        if let Some(round0) = proof.sumcheck_rounds.get(0) {
            use crate::sumcheck::poly_eval_k;
            let p0 = poly_eval_k(round0, K::ZERO);
            let p1 = poly_eval_k(round0, K::ONE);
            eprintln!("[pi-ccs][verify] round0: p(0)={}, p(1)={}, p(0)+p(1)={}",
                      format_ext(p0), format_ext(p1), format_ext(p0 + p1));
        }
    }
    
    // The sum-check protocol itself should enforce correctness of the full initial sum (F + NC + T).
    // No special handling for k=1 vs k=2 should be needed.
    
    // Bind initial_sum BEFORE verifying rounds (verifier side)
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[pi-ccs][verify] About to start sumcheck with initial_sum={}", format_ext(claimed_initial));
        let digest = {
            let mut tmp = tr.clone();
            tmp.digest32()
        };
        eprintln!("[pi-ccs][verify] Transcript state before sumcheck: {:?}", &digest[..4]);
    }
    
    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs][verify] d_round={}, proof.sumcheck_rounds.len()={}", d_round, proof.sumcheck_rounds.len());
    
    let (r_vec, running_sum, ok_rounds) =
        verify_sumcheck_rounds(tr, d_round, claimed_initial, &proof.sumcheck_rounds);
    
    #[cfg(feature = "debug-logs")]
    {
        // NOTE: digest32() is mutating! Don't call it for debug logging
        eprintln!("[pi-ccs][verify] Completed verify_sumcheck_rounds");
    }
    
    if !ok_rounds {
        return Err(PiCcsError::SumcheckError(format!(
            "sum-check rounds invalid (d_round={}, initial_sum={})", 
            d_round, format_ext(claimed_initial)
        )));
    }

    // Split r_vec into (r', α') - Row-first ordering (row rounds first)
    // First ell_n challenges are row bits, last ell_d challenges are Ajtai bits
    if r_vec.len() != ell {
        return Err(PiCcsError::SumcheckError(format!(
            "r_vec length mismatch: have {}, expect {}", 
            r_vec.len(), ell
        )));
    }
    let (r_prime, alpha_prime) = r_vec.split_at(ell_n);

    // =========================================================================
    // STEP 3: TRANSCRIPT BINDING & STRUCTURAL CHECKS
    // =========================================================================
    // Security: Ensure proof and outputs are bound to this specific transcript
    
    // Verifier-side terminal decomposition logs
    #[cfg(feature = "neo-logs")]
    {
        use std::cmp::min;
        eprintln!(
            "[pi-ccs][verify] ell={}, d_round={}, claimed_initial={}",
            ell, d_round, format_ext(claimed_initial)
        );
        eprintln!("[pi-ccs][verify] r_vec[0..{}]={:?}",
                  min(4, r_vec.len()), r_vec.iter().take(4).collect::<Vec<_>>());
        // no batching alphas in paper-style Q
        eprintln!("[pi-ccs][verify] running_sum   = {}", format_ext(running_sum));
    }
    
    // === Transcript binding and structural checks ===
    // Only apply transcript binding when we have sum-check rounds
    // For trivial cases (ell = 0, no rounds), skip binding checks
    if !proof.sumcheck_rounds.is_empty() {
        // The prover calls build_me_outputs which internally calls tr.digest32()
        // We need to get the digest at the same point
        #[cfg(feature = "debug-logs")]
        eprintln!("[pi-ccs][verify] About to compute header_digest after sumcheck");
        
        let digest = tr.digest32();
        
        #[cfg(feature = "debug-logs")]
        eprintln!("[pi-ccs][verify] header_digest computed: {:?}", &digest[..4]);
        // Verify proof header matches transcript state
        if proof.header_digest != digest {
            return Err(PiCcsError::InvalidStructure(format!(
                "header digest mismatch (proof={:?}, verifier={:?})",
                &proof.header_digest[..4], &digest[..4]
            )));
        }
        // Verify all output ME instances are bound to this transcript
        if !out_me.iter().all(|me| me.fold_digest == digest) {
            return Err(PiCcsError::InvalidStructure(
                "output ME instances have mismatched fold_digest".to_string()
            ));
        }
    }

    // Light structural sanity: every output ME must carry r' (row part) only
    if !out_me.iter().all(|me| me.r == r_prime) { 
        return Err(PiCcsError::InvalidStructure(
            "output ME instances have incorrect r values (should equal r_prime from sumcheck)".to_string()
        ));
    }
    
    // === Verify output ME instances are bound to the correct inputs ===
    // This prevents attacks where unrelated ME outputs pass RLC/DEC algebra
    let expected_outputs = mcs_list.len() + me_inputs.len();
    if out_me.len() != expected_outputs {
        #[cfg(feature = "debug-logs")]
        eprintln!("[pi-ccs] out_me.len() {} != expected {} (mcs={} + me_inputs={})", 
                  out_me.len(), expected_outputs, mcs_list.len(), me_inputs.len());
        return Err(PiCcsError::InvalidInput(format!(
            "out_me.len() {} != mcs_list.len() {} + me_inputs.len() {}", 
            out_me.len(), mcs_list.len(), me_inputs.len()
        )));
    }
    
    // Expected y vector length - should be padded to 2^ell_d
    let expected_len = 1usize << ell_d;
    
    // Verify MCS-derived outputs (i=1..mcs_list.len()) match input commitments
    for (i, (out, inp)) in out_me.iter().take(mcs_list.len()).zip(mcs_list.iter()).enumerate() {
        #[cfg(not(feature = "debug-logs"))]
        let _ = i;
        if out.c != inp.c {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] output {} commitment mismatch vs input", i);
            return Err(PiCcsError::InvalidInput(format!("output[{}].c != input.c", i)));
        }
        if out.m_in != inp.m_in {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] output {} m_in {} != input m_in {}", i, out.m_in, inp.m_in);
            return Err(PiCcsError::InvalidInput(format!(
                "output[{}].m_in {} != input.m_in {}", i, out.m_in, inp.m_in
            )));
        }
        
        // Shape/consistency checks: catch subtle mismatches before terminal verification
        if out.X.rows() != neo_math::D {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] out.X.rows {} != D {}", out.X.rows(), neo_math::D);
            return Err(PiCcsError::InvalidInput(format!(
                "output[{}].X.rows {} != D {}", i, out.X.rows(), neo_math::D
            )));
        }
        if out.X.cols() != inp.m_in {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] out.X.cols {} != m_in {}", out.X.cols(), inp.m_in);
            return Err(PiCcsError::InvalidInput(format!(
                "output[{}].X.cols {} != m_in {}", i, out.X.cols(), inp.m_in
            )));
        }
        if out.y.len() != s.t() {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] out.y.len {} != t {}", out.y.len(), s.t());
            return Err(PiCcsError::InvalidInput(format!(
                "output[{}].y.len {} != t {}", i, out.y.len(), s.t()
            )));
        } // Number of CCS matrices
        // Guard individual y[j] vector lengths - should be padded to 2^ell_d
        for (j, yj) in out.y.iter().enumerate() {
            if yj.len() != expected_len {
                return Err(PiCcsError::InvalidInput(format!(
                    "output[{}].y[{}].len {} != 2^ell_d = {}", i, j, yj.len(), expected_len
                )));
            }
        }
    }
    
    // Verify ME input-derived outputs (i=mcs_list.len()+1..k) match input ME instances
    let me_out_offset = mcs_list.len();
    for (idx, (out, inp)) in out_me.iter().skip(me_out_offset).zip(me_inputs.iter()).enumerate() {
        let i = me_out_offset + idx;
        #[cfg(not(feature = "debug-logs"))]
        let _ = i;
        if out.c != inp.c {
            #[cfg(feature = "debug-logs")]
            eprintln!("[pi-ccs] ME output {} commitment mismatch vs input ME", i);
            return Err(PiCcsError::InvalidInput(format!("me_output[{}].c != me_input.c", idx)));
        }
        if out.m_in != inp.m_in {
            return Err(PiCcsError::InvalidInput(format!("me_output[{}].m_in != me_input.m_in", idx)));
        }
        if out.y.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!("me_output[{}].y.len {} != t {}", idx, out.y.len(), s.t())));
        }
        for (j, yj) in out.y.iter().enumerate() {
            if yj.len() != expected_len {
                return Err(PiCcsError::InvalidInput(format!("me_output[{}].y[{}].len {} != 2^ell_d = {}", idx, j, yj.len(), expected_len)));
            }
        }
    }

    // =========================================================================
    // STEP 4: VERIFY TERMINAL IDENTITY v ?= Q(α', r') (paper-faithful)
    // =========================================================================
    use crate::pi_ccs::terminal::rhs_Q_apr;
    let rhs = rhs_Q_apr(
        &s,
        &crate::pi_ccs::transcript::Challenges {
            alpha: alpha_vec.clone(),
            beta_a: beta_a.to_vec().clone(),
            beta_r: beta_r.to_vec().clone(),
            gamma,
        },
        r_prime,
        alpha_prime,
        mcs_list,
        me_inputs,
        out_me,
        params,
    )?;
    if running_sum != rhs {
        // Provide a helpful error rather than returning Ok(false)
        return Err(PiCcsError::SumcheckError(format!(
            "terminal mismatch: running_sum != Q(α', r') (running_sum={}, rhs={})",
            format_ext(running_sum),
            format_ext(rhs)
        )));
    }

    Ok(true)
}

// ============================================================================
// CONVENIENCE WRAPPERS (Simple Use Cases)
// ============================================================================
// Simplified interfaces for common case of single MCS folding (k=1).

/// Convenience wrapper for `pi_ccs_prove` when folding a single MCS instance (k=1).
/// 
/// This is the common case for initial folds where there are no prior ME inputs.
/// Automatically passes empty slices for `me_inputs` and `me_witnesses`.
///
/// # Example
/// ```ignore
/// let (me_outputs, proof) = pi_ccs_prove_simple(
///     &mut transcript,
///     &params,
///     &ccs_structure,
///     &[mcs_instance],
///     &[mcs_witness],
///     &s_module,
/// )?;
/// ```
pub fn pi_ccs_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    l: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    pi_ccs_prove(tr, params, s, mcs_list, witnesses, &[], &[], l)
}

/// Convenience wrapper for `pi_ccs_verify` when verifying a single MCS instance (k=1).
/// 
/// This is the common case for initial folds where there are no prior ME inputs.
/// Automatically passes an empty slice for `me_inputs`.
///
/// # Example
/// ```ignore
/// let ok = pi_ccs_verify_simple(
///     &mut transcript,
///     &params,
///     &ccs_structure,
///     &[mcs_instance],
///     &me_outputs,
///     &proof,
/// )?;
/// ```
pub fn pi_ccs_verify_simple(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    pi_ccs_verify(tr, params, s, mcs_list, &[], out_me, proof)
}
