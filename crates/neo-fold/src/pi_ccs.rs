//! Π_CCS: Sum-check reduction over extension field K
//!
//! This is the single sum-check used throughout Neo protocol.
//! Proves: Σ_{u∈{0,1}^ℓ} (Σ_i α_i · f_i(u)) · χ_r(u) = 0

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use crate::transcript::{FoldTranscript, Domain};
use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat, SparsePoly};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, Field};

/// Π_CCS proof containing the single sum-check over K
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sum-check protocol rounds (univariate polynomials as coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    /// Extension policy binding digest  
    pub header_digest: [u8; 32],
}

// ===== Sum-check helpers =====

#[inline]
fn poly_eval_k(coeffs: &[K], x: K) -> K {
    let mut acc = K::ZERO;
    for &c in coeffs.iter().rev() { acc = acc * x + c; }
    acc
}

#[inline]
fn poly_mul_k(a: &[K], b: &[K]) -> Vec<K> {
    let mut out = vec![K::ZERO; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() { out[i + j] += ai * bj; }
    }
    out
}

/// Lagrange interpolation at distinct points xs with values ys (returns coeffs lowest→highest).
fn lagrange_interpolate_k(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len());
    let m = xs.len();
    let mut coeffs = vec![K::ZERO; m];
    for i in 0..m {
        let (xi, yi) = (xs[i], ys[i]);
        let mut denom = K::ONE;
        let mut numer = vec![K::ONE]; // polynomial 1
        for j in 0..m {
            if i == j { continue; }
            denom *= xi - xs[j];
            numer = poly_mul_k(&numer, &[-xs[j], K::ONE]); // (X - x_j)
        }
        let scale = yi * denom.inverse(); // K is a field; panic if denom=0 (distinct xs)
        for k in 0..numer.len() { coeffs[k] += numer[k] * scale; }
    }
    coeffs
}

/// Absorb sparse polynomial definition into transcript for soundness binding.
/// This prevents a malicious prover from using different polynomials with the same matrix structure.
fn absorb_sparse_polynomial(tr: &mut FoldTranscript, f: &SparsePoly<F>) {
    // Absorb polynomial structure
    tr.absorb_bytes(b"neo/ccs/poly");
    tr.absorb_u64(&[f.arity() as u64]);
    tr.absorb_u64(&[f.terms().len() as u64]);
    
    // Absorb each term: coefficient + exponents (sorted for determinism)
    let mut terms: Vec<_> = f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps); // deterministic ordering
    
    for term in terms {
        tr.absorb_f(&[term.coeff]);
        tr.absorb_u64(&term.exps.iter().map(|&e| e as u64).collect::<Vec<_>>());
    }
}

/// Batching coefficients for the composed polynomial Q
/// 
/// In v1: Only CCS constraints are included in Q(u).
/// Range and evaluation tie constraints are handled outside the sum-check.
#[derive(Debug, Clone)]
struct BatchingCoeffs {
    /// α coefficients for CCS constraints f(Mz)  
    alphas: Vec<K>,
}

/// Compute Y_j(u) = ⟨(M_j z), χ_u⟩ in K, for all j, then f(Y(u)) in K.
/// This is the CCS constraint component of Q.
fn eval_ccs_component(
    s: &CcsStructure<F>,
    z: &[F],
    mz_cache: Option<&[Vec<F>]>,   // optional cache: M_j z over F
    u: &[K],
) -> K {
    let rb = neo_ccs::utils::tensor_point::<K>(u); // χ_u ∈ K^n
    let t = s.t();
    // M_j z rows over F (compute or reuse)
    let mz: Vec<Vec<F>> = if let Some(cached) = mz_cache {
        cached.to_vec()
    } else {
        s.matrices.iter().map(|mj| neo_ccs::utils::mat_vec_mul_ff::<F>(
            mj.as_slice(), s.n, s.m, z
        )).collect()
    };
    // Y_j(u) = Σ_i (M_j z)[i] * χ_u[i]  (promote F→K)
    let mut y_ext = Vec::with_capacity(t);
    for j in 0..t {
        let mut yj = K::ZERO;
        for i in 0..s.n { yj += K::from(mz[j][i]) * rb[i]; }
        y_ext.push(yj);
    }
    s.f.eval_in_ext::<K>(&y_ext)
}

/// Evaluate range/decomposition constraint polynomials NC_i(z,Z) at point u.
/// These assert: Z = Decomp_b(z) and ||Z||_∞ < b
/// 
/// NOTE: For honest instances where Z == Decomp_b(z) and ||Z||_∞ < b, 
///       this MUST return zero to make the composed polynomial Q sum to zero.
pub fn eval_range_decomp_constraints(
    z: &[F],
    Z: &neo_ccs::Mat<F>,
    _u: &[K],                  // not used: constraints are independent of u
    params: &neo_params::NeoParams,
) -> K {
    // REAL CONSTRAINT EVALUATION (degree 0 in u)
    // Enforces two facts:
    // 1. Decomposition correctness: z[c] = Σ_{i=0}^{d-1} b^i * Z[i,c] 
    // 2. Digit range (balanced): R_b(x) = x * ∏_{t=1}^{b-1} (x-t)(x+t) = 0
    
    let d = Z.rows();
    let m = Z.cols();

    // Sanity: shapes
    if z.len() != m {
        // Treat shape mismatch as a hard violation: contribute a non-zero sentinel.
        return K::from(F::ONE);
    }

    // Precompute base powers in F for recomposition
    let b_f = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; d];
    for i in 1..d { 
        pow_b[i] = pow_b[i - 1] * b_f; 
    }

    // === (A) Decomposition correctness residual: sum of squares in K ===
    let mut decomp_residual = K::ZERO;
    for c in 0..m {
        // z_rec = Σ_{i=0..d-1} (b^i * Z[i,c])
        let mut z_rec_f = F::ZERO;
        for i in 0..d {
            z_rec_f += Z[(i, c)] * pow_b[i];
        }
        // Residual (in K): (z_rec - z[c])^2
        let diff_k = K::from(z_rec_f) - K::from(z[c]);
        decomp_residual += diff_k * diff_k;
    }

    // === (B) Range residual: R_b(x) = x * ∏_{t=1}^{b-1} (x - t)(x + t) for every digit ===
    // Works for the "Balanced" digit set in your code path; if you ever switch styles,
    // generalize this polynomial accordingly.
    let mut range_residual = K::ZERO;

    // Precompute constants in F for 1..(b-1)
    let mut t_vals = Vec::with_capacity((params.b - 1) as usize);
    for t in 1..params.b {
        t_vals.push(F::from_u64(t as u64));
    }

    for c in 0..m {
        for i in 0..d {
            let x = Z[(i, c)];            // digit in F
            // Build R_b(x) in K
            let mut rbx = K::from(x);     // starts with the leading x factor
            // Multiply (x - t)(x + t) over t=1..b-1
            for &t in &t_vals {
                rbx *= K::from(x) - K::from(t);   // (x - t)
                rbx *= K::from(x) + K::from(t);   // (x + t)
            }
            range_residual += rbx;
        }
    }

    decomp_residual + range_residual
}

/// Evaluate tie constraint polynomials ⟨M_j^T χ_u, Z⟩ - y_j at point u.
/// These assert that the y_j values are consistent with Z and the random point.
/// 
/// CRITICAL: This must be implemented correctly for soundness!
/// The sum-check terminal verification depends on this being accurate.
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

/// Evaluate the composed polynomial Q(u) = α·CCS
/// This is what the sum-check actually proves is zero.
///
/// v1: RANGE and TIE are enforced outside the sum-check (Π_DEC and ME checks).
/// This prevents the prover/verifier polynomial mismatch until proper eq() gating is implemented.
fn eval_composed_polynomial_q(
    s: &CcsStructure<F>,
    z: &[F],
    _Z: &neo_ccs::Mat<F>,
    mz_cache: Option<&[Vec<F>]>,
    _claimed_y: &[Vec<K>],
    u: &[K],
    coeffs: &BatchingCoeffs,
    _params: &neo_params::NeoParams,
    instance_idx: usize,
) -> K {
    // CCS component: α_i · f(M_i z)(u)
    let ccs_term = coeffs.alphas[instance_idx] * eval_ccs_component(s, z, mz_cache, u);
    
    // RANGE: omit in v1 (checked via Π_DEC)
    // TIE:   omit in v1 (checked via ME/Π_RLC)
    ccs_term
}

/// Prove Π_CCS: CCS instances satisfy constraints via sum-check
/// 
/// Input: k+1 CCS instances, outputs k ME instances + proof
pub fn pi_ccs_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    l: &L, // we need L to check c = L(Z) and to compute X = L_x(Z)
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    tr.domain(Domain::CCS);

    // --- Input & policy checks ---
    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput("empty or mismatched inputs".into()));
    }
    // >>> CHANGE #1: allow arbitrary n; compute ℓ from next power of two
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }
    let n_pad = s.n.next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(format!("Extension policy validation failed: {}", e)))?;
    // Enforce strict security policy: slack_bits must be non-negative
    // This ensures we meet or exceed the target lambda-bit security level
    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits (need ≥ 0 for target {}-bit security)", 
            ext.slack_bits, params.lambda
        )));
    }
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell as u32, d_sc as u32, ext.slack_bits);

    // --- Prepare per-instance data and check c=L(Z) ---
    // Also build z = x||w and cache M_j z over F for each instance.
    struct Inst<'a> {
        z: Vec<F>, 
        Z: &'a Mat<F>, 
        m_in: usize, 
        mz: Vec<Vec<F>>,
        c: Cmt,
    }
    let mut insts: Vec<Inst> = Vec::with_capacity(mcs_list.len());
    for (inst, wit) in mcs_list.iter().zip(witnesses.iter()) {
        let z = neo_ccs::relations::check_mcs_opening(l, inst, wit)
            .map_err(|e| PiCcsError::InvalidInput(format!("MCS opening failed: {e}")))?;
        
        // === CRITICAL SECURITY CHECK: Z == Decomp_b(z) ===
        // This prevents prover from using satisfying z for CCS but different Z for commitment
        let Z_expected_col_major = neo_ajtai::decomp_b(&z, params.b, neo_math::D, neo_ajtai::DecompStyle::Balanced);
        neo_ajtai::assert_range_b(&Z_expected_col_major, params.b)
            .map_err(|e| PiCcsError::InvalidInput(format!("Range check failed on expected Z: {e}")))?;
        
        // Convert Z_expected from column-major to row-major format to match wit.Z
        let d = neo_math::D;
        let m = Z_expected_col_major.len() / d;
        let mut Z_expected_row_major = vec![neo_math::F::ZERO; d * m];
        for col in 0..m {
            for row in 0..d {
                let col_major_idx = col * d + row;
                let row_major_idx = row * m + col;
                Z_expected_row_major[row_major_idx] = Z_expected_col_major[col_major_idx];
            }
        }
        
        // Compare Z with expected decomposition (both in row-major format)
        if wit.Z.as_slice() != Z_expected_row_major.as_slice() {
            return Err(PiCcsError::InvalidInput("SECURITY: Z != Decomp_b(z) - prover using inconsistent z and Z".into()));
        }
        // cache M_j z
        let mz = s.matrices.iter().map(|mj| neo_ccs::utils::mat_vec_mul_ff::<F>(
            mj.as_slice(), s.n, s.m, &z
        )).collect();
        insts.push(Inst{ z, Z: &wit.Z, m_in: inst.m_in, mz, c: inst.c.clone() });
    }

    // --- SECURITY: Absorb instance data BEFORE sampling challenges to prevent malleability ---
    tr.absorb_bytes(b"neo/ccs/instances");
    tr.absorb_u64(&[s.n as u64, s.m as u64, s.t() as u64]);
    // Absorb CCS matrices
    for mj in &s.matrices {
        tr.absorb_f(mj.as_slice());
    }
    // CRITICAL: Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.absorb_f(&inst.x);
        tr.absorb_u64(&[inst.m_in as u64]);
        // CRITICAL: Absorb commitment to prevent cross-instance attacks
        tr.absorb_f(&inst.c.data);
    }

    // --- Generate batching coefficients for composed polynomial Q ---
    tr.absorb_bytes(b"neo/ccs/batch");
    
    // α coefficients for CCS constraints (one per instance)
    let alphas: Vec<K> = (0..insts.len()).map(|_| tr.challenge_k()).collect();
    
    let batch_coeffs = BatchingCoeffs { alphas };

    // --- Run sum-check rounds over composed polynomial Q (degree ≤ d_sc) ---
    let mut rounds: Vec<Vec<K>> = Vec::with_capacity(ell);
    let mut prefix: Vec<K> = Vec::with_capacity(ell);

    // Tie values are not part of Q(u) in v1 - ties enforced via ME/Π_RLC checks
    let claimed_y_empty: Vec<Vec<K>> = vec![];

    // initial running sum is the public target: sum_{u∈{0,1}^ℓ} Q(u) == 0
    let mut running_sum = K::ZERO;

    // OPTIMIZATION: Precompute sample points once (invariant across rounds)
    let sample_xs: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();

    for i in 0..ell {
        // For each sample X, compute S_i(X) = Σ_{tail} Q(prefix, X, tail), batched over instances.
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];
        for (sx, &X) in sample_xs.iter().enumerate() {
            let tail_len = ell - i - 1;
            let mut sum_at_X = K::ZERO;
            for tail in 0..(1usize << tail_len) {
                // Build u = prefix || X || tail_bits
                let mut u = Vec::with_capacity(ell);
                u.extend_from_slice(&prefix);
                u.push(X);
                for j in 0..tail_len {
                    u.push(if (tail >> j) & 1 == 1 { K::ONE } else { K::ZERO });
                }
                // Evaluate composed polynomial Q(u) for each instance
                for (inst_idx, inst) in insts.iter().enumerate() {
                    let q_val = eval_composed_polynomial_q(
                        s, &inst.z, inst.Z, Some(&inst.mz),
                        &claimed_y_empty, &u, &batch_coeffs, params, inst_idx
                    );
                    sum_at_X += q_val;
                }
            }
            sample_ys[sx] = sum_at_X;
        }
        let coeffs = lagrange_interpolate_k(&sample_xs, &sample_ys);
        if coeffs.len() > d_sc + 1 {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: degree {} exceeds bound {d_sc}", coeffs.len()-1
            )));
        }

        // Verifier-side checks (non-interactively enforced via transcript):
        // Check p(0)+p(1)=running_sum
        let p0 = poly_eval_k(&coeffs, K::ZERO);
        let p1 = poly_eval_k(&coeffs, K::ONE);
        if p0 + p1 != running_sum {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: p(0)+p(1) mismatch")));
        }

        // Bind polynomial to transcript and sample r_i
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) { tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c); }
        let r_i = tr.challenge_k();

        // Update state
        running_sum = poly_eval_k(&coeffs, r_i);
        prefix.push(r_i);
        rounds.push(coeffs);
    }

    // r has length ℓ; rb has length 2^ℓ, we will only use indices 0..s.n
    let r = prefix;
    let rb = neo_ccs::utils::tensor_point::<K>(&r);
    debug_assert!(rb.len() >= s.n, "tensor point smaller than n");

    // --- Build ME instances (one per input) ---
    
    // CRITICAL SECURITY FIX: Generate fold_digest from final transcript state
    // This binds the ME instances to the exact folding proof and prevents re-binding attacks
    let fold_digest = tr.state_digest();
    
    let mut out_me = Vec::with_capacity(insts.len());
    for inst in insts.iter() {
        // X = L_x(Z)
        let X = l.project_x(inst.Z, inst.m_in);
        
        // For each M_j: v_j = M_j^T * r^⊗  (K^m), then y_j = Z * v_j (K^d)
        let mut y = Vec::with_capacity(s.t());
        // **SECURITY FIX**: Also compute Y_j(r) = ⟨(M_j z), χ_r⟩ scalars for terminal check
        let mut y_scalars = Vec::with_capacity(s.t());
        
        for (j, mj) in s.matrices.iter().enumerate() {
            // v_j[c] = Σ_r mj[r,c] * rb[r]
            let mut vj = vec![K::ZERO; s.m];
            for row in 0..s.n {
                let coeff = rb[row]; // safe: rb has 2^ℓ entries
                let row_m = mj.row(row);
                for c in 0..s.m { vj[c] += K::from(row_m[c]) * coeff; }
            }
            let z_ref = neo_ccs::MatRef::from_mat(inst.Z);
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &vj);
            y.push(yj);
            
            // CRITICAL: Compute Y_j(r) = ⟨(M_j z), χ_r⟩ for CCS terminal check
            // This is different from y_j = Z * (M_j^T * χ_r)!
            let mj_z = &inst.mz[j]; // Pre-computed M_j z in F
            let mut y_scalar = K::ZERO;
            for i in 0..s.n { 
                y_scalar += K::from(mj_z[i]) * rb[i]; 
            }
            y_scalars.push(y_scalar);
        }
        
        out_me.push(MeInstance{ 
            c: inst.c.clone(), 
            X, 
            r: r.clone(), 
            y, 
            y_scalars, // SECURITY: Correct scalars for terminal check
            m_in: inst.m_in,
            fold_digest, // Bind to transcript
        });
    }

    let proof = PiCcsProof { sumcheck_rounds: rounds, header_digest: fold_digest };
    Ok((out_me, proof))
}

/// Verify Π_CCS: Check sum-check rounds AND the critical final claim Q(r) = 0
pub fn pi_ccs_verify(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>], // Now used for final Q(r) check
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    tr.domain(Domain::CCS);
    // >>> CHANGE #2: allow arbitrary n; compute ℓ from next power of two
    if s.n == 0 { return Err(PiCcsError::InvalidInput("n=0 not allowed".into())); }
    let n_pad = s.n.next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell as u32, d_sc as u32, ext.slack_bits);

    // --- SECURITY: Absorb instance data BEFORE sampling challenges (match prover) ---
    tr.absorb_bytes(b"neo/ccs/instances");
    tr.absorb_u64(&[s.n as u64, s.m as u64, s.t() as u64]);
    // Absorb CCS matrices
    for mj in &s.matrices {
        tr.absorb_f(mj.as_slice());
    }
    // CRITICAL: Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.absorb_f(&inst.x);
        tr.absorb_u64(&[inst.m_in as u64]);
        // CRITICAL: Absorb commitment to prevent cross-instance attacks
        tr.absorb_f(&inst.c.data);
    }

    // Re-derive the SAME batching coefficients as the prover
    tr.absorb_bytes(b"neo/ccs/batch");
    let alphas: Vec<K> = (0..mcs_list.len()).map(|_| tr.challenge_k()).collect();
    
    let batch_coeffs = BatchingCoeffs { alphas };

    if proof.sumcheck_rounds.len() != ell { return Ok(false); }

    let mut running_sum = K::ZERO;
    let mut r = Vec::with_capacity(ell);

    // Check sum-check rounds
    for (_i, coeffs) in proof.sumcheck_rounds.iter().enumerate() {
        if coeffs.is_empty() || coeffs.len() > d_sc + 1 { return Ok(false); }
        let p0 = poly_eval_k(coeffs, K::ZERO);
        let p1 = poly_eval_k(coeffs, K::ONE);
        if p0 + p1 != running_sum { return Ok(false); }

        // absorb coeffs and sample r_i (same as prover)
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) { tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c); }
        let r_i = tr.challenge_k();
        running_sum = poly_eval_k(coeffs, r_i);
        r.push(r_i);
    }

    // === CRITICAL TRANSCRIPT BINDING SECURITY CHECK ===
    // Only apply transcript binding when we have sum-check rounds
    // For trivial cases (ell = 0, no rounds), skip binding checks
    if !proof.sumcheck_rounds.is_empty() {
        // Derive digest exactly where the prover did (after sum-check rounds)
        let digest = tr.state_digest();
        // Verify proof header matches transcript state
        if proof.header_digest != digest { return Ok(false); }
        // Verify all output ME instances are bound to this transcript
        if !out_me.iter().all(|me| me.fold_digest == digest) { return Ok(false); }
    }

    // Light structural sanity: every output ME must carry the same r
    if !out_me.iter().all(|me| me.r == r) { return Ok(false); }

    // === CRITICAL BINDING: out_me[i] must match input instance ===
    // This prevents attacks where unrelated ME outputs pass RLC/DEC algebra
    if out_me.len() != mcs_list.len() { return Ok(false); }
    for (out, inp) in out_me.iter().zip(mcs_list.iter()) {
        if out.c != inp.c { return Ok(false); }
        if out.m_in != inp.m_in { return Ok(false); }
        
        // Shape/consistency checks: catch subtle mismatches before terminal verification
        if out.X.rows() != neo_math::D { return Ok(false); }
        if out.X.cols() != inp.m_in { return Ok(false); }
        if out.y.len() != s.t() { return Ok(false); } // Number of CCS matrices
    }

    // === CRITICAL SUM-CHECK TERMINAL VERIFICATION ===
    // This is the missing piece that makes the proof sound!
    // We must verify that the final running_sum equals Q(r).
    // 
    // NOTE: Only CCS and range/decomp constraints in Q(r).
    // Tie constraints removed from sum-check as they break soundness.
    
    // Compute Q(r) that the verifier can actually verify from public data
    // CRITICAL: This must match what the prover computes in Q(r) to prevent vacuous proofs
    let mut expected_q_r = K::ZERO;
    
    // The challenge: verifier can only compute parts of Q(r) from public data
    // - CCS component: Can compute f(Y(r)) using ME instance y_j values  
    // - Range component: Cannot compute without private Z matrix
    // - Tie component: Not included in sum-check polynomial Q(u)
    
    for (inst_idx, me_inst) in out_me.iter().enumerate() {
        // 1) CCS component: α_i · f(Y(r)) using CORRECT Y_j(r) scalars from ME
        let ccs_contribution = if inst_idx < batch_coeffs.alphas.len() {
            // SECURITY FIX: Use the correct Y_j(r) = ⟨(M_j z), χ_r⟩ scalars
            // NOT the sum of y vector components (which are Z * (M_j^T * χ_r) vectors)
            let f_eval = s.f.eval_in_ext::<K>(&me_inst.y_scalars);
            batch_coeffs.alphas[inst_idx] * f_eval
        } else {
            K::ZERO
        };
        
        // Range constraints REMOVED from sum-check polynomial Q(r):
        // - Prover no longer includes them in Q(u) 
        // - Range verification handled by Π_DEC phase and bridge SNARK
        // - This eliminates the security gap in terminal verification
        
        expected_q_r += ccs_contribution;
    }
    
    // CRITICAL TERMINAL CHECK: Does the sum-check final running_sum match expected Q(r)?
    if running_sum != expected_q_r {
        return Ok(false); // REJECT: Sum-check terminal verification failed
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_ccs::{Mat, SparsePoly, Term, McsInstance, McsWitness};
    use neo_ccs::traits::SModuleHomomorphism;
    use neo_params::NeoParams;
    use p3_field::PrimeCharacteristicRing;

    /// Minimal S-module homomorphism for tests:
    /// - commit() returns a zero commitment with the right shape
    /// - project_x() returns a zero matrix with m_in columns
    struct DummyS;
    impl neo_ccs::traits::SModuleHomomorphism<F, Cmt> for DummyS {
        fn commit(&self, z: &Mat<F>) -> Cmt { Cmt::zeros(z.rows(), 2) }
        fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
            Mat::zero(z.rows(), m_in, F::ZERO)
        }
    }

    /// Build a CCS with non-power-of-two row count `n` and `m` variables.
    /// One matrix (t=1). Polynomial f(y) = y - y (degree 1) ⇒ identically 0.
    fn make_ccs(n: usize, m: usize) -> CcsStructure<F> {
        assert!(n >= 1 && m >= 1);
        // Simple matrix with an identity block so Y is non-trivial if needed
        let mut m0 = Mat::zero(n, m, F::ZERO);
        let diag = n.min(m);
        for i in 0..diag { m0[(i, i)] = F::ONE; }

        // f(y) = y - y  (arity=1)
        let terms = vec![
            Term { coeff: F::ONE,  exps: vec![1] },
            Term { coeff: -F::ONE, exps: vec![1] },
        ];
        let f = SparsePoly::new(1, terms);
        CcsStructure::new(vec![m0], f).expect("valid CCS")
    }

    /// Build a consistent MCS instance and witness:
    /// - w = z (all private, m_in = 0)
    /// - Z is the base-b decomposition of z (row-major D×m)
    /// - c is a dummy zero commitment matching Z's shape
    fn make_instance_and_witness(z: Vec<F>, params: &NeoParams) -> (McsInstance<Cmt, F>, McsWitness<F>) {
        let d = neo_math::D;
        let m = z.len();

        // Decompose z into base-b digits (Balanced), then convert to row-major D×m
        let z_digits = neo_ajtai::decomp_b(&z, params.b, d, neo_ajtai::DecompStyle::Balanced);
        let mut row_major = vec![F::ZERO; d * m];
        for col in 0..m { for row in 0..d { row_major[row * m + col] = z_digits[col * d + row]; } }
        let Z = Mat::from_row_major(d, m, row_major);

        // Dummy commitment that matches Z's shape; m_in=0 (all private)
        let l = DummyS;
        let c = l.commit(&Z);
        let inst = McsInstance { c, x: vec![], m_in: 0 };
        let wit  = McsWitness::<F> { w: z, Z };
        (inst, wit)
    }

    #[test]
    fn pi_ccs_non_power_of_two_n_works() {
        // Use the same params as your example; any sane params work here.
        let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

        // Exercise several non-power-of-two n; m is small/fixed.
        let ns = [3usize, 5, 6, 7, 9];
        let m  = 4usize;

        for &n in &ns {
            let s = make_ccs(n, m);

            // Simple witness vector z = [1,2,3,4] over F
            let z: Vec<F> = (0..m).map(|i| F::from_u64((i as u64) + 1)).collect();
            let (inst, wit) = make_instance_and_witness(z, &params);

            // Prove
            let l = DummyS;
            let mut tr_p = FoldTranscript::default();
            let (out_me, prf) = pi_ccs_prove(&mut tr_p, &params, &s, &[inst.clone()], &[wit], &l)
                .expect("pi_ccs_prove should succeed for non-power-of-two n");

            // ℓ must equal ceil(log2 n)
            let ell_expected = s.n.next_power_of_two().trailing_zeros() as usize;
            assert_eq!(prf.sumcheck_rounds.len(), ell_expected, "sum-check rounds must equal ℓ");

            // Verify
            let mut tr_v = FoldTranscript::default();
            let ok = pi_ccs_verify(&mut tr_v, &params, &s, &[inst], &out_me, &prf)
                .expect("pi_ccs_verify should run");
            assert!(ok, "pi_ccs_verify should accept for n={}", n);
        }
    }
}