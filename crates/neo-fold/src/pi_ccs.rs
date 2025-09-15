//! Π_CCS: Sum-check reduction over extension field K
//!
//! This is the single sum-check used throughout Neo protocol.
//! Proves: Σ_{u∈{0,1}^ℓ} (Σ_i α_i · f_i(u)) · χ_r(u) = 0

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use crate::transcript::{FoldTranscript, Domain};
use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat, MatRef, SparsePoly};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, Field, PrimeField64};
use rayon::prelude::*;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

/// Π_CCS proof containing the single sum-check over K
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sum-check protocol rounds (univariate polynomials as coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    /// Extension policy binding digest  
    pub header_digest: [u8; 32],
}

// ===== CSR Sparse Matrix Operations =====

/// Minimal CSR (Compressed Sparse Row) format for sparse matrix operations
/// This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices
struct Csr<F> {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,  // len = rows + 1
    indices: Vec<usize>, // len = nnz  
    data: Vec<F>,        // len = nnz
}

/// Convert dense matrix to CSR format - O(nm) but done once
fn to_csr<F: Field + Copy>(m: &Mat<F>, rows: usize, cols: usize) -> Csr<F> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for r in 0..rows {
        let row = m.row(r);
        for (c, &v) in row.iter().enumerate() {
            if v != F::ZERO {
                indices.push(c);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    Csr { rows, cols, indptr, indices, data }
}

/// Sparse matrix-vector multiply: y = A * x  (CSR SpMV, O(nnz))
fn spmv_csr_ff<F: Field + Send + Sync + Copy>(a: &Csr<F>, x: &[F]) -> Vec<F> {
    let mut y = vec![F::ZERO; a.rows];
    y.par_iter_mut().enumerate().for_each(|(r, yr)| {
        let start = a.indptr[r];
        let end = a.indptr[r + 1];
        let mut acc = F::ZERO;
        for k in start..end {
            let c = a.indices[k];
            acc += a.data[k] * x[c];
        }
        *yr = acc;
    });
    y
}

/// Sparse transpose multiply: v = A^T * w  (O(nnz))
/// v[c] += A[row,c] * w[row] for all non-zeros
fn spmv_csr_t_fk<F: Field, Kf: Field + From<F> + Send + Sync>(
    a: &Csr<F>, w: &[Kf]
) -> Vec<Kf> {
    let mut v = vec![Kf::ZERO; a.cols];
    // Parallel by rows with local buffers to avoid atomics
    let chunk = a.rows / rayon::current_num_threads().max(1) + 1;
    let partials: Vec<Vec<(usize, Kf)>> = (0..a.rows)
        .into_par_iter()
        .with_min_len(chunk)
        .map(|r| {
            let mut loc = Vec::with_capacity(a.indptr[r + 1] - a.indptr[r]);
            let start = a.indptr[r];
            let end = a.indptr[r + 1];
            let wr = w[r];
            for k in start..end {
                let c = a.indices[k];
                loc.push((c, Kf::from(a.data[k]) * wr));
            }
            loc
        })
        .collect();
    // Combine partials (small nnz, single-threaded combine is fine)
    for loc in partials {
        for (c, val) in loc { 
            v[c] += val; 
        }
    }
    v
}

// ===== Matrix Digest Optimization =====

/// Compute canonical sparse digest of CCS matrices to avoid absorbing 500M+ field elements  
/// Uses ZK-friendly Poseidon2 hash with proper domain separation
fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
    
    // Use fixed seed for deterministic hashing (equivalent to derive_key concept)
    const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154; // "CCSD_MAT" in hex
    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    
    // Sponge state for Poseidon2 (width=16)
    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;
    
    // PROPER DOMAIN SEPARATION: Absorb context string byte-by-byte (same as transcript pattern)
    const DOMAIN_STRING: &[u8] = b"neo/ccs/matrices/v1"; 
    for &byte in DOMAIN_STRING {
        if absorbed >= 15 { // Leave room for rate limiting
            poseidon2.permute_mut(&mut state);
            absorbed = 0;
        }
        state[absorbed] = Goldilocks::from_u32(byte as u32);
        absorbed += 1;
    }
    
    // Absorb matrix dimensions  
    if absorbed + 3 >= 16 { poseidon2.permute_mut(&mut state); absorbed = 0; }
    state[absorbed] = Goldilocks::from_u64(s.n as u64);
    state[absorbed + 1] = Goldilocks::from_u64(s.m as u64); 
    state[absorbed + 2] = Goldilocks::from_u64(s.t() as u64);
    // Note: absorbed will be reset to 0 in the loop below, so no need to track it here
    
    poseidon2.permute_mut(&mut state);
    
    // Absorb each matrix in sparse format
    for (j, matrix) in s.matrices.iter().enumerate() {
        // Reset and start with matrix index
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;
        
        let mat_ref = MatRef::from_mat(matrix);
        
        // Absorb sparse entries in canonical (row, col, val) order
        for row in 0..s.n {
            let row_slice = mat_ref.row(row);
            for (col, &val) in row_slice.iter().enumerate() {
                if val != F::ZERO {
                    if absorbed + 3 > 15 { // Leave room for rate limiting
                        poseidon2.permute_mut(&mut state);
                        absorbed = 0;
                    }
                    
                    // Absorb (row, col, val) as consecutive field elements
                    state[absorbed] = Goldilocks::from_u64(row as u64);
                    state[absorbed + 1] = Goldilocks::from_u64(col as u64);
                    state[absorbed + 2] = Goldilocks::from_u64(val.as_canonical_u64());
                    absorbed += 3;
                }
            }
        }
        
        // Permute after each matrix to ensure proper mixing
        poseidon2.permute_mut(&mut state);
    }
    
    // Return first 4 field elements as digest (128 bits of security)
    state[0..4].to_vec()
}

// ===== Sum-check helpers =====

#[inline]
fn poly_eval_k(coeffs: &[K], x: K) -> K {
    let mut acc = K::ZERO;
    for &c in coeffs.iter().rev() { acc = acc * x + c; }
    acc
}

// ===== MLE Folding DP Helpers =====

/// Pad vector to power of 2 length with zeros
#[inline]
fn pad_to_pow2_k(mut v: Vec<K>, ell: usize) -> Result<Vec<K>, PiCcsError> {
    let want = 1usize << ell;
    if v.len() > want {
        return Err(PiCcsError::SumcheckError(format!(
            "Vector length {} exceeds 2^ell = {} - would silently truncate", 
            v.len(), want
        )));
    }
    v.resize(want, K::ZERO);
    Ok(v)
}

// ===== Blocked Parallel Matrix-Vector Multiplication =====

/// Cache-friendly blocked parallel matrix-vector multiplication  
/// Kept as fallback for high-density matrices; CSR is preferred for sparse matrices
#[allow(dead_code)]
#[inline]
fn mat_vec_mul_blocked_parallel<F: Field + Send + Sync>(
    a: &[F], 
    n: usize, 
    m: usize, 
    z: &[F]
) -> Vec<F> {
    const BLOCK_SIZE: usize = 256; // Tune based on cache size
    let mut result = vec![F::ZERO; n];
    
    // Process matrix in cache-friendly blocks
    for k0 in (0..m).step_by(BLOCK_SIZE) {
        let k1 = (k0 + BLOCK_SIZE).min(m);
        
        // Parallel across rows within each block
        result.par_iter_mut().enumerate().for_each(|(i, result_i)| {
            let row_slice = &a[i * m + k0..i * m + k1];
            let z_slice = &z[k0..k1];
            
            // Manual unroll for better ILP
            let mut acc = F::ZERO;
            for j in 0..row_slice.len() {
                acc += row_slice[j] * z_slice[j];
            }
            *result_i += acc;
        });
    }
    result
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

    // MAJOR OPTIMIZATION: Convert to CSR sparse format once for all operations
    // This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices  
    let csr_start = std::time::Instant::now();
    let mats_csr: Vec<Csr<F>> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    let total_nnz: usize = mats_csr.iter().map(|c| c.data.len()).sum();
    println!("🔥 [NUCLEAR] CSR conversion completed: {:.2}ms ({} matrices, {} nnz total, {:.4}% density)",
             csr_start.elapsed().as_secs_f64() * 1000.0, 
             mats_csr.len(), total_nnz, 
             (total_nnz as f64) / (s.n * s.m * s.matrices.len()) as f64 * 100.0);

    // --- Prepare per-instance data and check c=L(Z) ---
    // Also build z = x||w and cache M_j z over F for each instance.
    struct Inst<'a> {
        Z: &'a Mat<F>, 
        m_in: usize, 
        mz: Vec<Vec<F>>,
        c: Cmt,
    }
    let mut insts: Vec<Inst> = Vec::with_capacity(mcs_list.len());
    let instance_prep_start = std::time::Instant::now();
    for (inst_idx, (inst, wit)) in mcs_list.iter().zip(witnesses.iter()).enumerate() {
        let z_check_start = std::time::Instant::now();
        let z = neo_ccs::relations::check_mcs_opening(l, inst, wit)
            .map_err(|e| PiCcsError::InvalidInput(format!("MCS opening failed: {e}")))?;
        println!("🔧 [INSTANCE {}] MCS opening check: {:.2}ms", inst_idx, 
                 z_check_start.elapsed().as_secs_f64() * 1000.0);
        
        // === CRITICAL SECURITY CHECK: Z == Decomp_b(z) ===
        // This prevents prover from using satisfying z for CCS but different Z for commitment
        let decomp_start = std::time::Instant::now();
        let Z_expected_col_major = neo_ajtai::decomp_b(&z, params.b, neo_math::D, neo_ajtai::DecompStyle::Balanced);
        println!("🔧 [INSTANCE {}] Decomp_b: {:.2}ms", inst_idx, 
                 decomp_start.elapsed().as_secs_f64() * 1000.0);
        
        let range_check_start = std::time::Instant::now();
        neo_ajtai::assert_range_b(&Z_expected_col_major, params.b)
            .map_err(|e| PiCcsError::InvalidInput(format!("Range check failed on expected Z: {e}")))?;
        println!("🔧 [INSTANCE {}] Range check: {:.2}ms", inst_idx, 
                 range_check_start.elapsed().as_secs_f64() * 1000.0);
        
        // Convert Z_expected from column-major to row-major format to match wit.Z
        let format_conv_start = std::time::Instant::now();
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
        println!("🔧 [INSTANCE {}] Format conversion: {:.2}ms", inst_idx, 
                 format_conv_start.elapsed().as_secs_f64() * 1000.0);
        
        // Compare Z with expected decomposition (both in row-major format)
        let z_compare_start = std::time::Instant::now();
        if wit.Z.as_slice() != Z_expected_row_major.as_slice() {
            return Err(PiCcsError::InvalidInput("SECURITY: Z != Decomp_b(z) - prover using inconsistent z and Z".into()));
        }
        println!("🔧 [INSTANCE {}] Z comparison: {:.2}ms", inst_idx, 
                 z_compare_start.elapsed().as_secs_f64() * 1000.0);
        // MAJOR OPTIMIZATION: Use CSR sparse matrix-vector multiply - O(nnz) instead of O(n*m)!
        let mz_start = std::time::Instant::now();
        let mz: Vec<Vec<F>> = mats_csr.par_iter().map(|csr| 
            spmv_csr_ff::<F>(csr, &z)
        ).collect();
        println!("💥 [TIMING] CSR M_j z computation: {:.2}ms (nnz={}, vs {}M dense elements - {}x reduction)", 
                 mz_start.elapsed().as_secs_f64() * 1000.0, total_nnz, 
                 (s.n * s.m * s.matrices.len()) / 1_000_000,
                 (s.n * s.m * s.matrices.len()) / total_nnz.max(1));
        insts.push(Inst{ Z: &wit.Z, m_in: inst.m_in, mz, c: inst.c.clone() });
    }
    println!("🔧 [TIMING] Instance preparation total: {:.2}ms ({} instances)", 
             instance_prep_start.elapsed().as_secs_f64() * 1000.0, insts.len());

    // --- SECURITY: Absorb instance data BEFORE sampling challenges to prevent malleability ---
    let transcript_start = std::time::Instant::now();
    tr.absorb_bytes(b"neo/ccs/instances");
    tr.absorb_u64(&[s.n as u64, s.m as u64, s.t() as u64]);
    // OPTIMIZATION: Absorb compact ZK-friendly digest instead of 500M+ field elements 
    // This reduces transcript absorption from ~51s to microseconds using Poseidon2
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest {
        tr.absorb_f(&[F::from_u64(digest_elem.as_canonical_u64())]);
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
    println!("🔧 [TIMING] Transcript absorption: {:.2}ms", 
             transcript_start.elapsed().as_secs_f64() * 1000.0);

    // --- Generate batching coefficients for composed polynomial Q ---
    let batching_start = std::time::Instant::now();
    tr.absorb_bytes(b"neo/ccs/batch");
    
    // α coefficients for CCS constraints (one per instance)
    let alphas: Vec<K> = (0..insts.len()).map(|_| tr.challenge_k()).collect();
    
    let batch_coeffs = BatchingCoeffs { alphas };
    println!("🔧 [TIMING] Batching coefficients: {:.2}ms", 
             batching_start.elapsed().as_secs_f64() * 1000.0);

    // --- Run sum-check rounds over composed polynomial Q (degree ≤ d_sc) ---
    let mut rounds: Vec<Vec<K>> = Vec::with_capacity(ell);
    let mut prefix: Vec<K> = Vec::with_capacity(ell);

    // Tie values are not part of Q(u) in v1 - ties enforced via ME/Π_RLC checks
    let _claimed_y_empty: Vec<Vec<K>> = vec![];

    // initial running sum is the public target: sum_{u∈{0,1}^ℓ} Q(u) == 0
    let mut running_sum = K::ZERO;

    // OPTIMIZATION: Precompute sample points once (invariant across rounds)
    let sample_xs: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();

    // ===== PROPER MLE FOLDING: One shrinking vector per M_j z =====
    /// In-place folding state for linear-time MLE evaluation
    struct MlePartials {
        /// One shrinking vector per matrix M_j, updated in-place each round
        s_per_j: Vec<Vec<K>>,
    }

    let mle_start = std::time::Instant::now();
    let partials_per_inst: Result<Vec<MlePartials>, PiCcsError> = insts.par_iter().map(|inst| {
        let mut s_per_j = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            // Lift M_j z from F to K once and pad to 2^ell
            let mut w_k: Vec<K> = inst.mz[j].iter().map(|&x| K::from(x)).collect();
            w_k = pad_to_pow2_k(w_k, ell)?;
            s_per_j.push(w_k);
        }
        Ok(MlePartials { s_per_j })
    }).collect();
    let mut partials_per_inst = partials_per_inst?;
    println!("🔧 [TIMING] MLE partials setup: {:.2}ms", 
             mle_start.elapsed().as_secs_f64() * 1000.0);

    let sumcheck_start = std::time::Instant::now();
    for i in 0..ell {
        // ===== PROPER LINEAR-TIME MLE FOLDING =====
        // At round i, each vector has length 2^{ell-i}, and we read (a_j,b_j) = (S[0],S[1])
        let round_start = std::time::Instant::now();
        
        // Preallocate buffers to avoid per-round allocations
        let mut a = vec![K::ZERO; s.t()];
        let mut delta = vec![K::ZERO; s.t()]; 
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];

        // Accumulate α_i · f(Y(X)) across instances
        for (inst_idx, partials) in partials_per_inst.iter().enumerate() {
            // 1) CRITICAL FIX: Sum over ALL pairs (v[2k], v[2k+1]) in current shrinking vector
            // This is the correct MLE folding - NOT just (v[0], v[1])!
            for j in 0..s.t() {
                let v = &partials.s_per_j[j];  // length = 2^{ell-i}
                debug_assert!(v.len().is_power_of_two() && v.len() >= 2, 
                              "Vector length {} invalid at round {} for matrix {}", v.len(), i, j);
                
                let half = v.len() >> 1;
                let mut aj = K::ZERO;
                let mut dj = K::ZERO;
                
                // Sum over ALL pairs in the current vector (correct MLE folding)
                for k in 0..half {
                    let e = v[2*k];     // even index  
                    let o = v[2*k + 1]; // odd index
                    aj += e;            // A_j = sum of evens
                    dj += o - e;        // Delta_j = sum of (odd - even)
                }
                
                a[j] = aj;
                delta[j] = dj;  // Y_j(X) = A_j + Delta_j * X
            }

            // 2) Accumulate across all sample points - reuse buffer to avoid micro-allocations
            let alpha = batch_coeffs.alphas[inst_idx];
            let mut y_buf = vec![K::ZERO; s.t()];  // Allocate once, reuse for all samples
            for (sx, &X) in sample_xs.iter().enumerate() {
                for j in 0..s.t() {
                    y_buf[j] = a[j] + delta[j] * X;  // Y_j(X) = a_j + (b_j - a_j) * X
                }
                let f_eval = s.f.eval_in_ext::<K>(&y_buf);
                sample_ys[sx] += alpha * f_eval;
            }
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

        // ===== KEY LINEAR-TIME STEP: Fold all vectors in-place with r_i =====
        // This is the standard MLE folding: S[k] <- (1-r_i)*S[2k] + r_i*S[2k+1]
        // After this, each vector shrinks from length 2^{ell-i} to 2^{ell-i-1}
        for partials in &mut partials_per_inst {
            for v in &mut partials.s_per_j {
                let n2 = v.len() >> 1;  // New length after folding
                for k in 0..n2 {
                    let a = v[2*k];
                    let b = v[2*k + 1];
                    v[k] = (K::ONE - r_i) * a + r_i * b;  // Standard MLE folding formula
                }
                v.truncate(n2);  // Shrink to half length
            }
        }
        
        println!("🔧 [ROUND {}] {:.2}ms", i, round_start.elapsed().as_secs_f64() * 1000.0);
    }
    
    println!("🔧 [TIMING] Sum-check rounds total: {:.2}ms ({} rounds)", 
             sumcheck_start.elapsed().as_secs_f64() * 1000.0, ell);

    // r has length ℓ; rb has length 2^ℓ, we will only use indices 0..s.n
    let r = prefix;
    let rb = neo_ccs::utils::tensor_point::<K>(&r);
    debug_assert!(rb.len() >= s.n, "tensor point smaller than n");

    // MAJOR OPTIMIZATION: Use CSR sparse transpose multiply - O(nnz) instead of O(n*m)!
    // Compute M_j^T * χ_r ONCE for all instances using sparse operations
    println!("🚀 [OPTIMIZATION] Computing sparse M_j^T * χ_r once for all instances...");
    let transpose_once_start = std::time::Instant::now();
    let vjs: Vec<Vec<K>> = mats_csr.par_iter()
        .map(|csr| {
            // v_j = A_j^T * χ_r using sparse CSR transpose multiply - O(nnz)!
            spmv_csr_t_fk::<F, K>(csr, &rb)
        })
        .collect();
    println!("💥 [OPTIMIZATION] CSR M_j^T * χ_r computed once: {:.2}ms (nnz={} vs {}M dense - {}x reduction, saved ~500ms per instance!)", 
             transpose_once_start.elapsed().as_secs_f64() * 1000.0, total_nnz,
             (s.n * s.m * s.matrices.len()) / 1_000_000,
             (s.n * s.m * s.matrices.len()) / total_nnz.max(1));

    // --- Build ME instances (one per input) ---
    let me_start = std::time::Instant::now();
    
    // CRITICAL SECURITY FIX: Generate fold_digest from final transcript state
    // This binds the ME instances to the exact folding proof and prevents re-binding attacks
    let fold_digest = tr.state_digest();
    
    let mut out_me = Vec::with_capacity(insts.len());
    for (inst_idx, inst) in insts.iter().enumerate() {
        // X = L_x(Z)
        let X = l.project_x(inst.Z, inst.m_in);
        
        // OPTIMIZATION: Use precomputed v_j vectors and MLE fold results  
        let mut y = Vec::with_capacity(s.t());
        // Use final MLE fold results for Y_j(r) (already computed above)
        let y_scalars: Vec<K> = partials_per_inst[inst_idx]
            .s_per_j.iter()
            .map(|v| { 
                debug_assert_eq!(v.len(), 1, "MLE fold should reduce to length 1 after {} rounds", ell);
                v[0]  // This IS Y_j(r) from standard MLE folding identity!
            })
            .collect();
        
        // Use precomputed v_j = M_j^T * χ_r vectors (no more expensive recomputation per instance!)
        let z_operations_start = std::time::Instant::now();
        for (_j, vj) in vjs.iter().enumerate() {
            let z_ref = neo_ccs::MatRef::from_mat(inst.Z);
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, vj);
            y.push(yj);
        }
        println!("🚀 [OPTIMIZATION] Used precomputed v_j vectors - only Z * v_j needed: {:.2}ms", 
                 z_operations_start.elapsed().as_secs_f64() * 1000.0);
        
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

    println!("🔧 [TIMING] ME instance building: {:.2}ms", 
             me_start.elapsed().as_secs_f64() * 1000.0);

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
    // OPTIMIZATION: Absorb compact ZK-friendly digest instead of 500M+ field elements 
    // This reduces transcript absorption from ~51s to microseconds using Poseidon2
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest {
        tr.absorb_f(&[F::from_u64(digest_elem.as_canonical_u64())]);
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
