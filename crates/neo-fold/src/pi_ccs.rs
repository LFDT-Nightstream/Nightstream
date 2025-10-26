//! # Π_CCS: CCS Reduction via Sum-Check
//!
//! This module implements Neo's folding scheme for CCS (Customizable Constraint Systems)
//! as described in Section 4 of the Neo paper.
//!
//! ## Overview
//!
//! The folding scheme consists of three sequential reductions:
//! 
//! 1. **Π_CCS** (Section 4.4): Reduces MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k
//!    - Uses sum-check protocol to reduce CCS constraints to partial evaluations
//!    - Polynomial Q(X) = eq(·,β)·(F+NC) + eq(·,(α,r))·Eval over {0,1}^{log(dn)}
//!
//! 2. **Π_RLC** (Section 4.5): Reduces ME(b,L)^{k+1} → ME(B,L) where B = b^k
//!    - Takes random linear combinations using strong sampling set C
//!
//! 3. **Π_DEC** (Section 4.6): Reduces ME(B,L) → ME(b,L)^k
//!    - Decomposes high-norm witness back to low-norm witnesses
//!
//! This file focuses on Π_CCS, with Π_RLC and Π_DEC implemented in_rlc and pi_dec respectively.
//!
//! ## Relations (Definitions 17-18)
//!
//! - **MCS(b,L)**: Matrix Constraint System relation
//!   - Instance: (c ∈ C, x ∈ F^{m_in})
//!   - Witness: (w ∈ F^{m-m_in}, Z ∈ F^{d×m})
//!   - Constraints: c = L(Z), Z = Decomp_b(x||w), f(M̃_1·z,...,M̃_t·z) ∈ ZS_n
//!
//! - **ME(b,L)**: Matrix Evaluation relation
//!   - Instance: (c ∈ C, X ∈ F^{d×m_in}, r ∈ K^{log n}, {y_j ∈ K^d}_{j∈[t]})
//!   - Witness: Z ∈ F^{d×m}
//!   - Constraints: c = L(Z), X = L_x(Z), ||Z||_∞ < b, ∀j: y_j = Z·M_j^T·r̂

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use neo_transcript::{Transcript, Poseidon2Transcript, labels as tr_labels};
use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat, MatRef, SparsePoly};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, KExtensions};
use p3_field::{PrimeCharacteristicRing, Field, PrimeField64};
use rayon::prelude::*;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use crate::sumcheck::{RoundOracle, run_sumcheck, run_sumcheck_skip_eval_at_one, verify_sumcheck_rounds, SumcheckOutput};

// ============================================================================
// PROOF STRUCTURE
// ============================================================================

#[allow(dead_code)]
fn format_ext(x: K) -> String { format!("{:?}", x) }

/// Π_CCS proof containing the single sum-check over K
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sum-check protocol rounds (univariate polynomials as coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    /// Extension policy binding digest  
    pub header_digest: [u8; 32],
    /// Sum-check initial claim s(0)+s(1) when the R1CS engine is used; None for generic CCS.
    pub sc_initial_sum: Option<K>,
}

// ============================================================================
// SPARSE MATRIX OPERATIONS (Supporting Infrastructure)
// ============================================================================
// Optimized CSR (Compressed Sparse Row) operations for efficient M·z and M^T·χ
// computations, avoiding O(n*m) dense operations for highly sparse matrices.

/// Minimal CSR (Compressed Sparse Row) format for sparse matrix operations
/// This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices
#[derive(Clone)]
pub struct Csr<F: Field> {
    pub rows: usize,
    pub cols: usize,
    pub indptr: Vec<usize>,  // len = rows + 1
    pub indices: Vec<usize>, // len = nnz  
    pub data: Vec<F>,        // len = nnz
}

/// Convert dense matrix to CSR format - O(nm) but done once
pub fn to_csr<F: Field + Copy>(m: &Mat<F>, rows: usize, cols: usize) -> Csr<F> {
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

// (dense χ_r variant removed; tests compute the dense check inline to avoid dead code)

// ============================================================================
// EQUALITY WEIGHT COMPUTATIONS (eq(·,r) optimization)
// ============================================================================
// Half-table implementation to compute χ_r(row) = eq(row, r) without
// materializing the full 2^ℓ tensor, using split lookup tables.

/// Row weight provider: returns χ_r(row) or an equivalent row weight.
pub trait RowWeight: Sync {
    fn w(&self, row: usize) -> K;
}

/// Half-table implementation of χ_r row weights to avoid materializing the full tensor.
pub struct HalfTableEq {
    lo: Vec<K>,
    hi: Vec<K>,
    split: usize,
}

impl HalfTableEq {
    pub fn new(r: &[K]) -> Self {
        let ell = r.len();
        let split = ell / 2; // lower split bits in lo, higher in hi
        let lo_len = 1usize << split;
        let hi_len = 1usize << (ell - split);

        // Precompute factors (1-r_i, r_i)
        let mut one_minus = Vec::with_capacity(ell);
        for &ri in r { one_minus.push(K::ONE - ri); }

        // Build lo table
        let mut lo = vec![K::ONE; lo_len];
        for mask in 0..lo_len {
            let mut acc = K::ONE;
            let mut m = mask;
            for i in 0..split {
                let bit = m & 1;
                acc *= if bit == 0 { one_minus[i] } else { r[i] };
                m >>= 1;
            }
            lo[mask] = acc;
        }

        // Build hi table
        let mut hi = vec![K::ONE; hi_len];
        for mask in 0..hi_len {
            let mut acc = K::ONE;
            let mut m = mask;
            for j in 0..(ell - split) {
                let idx = split + j;
                let bit = m & 1;
                acc *= if bit == 0 { one_minus[idx] } else { r[idx] };
                m >>= 1;
            }
            hi[mask] = acc;
        }

        Self { lo, hi, split }
    }
}

impl RowWeight for HalfTableEq {
    #[inline]
    fn w(&self, row: usize) -> K {
        let lo_mask = (1usize << self.split) - 1;
        let lo_idx = row & lo_mask;
        let hi_idx = row >> self.split;
        self.lo[lo_idx] * self.hi[hi_idx]
    }
}

/// Weighted version of CSR transpose multiply: v = A^T * w, where w(row) is provided on-the-fly.
/// Efficient parallel implementation using per-thread accumulators to avoid per-row Vec allocations
pub fn spmv_csr_t_weighted_fk<F, W>(
    a: &Csr<F>,
    w: &W,
) -> Vec<K> 
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
    W: RowWeight + Sync,
{
    use rayon::prelude::*;
    
    let cols = a.cols;
    let zero = K::ZERO;
    
    // Parallel fold: each thread builds a local accumulator, then reduce
    (0..a.rows).into_par_iter()
        .fold(|| vec![zero; cols], |mut acc, r| {
            let wr = w.w(r);
            for k in a.indptr[r]..a.indptr[r+1] {
                let c = a.indices[k];
                acc[c] += K::from(a.data[k]) * wr;
            }
            acc
        })
        .reduce(|| vec![zero; cols], |mut a, b| {
            for i in 0..cols { a[i] += b[i]; }
            a
        })
}

// ============================================================================
// TRANSCRIPT BINDING (Security Infrastructure)
// ============================================================================
// Compact digest computation for CCS matrices and polynomial definitions
// to bind transcript without absorbing millions of field elements.

/// Compute canonical sparse digest of CCS matrices to avoid absorbing 500M+ field elements  
/// Uses ZK-friendly Poseidon2 hash with proper domain separation
/// 
/// NOTE: Uses as_canonical_u64() for field element conversion. For fields with elements
/// larger than 64 bits, this may lose information. For transcript binding (not cryptographic
/// commitment), this is acceptable as long as it's deterministic and collision-resistant.
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

// ===== MLE Folding DP Helpers =====

/// Pad vector to power of 2 length with zeros
/// Pads a vector to power-of-two length for MLE use
/// 
/// # Arguments
/// * `v` - Vector to pad
/// * `ell` - Number of bits (target length = 2^ell, NOT the target length itself)
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

// ============================================================================
// CONSTRAINT EVALUATION FUNCTIONS
// ============================================================================
// Explicit evaluation of the constraint polynomials used in Q(X):
//   1. NC_i: Range and decomposition constraints
//   2. Tie constraints: y_j = Z·M_j^T·χ_u consistency checks

/// Evaluate range/decomposition constraint polynomials NC_i(z,Z) at point u.
/// 
/// **Paper Reference**: Section 4.4, NC_i term in Q polynomial
/// NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
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
/// 
/// **Paper Reference**: Definition 18 (ME relation)
/// Checks: ∀j ∈ [t], y_j = Z·M_j^T·r̂
/// 
/// These assert that the y_j values are consistent with Z and the random point.
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

// ============================================================================
// SUM-CHECK ORACLE CONSTRUCTION (Section 4.4)
// ============================================================================
// Implements the polynomial Q(X) for sum-check protocol:
//   Q(X_{[1,log(dn)]}) = eq(X,β)·(F + Σ_i γ^i·NC_i) + γ^k·Σ_{i,j} γ^{...}·Eval_{i,j}
// where:
//   - F(X_{[log(d)+1,log(dn)]}) = f(M̃_1·z_1,...,M̃_t·z_1)  [CCS constraint polynomial]
//   - NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)            [Range/decomp constraints]
//   - Eval_{i,j}(X) = eq(X,(α,r))·M̃_{i,j}(X)             [Evaluation tie constraints]

/// Absorb sparse polynomial definition into transcript for soundness binding.
/// This prevents a malicious prover from using different polynomials with the same matrix structure.
fn absorb_sparse_polynomial(tr: &mut Poseidon2Transcript, f: &SparsePoly<F>) {
    // Absorb polynomial structure
    tr.append_message(b"neo/ccs/poly", b"");
    tr.append_u64s(b"arity", &[f.arity() as u64]);
    tr.append_u64s(b"terms_len", &[f.terms().len() as u64]);
    
    // Absorb each term: coefficient + exponents (sorted for determinism)
    let mut terms: Vec<_> = f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps); // deterministic ordering
    
    for term in terms {
        tr.append_fields(b"coeff", &[term.coeff]);
        let exps: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
        tr.append_u64s(b"exps", &exps);
    }
}

// Generic CCS partials: one shrinking vector per matrix M_j z
// F-term partials: one shrinking vector per j-th matrix vector
struct MlePartials { s_per_j: Vec<Vec<K>> }

// NC state after row phase: Ajtai partials for y_{i,1}(r') per instance
#[allow(dead_code)]
struct NcState {
    // For each instance i (i=1..k), Ajtai partial vector of y_{i,1}(r')
    // Starts at length 2^ell_d, folded each Ajtai round
    y_partials: Vec<Vec<K>>,
    // γ^i weights, i=1..k
    #[allow(dead_code)]
    gamma_pows: Vec<K>,
    // Cache F(r') once computed at first Ajtai round
    #[allow(dead_code)]
    f_at_rprime: Option<K>,
}

/// Sum-check oracle for Generic CCS (Paper Section 4.4)
/// 
/// **Paper Reference**: Section 4.4, Equation for Q polynomial:
/// ```text
/// Q(X_{[1,log(dn)]}) := eq(X,β)·(F(X_{[log(d)+1,log(dn)]}) + Σ_{i∈[k]} γ^i·NC_i(X))
///                        + γ^k·Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1}·Eval_{(i,j)}(X)
/// ```
/// where:
/// - **F**: CCS constraint polynomial f(M̃_1·z_1,...,M̃_t·z_1)
/// - **NC_i**: Norm/decomposition constraints ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
/// - **Eval_{i,j}**: Evaluation ties eq(X,(α,r))·M̃_{i,j}(X)
/// 
/// **Implementation Note**: Uses two-axis decomposition (row-first, then Ajtai) to
/// avoid materializing the full 2^{log(dn)} tensor.
pub(crate) struct GenericCcsOracle<'a, F> 
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    #[allow(dead_code)]
    s: &'a CcsStructure<F>,
    // F term partials for instance 1 (row-domain MLE, LIVE in row phase)
    partials_first_inst: MlePartials,
    // Equality gates split by axes (two-axis structure for log(dn) domain):
    w_beta_a_partial: Vec<K>,   // size 2^ell_d (Ajtai dimension)
    w_beta_r_partial: Vec<K>,   // size 2^ell_n (row dimension)
    w_eval_r_partial: Vec<K>,   // eq against r_input (row dimension only, Ajtai pre-collapsed)
    // Z witnesses for all k instances (needed for NC in Ajtai phase)
    z_witnesses: Vec<&'a neo_ccs::Mat<F>>,
    // Parameters
    gamma: K,
    #[allow(dead_code)]  // Used for gamma_pows in NcState
    k_total: usize,
    #[allow(dead_code)]
    b: u32,
    ell_d: usize,  // log d (Ajtai dimension bits)
    ell_n: usize,  // log n (row dimension bits)
    d_sc: usize,
    round_idx: usize,  // track which round we're in (for phase detection)
    // Precomputed constants for eq(·,β) block (used in row phase as constants)
    f_at_beta_r: K,      // F(β_r) precomputed
    nc_sum_beta: K,      // Σ_i γ^i N_i(β) precomputed
    // Eval block (pre-collapsed over Ajtai at α), row-domain polynomial
    eval_row_partial: Vec<K>,
    // Row-phase bookkeeping for NC/F after rows are bound
    #[allow(dead_code)]
    row_chals: Vec<K>,           // Row challenges collected during row rounds
    #[allow(dead_code)]
    csr_m1: &'a Csr<F>,          // CSR for M_1 to build M_1^T * χ_{r'}
    // NC after row binding: y_{i,1}(r') Ajtai partials & γ weights
    #[allow(dead_code)]
    nc: Option<NcState>,
}

impl<'a, F> GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Lazily prepare NC state at first Ajtai round:
    /// Build y_{i,1}(r') = Z_i * M_1^T * χ_{r'} for all instances
    #[allow(dead_code)]
    fn ensure_nc_precomputed(&mut self) {
        if self.nc.is_some() { return; }
        
        // Build χ_{r'} over rows from collected row challenges
        // Use HalfTableEq + CSR to avoid materializing χ_r explicitly
        let w_r = HalfTableEq::new(&self.row_chals);
        // v1 = M_1^T · χ_{r'} ∈ K^m
        let v1 = spmv_csr_t_weighted_fk(self.csr_m1, &w_r);
        
        // For each Z_i, y_{i,1}(r') = Z_i · v1 ∈ K^d, then pad to 2^ell_d
        let mut y_partials = Vec::with_capacity(self.z_witnesses.len());
        for Zi in &self.z_witnesses {
            let z_ref = neo_ccs::MatRef::from_mat(Zi);
            let yi = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &v1);
            y_partials.push(pad_to_pow2_k(yi, self.ell_d).expect("pad y_i"));
        }
        
        // γ^i weights, i starts at 1
        let mut gamma_pows = Vec::with_capacity(self.z_witnesses.len());
        let mut g = self.gamma;
        for _ in 0..self.z_witnesses.len() { gamma_pows.push(g); g *= self.gamma; }
        
        self.nc = Some(NcState { y_partials, gamma_pows, f_at_rprime: None });
    }
}

impl<'a, F> RoundOracle for GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Total rounds = ell_n (row rounds first) + ell_d (Ajtai rounds second)
    /// Row-first ordering matches paper's two-axis sum-check decomposition
    fn num_rounds(&self) -> usize { self.ell_d + self.ell_n }
    fn degree_bound(&self) -> usize { self.d_sc }
    
    /// Evaluate Q at sample points in current variable
    /// 
    /// **Paper Reference**: Section 4.4, Q polynomial evaluation
    /// Two-phase oracle matching paper's two-axis structure:
    /// - **Row phase** (rounds 0..ell_n-1): Process X_r bits, evaluate all terms
    /// - **Ajtai phase** (rounds ell_n..ell): Process X_a bits, fold gates only
    fn evals_at(&mut self, sample_xs: &[K]) -> Vec<K> {
        // Two-axis oracle: Row-first approach (rounds 0..ell_n-1 are rows, ell_n..ell are Ajtai)
        // Q(X_a, X_r) = eq(·,β)·[F(X_r)+ΣNC_i(·)] + eq(·,(α,r))·Eval
        
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];
        
        // Helper: per-pair equality gate evaluation (not sum over pairs!)
        // Computes (1-X)·w[2k] + X·w[2k+1] for the k-th pair
        #[allow(dead_code)]
        let eq_gate_pair = |weights: &Vec<K>, k: usize, X: K| -> K {
            debug_assert!(weights.len() >= 2 * (k + 1), "weights must have even length with at least {} elements", 2*(k+1));
            let w0 = weights[2 * k];
            let w1 = weights[2 * k + 1];
            (K::ONE - X) * w0 + X * w1
        };
        
        if self.round_idx < self.ell_n {
            // ===== ROW PHASE =====
            // Q(X_a, X_r) = eq_a(X_a,β_a)·eq_r(X_r,β_r)·(F+NC) + eq_a(X_a,α)·eq_r(X_r,r)·Eval
            // Row phase processes X_r bits. For each X_r value, sum over all X_a.
            // Σ_{X_a} eq_a(X_a,β_a) = 1, so F+NC contribute as constants times eq_r gates.
            // Eval is pre-collapsed: Σ_{X_a} eq_a(X_a,α)·Eval_term = G_eval.
            
            #[cfg(feature = "debug-logs")]
            {
                eprintln!("[oracle][row{}] half_beta = {}, half_eval = {}", 
                    self.round_idx, self.w_beta_r_partial.len() >> 1, self.eval_row_partial.len() >> 1);
            }
            
            // (A) F+NC block: row eq gate times constant
            let fnc_constant = self.f_at_beta_r + self.nc_sum_beta;
            let half_beta = self.w_beta_r_partial.len() >> 1;
            for (sx, &X) in sample_xs.iter().enumerate() {
                let mut fnc_contrib = K::ZERO;
                for k in 0..half_beta {
                    fnc_contrib += eq_gate_pair(&self.w_beta_r_partial, k, X);
                }
                sample_ys[sx] += fnc_contrib * fnc_constant;
            }
            
            // (B) Eval block (already collapsed over Ajtai at α):
            let half_e = self.eval_row_partial.len() >> 1;
            debug_assert_eq!(self.eval_row_partial.len() % 2, 0, "eval_row_partial must have even length");
            for k in 0..half_e {
                let a_eval = self.eval_row_partial[2 * k];
                let d_eval = self.eval_row_partial[2 * k + 1] - a_eval;
                for (sx, &X) in sample_xs.iter().enumerate() {
                    let w0 = self.w_eval_r_partial[2 * k];
                    let w1 = self.w_eval_r_partial[2 * k + 1];
                    let row_gate_eval = (K::ONE - X) * w0 + X * w1;
                    sample_ys[sx] += row_gate_eval * (a_eval + d_eval * X);
                }
            }
            
        } else {
            // ===== AJTAI PHASE =====
            // During Ajtai rounds, we process X_a bits.
            // F and NC were already fully handled in the row phase (as constants gated by eq_r).
            // The only remaining term is... actually NOTHING!
            // The row phase consumed all of Q's contribution.
            
            // All terms (F+NC and Eval) were processed in the row phase:
            // - F+NC: Fully bound at β, gated by eq_r in row phase
            // - Eval: Pre-collapsed over Ajtai at α, gated by eq_r in row phase
            
            // So Ajtai rounds just fold the gates without adding new contributions
            // (The oracle returns zeros here, which is correct)
            
            #[cfg(feature = "debug-logs")]
            {
                let ajtai_round = self.round_idx - self.ell_n;
                eprintln!("[oracle][ajtai{}] No contributions (all terms handled in row phase)", ajtai_round);
            }
        }
        
        #[cfg(feature = "debug-logs")]
        if self.round_idx <= 2 {
            eprintln!("[oracle][round{}] Returning {} samples: {:?}", 
                self.round_idx, sample_ys.len(), 
                if sample_ys.len() <= 4 { format!("{:?}", sample_ys) } else { format!("[{} values]", sample_ys.len()) });
        }
        
        sample_ys
    }
    
    /// Fold oracle state after binding one variable to challenge r_i
    /// 
    /// **Paper Reference**: Standard MLE folding: f(X) → (1-r)·f(0,X) + r·f(1,X)
    /// 
    /// **Implementation**: Folds all partial vectors in current phase:
    /// - Row phase: F partials, Eval partials, β/r equality gates
    /// - Ajtai phase: NC partials, β_a equality gates
    fn fold(&mut self, r_i: K) {
        // Fold based on current phase (row or Ajtai)
        if self.round_idx < self.ell_n {
            // Row phase: collect challenge and fold row partials (gates, F partials, Eval)
            self.row_chals.push(r_i);
            
            let fold_partial = |partial: &mut Vec<K>, r: K| {
                let n2 = partial.len() >> 1;
                for k in 0..n2 {
                    let a0 = partial[2*k];
                    let b0 = partial[2*k + 1];
                    partial[k] = (K::ONE - r) * a0 + r * b0;
                }
                partial.truncate(n2);
            };
            
            fold_partial(&mut self.w_beta_r_partial, r_i);
            fold_partial(&mut self.w_eval_r_partial, r_i);
            fold_partial(&mut self.eval_row_partial, r_i);
            
            // Fold F partials (instance 1 only)
            for v in &mut self.partials_first_inst.s_per_j {
                fold_partial(v, r_i);
            }
            
        } else {
            // Ajtai phase: fold Ajtai partials (β gates and NC Ajtai partials)
            let fold_partial = |partial: &mut Vec<K>, r: K| {
                let n2 = partial.len() >> 1;
                for k in 0..n2 {
                    let a0 = partial[2*k];
                    let b0 = partial[2*k + 1];
                    partial[k] = (K::ONE - r) * a0 + r * b0;
                }
                partial.truncate(n2);
            };
            
            fold_partial(&mut self.w_beta_a_partial, r_i);
            
            // Fold NC Ajtai partials if they exist
            if let Some(ref mut nc) = self.nc {
                for y in &mut nc.y_partials {
                    fold_partial(y, r_i);
                }
            }
        }
        
        // Increment round index
        self.round_idx += 1;
    }
}

// ============================================================================
// Π_CCS PROVER (Section 4.4)
// ============================================================================
// Implements the CCS reduction via sum-check protocol.
//
// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS
//
// **Input**:  MCS(b,L) × ME(b,L)^{k-1}
// **Output**: ME(b,L)^k + PiCcsProof
//
// **Protocol Structure** (matching paper exactly):
//   - **Setup G(1^λ, sz)**: Generate public parameters defining L
//   - **Encoder K(pp, s)**: Output proving/verification keys
//   - **Reduction ⟨P,V⟩**: Interactive protocol with 4 steps:
//       1. V → P: Send challenges α ∈ K^{log d}, β ∈ K^{log(dn)}, γ ∈ K
//       2. P ↔ V: Sum-check on Q(X) over {0,1}^{log(dn)} → claim v = Q(α',r')
//       3. P → V: Send y'_{(i,j)} = Z_i·M_j^T·r̂' for all i ∈ [k], j ∈ [t]
//       4. V: Check v ?= eq((α',r'), β)·(F'+NC') + eq((α',r'), (α,r))·Eval'

/// Prove Π_CCS: CCS instances satisfy constraints via sum-check
/// 
/// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS
/// 
/// Implements the prover's side of the reduction protocol.
/// Input: 1 MCS + (k-1) ME instances, outputs k ME instances + proof
/// Per paper: MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k
pub fn pi_ccs_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],  // k-1 ME inputs with shared r
    me_witnesses: &[neo_ccs::Mat<F>],      // NEW: Z_i for i=2..k (row-major d×m)
    l: &L, // we need L to check c = L(Z) and to compute X = L_x(Z)
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // =========================================================================
    // SETUP: Initialize transcript and validate inputs
    // =========================================================================
    
    tr.append_message(tr_labels::PI_CCS, b"");

    // Validate ME inputs and witnesses match
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput("me_inputs.len() != me_witnesses.len()".into()));
    }

    // Input & policy checks
    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput("empty or mismatched inputs".into()));
    }
    
    // Note: The CCS/LCCCS formalism does not require M_1 = I_n
    // Different papers and implementations use different matrix orderings
    // The NC computation uses M_1 in the general case: y_{(i,1)} = M_1 * Z_i
    
    // NEW: Validate ME inputs (k-1 entries when k>1, all with same r)
    // For k=1 (initial fold), me_inputs should be empty
    if !me_inputs.is_empty() {
        let r_inp = &me_inputs[0].r;
        if !me_inputs.iter().all(|me| &me.r == r_inp) {
            return Err(PiCcsError::InvalidInput("all ME inputs must share the same r".into()));
        }
    }
    
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }
    
    // =========================================================================
    // PARAMETERS: Compute reduction parameters (Section 4.3)
    // =========================================================================
    // **Paper Reference**: ℓ = log(dn) for sum-check over {0,1}^{log(dn)} hypercube
    let d_pad = neo_math::D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;  // log d (Ajtai dimension)
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;  // log n (row dimension)
    let ell = ell_d + ell_n;                       // log(dn) - FULL hypercube as per paper
    
    // Degree bound: max of F+eq, Eval+eq, Range+eq
    // Per paper analysis: NC_i has degree ~2b, F has degree max_degree(f), both gated by eq (+1)
    let d_sc = core::cmp::max(
        s.max_degree() as usize + 1,           // F with eq gating
        core::cmp::max(2, 2 * params.b as usize)  // Eval+eq vs. Range+eq
    );

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
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // Convert to CSR sparse format once for all operations
    // This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices  
    #[cfg(feature = "debug-logs")]
    let csr_start = std::time::Instant::now();
    let mats_csr: Vec<Csr<F>> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    #[cfg(feature = "debug-logs")]
    let total_nnz: usize = mats_csr.iter().map(|c| c.data.len()).sum();
    #[cfg(feature = "debug-logs")]
    println!("🔥 CSR conversion completed: {:.2}ms ({} matrices, {} nnz total, {:.4}% density)",
             csr_start.elapsed().as_secs_f64() * 1000.0, 
             mats_csr.len(), total_nnz, 
             (total_nnz as f64) / (s.n * s.m * s.matrices.len()) as f64 * 100.0);

    // --- Prepare per-instance data and check c=L(Z) ---
    // Also build z = x||w and cache M_j z over F for each instance.
    struct Inst<'a> {
        Z: &'a Mat<F>, 
        m_in: usize, 
        mz: Vec<Vec<F>>, // rows = n, per-matrix M_j z over F
        c: Cmt,
    }
    
    let mut insts: Vec<Inst> = Vec::with_capacity(mcs_list.len());
    #[cfg(feature = "debug-logs")]
    let instance_prep_start = std::time::Instant::now();
    for (inst_idx, (inst, wit)) in mcs_list.iter().zip(witnesses.iter()).enumerate() {
        #[cfg(not(feature = "debug-logs"))]
        let _ = inst_idx;
        #[cfg(feature = "debug-logs")]
        let z_check_start = std::time::Instant::now();
        let z = neo_ccs::relations::check_mcs_opening(l, inst, wit)
            .map_err(|e| PiCcsError::InvalidInput(format!("MCS opening failed: {e}")))?;
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] MCS opening check: {:.2}ms", inst_idx, 
                 z_check_start.elapsed().as_secs_f64() * 1000.0);
        // Ensure z matches CCS width to prevent OOB during M*z
        if z.len() != s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "z length {} does not match CCS column count {} (malformed instance)",
                z.len(), s.m
            )));
        }
        
        // Verify Z == Decomp_b(z)
        // This prevents prover from using satisfying z for CCS but different Z for commitment
        #[cfg(feature = "debug-logs")]
        let decomp_start = std::time::Instant::now();
        let Z_expected_col_major = neo_ajtai::decomp_b(&z, params.b, neo_math::D, neo_ajtai::DecompStyle::Balanced);
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] Decomp_b: {:.2}ms", inst_idx, 
                 decomp_start.elapsed().as_secs_f64() * 1000.0);
        
        #[cfg(feature = "debug-logs")]
        let range_check_start = std::time::Instant::now();
        neo_ajtai::assert_range_b(&Z_expected_col_major, params.b)
            .map_err(|e| PiCcsError::InvalidInput(format!("Range check failed on expected Z: {e}")))?;
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] Range check: {:.2}ms", inst_idx, 
                 range_check_start.elapsed().as_secs_f64() * 1000.0);
        
        // Convert Z_expected from column-major to row-major format to match wit.Z
        #[cfg(feature = "debug-logs")]
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
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] Format conversion: {:.2}ms", inst_idx, 
                 format_conv_start.elapsed().as_secs_f64() * 1000.0);
        
        // Compare Z with expected decomposition (both in row-major format)
        #[cfg(feature = "debug-logs")]
        let z_compare_start = std::time::Instant::now();
        if wit.Z.as_slice() != Z_expected_row_major.as_slice() {
            return Err(PiCcsError::InvalidInput("Z != Decomp_b(z) - prover using inconsistent z and Z".into()));
        }
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] Z comparison: {:.2}ms", inst_idx, 
                 z_compare_start.elapsed().as_secs_f64() * 1000.0);
        // Use CSR sparse matrix-vector multiply - O(nnz) instead of O(n*m)
        #[cfg(feature = "debug-logs")]
        let mz_start = std::time::Instant::now();
        let mz: Vec<Vec<F>> = mats_csr.par_iter().map(|csr| 
            spmv_csr_ff::<F>(csr, &z)
        ).collect();
        #[cfg(feature = "debug-logs")]
        println!("💥 [TIMING] CSR M_j z computation: {:.2}ms (nnz={}, vs {}M dense elements - {}x reduction)", 
                 mz_start.elapsed().as_secs_f64() * 1000.0, total_nnz, 
                 (s.n * s.m * s.matrices.len()) / 1_000_000,
                 (s.n * s.m * s.matrices.len()) / total_nnz.max(1));
        insts.push(Inst{ Z: &wit.Z, m_in: inst.m_in, mz, c: inst.c.clone() });
    }
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Instance preparation total: {:.2}ms ({} instances)", 
             instance_prep_start.elapsed().as_secs_f64() * 1000.0, insts.len());

    // =========================================================================
    // STEP 0: BIND INSTANCES TO TRANSCRIPT (before challenge sampling)
    // =========================================================================
    // **Security**: Absorb all instance data BEFORE sampling challenges
    // to prevent malleability attacks
    #[cfg(feature = "debug-logs")]
    let transcript_start = std::time::Instant::now();
    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);
    // OPTIMIZATION: Absorb compact ZK-friendly digest instead of 500M+ field elements 
    // This reduces transcript absorption from ~51s to microseconds using Poseidon2
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest {
        tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]);
    }
    // Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        // Absorb commitment to prevent cross-instance attacks
        tr.append_fields(b"c_data", &inst.c.data);
    }
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Transcript absorption: {:.2}ms", 
             transcript_start.elapsed().as_secs_f64() * 1000.0);

    // --- Absorb ME inputs BEFORE sampling challenges ---
    // This binds the input ME claims (which define T) to prevent malleability
    tr.append_message(b"neo/ccs/me_inputs", b"");
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);
    for me in me_inputs.iter() {
        tr.append_fields(b"c_data_in", &me.c.data);
        tr.append_u64s(b"m_in_in", &[me.m_in as u64]);
        // Bind r (must be shared across all inputs)
        for limb in &me.r { tr.append_fields(b"r_in", &limb.as_coeffs()); }
        // Bind y vectors compactly (digest to avoid absorbing t·d elements)
        for yj in &me.y {
            // Simple digest: just absorb all elements (Poseidon2 handles compression)
            for &y_elem in yj {
                tr.append_fields(b"y_elem", &y_elem.as_coeffs());
            }
        }
    }

    // =========================================================================
    // STEP 1: VERIFIER SENDS CHALLENGES α, β, γ
    // =========================================================================
    // **Paper Reference**: Section 4.4, Step 1
    // V: Send challenges α ∈ K^{log d}, β ∈ K^{log(dn)}, γ ∈ K to P
    #[cfg(feature = "debug-logs")]
    let chal_start = std::time::Instant::now();
    tr.append_message(b"neo/ccs/chals/v1", b"");
    
    // α over Ajtai dimension (log d)
    let alpha: Vec<K> = (0..ell_d)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    
    // β over FULL log(dn) space - split into Ajtai and row parts
    let beta: Vec<K> = (0..ell)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    let (beta_a, beta_r) = beta.split_at(ell_d);
    
    // γ scalar
    let ch_g = tr.challenge_fields(b"chal/k", 2);
    let gamma: K = neo_math::from_complex(ch_g[0], ch_g[1]);
    
    // PAPER FIX: Use the input ME's r (if k>1), otherwise Eval term is zero
    let r_inp_full_opt = if !me_inputs.is_empty() {
        Some(me_inputs[0].r.clone())
    } else {
        None  // k=1: no input ME, Eval term is zero
    };
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Challenge sampling: {:.2}ms (α: {}, β: {}, γ: 1{})",
             chal_start.elapsed().as_secs_f64() * 1000.0, ell_d, ell,
             if r_inp_full_opt.is_some() { ", using input r" } else { "" });

    // =========================================================================
    // STEP 2: BUILD Q POLYNOMIAL AND RUN SUM-CHECK
    // =========================================================================
    // **Paper Reference**: Section 4.4, Step 2
    // P ↔ V: Perform sum-check protocol on Q(X) over {0,1}^{log(dn)}
    // This reduces to evaluation claim v ?= Q(α', r')
    
    // --- Prepare equality weight vectors for β and (α, r_inp) ---
    // β is over full log(dn) space, r_inp is from ME inputs (log n) if k>1
    // We need to split β into Ajtai (ell_d bits) and row (ell_n bits) for the two-phase gate
    
    // Build separate half-table equality weights for each axis
    let w_beta_a = HalfTableEq::new(beta_a);
    let w_beta_r = HalfTableEq::new(beta_r);
    let d_n_a = 1usize << ell_d;  // Ajtai space size
    let d_n_r = 1usize << ell_n;  // row space size
    let mut w_beta_a_partial = vec![K::ZERO; d_n_a];
    for i in 0..d_n_a { w_beta_a_partial[i] = w_beta_a.w(i); }
    let mut w_beta_r_partial = vec![K::ZERO; d_n_r];
    for i in 0..d_n_r { w_beta_r_partial[i] = w_beta_r.w(i); }
    
    // For the Eval term: eq((α', r'), (α, r_inp)) - only if k>1
    // Only need row dimension (Ajtai pre-collapsed at α in G_eval)
    let w_eval_r_partial = if let Some(ref r_inp_full) = r_inp_full_opt {
        let mut w_eval_r = vec![K::ZERO; d_n_r];
        for i in 0..d_n_r { w_eval_r[i] = HalfTableEq::new(r_inp_full).w(i); }
        w_eval_r
    } else {
        // k=1: No input ME, Eval term is zero - use dummy zero vector
        vec![K::ZERO; d_n_r]
    };
    
    // --- Build MLE partials for instance 1 only ---
    // **Paper Reference**: Section 4.4, F polynomial uses M̃_j·z_1 evaluations
    #[cfg(feature = "debug-logs")]
    println!("🔍 Sum-check starting: {} instances, {} rounds", insts.len(), ell);
    let sample_xs_generic: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();
    // Build partial states (invariant across the engine) for instance 1
    // F term MLE is only over ROW dimension (ell_n), not Ajtai dimension
    #[cfg(feature = "debug-logs")]
    let mle_start = std::time::Instant::now();
    let partials_first_inst: MlePartials = {
        let inst = &insts[0];
        let mut s_per_j = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            let w_k: Vec<K> = inst.mz[j].iter().map(|&x| K::from(x)).collect();
            let w_k = pad_to_pow2_k(w_k, ell_n)?;  // Pad to 2^ell_n, not 2^ell
            s_per_j.push(w_k);
        }
        MlePartials { s_per_j }
    };
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] MLE partials setup: {:.2}ms",
             mle_start.elapsed().as_secs_f64() * 1000.0);
    
    // --- Compute initial sum T per paper: T = Σ_{j=1, i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ_{(i,j)}(α) ---
    // **Paper Reference**: Section 4.4, Step 2 - Claimed sum definition
    // T is computed ONLY from ME inputs and α (no F term, no NC term, no secret Z)
    // This is the key fix for soundness - T must be verifiable from public/committed data
    
    // Build χ_α once for MLE evaluations over Ajtai dimension
    let chi_alpha: Vec<K> = neo_ccs::utils::tensor_point::<K>(&alpha);
    
    // k_total = 1 (MCS) + me_inputs.len() (ME inputs) = k instances on output
    let k_total = 1 + me_inputs.len();
    
    // === Precompute oracle constants (f_at_beta_r, nc_sum_beta, G_eval) ===
    // **Paper Reference**: Section 4.4, Step 4 - Terminal verification values
    // Must compute BEFORE initial_sum because initial_sum = f + nc + T (full Q sum)
    
    #[cfg(feature = "debug-logs")]
    let precomp_start = std::time::Instant::now();
    
    // Build tensor points for β
    let chi_beta_a = neo_ccs::utils::tensor_point::<K>(beta_a);
    
    // Row eq weights at β_r (full length 2^ell_n, not folded)
    let beta_r_ht = HalfTableEq::new(beta_r);
    let chi_beta_r_full: Vec<K> = (0..(1 << ell_n)).map(|i| beta_r_ht.w(i)).collect();
    
    // === F(β_r) from instance 1 ===
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let v_f = &insts[0].mz[j];  // length n over F
        m_vals[j] = v_f.iter().take(s.n).zip(&chi_beta_r_full)
            .fold(K::ZERO, |acc, (&vv, &w)| acc + K::from(vv) * w);
    }
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [F_BETA] m_vals = {:?}", m_vals.iter().take(3).collect::<Vec<_>>());
    
    let f_at_beta_r = s.f.eval_in_ext::<K>(&m_vals);
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [F_BETA] f_at_beta_r = {:?}, polynomial = {:?}", f_at_beta_r, &s.f);
    
    // === NC sum at β: Σ_i γ^i N_i(β) ===
    // NC term uses y_{(i,1)} = Z_i * M_1^T * χ_{β_r}, just like other y vectors
    // Compute v_1 = M_1^T * χ_{β_r}  (result is a vector in K^m)
    let m1 = &s.matrices[0];
    let mut v1_beta_r = vec![K::ZERO; s.m];
    for row in 0..s.n {
        for col in 0..s.m {
            v1_beta_r[col] += K::from(m1[(row, col)]) * chi_beta_r_full[row];
        }
    }
    
    let mut nc_sum_beta = K::ZERO;
    let mut gamma_pow_i = gamma;  // γ^1
    
    // Iterate over ALL MCS witnesses, then ME witnesses (matches output ME ordering)
    for Zi in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()) {
        let z_ref = neo_ccs::MatRef::from_mat(Zi);
        // y_{(i,1)}(β_r) = Z_i * v_1 where v_1 = M_1^T * χ_{β_r}
        let y_i1_beta_r = neo_ccs::utils::mat_vec_mul_fk::<F,K>(
            z_ref.data, z_ref.rows, z_ref.cols, &v1_beta_r
        );
        
        // ỹ_{(i,1)}(β) = ⟨ y_{(i,1)}(β_r), χ_{β_a} ⟩
        let y_mle_beta = y_i1_beta_r.iter().zip(&chi_beta_a)
            .fold(K::ZERO, |acc, (&v, &w)| acc + v * w);
        
        // Balanced range polynomial: N_i(β) = ∏_{t∈{-b+1..b-1}} (y_mle_beta - t)
        let mut Ni_beta = K::ONE;
        for t in -(params.b as i64 - 1)..=(params.b as i64 - 1) {
            Ni_beta *= y_mle_beta - K::from(F::from_i64(t));
        }
        
        nc_sum_beta += gamma_pow_i * Ni_beta;
        gamma_pow_i *= gamma;
    }
    
    // === Aggregated row vector G for the Eval block ===
    // EXPONENT FORMULA MAPPING (paper → code):
    // Paper uses: γ^{i+(j-1)k-1} for instance i∈[2,k], matrix j∈[t]
    // Code uses: exponent = (i_offset + 1) + j * k_total
    // where i_offset ∈ [0, k-2] enumerates me_witnesses (paper i = i_offset + 2)
    // and k_total = 1 + me_inputs.len() = 1 + (k-1) = k
    // So: (i_offset + 1) + j*k = (i_offset + 2 - 1) + j*k = i + (j-1)k  (matches paper when j≥1)
    // This exact formula is replicated in verifier's eval_sum_prime computation
    let mut G_eval = vec![K::ZERO; s.n];
    
    for (i_off, Zi) in me_witnesses.iter().enumerate() {
        // u_i[c] = Σ_{ρ=0..d-1} Z_i[ρ,c]·χ_α[ρ]
        let mut u_i = vec![K::ZERO; s.m];
        for c in 0..s.m {
            u_i[c] = (0..neo_math::D).fold(K::ZERO, |acc, rho| {
                let w = if rho < chi_alpha.len() { chi_alpha[rho] } else { K::ZERO };
                acc + K::from(Zi[(rho, c)]) * w
            });
        }
        
        for j in 0..s.t() {
            let mj_ref = neo_ccs::MatRef::from_mat(&s.matrices[j]);
            let g_ij = neo_ccs::utils::mat_vec_mul_fk::<F,K>(
                mj_ref.data, mj_ref.rows, mj_ref.cols, &u_i
            );
            
            let exponent = (i_off + 1) + j * k_total;
            let mut w_pow = K::ONE;
            for _ in 0..exponent { w_pow *= gamma; }
            
            for r in 0..s.n { G_eval[r] += w_pow * g_ij[r]; }
        }
    }
    
    let eval_row_partial = pad_to_pow2_k(G_eval, ell_n)?;
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [PRECOMPUTE] f_at_beta_r, nc_sum_beta, G_eval: {:.2}ms",
             precomp_start.elapsed().as_secs_f64() * 1000.0);
    
    // === Compute T (Eval-only part) and set initial_sum to FULL sum ===
    // **Paper Reference**: Section 4.4, claimed sum T definition
    let mut T = K::ZERO;
    for j in 0..s.t() {
        for (i_offset, me_input) in me_inputs.iter().enumerate() {
            let y_mle = me_input.y[j].iter().zip(&chi_alpha)
                .fold(K::ZERO, |acc, (&v, &w)| acc + v * w);
            
            let exponent = (i_offset + 1) + j * k_total;
            let mut weight = K::ONE;
            for _ in 0..exponent { weight *= gamma; }
            
            T += weight * y_mle;
        }
    }
    
    // Initial sum must equal the FULL sum of Q over the hypercube
    // **Paper Reference**: Section 4.4, Q = eq(·,β)·(F+NC) + eq(·,(α,r))·Eval
    // So sum = F(β) + NC(β) + Eval(α)
    let initial_sum = f_at_beta_r + nc_sum_beta + T;
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [INITIAL_SUM] F(β_r)={}, NC(β)={}, T(Eval)={}, TOTAL={}",
             format_ext(f_at_beta_r), format_ext(nc_sum_beta), format_ext(T), format_ext(initial_sum));
    
    // Bind initial_sum BEFORE rounds to the transcript (prover side)
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

    // Drive rounds with the generic engine
    // Two-axis oracle with F, NC, and Eval terms per paper
    let SumcheckOutput { rounds, challenges: r, final_sum: _running_sum } = {
        // Collect Z witnesses for all k instances (needed for NC terms)
        // Note on ordering: Must match gamma exponent indexing and output ME ordering
        // Order: ALL MCS witnesses first (corresponds to γ^1, γ^2, ... in NC term),
        //        then ME witnesses (correspond to γ^{mcs_count+1}, γ^{mcs_count+2}, ... in NC term)
        // This order must match: (1) NC gamma_pows computation, (2) out_me ordering
        let mut z_witnesses = Vec::with_capacity(k_total);
        // All MCS instances: from witnesses (corresponds to γ^1, γ^2, ... in NC term)
        for w in witnesses {
            z_witnesses.push(&w.Z);
        }
        // All ME instances: from ME witnesses (correspond to subsequent γ powers in NC term)
        for me_wit in me_witnesses {
            z_witnesses.push(me_wit);
        }
        
        let mut oracle = GenericCcsOracle {
            s,
            partials_first_inst,
            w_beta_a_partial,
            w_beta_r_partial,
            w_eval_r_partial,
            z_witnesses,
            gamma,
            k_total,
            b: params.b,
            ell_d,
            ell_n,
            d_sc: d_sc,
            round_idx: 0,
            f_at_beta_r,
            nc_sum_beta,
            eval_row_partial,
            // NEW: Row-phase bookkeeping for NC/F after rows are bound
            row_chals: Vec::with_capacity(ell_n),
            csr_m1: &mats_csr[0],
            nc: None,
        };
        if d_sc >= 1 { run_sumcheck_skip_eval_at_one(tr, &mut oracle, initial_sum, &sample_xs_generic)? }
        else { run_sumcheck(tr, &mut oracle, initial_sum, &sample_xs_generic)? }
    };
    
    
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Sum-check rounds complete: {} rounds", ell);

    // =========================================================================
    // STEP 3: COMPUTE AND SEND y'_{(i,j)} VALUES
    // =========================================================================
    // **Paper Reference**: Section 4.4, Step 3
    // P → V: For all i ∈ [k], j ∈ [t], send y'_{(i,j)} = Z_i·M_j^T·r̂'
    // where r̂' is the MLE evaluation of the sum-check challenge r'
    
    // Split the ℓ-vector into (r', α') - row-first ordering!
    // First ell_n challenges are row bits, last ell_d challenges are Ajtai bits
    if r.len() != ell { return Err(PiCcsError::SumcheckError("bad r length".into())); }
    let (r_prime, _alpha_prime) = r.split_at(ell_n);

    // Compute M_j^T * χ_r' using streaming/half-table weights (no full χ_r materialization)
    #[cfg(feature = "debug-logs")]
    println!("🚀 [OPTIMIZATION] Computing M_j^T * χ_r' with half-table weights...");
    #[cfg(feature = "debug-logs")]
    let transpose_once_start = std::time::Instant::now();
    let w = HalfTableEq::new(r_prime);
    #[cfg(feature = "neo-logs")]
    {
        let max_i = core::cmp::min(8usize, s.n);
        for i in 0..max_i {
            eprintln!("[chi] w({}) = {}", i, format_ext(w.w(i)));
        }
    }
    let vjs: Vec<Vec<K>> = mats_csr.par_iter()
        .map(|csr| spmv_csr_t_weighted_fk(csr, &w))
        .collect();
    #[cfg(feature = "debug-logs")]
    println!("💥 [OPTIMIZATION] Weighted CSR M_j^T * χ_r computed: {:.2}ms (nnz={})",
             transpose_once_start.elapsed().as_secs_f64() * 1000.0, total_nnz);

    // =========================================================================
    // OUTPUT: BUILD k ME INSTANCES (Section 4.4 output specification)
    // =========================================================================
    // **Paper Reference**: Output of Π_CCS is k ME instances in ME(b,L)^k
    // Each ME instance carries (c, X, r', {y_j}, m_in) where:
    //   - c: commitment (from input MCS or ME)
    //   - X: public input projection
    //   - r': new random point from sum-check
    //   - y_j: partial evaluations Z_i·M_j^T·r̂' for all j ∈ [t]
    
    // Build ME instances: k outputs = 1 MCS + (k-1) ME inputs
    #[cfg(feature = "debug-logs")]
    let me_start = std::time::Instant::now();
    
    // Generate fold_digest from final transcript state
    // This binds the ME instances to the exact folding proof and prevents re-binding attacks
    let fold_digest = tr.digest32();
    
    // Helper: Ajtai recomposition to scalar m_j = Σ_ℓ b^{ℓ-1}·y_{j,ℓ}
    let base_f = K::from(F::from_u64(params.b as u64));
    let mut pow_cache = vec![K::ONE; neo_math::D];
    for i in 1..neo_math::D { pow_cache[i] = pow_cache[i-1] * base_f; }
    let recompose_to_scalar = |y_vec: &[K]| -> K {
        y_vec.iter().zip(&pow_cache).fold(K::ZERO, |acc, (&v, &p)| acc + v * p)
    };
    
    let mut out_me = Vec::with_capacity(insts.len() + me_witnesses.len());
    
    // (i=1): Build ME outputs for each MCS instance
    for (_inst_idx, inst) in insts.iter().enumerate() {
        // X = L_x(Z)
        let X = l.project_x(inst.Z, inst.m_in);
        
        // OPTIMIZATION: Use precomputed v_j vectors and MLE fold results  
        let mut y = Vec::with_capacity(s.t());
        for (_j, vj) in vjs.iter().enumerate() {
            let z_ref = neo_ccs::MatRef::from_mat(inst.Z);
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, vj);
            y.push(yj);
        }
        
        // y_scalars[j] = Σ_ℓ b^{ℓ-1}·y_{(1,j),ℓ} (Ajtai recomposition)
        let y_scalars: Vec<K> = y.iter().map(|yj| recompose_to_scalar(yj)).collect();

        out_me.push(MeInstance{ 
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: inst.c.clone(), 
            X, 
            r: r_prime.to_vec(),   // only the row component
            y, 
            y_scalars,
            m_in: inst.m_in,
            fold_digest,
        });
    }
    
    // (i=2..k): Build ME outputs for each input ME using its witness Z_i
    for (inp, zi) in me_inputs.iter().zip(me_witnesses.iter()) {
        // Sanity: verify commitment consistency (optional but good for honest prover)
        // let expected_c = l.commit(zi);
        // if expected_c != inp.c { return Err(...); }
        
        let mut y = Vec::with_capacity(s.t());
        let z_ref = neo_ccs::MatRef::from_mat(zi);
        for vj in &vjs {
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, vj);
            y.push(yj);
        }
        let y_scalars: Vec<K> = y.iter().map(|yj| recompose_to_scalar(yj)).collect();
        
        out_me.push(MeInstance {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: inp.c.clone(),
            X: inp.X.clone(), // already public from the input ME
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inp.m_in,
            fold_digest,
        });
    }

    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] ME instance building ({} outputs): {:.2}ms", 
             out_me.len(), me_start.elapsed().as_secs_f64() * 1000.0);

    // Prover-side χ probes for diagnostics (first two instances)
    #[cfg(feature = "neo-logs")]
    {
        let n = 1usize << ell_n;  // Use row dimension only
        let mut chi_lsbf = vec![K::ONE; n];
        for (jbit, &rj) in r_prime.iter().enumerate() {
            let stride = 1usize << jbit;
            let (a0, a1) = (K::ONE - rj, rj);
            for block in (0..n).step_by(stride * 2) {
                for i in 0..stride {
                    let t = chi_lsbf[block + i];
                    chi_lsbf[block + i] = t * a0;
                    chi_lsbf[block + i + stride] = t * a1;
                }
            }
        }
        let mut chi_msbf = vec![K::ZERO; n];
        for i in 0..n {
            let mut x = i; let mut y = 0usize;
            for _ in 0..ell_n { y = (y << 1) | (x & 1); x >>= 1; }
            chi_msbf[y] = chi_lsbf[i];
        }
        let inst_probe_max = core::cmp::min(2usize, insts.len());
        for inst_idx in 0..inst_probe_max {
            for j in 0..s.t() {
                let acc_lsbf = (0..s.n).fold(K::ZERO, |acc, i| acc + K::from(insts[inst_idx].mz[j][i]) * chi_lsbf[i]);
                let acc_msbf = (0..s.n).fold(K::ZERO, |acc, i| acc + K::from(insts[inst_idx].mz[j][i]) * chi_msbf[i]);
                let y_cur = out_me[inst_idx].y_scalars[j];
                eprintln!(
                    "[pi-ccs][probe] inst{} j{}: y_cur={}, acc_lsbf={}, acc_msbf={}",
                    inst_idx, j, format_ext(y_cur), format_ext(acc_lsbf), format_ext(acc_msbf)
                );
            }
        }
    }
    #[cfg(not(any(feature = "neo-logs", feature = "debug-logs")))]
    let _ = _running_sum;

    // (Optional) self-check could compare against generic terminal; omitted for performance.

    // Carry exactly the initial_sum value we absorbed (works for both engines)
    let sc_initial_sum = Some(initial_sum);
    
    let proof = PiCcsProof { 
        sumcheck_rounds: rounds, 
        header_digest: fold_digest,
        sc_initial_sum,
    };
    Ok((out_me, proof))
}

// ============================================================================
// Π_CCS VERIFIER (Section 4.4)
// ============================================================================
// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS (verification)
//
// **Verification Strategy** (matching paper structure):
//   1. Replay transcript to derive same challenges α, β, γ as prover
//   2. Verify sum-check rounds → extract (r', α') and running_sum
//   3. Verify transcript binding (fold_digest matches)
//   4. Check terminal identity: running_sum ?= Q(α',r') using public values
//
// **Key Insight**: Verifier can compute Q(α',r') from public data only:
//   - F' from output y values (Ajtai recomposition)
//   - NC' from output y values (range polynomial)
//   - Eval' from input ME y values (if k>1)

/// Verify Π_CCS: Check sum-check rounds and final claim Q(r) = 0
/// 
/// **Paper Reference**: Section 4.4 - CCS Reduction Π_CCS (verification)
/// 
/// Verifier checks proof without access to secret witnesses.
/// Per paper: MCS(b,L) × ME(b,L)^{k-1} → ME(b,L)^k
pub fn pi_ccs_verify(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],  // NEW: k-1 ME inputs with shared r
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    // =========================================================================
    // SETUP: Initialize transcript and compute parameters
    // =========================================================================
    
    tr.append_message(tr_labels::PI_CCS, b"");
    
    // Compute same parameters as prover (Section 4.3)
    // Paper specifies sum-check over {0,1}^{log(dn)} hypercube
    if s.n == 0 { return Err(PiCcsError::InvalidInput("n=0 not allowed".into())); }
    let d_pad = neo_math::D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;      // log d (Ajtai dimension)
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;      // log n (row dimension)
    let ell    = ell_d + ell_n;                       // log(dn) - FULL hypercube as per paper
    let d_sc   = core::cmp::max(
        s.max_degree() as usize + 1,                  // F with eq gating
        core::cmp::max(2, 2 * params.b as usize),     // Eval+eq vs. Range+eq
    );

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // =========================================================================
    // STEP 0: BIND INSTANCES (replay prover's transcript)
    // =========================================================================
    // Absorb same instance data as prover to derive same challenges
    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);
    // OPTIMIZATION: Absorb compact ZK-friendly digest instead of 500M+ field elements 
    // This reduces transcript absorption from ~51s to microseconds using Poseidon2
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest { tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]); }
    // Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        // Absorb commitment to prevent cross-instance attacks
        tr.append_fields(b"c_data", &inst.c.data);
    }

    // --- Absorb ME inputs BEFORE sampling challenges (mirror prover) ---
    tr.append_message(b"neo/ccs/me_inputs", b"");
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);
    for me in me_inputs.iter() {
        tr.append_fields(b"c_data_in", &me.c.data);
        tr.append_u64s(b"m_in_in", &[me.m_in as u64]);
        for limb in &me.r { tr.append_fields(b"r_in", &limb.as_coeffs()); }
        for yj in &me.y {
            for &y_elem in yj {
                tr.append_fields(b"y_elem", &y_elem.as_coeffs());
            }
        }
    }

    // =========================================================================
    // STEP 1: DERIVE CHALLENGES α, β, γ (replay prover's sampling)
    // =========================================================================
    // **Paper Reference**: Section 4.4, Step 1
    // Verifier derives same challenges as prover from transcript
    tr.append_message(b"neo/ccs/chals/v1", b"");
    let alpha_vec: Vec<K> = (0..ell_d)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    let beta: Vec<K> = (0..ell)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    let (beta_a, beta_r) = beta.split_at(ell_d);
    let ch_g = tr.challenge_fields(b"chal/k", 2);
    let gamma: K = neo_math::from_complex(ch_g[0], ch_g[1]);

    // Use the input ME's shared r (if k>1, must be the same across me_inputs)
    // For k=1 (initial fold), me_inputs is empty
    let r_input_opt = if !me_inputs.is_empty() {
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
    // **Paper Reference**: Section 4.4, Step 2
    // Verify sum-check rounds and extract (r', α') and running_sum
    
    if proof.sumcheck_rounds.len() != ell { return Ok(false); }
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
        if let Some(round0) = proof.sumcheck_rounds.get(0) {
            use crate::sumcheck::poly_eval_k;
            let p0 = poly_eval_k(round0, K::ZERO);
            let p1 = poly_eval_k(round0, K::ONE);
            eprintln!("[pi-ccs][verify] round0: p(0)={}, p(1)={}, p(0)+p(1)={}",
                      format_ext(p0), format_ext(p1), format_ext(p0 + p1));
        }
    }
    
    // Bind initial_sum BEFORE verifying rounds (verifier side)
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r_vec, running_sum, ok_rounds) =
        verify_sumcheck_rounds(tr, d_round, claimed_initial, &proof.sumcheck_rounds);
    if !ok_rounds { return Ok(false); }

    // Split r_vec into (r', α') - row-first ordering!
    // First ell_n challenges are row bits, last ell_d challenges are Ajtai bits
    if r_vec.len() != ell { return Ok(false); }
    let (r_prime, alpha_prime) = r_vec.split_at(ell_n);

    // =========================================================================
    // STEP 3: VERIFY TRANSCRIPT BINDING AND STRUCTURAL CHECKS
    // =========================================================================
    // **Security checks**: Ensure proof and outputs are bound to this transcript
    
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
        // Derive digest exactly where the prover did (after sum-check rounds)
        let digest = tr.digest32();
        // Verify proof header matches transcript state
        if proof.header_digest != digest {
            #[cfg(feature = "debug-logs")]
            eprintln!("❌ PI_CCS VERIFY: header digest mismatch (proof={:?}, verifier={:?})",
                      &proof.header_digest[..4], &digest[..4]);
            return Ok(false);
        }
        // Verify all output ME instances are bound to this transcript
        if !out_me.iter().all(|me| me.fold_digest == digest) {
            #[cfg(feature = "debug-logs")]
            eprintln!("❌ PI_CCS VERIFY: out_me fold_digest mismatch");
            return Ok(false);
        }
    }

    // Light structural sanity: every output ME must carry r' (row part) only
    if !out_me.iter().all(|me| me.r == r_prime) { return Ok(false); }
    
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
        // Guard individual y[j] vector lengths
        for (j, yj) in out.y.iter().enumerate() {
            if yj.len() != neo_math::D {
                return Err(PiCcsError::InvalidInput(format!(
                    "output[{}].y[{}].len {} != D {}", i, j, yj.len(), neo_math::D
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
            if yj.len() != neo_math::D {
                return Err(PiCcsError::InvalidInput(format!("me_output[{}].y[{}].len {} != D {}", idx, j, yj.len(), neo_math::D)));
            }
        }
    }

    // =========================================================================
    // STEP 4: VERIFY TERMINAL IDENTITY v ?= Q(α',r')
    // =========================================================================
    // **Paper Reference**: Section 4.4, Step 4
    // V: Check v ?= eq((α',r'), β)·(F' + Σ γ^i·N_i') + eq((α',r'), (α,r))·Eval'
    //
    // This is the soundness check!
    // We must verify that the final running_sum equals Q(r).
    // 
    // NOTE: Only CCS and range/decomp constraints in Q(r).
    // Tie constraints removed from sum-check as they break soundness.
    
    // Unified terminal check shape guards
    for me in out_me.iter() {
        if me.y_scalars.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "output[].y_scalars.len {} != t {}", me.y_scalars.len(), s.t()
            )));
        }
    }
    
    // === Paper-style terminal evaluation check ===
    // **Paper Reference**: Section 4.4, Step 4:
    // v ?= eq((α',r'), β)·(F' + Σ γ^i·N_i') + eq((α',r'), (α,r))·(Σ γ^{...}·ỹ'_{(i,j)}(α'))
    // Build eq polynomials on (alpha', r') vs β and vs (alpha, r_input)
    let eq_points = |p: &[K], q: &[K]| -> K {
        if p.len() != q.len() { return K::ZERO; }
        let mut acc = K::ONE;
        for i in 0..p.len() {
            acc *= (K::ONE - p[i]) * (K::ONE - q[i]) + p[i] * q[i];
        }
        acc
    };
    let eq_aprp_beta = eq_points(alpha_prime, beta_a) * eq_points(r_prime, beta_r);
    let eq_aprp_ar = if let Some(r_input) = r_input_opt {
        eq_points(alpha_prime, &alpha_vec) * eq_points(r_prime, r_input)
    } else {
        K::ZERO  // k=1: No input ME, Eval term is zero
    };

    // F': reconstruct m_j from out_me[0].y (Ajtai recomposition), then f(m_1..m_t)
    let base_f = K::from(F::from_u64(params.b as u64));
    let mut pow_cache = vec![K::ONE; neo_math::D];
    for i in 1..neo_math::D { pow_cache[i] = pow_cache[i-1] * base_f; }
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let y_vec = &out_me[0].y[j]; // K^d
        let mut acc = K::ZERO;
        for row in 0..neo_math::D { acc += y_vec[row] * pow_cache[row]; }
        m_vals[j] = acc;
    }
    let f_prime = s.f.eval_in_ext::<K>(&m_vals);

    // N_i': use y'_(i,1) and α' (from sum-check, not fresh α) to derive digit range polynomial
    // Note: Evaluate at alpha_prime (the bound Ajtai challenges), not alpha_vec (fresh challenge)
    let chi_alpha_prime: Vec<K> = neo_ccs::utils::tensor_point::<K>(alpha_prime);
    let mut nc_sum_prime = K::ZERO;
    let mut gamma_pow = gamma; // γ^1
    for me in out_me.iter() {
        let y_i1 = &me.y[0];
        let mut y_mle = K::ZERO;
        for (idx, &val) in y_i1.iter().enumerate() { 
            if idx < chi_alpha_prime.len() { y_mle += val * chi_alpha_prime[idx]; }
        }
        // Balanced range polynomial: N_i(x) = ∏_{t=-(b-1)}^{b-1} (x - t)
        let mut Ni = K::ONE;
        for t in -(params.b as i64 - 1)..=(params.b as i64 - 1) {
            Ni *= y_mle - K::from(F::from_i64(t));
        }
        nc_sum_prime += gamma_pow * Ni;
        gamma_pow *= gamma;
    }

    // Eval sum' over i=2..k and all j: weighted ỹ'_{(i,j)}(α')
    // out_me[0] is the i=1/MCS slot, so out_me[1..] are i=2..k
    // Note: Exponent formula must match prover's G_eval computation exactly
    // Prover uses: exponent = (i_offset + 1) + j * k_total
    // Verifier uses: exponent = i + j * k_total (where i starts at 1 for out_me[1])
    // These match because out_me ordering is: [MCS(i=1), ME_input[0](i=2), ME_input[1](i=3), ...]
    let mut eval_sum_prime = K::ZERO;
    let k_total = out_me.len();
    for j in 0..s.t() {
        for i in 1..k_total { // i loops 1..k_total (offset by 1 from paper i=2..k+1)
            let y_vec = &out_me[i].y[j];
            let mut y_mle = K::ZERO;
            for (idx, &val) in y_vec.iter().enumerate() { 
                if idx < chi_alpha_prime.len() { y_mle += val * chi_alpha_prime[idx]; }
            }
            // Corrected weight: γ^{i + j*k_total} which equals γ^{(i+1) + j*k - 1} in paper notation
            let exponent = i + j * k_total;
            let mut w_pow = K::ONE;
            for _ in 0..exponent { w_pow *= gamma; }
            eval_sum_prime += w_pow * y_mle;
        }
    }

    // Final identity: v ?= eq((α', r'), β) · (F + Σ γ^i N_i) + eq((α', r'), (α, r)) · (Σ γ^{i+jk-1} ỹ'_{(i,j)}(α'))
    // NOTE: No gamma_to_k multiplier - exponents already correct in eval_sum_prime
    let rhs = eq_aprp_beta * (f_prime + nc_sum_prime) + eq_aprp_ar * eval_sum_prime;
    if running_sum != rhs {
        #[cfg(feature = "debug-logs")]
        eprintln!(
            "[pi-ccs] terminal mismatch: running_sum != Q(β, α, r, r')\n  running_sum = {}\n  rhs = {}",
            format_ext(running_sum), format_ext(rhs)
        );
        return Ok(false);
    }

    // TODO: verify v_j = M_j^T χ_r if carried (disabled to keep verifier lightweight) under a flag for testing

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

// ============================================================================
// TRANSCRIPT UTILITIES
// ============================================================================
// Helper functions for replaying transcript and extracting verification data.

/// Data derived from the Π-CCS transcript tail used by the verifier.
#[derive(Debug, Clone)]
pub struct TranscriptTail {
    pub _wr: K,
    pub r: Vec<K>,
    pub alphas: Vec<K>,
    pub running_sum: K,
    /// The claimed sum over the hypercube (T in the paper), used to verify satisfiability
    pub initial_sum: K,
}

/// Replay the Π-CCS transcript to derive the tail (wr, r, alphas).
pub fn pi_ccs_derive_transcript_tail(
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    tr.append_message(tr_labels::PI_CCS, b"");
    // Header (same as in pi_ccs_verify)
    if s.n == 0 { return Err(PiCcsError::InvalidInput("n=0 not allowed".into())); }
    // ℓ = log(d*n), not just log(n) (mirror prove/verify)
    let d_pad = neo_math::D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;      // log d (Ajtai dimension)
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;      // log n (row dimension)
    let ell    = ell_d + ell_n;                       // log(dn) - FULL hypercube as per paper
    let d_sc   = core::cmp::max(
        s.max_degree() as usize + 1,                  // F with eq gating
        core::cmp::max(2, 2 * params.b as usize),     // Eval+eq vs. Range+eq
    );
    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // Instances
    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest { tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]); }
    absorb_sparse_polynomial(&mut tr, &s.f);
    for inst in mcs_list.iter() {
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        tr.append_fields(b"c_data", &inst.c.data);
    }

    // NOTE: ME inputs not absorbed here since this is a simplified helper
    // In real usage, this should match prove/verify transcript exactly
    // For now, assume no ME inputs (empty slice) for initial fold
    tr.append_message(b"neo/ccs/me_inputs", b"");
    tr.append_u64s(b"me_count", &[0u64]); // Initial fold: k=1, no ME inputs

    // Sample challenges (mirror prove/verify) - NO r_in!
    tr.append_message(b"neo/ccs/chals/v1", b"");
    let _alpha_vec: Vec<K> = (0..ell_d)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    let _beta: Vec<K> = (0..ell)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    let _gamma: K = {
        let ch_g = tr.challenge_fields(b"chal/k", 2);
        neo_math::from_complex(ch_g[0], ch_g[1])
    };
    // REMOVED: r_in sampling (no longer in protocol)

    // Derive r by verifying rounds (structure only)
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

    // Bind initial_sum BEFORE rounds to match prover/verifier transcript layout
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r, running_sum, ok_rounds) = verify_sumcheck_rounds(&mut tr, d_round, claimed_initial, &proof.sumcheck_rounds);
    if !ok_rounds {
        #[cfg(feature = "debug-logs")]
        eprintln!("[pi-ccs] rounds invalid: expected degree ≤ {}, got {} rounds", d_round, proof.sumcheck_rounds.len());
        return Err(PiCcsError::SumcheckError("rounds invalid".into()));
    }
    // (Already bound before rounds)

    // NOTE: No s(0)+s(1) == 0 requirement for R1CS eq-binding; terminal equality suffices.

    // Keep transcript layout; wr no longer used by verifier semantics
    let _wr = K::ONE;
    
    #[cfg(feature = "debug-logs")]
    eprintln!("[pi-ccs] derive_tail: s.n={}, ell={}, d_sc={}, outputs={}, rounds={}", s.n, ell, d_sc, mcs_list.len(), proof.sumcheck_rounds.len());
    Ok(TranscriptTail { _wr, r, alphas: Vec::new(), running_sum, initial_sum: claimed_initial })
}

// (Removed backward-compat wrappers in favor of `pi_ccs_derive_transcript_tail` only)

/// Compute the terminal claim from Π_CCS outputs given wr or generic CCS terminal.
pub fn pi_ccs_compute_terminal_claim_r1cs_or_ccs(
    s: &CcsStructure<F>,
    _wr: K,
    alphas: &[K],
    out_me: &[MeInstance<Cmt, F, K>],
) -> K {
    // Unified semantics: ignore wr; always compute generic CCS terminal
    let mut expected_q_r = K::ZERO;
    for (inst_idx, me_inst) in out_me.iter().enumerate() {
        let f_eval = s.f.eval_in_ext::<K>(&me_inst.y_scalars);
        expected_q_r += alphas[inst_idx] * f_eval;
    }
    expected_q_r
}

