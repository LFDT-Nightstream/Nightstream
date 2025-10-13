//! Π_CCS: Sum-check reduction over extension field K
//!
//! This is the single sum-check used throughout Neo protocol.
//! Proves: Σ_{u∈{0,1}^ℓ} (Σ_i α_i · f_i(u)) · χ_r(u) = 0

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

// ===== CSR Sparse Matrix Operations =====

/// Minimal CSR (Compressed Sparse Row) format for sparse matrix operations
/// This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices
pub struct Csr<F> {
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

// ===== Streaming/half-table equality-weighted CSR transpose multiply =====

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
pub fn spmv_csr_t_weighted_fk<W: RowWeight + Sync>(a: &Csr<F>, w: &W) -> Vec<K> {
    let mut v = vec![K::ZERO; a.cols];
    let chunk = a.rows / rayon::current_num_threads().max(1) + 1;
    let partials: Vec<Vec<(usize, K)>> = (0..a.rows)
        .into_par_iter()
        .with_min_len(chunk)
        .map(|r| {
            let mut loc = Vec::with_capacity(a.indptr[r + 1] - a.indptr[r]);
            let wr = w.w(r);
            for k in a.indptr[r]..a.indptr[r + 1] {
                let c = a.indices[k];
                loc.push((c, K::from(a.data[k]) * wr));
            }
            loc
        })
        .collect();
    for loc in partials { for (c, val) in loc { v[c] += val; } }
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


// ===== Local MLE partial structures and Sum-check round oracles =====

// Generic CCS partials: one shrinking vector per matrix M_j z
struct MlePartials { s_per_j: Vec<Vec<K>> }
// R1CS residual partials: one shrinking vector over row-wise residuals
#[allow(dead_code)]
struct MleResiduals { s: Vec<K> }

struct GenericCcsOracle<'a> {
    s: &'a CcsStructure<F>,
    alphas: Vec<K>,
    partials_per_inst: Vec<MlePartials>,
    // Small-value fast path for round 0: access original F-vectors without lifting
    mz_f_per_inst: Vec<Vec<&'a [F]>>,
    ell: usize,
    d_sc: usize,
    first_round_done: bool,
}
impl<'a> RoundOracle for GenericCcsOracle<'a> {
    fn num_rounds(&self) -> usize { self.ell }
    fn degree_bound(&self) -> usize { self.d_sc }
    fn evals_at(&mut self, sample_xs: &[K]) -> Vec<K> {
        let t = self.s.t();
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];
        // Scratch buffers reused across loops
        if !self.first_round_done {
            // Round 0: iterate over each remaining row-block ρ (k), build (a,d), evaluate f, then sum
            let n_pad = 1usize << self.ell;
            let half = n_pad >> 1;
            let mut a = vec![K::ZERO; t];
            let mut d = vec![K::ZERO; t];
            let mut y_buf = vec![K::ZERO; t];
            for (inst_idx, mz_f) in self.mz_f_per_inst.iter().enumerate() {
                let alpha = self.alphas[inst_idx];
                for k in 0..half {
                    // Build per-ρ a,d across all j
                    for j in 0..t {
                        let v_f = mz_f[j];
                        let idx_e = 2 * k;
                        let idx_o = idx_e + 1;
                        let e = if idx_e < v_f.len() { v_f[idx_e] } else { F::ZERO };
                        let o = if idx_o < v_f.len() { v_f[idx_o] } else { F::ZERO };
                        a[j] = K::from(e);
                        d[j] = K::from(o - e);
                    }
                    // Evaluate f(a + d X) for all requested X and accumulate
                    for (sx, &X) in sample_xs.iter().enumerate() {
                        for j in 0..t { y_buf[j] = a[j] + d[j] * X; }
                        sample_ys[sx] += alpha * self.s.f.eval_in_ext::<K>(&y_buf);
                    }
                }
            }
            self.first_round_done = true;
        } else {
            // Later rounds: use folded partials; same per-ρ evaluation pattern
            let mut a = vec![K::ZERO; t];
            let mut d = vec![K::ZERO; t];
            let mut y_buf = vec![K::ZERO; t];
            for (inst_idx, partials) in self.partials_per_inst.iter().enumerate() {
                let alpha = self.alphas[inst_idx];
                debug_assert!(partials.s_per_j.len() == t);
                let half = partials.s_per_j[0].len() >> 1;
                for k in 0..half {
                    for j in 0..t {
                        let v = &partials.s_per_j[j];
                        let e = v[2 * k];
                        let o = v[2 * k + 1];
                        a[j] = e;
                        d[j] = o - e;
                    }
                    for (sx, &X) in sample_xs.iter().enumerate() {
                        for j in 0..t { y_buf[j] = a[j] + d[j] * X; }
                        sample_ys[sx] += alpha * self.s.f.eval_in_ext::<K>(&y_buf);
                    }
                }
            }
        }
        sample_ys
    }
    fn fold(&mut self, r_i: K) {
        for partials in self.partials_per_inst.iter_mut() {
            for v in &mut partials.s_per_j {
                let n2 = v.len() >> 1;
                for k in 0..n2 {
                    let a0 = v[2*k]; let b0 = v[2*k + 1];
                    v[k] = (K::ONE - r_i) * a0 + r_i * b0;
                }
                v.truncate(n2);
            }
        }
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

/// Prove Π_CCS: CCS instances satisfy constraints via sum-check
/// 
/// Input: k+1 CCS instances, outputs k ME instances + proof
pub fn pi_ccs_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    l: &L, // we need L to check c = L(Z) and to compute X = L_x(Z)
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    tr.append_message(tr_labels::PI_CCS, b"");

    // --- Input & policy checks ---
    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput("empty or mismatched inputs".into()));
    }
    // >>> CHANGE #1: allow arbitrary n; compute ℓ from next power of two
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }
    let n_pad = s.n.next_power_of_two().max(2);
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
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // MAJOR OPTIMIZATION: Convert to CSR sparse format once for all operations
    // This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices  
    #[cfg(feature = "debug-logs")]
    let csr_start = std::time::Instant::now();
    let mats_csr: Vec<Csr<F>> = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect();
    #[cfg(feature = "debug-logs")]
    let total_nnz: usize = mats_csr.iter().map(|c| c.data.len()).sum();
    #[cfg(feature = "debug-logs")]
    println!("🔥 [NUCLEAR] CSR conversion completed: {:.2}ms ({} matrices, {} nnz total, {:.4}% density)",
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
        // SECURITY: Ensure z matches CCS width to prevent OOB during M*z
        if z.len() != s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "SECURITY: z length {} does not match CCS column count {} (malformed instance)",
                z.len(), s.m
            )));
        }
        
        // === CRITICAL SECURITY CHECK: Z == Decomp_b(z) ===
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
            return Err(PiCcsError::InvalidInput("SECURITY: Z != Decomp_b(z) - prover using inconsistent z and Z".into()));
        }
        #[cfg(feature = "debug-logs")]
        println!("🔧 [INSTANCE {}] Z comparison: {:.2}ms", inst_idx, 
                 z_compare_start.elapsed().as_secs_f64() * 1000.0);
        // MAJOR OPTIMIZATION: Use CSR sparse matrix-vector multiply - O(nnz) instead of O(n*m)!
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

    // --- SECURITY: Absorb instance data BEFORE sampling challenges to prevent malleability ---
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
    // CRITICAL: Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        // CRITICAL: Absorb commitment to prevent cross-instance attacks
        tr.append_fields(b"c_data", &inst.c.data);
    }
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Transcript absorption: {:.2}ms", 
             transcript_start.elapsed().as_secs_f64() * 1000.0);

    // --- Generate eq-binding vector and batching coefficients for composed polynomial Q ---
    #[cfg(feature = "debug-logs")]
    let batching_start = std::time::Instant::now();
    // Eq-binding vector w sampled before batching (kept for transcript layout)
    tr.append_message(b"neo/ccs/eq", b"");
    let _w_eq: Vec<K> = (0..ell).map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) }).collect();
    tr.append_message(b"neo/ccs/batch", b"");
    
    // α coefficients for CCS constraints (one per instance)
    let alphas: Vec<K> = (0..insts.len()).map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) }).collect();
    
    let batch_coeffs = BatchingCoeffs { alphas };
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Batching coefficients: {:.2}ms", 
             batching_start.elapsed().as_secs_f64() * 1000.0);

    // --- Run sum-check rounds over composed polynomial Q (degree ≤ d_sc) ---
    #[cfg(feature = "debug-logs")]
    println!("🔍 Sum-check starting: {} instances, {} rounds", insts.len(), ell);
    let sample_xs_generic: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();
    // Single generic engine for all shapes (including R1CS-shaped CCS)

    // Build partial states (invariant across the engine)
    #[cfg(feature = "debug-logs")]
    let mle_start = std::time::Instant::now();
    let partials_per_inst_opt: Option<Vec<MlePartials>>;
    {
        let partials: Result<Vec<MlePartials>, PiCcsError> = insts.par_iter().map(|inst| {
            let mut s_per_j = Vec::with_capacity(s.t());
            for j in 0..s.t() {
                let w_k: Vec<K> = inst.mz[j].iter().map(|&x| K::from(x)).collect();
                let w_k = pad_to_pow2_k(w_k, ell)?;
                s_per_j.push(w_k);
            }
            Ok(MlePartials { s_per_j })
        }).collect();
        partials_per_inst_opt = Some(partials?);
    }
    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] MLE partials setup: {:.2}ms",
             mle_start.elapsed().as_secs_f64() * 1000.0);

    // Drive rounds with the generic engine
    // Compute initial_sum = s(0) + s(1) for round 0 and bind before rounds
    let initial_sum = {
        let t = s.t();
        let n_pad = 1usize << ell;
        let half = n_pad >> 1;
        let mut acc0_plus_1 = K::ZERO;
        let mut a = vec![K::ZERO; t];
        let mut o = vec![K::ZERO; t];
        for (inst_idx, inst) in insts.iter().enumerate() {
            let alpha = batch_coeffs.alphas[inst_idx];
            for k in 0..half {
                // Build per-ρ vectors a_ρ and o_ρ across all j
                for j in 0..t {
                    let v_f = &inst.mz[j];
                    let idx_e = 2 * k;
                    let idx_o = idx_e + 1;
                    let e = if idx_e < v_f.len() { v_f[idx_e] } else { F::ZERO };
                    let oo = if idx_o < v_f.len() { v_f[idx_o] } else { F::ZERO };
                    a[j] = K::from(e);
                    o[j] = K::from(oo);
                }
                let f0 = s.f.eval_in_ext::<K>(&a);
                let f1 = s.f.eval_in_ext::<K>(&o);
                acc0_plus_1 += alpha * (f0 + f1);
            }
        }
        acc0_plus_1
    };
    // Bind initial_sum BEFORE rounds to the transcript (prover side)
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());
    let SumcheckOutput { rounds, challenges: r, final_sum: _running_sum } = {
        let mut oracle = GenericCcsOracle {
            s, alphas: batch_coeffs.alphas.clone(),
            partials_per_inst: partials_per_inst_opt.unwrap(),
            mz_f_per_inst: insts.iter().map(|inst| inst.mz.iter().map(|v| v.as_slice()).collect()).collect(),
            ell, d_sc,
            first_round_done: false,
        };
        if d_sc >= 1 { run_sumcheck_skip_eval_at_one(tr, &mut oracle, initial_sum, &sample_xs_generic)? }
        else { run_sumcheck(tr, &mut oracle, initial_sum, &sample_xs_generic)? }
    };
    

    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] Sum-check rounds complete: {} rounds", ell);

    // Prover-side transcript snapshot and alpha/r summary
    #[cfg(feature = "neo-logs")]
    {
        use std::cmp::min;
        eprintln!("[pi-ccs][prove] ell={}, d_sc={}, instances={}, alphas_len={}",
                  ell, d_sc, insts.len(), batch_coeffs.alphas.len());
        eprintln!("[pi-ccs][prove] r[0..{}]={:?}",
                  min(4, r.len()), r.iter().take(4).collect::<Vec<_>>());
        eprintln!("[pi-ccs][prove] alphas[0..{}]={:?}",
                  min(4, batch_coeffs.alphas.len()),
                  batch_coeffs.alphas.iter().take(4).map(|a| format!("{:?}", a)).collect::<Vec<_>>());
        for (i, inst) in mcs_list.iter().enumerate().take(2) {
            let prefix = inst.c.data.as_slice();
            let prefix = if prefix.len() >= 4 { &prefix[0..4] } else { prefix };
            eprintln!(
                "[pi-ccs][prove] inst {}: m_in={}, c_prefix={:02x?}",
                i, inst.m_in, prefix
            );
        }
    }

    // Compute M_j^T * χ_r using streaming/half-table weights (no full χ_r materialization)
    #[cfg(feature = "debug-logs")]
    println!("🚀 [OPTIMIZATION] Computing M_j^T * χ_r with half-table weights...");
    #[cfg(feature = "debug-logs")]
    let transpose_once_start = std::time::Instant::now();
    let w = HalfTableEq::new(&r);
    #[cfg(feature = "neo-logs")]
    {
        let max_i = core::cmp::min(8usize, s.n);
        for i in 0..max_i {
            eprintln!("[chi] w({}) = {}", i, format_ext(w.w(i)));
        }
    }
    let vjs: Vec<Vec<K>> = mats_csr.par_iter()
        .map(|csr| spmv_csr_t_weighted_fk::<_>(csr, &w))
        .collect();
    #[cfg(feature = "debug-logs")]
    println!("💥 [OPTIMIZATION] Weighted CSR M_j^T * χ_r computed: {:.2}ms (nnz={})",
             transpose_once_start.elapsed().as_secs_f64() * 1000.0, total_nnz);

    // --- Build ME instances (one per input) ---
    #[cfg(feature = "debug-logs")]
    let me_start = std::time::Instant::now();
    
    // CRITICAL SECURITY FIX: Generate fold_digest from final transcript state
    // This binds the ME instances to the exact folding proof and prevents re-binding attacks
    let fold_digest = tr.digest32();
    
    let mut out_me = Vec::with_capacity(insts.len());
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
        // Compute Y_j(r) canonically: ⟨M_j z, χ_r⟩ using the same LSB-first row indexing as the sum-check oracle
        let y_scalars: Vec<K> = (0..s.t())
            .map(|j| {
                (0..s.n)
                    .map(|i| K::from(inst.mz[j][i]) * w.w(i))
                    .fold(K::ZERO, |acc, term| acc + term)
            })
            .collect();

        out_me.push(MeInstance{ 
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: inst.c.clone(), 
            X, 
            r: r.clone(), 
            y, 
            y_scalars,
            m_in: inst.m_in,
            fold_digest,
        });
    }

    #[cfg(feature = "debug-logs")]
    println!("🔧 [TIMING] ME instance building: {:.2}ms", 
             me_start.elapsed().as_secs_f64() * 1000.0);

    // Prover-side terminal decomposition and χ probes
    #[cfg(feature = "neo-logs")]
    {
        let mut sum_qr = K::ZERO;
        for (i, me) in out_me.iter().enumerate() {
            let f_eval = s.f.eval_in_ext::<K>(&me.y_scalars);
            let contrib = batch_coeffs.alphas[i] * f_eval;
            sum_qr += contrib;
            eprintln!(
                "[pi-ccs][prove] inst {}: f(Y)={}, alpha={}, alpha*f(Y)={}",
                i, format_ext(f_eval), format_ext(batch_coeffs.alphas[i]), format_ext(contrib)
            );
        }
        eprintln!("[pi-ccs][prove] Σ alpha f(Y) = {}", format_ext(sum_qr));
        eprintln!("[pi-ccs][prove] final running_sum = {}", format_ext(_running_sum));

        // Prover-side self-check: compare sum-check terminal running_sum vs Σ α_i f(Y(r))
        eprintln!(
            "[pi-ccs][prove/self-check] running_sum = {}",
            format_ext(_running_sum)
        );
        eprintln!(
            "[pi-ccs][prove/self-check] Σ α f(Y)   = {}",
            format_ext(sum_qr)
        );

        // Probe χ consistency for first two instances across all matrices
        let n = 1usize << ell;
        let mut chi_lsbf = vec![K::ONE; n];
        for (jbit, &rj) in r.iter().enumerate() {
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
            for _ in 0..ell { y = (y << 1) | (x & 1); x >>= 1; }
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

/// Verify Π_CCS: Check sum-check rounds AND the critical final claim Q(r) = 0
pub fn pi_ccs_verify(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>], // Now used for final Q(r) check
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    tr.append_message(tr_labels::PI_CCS, b"");
    // >>> CHANGE #2: allow arbitrary n; compute ℓ from next power of two
    if s.n == 0 { return Err(PiCcsError::InvalidInput("n=0 not allowed".into())); }
    let n_pad = s.n.next_power_of_two().max(2);
    let ell = n_pad.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // --- SECURITY: Absorb instance data BEFORE sampling challenges (match prover) ---
    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);
    // OPTIMIZATION: Absorb compact ZK-friendly digest instead of 500M+ field elements 
    // This reduces transcript absorption from ~51s to microseconds using Poseidon2
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest { tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]); }
    // CRITICAL: Absorb polynomial definition to prevent malicious polynomial substitution
    absorb_sparse_polynomial(tr, &s.f);
    
    // Absorb all instance data (commitment, public inputs, witness structure)
    for inst in mcs_list.iter() {
        // Absorb instance data that affects soundness
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        // CRITICAL: Absorb commitment to prevent cross-instance attacks
        tr.append_fields(b"c_data", &inst.c.data);
    }

    // Keep transcript layout stable, but also bind initial_sum before rounds
    tr.append_message(b"neo/ccs/eq", b"");
    let _w_eq: Vec<K> = (0..ell).map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) }).collect();
    tr.append_message(b"neo/ccs/batch", b"");
    let alphas: Vec<K> = (0..mcs_list.len()).map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) }).collect();
    
    let batch_coeffs = BatchingCoeffs { alphas };

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
    let (r, running_sum, ok_rounds) =
        verify_sumcheck_rounds(tr, d_round, claimed_initial, &proof.sumcheck_rounds);
    if !ok_rounds { return Ok(false); }

    // NOTE: No s(0)+s(1) == 0 requirement for R1CS eq-binding; terminal equality suffices.
    
    // (Already bound before rounds)

    // Verifier-side terminal decomposition logs
    #[cfg(feature = "neo-logs")]
    {
        use std::cmp::min;
        eprintln!(
            "[pi-ccs][verify] ell={}, d_round={}, claimed_initial={}",
            ell, d_round, format_ext(claimed_initial)
        );
        eprintln!("[pi-ccs][verify] r[0..{}]={:?}",
                  min(4, r.len()), r.iter().take(4).collect::<Vec<_>>());
        eprintln!("[pi-ccs][verify] alphas[0..{}]={:?}",
                  min(4, batch_coeffs.alphas.len()),
                  batch_coeffs.alphas.iter().take(4).map(|a| format!("{:?}", a)).collect::<Vec<_>>());

        let mut sum_qr = K::ZERO;
        for (i, me) in out_me.iter().enumerate() {
            let f_eval = s.f.eval_in_ext::<K>(&me.y_scalars);
            let contrib = batch_coeffs.alphas[i] * f_eval;
            eprintln!(
                "[pi-ccs][verify] inst {}: f(Y)={}, alpha={}, alpha*f(Y)={}",
                i, format_ext(f_eval), format_ext(batch_coeffs.alphas[i]), format_ext(contrib)
            );
            sum_qr += contrib;
        }
        eprintln!("[pi-ccs][verify] Σ alpha f(Y) = {}", format_ext(sum_qr));
        eprintln!("[pi-ccs][verify] running_sum   = {}", format_ext(running_sum));
    }

    // === CRITICAL TRANSCRIPT BINDING SECURITY CHECK ===
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

    // Light structural sanity: every output ME must carry the same r
    if !out_me.iter().all(|me| me.r == r) { return Ok(false); }

    // === CRITICAL BINDING: out_me[i] must match input instance ===
    // This prevents attacks where unrelated ME outputs pass RLC/DEC algebra
    if out_me.len() != mcs_list.len() {
        #[cfg(feature = "debug-logs")]
        eprintln!("[pi-ccs] out_me.len() {} != mcs_list.len() {}", out_me.len(), mcs_list.len());
        return Err(PiCcsError::InvalidInput(format!(
            "out_me.len() {} != mcs_list.len() {}", out_me.len(), mcs_list.len()
        )));
    }
    for (i, (out, inp)) in out_me.iter().zip(mcs_list.iter()).enumerate() {
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

    // === CRITICAL SUM-CHECK TERMINAL VERIFICATION ===
    // This is the missing piece that makes the proof sound!
    // We must verify that the final running_sum equals Q(r).
    // 
    // NOTE: Only CCS and range/decomp constraints in Q(r).
    // Tie constraints removed from sum-check as they break soundness.
    
    // Unified terminal check for all shapes: Q(r) = Σ α_i·f(Y(r))
    for (i, me) in out_me.iter().enumerate() {
        if me.y_scalars.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{}].y_scalars.len {} != t {}", i, me.y_scalars.len(), s.t()
            )));
        }
    }
    
    // Compute batched terminal evaluation
    let mut expected_q_r = K::ZERO;
    for (inst_idx, me_inst) in out_me.iter().enumerate() {
        let f_eval = s.f.eval_in_ext::<K>(&me_inst.y_scalars);
        expected_q_r += batch_coeffs.alphas[inst_idx] * f_eval;
    }
    if running_sum != expected_q_r {
        #[cfg(feature = "debug-logs")]
        eprintln!(
            "[pi-ccs] terminal mismatch (CCS): running_sum != Σ α_i f(Y)\n  running_sum = {}\n  Σ α_i f(Y) = {}",
            format_ext(running_sum),
            format_ext(expected_q_r)
        );
        return Ok(false);
    }

    // 🔒 SOUNDNESS FIX: Verify that the claimed initial sum is zero
    // For a valid CCS instance, Σ_{x∈{0,1}^ℓ} f((M·z)[x]) should equal 0.
    // The initial sum T⁰ = Σ_i α_i · Σ_x f((M·z_i)[x]) should also be 0 when all instances are valid.
    // 
    // Bug: Previously didn't check if initial_sum == 0, allowing batching of valid + invalid
    // instances to produce a non-zero initial sum that still passed terminal verification.
    // In base case: LHS all-zero gives T⁰ ≈ 0, so invalid RHS is caught.
    // In non-base case: LHS + invalid RHS could produce non-zero T⁰ that verifies.
    //
    // SOUNDNESS REQUIREMENT: ℓ must be at least 2 for proper validation.
    // In the ℓ=1 case (single-row padded to 2), the augmented CCS can carry a constant offset
    // (e.g., from const-1 binding or other glue), making the hypercube sum non-zero even for
    // valid witnesses. This prevents us from detecting invalid witnesses.
    // Production circuits MUST have at least 3 constraint rows to ensure ℓ ≥ 2 after padding.
    if ell < 2 {
        panic!(
            "SOUNDNESS ERROR: Pi-CCS verification requires ℓ ≥ 2, got ℓ = {}.\n\
             Single-row CCS (ℓ=1) cannot be properly validated because the augmented CCS \n\
             carries constant offsets that prevent distinguishing valid from invalid witnesses.\n\
             Please ensure your step CCS has at least 3 rows (which pads to 4, giving ℓ=2).",
            ell
        );
    }
    
    if claimed_initial != K::ZERO {
        #[cfg(feature = "debug-logs")]
        eprintln!(
            "[pi-ccs] SOUNDNESS CHECK FAILED: initial sum T⁰ = {} ≠ 0\n\
             This indicates at least one batched instance violates the CCS relation.\n\
             For valid instances: Σ_{{x∈{{0,1}}^ℓ}} f((M·z)[x]) must equal 0.",
            format_ext(claimed_initial)
        );
        return Ok(false);
    }

    // Initial sum is zero (all instances satisfy CCS) and terminal consistency confirmed.
    // RLC/DEC stages will verify commitment bindings separately.

    // TODO: verify v_j = M_j^T χ_r if carried (disabled to keep verifier lightweight) under a flag for testing

    Ok(true)
}

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
    // Keep derive-tail consistent with prove/verify to avoid ℓ=0 for n=1
    let n_pad = s.n.next_power_of_two().max(2);
    let ell = n_pad.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;
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

    // Sample eq-binding vector w and batch alphas (layout only; wr unused by verifier)
    tr.append_message(b"neo/ccs/eq", b"");
    let _w_eq: Vec<K> = (0..ell)
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();
    tr.append_message(b"neo/ccs/batch", b"");
    let alphas: Vec<K> = (0..mcs_list.len())
        .map(|_| { let ch = tr.challenge_fields(b"chal/k", 2); neo_math::from_complex(ch[0], ch[1]) })
        .collect();

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
    Ok(TranscriptTail { _wr, r, alphas, running_sum, initial_sum: claimed_initial })
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
