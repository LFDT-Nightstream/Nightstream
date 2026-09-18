use p3_field::Field;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Validate n is a power-of-two.
pub fn validate_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Compute the tensor point r^b = ⊗_{i=1..ell} (r_i, 1-r_i) ∈ K^n where n = 2^ell.
pub fn tensor_point<K: Field>(r: &[K]) -> Vec<K> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];
    // Gray-code style expansion
    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = K::ONE - ri;
        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                let a = out[idx + j];
                out[idx + j] = a * one_minus;
            }
            for j in 0..stride {
                let a = out[idx + stride + j];
                out[idx + stride + j] = a * ri;
            }
            idx += 2 * stride;
        }
    }
    out
}

/// Compute [`tensor_point`] with parallel stage updates for large domains.
pub fn tensor_point_parallel<K>(r: &[K]) -> Vec<K>
where
    K: Field + Send + Sync,
{
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    const PAR_THRESHOLD: usize = 1 << 14;

    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block_len = 2 * stride;
        let one_minus = K::ONE - ri;

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            if n >= PAR_THRESHOLD && rayon::current_num_threads() > 1 {
                out.par_chunks_mut(block_len).for_each(|block| {
                    let (lo, hi) = block.split_at_mut(stride);
                    for slot in lo.iter_mut() {
                        *slot *= one_minus;
                    }
                    for slot in hi.iter_mut() {
                        *slot *= ri;
                    }
                });
                continue;
            }
        }

        let block_count = 1usize << (ell - i - 1);
        let mut idx = 0usize;
        for _ in 0..block_count {
            for j in 0..stride {
                out[idx + j] *= one_minus;
            }
            for j in 0..stride {
                out[idx + stride + j] *= ri;
            }
            idx += block_len;
        }
    }

    out
}

/// Multiply an F-matrix (n×m) by an F-vector (m) → F-vector (n).
pub fn mat_vec_mul_ff<F: Field>(m: &[F], n_rows: usize, n_cols: usize, v: &[F]) -> Vec<F> {
    debug_assert_eq!(n_cols, v.len());
    let mut out = vec![F::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc = F::ZERO;
        let row = &m[r * n_cols..(r + 1) * n_cols];
        for (a, b) in row.iter().zip(v.iter()) {
            acc += *a * *b;
        }
        out[r] = acc;
    }
    out
}

/// Multiply an F-matrix (d×m) by a K-vector (m) using the natural embedding F→K.
pub fn mat_vec_mul_fk<F: Field, K: Field + From<F>>(m: &[F], n_rows: usize, n_cols: usize, v: &[K]) -> Vec<K> {
    debug_assert_eq!(n_cols, v.len());
    let mut out = vec![K::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc = K::ZERO;
        let row = &m[r * n_cols..(r + 1) * n_cols];
        for (a_f, b_k) in row.iter().zip(v.iter()) {
            let a_k: K = (*a_f).into();
            acc += a_k * *b_k;
        }
        out[r] = acc;
    }
    out
}

/// Extract triplets from a [`CcsMatrix`], applying row/col offsets for block-diagonal embedding.
fn embed_matrix_parts<F: Field>(
    src: &crate::sparse::CcsMatrix<F>,
    row_offset: usize,
    col_offset: usize,
) -> (
    Vec<(usize, usize, F)>,
    Vec<crate::SeededPhi81LinearBlock>,
    Vec<crate::GeometricRowRun<F>>,
) {
    match src {
        crate::sparse::CcsMatrix::Identity { n } => (
            (0..*n)
                .map(|i| (row_offset + i, col_offset + i, F::ONE))
                .collect(),
            Vec::new(),
            Vec::new(),
        ),
        crate::sparse::CcsMatrix::Csc(m) => {
            let mut trips = Vec::with_capacity(m.vals.len());
            for col in 0..m.ncols {
                for k in m.column_range(col) {
                    trips.push((row_offset + m.row_index(k), col_offset + col, m.vals[k]));
                }
            }
            (trips, Vec::new(), Vec::new())
        }
        crate::sparse::CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut trips = Vec::with_capacity(csc.vals.len());
            for col in 0..csc.ncols {
                for k in csc.column_range(col) {
                    trips.push((row_offset + csc.row_index(k), col_offset + col, csc.vals[k]));
                }
            }
            (
                trips,
                blocks
                    .iter()
                    .map(|block| block.shifted(row_offset, col_offset))
                    .collect(),
                geometric_runs
                    .iter()
                    .map(|run| run.shifted(row_offset, col_offset))
                    .collect(),
            )
        }
    }
}

/// Build block-diagonal stacked matrices for two CCS structures.
///
/// ccs1's matrices occupy the top-left block, ccs2's the bottom-right.
/// All returned matrices have dimensions `n_total × m_total`.
fn build_stacked_matrices<F: Field>(
    ccs1: &crate::relations::CcsStructure<F>,
    ccs2: &crate::relations::CcsStructure<F>,
    n_total: usize,
    m_total: usize,
) -> Vec<crate::sparse::CcsMatrix<F>> {
    use crate::sparse::{CcsMatrix, CscMat};

    let t_total = ccs1.t() + ccs2.t();
    let mut stacked: Vec<CcsMatrix<F>> = Vec::with_capacity(t_total);

    // Top-left block (ccs1): no offset
    for j in 0..ccs1.t() {
        let (trips, blocks, geometric_runs) = embed_matrix_parts(&ccs1.matrices[j], 0, 0);
        let csc = CscMat::from_triplets(trips, n_total, m_total);
        stacked.push(
            CcsMatrix::csc_with_compact_rows(csc, blocks, geometric_runs)
                .expect("embedded compact terms must remain inside the direct-sum matrix"),
        );
    }

    // Bottom-right block (ccs2): offset by (n1, m1)
    for j in 0..ccs2.t() {
        let (trips, blocks, geometric_runs) = embed_matrix_parts(&ccs2.matrices[j], ccs1.n, ccs1.m);
        let csc = CscMat::from_triplets(trips, n_total, m_total);
        stacked.push(
            CcsMatrix::csc_with_compact_rows(csc, blocks, geometric_runs)
                .expect("embedded compact terms must remain inside the direct-sum matrix"),
        );
    }

    stacked
}

/// Compute the direct sum (block diagonal composition) of two CCS structures.
/// This creates a new CCS that enforces both systems independently by stacking
/// the constraint matrices in block-diagonal form and concatenating the witness vectors.
///
/// Used for IVC where we want to stack step CCS with embedded verifier CCS.
pub fn direct_sum<F: Field>(
    ccs1: &crate::relations::CcsStructure<F>,
    ccs2: &crate::relations::CcsStructure<F>,
) -> Result<crate::relations::CcsStructure<F>, crate::error::CcsError> {
    use crate::poly::{SparsePoly, Term};

    // If one is empty, return the other
    if ccs1.n == 0 || ccs1.m == 0 {
        return Ok(ccs2.clone());
    }
    if ccs2.n == 0 || ccs2.m == 0 {
        return Ok(ccs1.clone());
    }

    let t1 = ccs1.t();
    let n_total = ccs1.n + ccs2.n;
    let m_total = ccs1.m + ccs2.m;
    let t_total = t1 + ccs2.t();

    let stacked_matrices = build_stacked_matrices(ccs1, ccs2, n_total, m_total);

    // Build combined polynomial f
    // f_combined(X1, ..., X_{t1+t2}) = f1(X1, ..., X_{t1}) + f2(X_{t1+1}, ..., X_{t1+t2})
    let mut combined_terms = Vec::new();

    // Terms from f1 - pad exponent vector to length t1+t2
    for term in ccs1.f.terms() {
        let mut padded_exps = term.exps.clone();
        padded_exps.resize(t_total, 0); // pad with zeros
        combined_terms.push(Term {
            coeff: term.coeff,
            exps: padded_exps,
        });
    }

    // Terms from f2 - shift indices by t1
    for term in ccs2.f.terms() {
        let mut shifted_exps = vec![0; t1]; // zeros for first t1 variables
        shifted_exps.extend_from_slice(&term.exps); // f2's variables become X_{t1+1}, ...
        combined_terms.push(Term {
            coeff: term.coeff,
            exps: shifted_exps,
        });
    }

    let combined_f = SparsePoly::new(t_total, combined_terms);

    Ok(crate::relations::CcsStructure::new_sparse(
        stacked_matrices,
        combined_f,
    )?)
}

/// Block-diagonal direct sum with transcript-bound mixing (CANCELLATION-RESISTANT).
///
/// This prevents cross-cancellation between sub-systems in terminal polynomial checks.
/// f_total(X) = f1(X_1..X_t1) + beta * f2(X_{t1+1}..X_{t1+t2})
///
/// The mixing parameter β is derived from transcript state, preventing adversarial
/// cancellation between the two sub-systems in terminal polynomial checks.
///
/// # Security
/// - ✅ Prevents cross-cancellation in terminal polynomial evaluation
/// - ✅ Uses unpredictable, statement-bound mixing coefficient β
/// - ✅ Preserves block-diagonal constraint structure
/// - ✅ Recommended for combining independent CCS subsystems
///
/// # Arguments
/// - `ccs1`, `ccs2`: The two CCS structures to combine
/// - `beta`: Transcript-derived mixing coefficient (should be unpredictable)
pub fn direct_sum_mixed<F: Field>(
    ccs1: &crate::relations::CcsStructure<F>,
    ccs2: &crate::relations::CcsStructure<F>,
    beta: F,
) -> Result<crate::relations::CcsStructure<F>, crate::error::CcsError> {
    use crate::poly::{SparsePoly, Term};

    if ccs1.n == 0 || ccs1.m == 0 {
        // Scale the second CCS by beta
        let mut scaled_terms = Vec::new();
        for term in ccs2.f.terms() {
            scaled_terms.push(Term {
                coeff: term.coeff * beta,
                exps: term.exps.clone(),
            });
        }
        let scaled_f = SparsePoly::new(ccs2.f.arity(), scaled_terms);
        return Ok(crate::relations::CcsStructure::new_sparse(
            ccs2.matrices.clone(),
            scaled_f,
        )?);
    }
    if ccs2.n == 0 || ccs2.m == 0 {
        return Ok(ccs1.clone());
    }

    let t1 = ccs1.t();
    let n_total = ccs1.n + ccs2.n;
    let m_total = ccs1.m + ccs2.m;
    let t_total = t1 + ccs2.t();

    let stacked_matrices = build_stacked_matrices(ccs1, ccs2, n_total, m_total);

    // SECURE MIXING: f_total = f1 + β*f2 (prevents cancellation attacks)
    let mut mixed_terms = Vec::new();

    // f1 terms: pad to t_total
    for term in ccs1.f.terms() {
        let mut padded_exps = term.exps.clone();
        padded_exps.resize(t_total, 0);
        mixed_terms.push(Term {
            coeff: term.coeff,
            exps: padded_exps,
        });
    }

    // f2 terms: shift by t1 and multiply coefficients by beta
    for term in ccs2.f.terms() {
        let mut shifted_exps = vec![0; t1];
        shifted_exps.extend_from_slice(&term.exps);
        mixed_terms.push(Term {
            coeff: term.coeff * beta, // CRITICAL: multiply by beta
            exps: shifted_exps,
        });
    }

    let mixed_f = SparsePoly::new(t_total, mixed_terms);

    Ok(crate::relations::CcsStructure::new_sparse(stacked_matrices, mixed_f)?)
}

/// Convenience: derive β from transcript digest and call `direct_sum_mixed`.
///
/// This is the recommended way to securely combine CCS structures in production.
/// The mixing coefficient β is derived deterministically from transcript state,
/// ensuring unpredictability while maintaining reproducibility.
///
/// # Security
/// Uses the first 8 bytes of the transcript digest to derive β, preventing
/// adversarial selection of mixing coefficients.
pub fn direct_sum_transcript_mixed<F: Field>(
    ccs1: &crate::relations::CcsStructure<F>,
    ccs2: &crate::relations::CcsStructure<F>,
    transcript_digest: [u8; 32],
) -> Result<crate::relations::CcsStructure<F>, crate::error::CcsError> {
    // Derive β from transcript (use ALL 32 bytes for better entropy).
    // The digest is exactly 32 bytes; each 8-byte slice is guaranteed to
    // convert to [u8; 8], so these unwraps cannot fail.
    let limb0 = u64::from_le_bytes(transcript_digest[0..8].try_into().unwrap());
    let limb1 = u64::from_le_bytes(transcript_digest[8..16].try_into().unwrap());
    let limb2 = u64::from_le_bytes(transcript_digest[16..24].try_into().unwrap());
    let limb3 = u64::from_le_bytes(transcript_digest[24..32].try_into().unwrap());

    // Fold all 32 bytes with prime multipliers for better distribution
    let mut beta = F::from_u64(limb0)
        + F::from_u64(limb1) * F::from_u64(0x9E37_79B9)
        + F::from_u64(limb2) * F::from_u64(0xC2B2_AE35)
        + F::from_u64(limb3) * F::from_u64(0x1656_67B1);

    // Domain separation: prevent reuse of same digest in different contexts
    beta += F::from_u64(0x6E656F); // "neo" tag (ASCII: 0x6e656f)

    // Light nonlinearity: remove purely linear structure (cheap)
    beta = beta * beta;

    // CRITICAL: Ensure β ≠ 0 and β ≠ 1 to prevent cancellation attacks
    if beta == F::ZERO || beta == F::ONE {
        beta += F::from_u64(2);
        if beta == F::ZERO || beta == F::ONE {
            beta = F::from_u64(42);
        }
    }

    direct_sum_mixed(ccs1, ccs2, beta)
}
