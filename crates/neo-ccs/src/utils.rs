use p3_field::Field;

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

/// Compute the direct sum (block diagonal composition) of two CCS structures.
/// This creates a new CCS that enforces both systems independently by stacking
/// the constraint matrices in block-diagonal form and concatenating the witness vectors.
///
/// Used for IVC where we want to stack step CCS with embedded verifier CCS.
pub fn direct_sum<F: Field>(
    ccs1: &crate::relations::CcsStructure<F>,
    ccs2: &crate::relations::CcsStructure<F>,
) -> Result<crate::relations::CcsStructure<F>, crate::error::CcsError> {
    use crate::{
        matrix::Mat,
        poly::{SparsePoly, Term},
    };

    // If one is empty, return the other
    if ccs1.n == 0 || ccs1.m == 0 {
        return Ok(ccs2.clone());
    }
    if ccs2.n == 0 || ccs2.m == 0 {
        return Ok(ccs1.clone());
    }

    let n1 = ccs1.n;
    let m1 = ccs1.m;
    let n2 = ccs2.n;
    let m2 = ccs2.m;
    let t1 = ccs1.t();
    let t2 = ccs2.t();

    // Result dimensions
    let n_total = n1 + n2;
    let m_total = m1 + m2;
    let t_total = t1 + t2;

    // Build block-diagonal matrices
    let mut stacked_matrices = Vec::new();

    // First t1 matrices from ccs1
    for j in 0..t1 {
        let mut data = vec![F::ZERO; n_total * m_total];

        // Copy ccs1 matrix to top-left block
        for r in 0..n1 {
            for c in 0..m1 {
                data[r * m_total + c] = ccs1.matrices[j][(r, c)];
            }
        }

        stacked_matrices.push(Mat::from_row_major(n_total, m_total, data));
    }

    // Next t2 matrices from ccs2
    for j in 0..t2 {
        let mut data = vec![F::ZERO; n_total * m_total];

        // Copy ccs2 matrix to bottom-right block
        for r in 0..n2 {
            for c in 0..m2 {
                data[(n1 + r) * m_total + (m1 + c)] = ccs2.matrices[j][(r, c)];
            }
        }

        stacked_matrices.push(Mat::from_row_major(n_total, m_total, data));
    }

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

    Ok(crate::relations::CcsStructure::new(stacked_matrices, combined_f)?)
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
    use crate::{
        matrix::Mat,
        poly::{SparsePoly, Term},
    };

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
        return Ok(crate::relations::CcsStructure::new(ccs2.matrices.clone(), scaled_f)?);
    }
    if ccs2.n == 0 || ccs2.m == 0 {
        return Ok(ccs1.clone());
    }

    let n1 = ccs1.n;
    let m1 = ccs1.m;
    let n2 = ccs2.n;
    let m2 = ccs2.m;
    let t1 = ccs1.t();
    let t2 = ccs2.t();

    let n_total = n1 + n2;
    let m_total = m1 + m2;
    let t_total = t1 + t2;

    // Build block-diagonal matrices (same as direct_sum)
    let mut stacked_matrices = Vec::new();

    for j in 0..t1 {
        let mut data = vec![F::ZERO; n_total * m_total];
        for r in 0..n1 {
            for c in 0..m1 {
                data[r * m_total + c] = ccs1.matrices[j][(r, c)];
            }
        }
        stacked_matrices.push(Mat::from_row_major(n_total, m_total, data));
    }

    for j in 0..t2 {
        let mut data = vec![F::ZERO; n_total * m_total];
        for r in 0..n2 {
            for c in 0..m2 {
                data[(n1 + r) * m_total + (m1 + c)] = ccs2.matrices[j][(r, c)];
            }
        }
        stacked_matrices.push(Mat::from_row_major(n_total, m_total, data));
    }

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

    Ok(crate::relations::CcsStructure::new(stacked_matrices, mixed_f)?)
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
    // Derive β from transcript (use ALL 32 bytes for better entropy)
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
