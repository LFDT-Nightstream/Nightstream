use p3_goldilocks::Goldilocks as Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{RngCore, CryptoRng};
use crate::types::{PP, Commitment};
use crate::error::{AjtaiError, AjtaiResult};


/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{Rq as RqEl, cf_inv as cf_unmap, cf, D, ETA};
use neo_math::s_action::SAction;

// Compile-time guards: this file's rot_step assumes Φ₈₁ (η=81 ⇒ D=54)
const _: () = assert!(ETA == 81, "rot_step is specialized for η=81 (D=54)");
const _: () = assert!(D == 54, "D must be 54 when η=81");

/// Sample a uniform element from F_q using rejection sampling to avoid bias.
#[inline]
fn sample_uniform_fq<R: RngCore + CryptoRng>(rng: &mut R) -> Fq {
    // Rejection sampling: draw u64; accept if < q; otherwise redraw.
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64; // 2^64 - 2^32 + 1
    loop {
        let x = rng.next_u64();
        if x < Q { return Fq::from_u64(x); }
    }
}

/// Rotation "one-step" for Φ₈₁(X) = X^54 + X^27 + 1
/// 
/// Turns column t into column t+1 in O(d) (no ring multiply).
/// For Φ₈₁, the step is: next[0] = -v_{d-1}, next[27] = v_{26} - v_{d-1},
/// next[k] = v_{k-1} for k ∈ {1,...,d-1}\{27}.
#[inline]
fn rot_step_phi_81(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    // shift: next[k] = cur[k-1] for k>=1; next[0] = 0
    next[0] = Fq::ZERO;
    for k in 1..D { next[k] = cur[k - 1]; }
    // cyclotomic corrections for X^54 ≡ -X^27 - 1
    next[0] -= last;        // -1 * last
    next[27] -= last;       // -X^27 * last
}

/// Optional: if you ever support X^D + 1 rings (AGL/Mersenne), use this fallback
#[allow(dead_code)]
#[inline]
fn rot_step_xd_plus_1(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    next[0] = Fq::ZERO;
    for k in 1..D { next[k] = cur[k - 1]; }
    next[0] -= last; // X^D ≡ -1
}

/// Rotation step dispatcher - compile-time constant for η=81 ⇒ D=54  
/// 
/// **WARNING**: This function is exposed publicly only for integration testing.
/// Do not use in external code - it may change without notice.
/// Prefer the high-level commitment APIs in this crate instead.
#[inline]
#[cfg(any(test, feature = "testing"))]
pub fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    rot_step_impl(cur, next)
}

/// Rotation step dispatcher - internal crate version
#[inline]
#[cfg(not(any(test, feature = "testing")))]
pub(crate) fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    rot_step_impl(cur, next)
}

/// Shared implementation for rotation step
/// This implementation is specialized for η=81 (D=54) as enforced by compile-time assertions.
#[inline]
fn rot_step_impl(cur: &[Fq; D], next: &mut [Fq; D]) {
    // Note: ETA == 81 is guaranteed at compile-time by const assertions at module top
    rot_step_phi_81(cur, next)
}

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions("d parameter must match ring dimension D"));
    }
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            // sample ring element uniformly by sampling d random coefficients in F_q and mapping via cf^{-1}
            let coeffs_vec: Vec<Fq> = (0..neo_math::ring::D)
                .map(|_| sample_uniform_fq(rng))
                .collect();
            let coeffs: [Fq; neo_math::ring::D] = coeffs_vec
                .try_into()
                .map_err(|_| AjtaiError::InvalidDimensions("Failed to create coefficient array"))?;
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP { kappa, m, d, m_rows: rows })
}

// Variable-time optimization removed for security and simplicity

/// OFFICIAL API: Extract Ajtai binding rows for Neo circuit integration.
/// 
/// **TEMPORARY IMPLEMENTATION**: This function returns synthetic binding rows
/// that satisfy the circuit's requirements by construction. Each row `L_i` 
/// satisfies `<L_i, z_digits> = c_coords[i]` for any witness/commitment pair.
/// 
/// **TODO**: Replace with true mathematical extraction from the Ajtai public parameters
/// once the mapping between witness decomposition and commitment coordinates is clarified.
///
/// # Arguments  
/// * `_pp` - Ajtai public parameters (currently unused in synthetic implementation)
/// * `z_len` - Length of z_digits vector 
/// * `num_coords` - Number of coordinate rows to return
///
/// # Returns
/// Authentic rows from the Ajtai matrix, or an error if extraction is not implemented.
///
/// # Security Note
/// Currently returns an error until proper matrix extraction from PP is implemented.
/// This ensures no synthetic rows can accidentally be used in production.
pub fn rows_for_coords(
    pp: &PP<RqEl>, 
    z_len: usize, 
    num_coords: usize
) -> AjtaiResult<Vec<Vec<Fq>>> {
    let d = pp.d;
    let m = pp.m; 
    let kappa = pp.kappa;
    
    // Validate inputs
    if z_len != d * m {
        return Err(AjtaiError::SizeMismatch { 
            expected: d * m, 
            actual: z_len 
        });
    }
    
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa"));
    }
    
    let mut rows = Vec::with_capacity(num_coords);
    
    // Extract the first num_coords rows from the linearized commitment matrix
    for coord_idx in 0..num_coords {
        let mut row = vec![Fq::ZERO; z_len];
        
        // Determine which commitment coordinate this row corresponds to  
        let commit_col = coord_idx / d;  // Which column of commitment (0..kappa)
        let commit_row = coord_idx % d;  // Which row within that column (0..d)
        
        if commit_col >= kappa {
            break; // Don't go beyond available commitment coordinates
        }
        
        // For this commitment coordinate, compute its dependency on z_digits
        // The commitment is computed as: c[commit_col][commit_row] = Σ_j S-action(M[commit_col][j])[commit_row] · Z_col_j
        
        for j in 0..m {
            // Get the ring element M[commit_col][j]
            let a_ij = pp.m_rows[commit_col][j];
            
            // Convert to S-action (d×d field linear map)
            let s_action = SAction::from_ring(a_ij);
            
            // For each input digit position in column j
            for input_row in 0..d {
                let input_idx = j * d + input_row;  // Column-major indexing in z_digits
                
                // Get the S-action coefficient that multiplies z_digits[input_idx] 
                // to contribute to commitment coordinate [commit_col][commit_row]
                // Apply the S-action to a unit vector to extract the coefficient
                let mut unit_vec = [Fq::ZERO; D];
                unit_vec[input_row] = Fq::ONE;
                let result = s_action.apply_vec(&unit_vec);
                let coeff = result[commit_row];
                row[input_idx] = coeff;
            }
        }
        
        rows.push(row);
    }
    
    Ok(rows)
}

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
/// Uses constant-time dense computation for all inputs (audit-ready).
/// Returns error if Z dimensions don't match expected d×m.
#[allow(non_snake_case)]
pub fn try_commit(pp: &PP<RqEl>, Z: &[Fq]) -> AjtaiResult<Commitment> {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d; let m = pp.m;
    if Z.len() != d*m {
        return Err(AjtaiError::SizeMismatch { 
            expected: d*m, 
            actual: Z.len() 
        });
    }
    
    // Route to the audited constant-time variant by default for security
    Ok(commit_masked_ct(pp, Z))
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    try_commit(pp, Z).expect("commit: Z dimensions must match d×m")
}

/// Commit implementation via SAction::apply_vec.
/// 
/// **Constant-time depends on SAction implementation** - this function has fixed loops
/// but relies on SAction::apply_vec being constant-time. For guaranteed constant-time
/// behavior, prefer commit_masked_ct() or commit_precomp_ct() which are constant-time
/// by construction.
#[cfg(any(test, feature = "testing"))]
#[allow(non_snake_case, dead_code)]
fn commit_via_saction(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;

    // Pre-extract columns of Z (digits per column)
    let cols: Vec<&[Fq]> = (0..m).map(|j| &Z[j*d .. (j+1)*d]).collect();

    let mut c = Commitment::zeros(d, kappa);

    // For each Ajtai row i, compute Σ_j S-action(a_ij) · Z_col_j into c_col_i.
    for i in 0..kappa {
        let acc = c.col_mut(i);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            let a_ij = &pp.m_rows[i][j];           // R_q element
            let s_action = SAction::from_ring(*a_ij);  // Create S-action from ring element
            let v: [Fq; neo_math::ring::D] = cols[j].try_into().expect("column length should be d"); // digits for column j

            // Apply S-action to the coefficient vector
            let result = s_action.apply_vec(&v);
            
            // Constant-time accumulation (no secret-dependent branching).
            for (a, &r) in acc.iter_mut().zip(&result) {
                *a += r;
            }
        }
    }
    c
}

/// MUST: Verify opening by recomputing commitment (binding implies uniqueness).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_open(pp: &PP<RqEl>, c: &Commitment, Z: &[Fq]) -> bool {
    &commit(pp, Z) == c
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    if k != Z_is.len() { return false; }
    // Check shapes
    for ci in c_is { if ci.d != c.d || ci.kappa != c.kappa { return false; } }
    // Recompose commitment
    let mut acc = Commitment::zeros(c.d, c.kappa);
    let mut pow = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for (a, &x) in acc.data.iter_mut().zip(&c_is[i].data) { *a += x * pow; }
        pow *= b_f;
    }
    if &acc != c { return false; }
    // Recompose Z and check commit again
    let d = pp.d; let m = pp.m;
    let mut Z = vec![Fq::ZERO; d*m];
    let mut pow = Fq::ONE;
    for Zi in Z_is {
        if Zi.len() != d*m { return false; }
        for (a, &x) in Z.iter_mut().zip(Zi) { *a += x * pow; }
        pow *= b_f;
    }
    &commit(pp, &Z) == c
}

/// MUST: S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
/// Since we don't have direct access to SMatrix, we use SAction to operate on the commitment data.
pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d; let kappa = c.kappa;
    let mut out = Commitment::zeros(d, kappa);
    let s_action = SAction::from_ring(*rho_ring);
    
    for col in 0..kappa {
        let src: [Fq; neo_math::ring::D] = c.col(col).try_into().expect("column length should be d");
        let dst_result = s_action.apply_vec(&src);
        let dst = out.col_mut(col);
        dst.copy_from_slice(&dst_result);
    }
    out
}

/// Reference implementation for differential testing against the optimized commit
/// 
/// Implements the specification directly: c = cf(M · cf^{-1}(Z))
/// This verifies the fundamental S-action isomorphism cf(a·b) = rot(a)·cf(b) 
/// at the commitment level - exactly the algebra the construction relies on.
/// 
/// ⚠️  FOR TESTING ONLY - NOT CONSTANT TIME ⚠️
/// Available for unit tests and with the 'testing' feature for integration tests
#[cfg(any(test, feature = "testing"))]
#[allow(non_snake_case)]
pub fn commit_spec(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m;
    let mut c = Commitment::zeros(d, pp.kappa);
    
    for i in 0..pp.kappa {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let s = SAction::from_ring(pp.m_rows[i][j]);
            let v: [Fq; neo_math::ring::D] = Z[j*d..(j+1)*d].try_into().unwrap();
            let w = s.apply_vec(&v);
            for (a, &x) in acc_i.iter_mut().zip(&w) { *a += x; }
        }
    }
    c
}

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> AjtaiResult<Commitment> {
    if rhos.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    if rhos.len() != cs.len() {
        return Err(AjtaiError::SizeMismatch { 
            expected: rhos.len(), 
            actual: cs.len() 
        });
    }
    if cs.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        let term = s_mul(rho, c);
        acc.add_inplace(&term);
    }
    Ok(acc)
}

/// Constant-time masked columns accumulation (streaming).
///
/// c = cf(M · cf^{-1}(Z)) computed as:
///   for i in 0..kappa, j in 0..m:
///     col <- cf(a_ij)       // column 0 of rot(a_ij)
///     for t in 0..d-1:
///       acc += Z[j*d + t] * col
///       col <- next column via rot_step()
///
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
/// 
/// This implements the identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
#[allow(non_snake_case)]
pub fn commit_masked_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    debug_assert_eq!(d, D);
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    // For each Ajtai row i and message column j
    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            // Start from col_0 = cf(a_ij)
            let mut col = cf(pp.m_rows[i][j]);
            let mut nxt = [Fq::ZERO; D];

            // Loop over all base-digits t (constant-time)
            let base = j * d;
            for t in 0..d {
                let mask = Z[base + t];        // any Fq digit (0, ±1, small, or general)
                // acc += mask * col   (branch-free masked add)
                for r in 0..d {
                    // single FMA-like op on the field
                    acc_i[r] += col[r] * mask;
                }
                // Advance to the next rotation column in O(d)
                rot_step(&col, &mut nxt);
                core::mem::swap(&mut col, &mut nxt); // Cheaper than copying [Fq; D]
            }
        }
    }
    C
}

/// Fill `cols` with the d rotation columns of rot(a): cols[t] = cf(a * X^t).
#[inline]
fn precompute_rot_columns(a: RqEl, cols: &mut [[Fq; D]]) {
    let mut col = cf(a);
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        cols[t] = col;
        rot_step(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt); // Avoid copying 54 elements
    }
}

/// Constant-time commit using precomputed rotation columns per (i,j).
///
/// Space/time trade: uses a stack-allocated `[[Fq; D]; D]` scratch per (i,j) to
/// remove per-step rot_step(), keeping the same constant-time masked adds.
/// 
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)  
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
/// 
/// This implements the same identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
/// but precomputes all rotation columns once per (i,j) pair for better cache locality.
#[allow(non_snake_case)]
pub fn commit_precomp_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    debug_assert_eq!(d, D);
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    // Heap-allocated scratch for columns of rot(a_ij) to avoid stack overflow
    // 54×54×8 ≈ 23 KiB per allocation, hoisted outside inner loop for reuse
    let mut cols = vec![[Fq::ZERO; D]; D].into_boxed_slice();

    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            precompute_rot_columns(pp.m_rows[i][j], &mut cols);
            let base = j * d;
            // Constant schedule: always loop over all t
            for t in 0..d {
                let mask = Z[base + t];
                let col_t = &cols[t];
                for r in 0..d {
                    acc_i[r] += col_t[r] * mask;
                }
            }
        }
    }
    C
}

/// Linear opening proof for Ajtai commitments
/// Proves that y_j = Z * v_j for given linear functionals v_j without revealing Z
#[cfg(any(test, feature = "testing"))]
#[derive(Debug, Clone)]
pub struct LinearOpeningProof {
    /// The opened values y_j = Z * v_j for each linear functional
    pub opened_values: Vec<Vec<neo_math::K>>,
    /// Placeholder for additional proof data (future extension)
    pub proof_data: Vec<u8>,
}

/// Generate a linear opening proof for multiple linear functionals
/// 
/// Given commitment c = L(Z) and linear functionals v_j, proves that y_j = Z * v_j
/// 
/// SECURITY NOTE: Since Ajtai is S-homomorphic and linear, this is a deterministic
/// verification that doesn't require zero-knowledge. The "proof" is just the claimed
/// values y_j, and verification checks consistency with the commitment via linearity.
/// 
/// # Arguments
/// * `pp` - Public parameters
/// * `c` - Commitment to witness Z
/// * `Z` - The witness matrix (prover-side only)
/// * `v_slices` - Linear functionals v_j ∈ F^m (each promoted to K inside)
/// 
/// # Returns
/// * Tuple of (opened values y_j, proof)
#[cfg(any(test, feature = "testing"))]
#[allow(non_snake_case)]
pub fn open_linear(
    _pp: &PP<RqEl>,
    _c: &Commitment,
    Z: &[Fq],  // flattened witness (d*m elements)
    v_slices: &[Vec<Fq>],  // each v_j ∈ F^m
) -> (Vec<Vec<neo_math::K>>, LinearOpeningProof) {
    let d = neo_math::D;
    let m = v_slices.first().map(|v| v.len()).unwrap_or(0);
    
    // Debug assertion: Z must be d×m (dimension check)
    debug_assert_eq!(Z.len(), d * m, "open_linear: Z must be d×m ({}×{})", d, m);
    
    let mut opened_values = Vec::with_capacity(v_slices.len());
    
    for v_j in v_slices {
        // Debug assertion: each v_j must have length m
        debug_assert_eq!(v_j.len(), m, "open_linear: v_j must have length m ({})", m);
        
        // Compute y_j = Z * v_j where Z is d×m matrix and v_j is m-vector
        // Result is d-dimensional vector in extension field K
        let mut y_j = Vec::with_capacity(d);
        
        for row in 0..d {
            let mut sum = neo_math::K::ZERO;
            for col in 0..m {
                // Z is d×m in **column-major** order: (col * d + row)
                let z_element = Z[col * d + row];
                let v_element = v_j[col];
                sum += neo_math::K::from(z_element) * neo_math::K::from(v_element);
            }
            y_j.push(sum);
        }
        
        opened_values.push(y_j);
    }
    
    let proof = LinearOpeningProof {
        opened_values: opened_values.clone(),
        proof_data: Vec::new(),  // Deterministic verification needs no extra proof
    };
    
    (opened_values, proof)
}

// NOTE: verify_linear is intentionally NOT PROVIDED in the Ajtai commitment layer.
//
// ### Why no verify_linear?
// Proving a generic linear evaluation `y = Z · v` from **only one** Ajtai commitment
// `c = L(Z)` is not possible in general without additional openings/structure:
// it requires reweighting per‑column contributions inside `Σ_j a_{ij} · ẑ_j`,
// which are **lost** in the compressed commitment.
//
// ### Where to find linear verification:
// In Neo, linear evaluation ties are enforced inside the FS transcript via:
// - Π_CCS (sum‑check over extension field)
// - Π_RLC (random linear combination - see neo_fold::verify_linear)  
// - Π_DEC recomposition checks
//
// Use neo_fold::verify_linear for Π_RLC verification and verify_split_open for
// recomposition checks. The commitment layer only provides S-homomorphic binding.
