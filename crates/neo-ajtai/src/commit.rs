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
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
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
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
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
        return Err(AjtaiError::InvalidDimensions("d parameter must match ring dimension D".to_string()));
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
                .map_err(|_| AjtaiError::InvalidDimensions("Failed to create coefficient array".to_string()))?;
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP { kappa, m, d, m_rows: rows })
}

// Variable-time optimization removed for security and simplicity

///
/// **PRODUCTION IMPLEMENTATION** — true extraction from Ajtai public parameters.
///
/// Builds the linear map `L ∈ F_q^{(d·κ) × (d·m)}` such that, for **column‑major**
/// encodings of `Z ∈ F_q^{d×m}` and commitment `c ∈ F_q^{d×κ}`:
///
///   `vec(c) = L · vec(Z)`  where  `c = cf(M · cf^{-1}(Z))`.
///
/// Using the identity `cf(a·b) = rot(a) · cf(b)`, the row corresponding to
/// `(commit_col=i, commit_row=r)` has blocks
///
///   `L[(i,r), (j, 0..d-1)] = col_t(rot(a_ij))[r]  for t=0..d-1`,
///
/// i.e. the `t`‑th rotation column of `a_ij` evaluated at row `r`. This matches the
/// prover's constant‑time path (`commit_masked_ct`) exactly.
///
/// **Indexing conventions**
/// - `z_len` must be `d·m`. The caller pads rows externally if the circuit padded `z_digits`
///   to a power of two (we intentionally keep the extractor strict).
/// - `num_coords` ≤ `d·κ`; typical usage requests all `d·κ` rows.
///
/// **Performance**
/// - Deduplicates equal `a_ij` by hashing `cf(a_ij)` (O(1) average).
/// - Precomputes all rotation columns per unique ring element via `rot_step`.
/// - Builds rows in parallel with Rayon.
///
/// **Security**
/// - Strict runtime dimension checks (prevents binding bugs).
/// - No secret‑dependent branching/memory access; depends only on public `pp`.
///
pub fn rows_for_coords(
    pp: &PP<RqEl>, 
    z_len: usize, 
    num_coords: usize
) -> AjtaiResult<Vec<Vec<Fq>>> {
    use rayon::prelude::*;

    // SECURITY: Gated logging to prevent CWE-532 (sensitive info leakage)
    #[cfg(feature = "neo-logs")]
    use tracing::{debug, info};
    
    #[cfg(not(feature = "neo-logs"))]
    macro_rules! debug { ($($tt:tt)*) => {} }
    #[cfg(not(feature = "neo-logs"))]
    macro_rules! info  { ($($tt:tt)*) => {} }
    
    let d = pp.d;
    let m = pp.m; 
    let kappa = pp.kappa;
    
    // CRITICAL SECURITY: Ensure compile-time ring dimension matches runtime PP dimension
    // This MUST be a runtime check to prevent binding bugs in release builds
    if D != d {
        return Err(AjtaiError::InvalidInput(
            format!("Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})", D, d)
        ));
    }
    
    // CRITICAL: Validate dimensions to prevent binding bugs (B4).
    // Keep extractor strict; the caller can pad rows externally if z_digits was power-of-two padded.
    if z_len != d * m {
        return Err(AjtaiError::InvalidInput("z_len must equal d*m".to_string()));
    }
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa".to_string()));
    }
    
    info!("Starting Ajtai binding rows computation (true extraction)...");
    let total_start = std::time::Instant::now();

    // 🚀 OPTIMIZATION 1: Deduplicate identical ring elements
    debug!("Deduplicating {} ring elements...", kappa * m);
    let cache_start = std::time::Instant::now();
    
    // Use HashMap with cf() canonical form for O(1) deduplication instead of O(n²) Vec::position
    use std::collections::HashMap;
    
    let mut unique_map = HashMap::new();
    let mut all_elements: Vec<RqEl> = Vec::new();
    let mut element_indices = vec![vec![0; m]; kappa];
    
    for commit_col in 0..kappa {
        for j in 0..m {
            let a_ij = pp.m_rows[commit_col][j];
            
            // Use canonical form of ring element as key (arrays implement Hash+Eq)
            let canonical_key = cf(a_ij);
            
            // O(1) average lookup instead of O(n) - HUGE performance win!
            let idx = *unique_map.entry(canonical_key).or_insert_with(|| {
                let new_idx = all_elements.len();
                all_elements.push(a_ij);
                new_idx
            });
            
            element_indices[commit_col][j] = idx;
        }
    }
    
    debug!("Found {} unique ring elements (vs {} total)", all_elements.len(), kappa * m);
    
    // 🚀 OPTIMIZATION 2: Precompute all rotation columns for each unique element.
    // cols[t] = cf(a * X^t) for t ∈ [0..d-1]; computed via rot_step in O(d^2) once.
    let precomp_start = std::time::Instant::now();
    let mut rot_columns: Vec<Box<[[Fq; D]]>> = Vec::with_capacity(all_elements.len());
    rot_columns.par_extend(
        all_elements.par_iter().map(|&a_ij| {
            let mut cols = vec![[Fq::ZERO; D]; D].into_boxed_slice();
            precompute_rot_columns(a_ij, &mut cols);
            cols
        })
    );
    let _precomp_time = precomp_start.elapsed();
    
    let _cache_time = cache_start.elapsed();
    debug!("Precomputation completed: {:.2}ms", _precomp_time.as_secs_f64() * 1000.0);

    // 🚀 OPTIMIZATION 3: Parallel row computation
    debug!("Computing {} rows in parallel across {} threads...", 
             num_coords, rayon::current_num_threads());
    let parallel_start = std::time::Instant::now();
    
    let rows: Vec<Vec<Fq>> = (0..num_coords)
        .into_par_iter()
        .map(|coord_idx| {
            let mut row = vec![Fq::ZERO; z_len];
            
            // Determine which commitment coordinate this row corresponds to  
            let commit_col = coord_idx / d;  // Which column of commitment (0..kappa)
            let commit_row = coord_idx % d;  // Which row within that column (0..d)
            
            // SECURITY: This should be unreachable with proper num_coords validation
            debug_assert!(commit_col < kappa, 
                "coord_idx {} out of range: commit_col {} >= kappa {}", 
                coord_idx, commit_col, kappa);
            if commit_col >= kappa {
                return row; // Defensive fallback for release builds
            }
            
            // 🚀 OPTIMIZATION 4: Vectorized coefficient extraction from precomputed rot columns
            for j in 0..m {
                let element_idx = element_indices[commit_col][j];
                let cols = &rot_columns[element_idx];
                
                // Extract entire column of coefficients at once
                let base_idx = j * d;
                for input_row in 0..d {
                    let input_idx = base_idx + input_row;      // column-major in z_digits
                    row[input_idx] = cols[input_row][commit_row];
                }
            }
            
            row
        })
        .collect();
    
    let _parallel_time = parallel_start.elapsed();
    let _total_time = total_start.elapsed();
    
    debug!("Parallel computation completed: {:.2}ms", _parallel_time.as_secs_f64() * 1000.0);
    info!("Total optimized time: {:.2}ms (vs ~{:.0}s unoptimized)", 
             _total_time.as_secs_f64() * 1000.0, 
             (kappa * m * num_coords * d) as f64 / 1_000_000.0); // Rough estimate of old time
    
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
    
    // 🚀 PERFORMANCE OPTIMIZATION: Use precomputed rotations for large m
    // For small m, masked CT is faster due to lower setup cost
    // For large m, precomputed CT amortizes the rotation computation cost
    const PRECOMP_THRESHOLD: usize = 256; // Threshold tuned for D=54: precomp pays off when m*D > 16k
    
    if m >= PRECOMP_THRESHOLD {
        Ok(commit_precomp_ct(pp, Z))
    } else {
        Ok(commit_masked_ct(pp, Z))
    }
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
    
    // CRITICAL SECURITY: Runtime dimension checks to prevent binding bugs
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
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
    
    // CRITICAL SECURITY: Runtime dimension checks to prevent binding bugs
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
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

/// Compute a single Ajtai binding row on-demand from PP
/// 
/// This is the streaming version of `rows_for_coords` that computes only one row
/// to avoid materializing the entire row matrix in memory.
/// 
/// # Arguments
/// * `pp` - Ajtai public parameters
/// * `coord_idx` - Index of the coordinate (row) to compute (0..num_coords)
/// * `z_len` - Length of the witness vector (must equal d*m)
/// * `num_coords` - Total number of coordinates (for validation, must be <= d*kappa)
/// 
/// # Returns
/// A single row vector of length `z_len` such that `<row, z_digits> = c_coords[coord_idx]`

/// Compute an aggregated Ajtai row for RLC binding: G = Σ r_i * L_i
/// This enables a single linear constraint that binds c_step to the witness
/// via: ⟨G, z⟩ = Σ r_i * c_step[i], avoiding the need for d*κ constraints.
/// 
/// # Arguments
/// * `pp` - Ajtai public parameters
/// * `r_coeffs` - Random coefficients for linear combination (length = num_coords)
/// * `z_len` - Length of the witness vector z
/// * `num_coords` - Number of commitment coordinates (d * κ)
/// 
/// # Returns
/// Aggregated row G where G[j] = Σ r_i * L_i[j] for all coordinates i
pub fn compute_aggregated_ajtai_row(
    pp: &PP<RqEl>,
    r_coeffs: &[Fq],
    z_len: usize,
    num_coords: usize,
) -> AjtaiResult<Vec<Fq>> {
    if r_coeffs.len() != num_coords {
        return Err(AjtaiError::InvalidDimensions(format!(
            "r_coeffs length {} must equal num_coords {}", 
            r_coeffs.len(), num_coords
        )));
    }
    
    let mut aggregated_row = vec![Fq::ZERO; z_len];
    
    // For each coordinate i, compute r_i * L_i and add to aggregated_row
    for (coord_idx, &r_i) in r_coeffs.iter().enumerate() {
        let row_i = compute_single_ajtai_row(pp, coord_idx, z_len, num_coords)?;
        
        // Add r_i * L_i to the aggregated row
        for (j, &l_ij) in row_i.iter().enumerate() {
            aggregated_row[j] += r_i * l_ij;
        }
    }
    
    Ok(aggregated_row)
}

pub fn compute_single_ajtai_row(
    pp: &PP<RqEl>,
    coord_idx: usize,
    z_len: usize,
    num_coords: usize,
) -> AjtaiResult<Vec<Fq>> {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;
    
    // Validation (same as rows_for_coords)
    if D != d {
        return Err(AjtaiError::InvalidInput(
            format!("Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})", D, d)
        ));
    }
    if z_len != d * m {
        return Err(AjtaiError::InvalidInput("z_len must equal d*m".to_string()));
    }
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa".to_string()));
    }
    if coord_idx >= num_coords {
        return Err(AjtaiError::InvalidInput("coord_idx out of range".to_string()));
    }
    
    // Determine which commitment coordinate this row corresponds to  
    let commit_col = coord_idx / d;  // Which column of commitment (0..kappa)
    let commit_row = coord_idx % d;  // Which row within that column (0..d)
    
    if commit_col >= kappa {
        return Err(AjtaiError::InvalidInput("coord_idx out of range w.r.t. kappa".to_string()));
    }
    
    let mut row = vec![Fq::ZERO; z_len];
    
    // 🚀 STREAMING OPTIMIZATION: O(m·D) using proper cyclotomic rotation
    // Use the same rot_step logic as the original, but compute on-demand
    for j in 0..m {
        let a_ij = pp.m_rows[commit_col][j]; // Get ring element a_{i,j}
        
        // Start with cf(a_ij) and rotate through positions
        let mut col = cf(a_ij); // [Fq; D] - coefficient representation  
        let mut nxt = [Fq::ZERO; D];
        
        // Fill row positions for this j, rotating col for each t
        for t in 0..d {
            row[j * d + t] = col[commit_row];
            // Rotate to next position: col := rot_step(col)
            rot_step(&col, &mut nxt);
            core::mem::swap(&mut col, &mut nxt);
        }
    }
    
    Ok(row)
}
