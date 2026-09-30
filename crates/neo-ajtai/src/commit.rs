use crate::error::{AjtaiError, AjtaiResult};
use crate::types::{Commitment, PP};
use neo_ccs::Mat;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Fq;
use rand::{CryptoRng, RngCore};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{cf, cf_inv as cf_unmap, Rq as RqEl, D, ETA};

// Compile-time guards: this file's rot_step assumes Φ₈₁ (η=81 ⇒ D=54)
const _: () = assert!(ETA == 81, "rot_step is specialized for η=81 (D=54)");
const _: () = assert!(D == 54, "D must be 54 when η=81");

/// Sample a uniform element from F_q using rejection sampling to avoid bias.
#[inline]
fn sample_uniform_fq<R: RngCore + CryptoRng>(rng: &mut R) -> Fq {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64; // 2^64 - 2^32 + 1
    loop {
        let x = rng.next_u64();
        if x < Q {
            return Fq::from_u64(x);
        }
    }
}

/// Sample a uniform element from R_q by sampling D uniform coefficients in F_q and mapping with `cf^{-1}`.
#[doc(hidden)]
#[inline]
pub fn sample_uniform_rq<R: RngCore + CryptoRng>(rng: &mut R) -> RqEl {
    let coeffs = sample_uniform_rq_coeffs(rng);
    cf_unmap(coeffs)
}

/// Sample a uniform coefficient vector in F_q^D that corresponds to a uniform R_q element.
#[inline]
pub(crate) fn sample_uniform_rq_coeffs<R: RngCore + CryptoRng>(rng: &mut R) -> [Fq; D] {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64;
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    core::array::from_fn(|idx| {
        let start = idx * 8;
        let x = u64::from_le_bytes(bytes[start..start + 8].try_into().expect("8-byte chunk"));
        if x < Q {
            Fq::from_u64(x)
        } else {
            sample_uniform_fq(rng)
        }
    })
}

/// Rotation "one-step" for Φ₈₁(X) = X^54 + X^27 + 1
///
/// Turns column t into column t+1 in O(d) (no ring multiply).
/// For Φ₈₁, the step is: next[0] = -v_{d-1}, next[27] = v_{26} - v_{d-1},
/// next[k] = v_{k-1} for k ∈ {1,...,d-1}\{27}.
#[inline(always)]
fn rot_step_phi_81(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    // shift: next[k] = cur[k-1] for k>=1; next[0] = 0
    next[0] = Fq::ZERO;
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
    // cyclotomic corrections for X^54 ≡ -X^27 - 1
    next[0] -= last; // -1 * last
    next[27] -= last; // -X^27 * last
}

/// Rotation step for internal use by commit implementations.
///
/// This implementation is specialized for η=81 (D=54) as enforced by compile-time assertions.
/// Re-exported publicly only when the `testing` feature is enabled (see lib.rs).
#[inline(always)]
pub fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    rot_step_phi_81(cur, next)
}

#[inline(always)]
fn acc_add_inplace(acc: &mut [Fq; D], col: &[Fq; D]) {
    let mut r = 0usize;
    while r + 3 < D {
        acc[r] += col[r];
        acc[r + 1] += col[r + 1];
        acc[r + 2] += col[r + 2];
        acc[r + 3] += col[r + 3];
        r += 4;
    }
    while r < D {
        acc[r] += col[r];
        r += 1;
    }
}

#[inline(always)]
fn acc_mul_add_inplace(acc: &mut [Fq; D], col: &[Fq; D], scalar: Fq) {
    // Fast paths for the common balanced-digit case (b ∈ {2,3} ⇒ scalar ∈ {-1,0,1}).
    //
    // NOTE: This is intentionally variable-time w.r.t. `scalar`. Its only caller,
    // `s_mul_add_from_rot_col`, takes `scalar` from a public commitment.
    if scalar == Fq::ZERO {
        return;
    }
    if scalar == Fq::ONE {
        acc_add_inplace(acc, col);
        return;
    }
    let neg_one = Fq::ZERO - Fq::ONE;
    if scalar == neg_one {
        let mut r = 0usize;
        while r + 3 < D {
            acc[r] -= col[r];
            acc[r + 1] -= col[r + 1];
            acc[r + 2] -= col[r + 2];
            acc[r + 3] -= col[r + 3];
            r += 4;
        }
        while r < D {
            acc[r] -= col[r];
            r += 1;
        }
        return;
    }

    // Fallback: generic scalar multiply-add.
    // Unrolled to encourage LLVM auto-vectorization on platforms that support it.
    let mut r = 0usize;
    while r + 3 < D {
        acc[r] += col[r] * scalar;
        acc[r + 1] += col[r + 1] * scalar;
        acc[r + 2] += col[r + 2] * scalar;
        acc[r + 3] += col[r + 3] * scalar;
        r += 4;
    }
    while r < D {
        acc[r] += col[r] * scalar;
        r += 1;
    }
}

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions(
            "d parameter must match ring dimension D".to_string(),
        ));
    }
    if kappa == 0 || m == 0 {
        return Err(AjtaiError::InvalidDimensions(
            "kappa and m must both be nonzero".to_string(),
        ));
    }
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(rng));
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP {
        kappa,
        m,
        d,
        m_rows: rows,
    })
}

/// Commit `Z` with the Ajtai map `cf(M · cf⁻¹(Z))`.
#[allow(non_snake_case)]
#[inline]
pub fn try_commit(pp: &PP<RqEl>, Z: &[Fq]) -> AjtaiResult<Commitment> {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d;
    let m = pp.m;
    if Z.len() != d * m {
        return Err(AjtaiError::SizeMismatch {
            expected: d * m,
            actual: Z.len(),
        });
    }

    Ok(commit_precomp_ct(pp, Z))
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    try_commit(pp, Z).expect("commit: Z dimensions must match d×m")
}

/// Commit to a **row-major** `Mat<Fq>` without materializing a full column-major buffer.
///
/// This is equivalent to:
/// 1) transposing `Z` (row-major) into a column-major `Vec<Fq>`, then
/// 2) calling [`commit`].
///
/// It exists to avoid a multi-hundred-MB temporary allocation in prover hot paths.
#[allow(non_snake_case)]
pub fn try_commit_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> AjtaiResult<Commitment> {
    let d = pp.d;
    let m = pp.m;
    if Z.rows() != d || Z.cols() != m {
        return Err(AjtaiError::InvalidDimensions(format!(
            "Z must have shape d×m = {}×{} (got {}×{})",
            d,
            m,
            Z.rows(),
            Z.cols()
        )));
    }

    Ok(if Z.is_packed_signed_unit() || Z.is_virtual_constant() {
        commit_masked_ct_row_major(pp, Z)
    } else {
        commit_precomp_ct_row_major(pp, Z)
    })
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    try_commit_row_major(pp, Z).expect("commit_row_major: Z dimensions must match d×m")
}

/// MUST: Verify opening by recomputing commitment (binding implies uniqueness).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_open(pp: &PP<RqEl>, c: &Commitment, Z: &[Fq]) -> bool {
    try_commit(pp, Z).is_ok_and(|opened| &opened == c)
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    let Some(commitment_len) = pp.d.checked_mul(pp.kappa) else {
        return false;
    };
    if k == 0 || k != Z_is.len() || b < 2 || c.d != pp.d || c.kappa != pp.kappa || c.data.len() != commitment_len {
        return false;
    }
    // Check shapes
    for (ci, zi) in c_is.iter().zip(Z_is) {
        if ci.d != c.d || ci.kappa != c.kappa || !verify_open(pp, ci, zi) {
            return false;
        }
    }
    // Recompose commitment
    let mut acc = Commitment::zeros(c.d, c.kappa);
    let mut pow = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for (a, &x) in acc.data.iter_mut().zip(&c_is[i].data) {
            *a += x * pow;
        }
        pow *= b_f;
    }
    if &acc != c {
        return false;
    }
    // Recompose Z and check commit again
    let d = pp.d;
    let m = pp.m;
    let mut Z = vec![Fq::ZERO; d * m];
    let mut pow = Fq::ONE;
    for Zi in Z_is {
        if Zi.len() != d * m {
            return false;
        }
        for (a, &x) in Z.iter_mut().zip(Zi) {
            *a += x * pow;
        }
        pow *= b_f;
    }
    &commit(pp, &Z) == c
}

/// S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
pub fn s_mul_add_from_rot_col(acc: &mut Commitment, first_rot_col: &[Fq; D], c: &Commitment) {
    let d = c.d;
    let kappa = c.kappa;
    debug_assert_eq!(d, D, "Ajtai commitment columns must have D rows");
    debug_assert_eq!(acc.d, d);
    debug_assert_eq!(acc.kappa, kappa);

    let (acc_cols, acc_rem) = acc.data.as_chunks_mut::<D>();
    let (c_cols, c_rem) = c.data.as_chunks::<D>();
    debug_assert!(acc_rem.is_empty(), "accumulator commitment columns must be D-wide");
    debug_assert!(c_rem.is_empty(), "input commitment columns must be D-wide");
    debug_assert_eq!(acc_cols.len(), kappa);
    debug_assert_eq!(c_cols.len(), kappa);

    let mut rot_col = *first_rot_col;
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        for (dst, src) in acc_cols.iter_mut().zip(c_cols.iter()) {
            acc_mul_add_inplace(dst, &rot_col, src[t]);
        }
        rot_step(&rot_col, &mut nxt);
        rot_col = nxt;
    }
}

/// S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
pub fn s_mul_add(acc: &mut Commitment, rho_ring: &RqEl, c: &Commitment) {
    let rot_col = cf(*rho_ring);
    s_mul_add_from_rot_col(acc, &rot_col, c);
}

/// Add a field-scalar multiple of a commitment into an accumulator.
///
/// This is the constant-ring-element `S`-action specialized to the common verifier-side case
/// where the multiplier is just a base-field scalar instead of a full rotation element.
pub fn scale_commitment_add_inplace(acc: &mut Commitment, scalar: Fq, c: &Commitment) {
    debug_assert_eq!(acc.d, c.d);
    debug_assert_eq!(acc.kappa, c.kappa);

    if scalar == Fq::ZERO {
        return;
    }
    if scalar == Fq::ONE {
        acc.add_inplace(c);
        return;
    }
    let neg_one = Fq::ZERO - Fq::ONE;
    if scalar == neg_one {
        for (dst, src) in acc.data.iter_mut().zip(c.data.iter()) {
            *dst -= *src;
        }
        return;
    }

    for (dst, src) in acc.data.iter_mut().zip(c.data.iter()) {
        *dst += *src * scalar;
    }
}

pub fn scale_commitment(scalar: Fq, c: &Commitment) -> Commitment {
    let mut out = Commitment::zeros(c.d, c.kappa);
    scale_commitment_add_inplace(&mut out, scalar, c);
    out
}

pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d;
    let kappa = c.kappa;
    let mut out = Commitment::zeros(d, kappa);
    s_mul_add(&mut out, rho_ring, c);
    out
}

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> AjtaiResult<Commitment> {
    if rhos.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    if rhos.len() != cs.len() {
        return Err(AjtaiError::SizeMismatch {
            expected: rhos.len(),
            actual: cs.len(),
        });
    }
    if cs.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }

    let expected_d = cs[0].d;
    let expected_kappa = cs[0].kappa;
    let expected_len = expected_d
        .checked_mul(expected_kappa)
        .ok_or_else(|| AjtaiError::InvalidDimensions("commitment shape overflows usize".to_string()))?;
    if cs
        .iter()
        .any(|c| c.d != expected_d || c.kappa != expected_kappa || c.data.len() != expected_len)
    {
        return Err(AjtaiError::InvalidDimensions(
            "all commitments must have the same canonical shape".to_string(),
        ));
    }

    let mut acc = Commitment::zeros(expected_d, expected_kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        s_mul_add(&mut acc, rho, c);
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
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

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
                let mask = Z[base + t]; // any Fq digit (0, ±1, small, or general)
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

/// Row-major variant of [`commit_masked_ct`].
#[allow(non_snake_case)]
fn commit_masked_ct_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            let mut col = cf(pp.m_rows[i][j]);
            let mut nxt = [Fq::ZERO; D];

            for t in 0..d {
                let mask = Z[(t, j)];
                for r in 0..d {
                    acc_i[r] += col[r] * mask;
                }
                rot_step(&col, &mut nxt);
                core::mem::swap(&mut col, &mut nxt);
            }
        }
    }
    C
}

/// Fill `cols` with the d rotation columns of rot(a): cols[t] = cf(a * X^t).
///
/// This is an internal building block for high-performance folding code paths that need
/// to batch multiple Ajtai commitments without materializing full digit matrices.
#[doc(hidden)]
#[inline]
pub fn precompute_rot_columns(a: RqEl, cols: &mut [[Fq; D]]) {
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
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    // CRITICAL SECURITY: Runtime dimension checks to prevent binding bugs
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
        cols: Box<[[Fq; D]]>,
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self {
                acc: [Fq::ZERO; D],
                cols: vec![[Fq::ZERO; D]; D].into_boxed_slice(),
            }
        }
    }

    for i in 0..kappa {
        let row = &pp.m_rows[i];
        debug_assert_eq!(row.len(), m);

        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                row.par_iter()
                    .zip(Z.par_chunks_exact(d))
                    .fold(Acc::new, |mut st, (&a_ij, z_col)| {
                        precompute_rot_columns(a_ij, &mut st.cols);
                        // Constant schedule: always loop over all t
                        for t in 0..d {
                            let mask = z_col[t];
                            let col_t = &st.cols[t];
                            for r in 0..d {
                                st.acc[r] += col_t[r] * mask;
                            }
                        }
                        st
                    })
                    .reduce_with(|mut a, b| {
                        for r in 0..d {
                            a.acc[r] += b.acc[r];
                        }
                        a
                    })
                    .unwrap_or_else(Acc::new)
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut st = Acc::new();
                for (&a_ij, z_col) in row.iter().zip(Z.chunks_exact(d)) {
                    precompute_rot_columns(a_ij, &mut st.cols);
                    for t in 0..d {
                        let mask = z_col[t];
                        let col_t = &st.cols[t];
                        for r in 0..d {
                            st.acc[r] += col_t[r] * mask;
                        }
                    }
                }
                st
            }
        };

        C.col_mut(i).copy_from_slice(&acc.acc);
    }

    C
}

/// Row-major variant of [`commit_precomp_ct`].
#[allow(non_snake_case)]
fn commit_precomp_ct_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);
    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
        cols: Box<[[Fq; D]]>,
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self {
                acc: [Fq::ZERO; D],
                cols: vec![[Fq::ZERO; D]; D].into_boxed_slice(),
            }
        }
    }

    // Grab row slices once; avoids repeated bounds checks in the inner column loop.
    let z_rows: Vec<&[Fq]> = (0..d).map(|r| Z.row(r)).collect();

    for i in 0..kappa {
        let row = &pp.m_rows[i];
        debug_assert_eq!(row.len(), m);

        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                row.par_iter()
                    .enumerate()
                    .fold(Acc::new, |mut st, (j, &a_ij)| {
                        precompute_rot_columns(a_ij, &mut st.cols);
                        for t in 0..d {
                            let mask = z_rows[t][j];
                            let col_t = &st.cols[t];
                            for r in 0..d {
                                st.acc[r] += col_t[r] * mask;
                            }
                        }
                        st
                    })
                    .reduce_with(|mut a, b| {
                        for r in 0..d {
                            a.acc[r] += b.acc[r];
                        }
                        a
                    })
                    .unwrap_or_else(Acc::new)
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut st = Acc::new();
                for (j, &a_ij) in row.iter().enumerate() {
                    precompute_rot_columns(a_ij, &mut st.cols);
                    for t in 0..d {
                        let mask = z_rows[t][j];
                        let col_t = &st.cols[t];
                        for r in 0..d {
                            st.acc[r] += col_t[r] * mask;
                        }
                    }
                }
                st
            }
        };

        C.col_mut(i).copy_from_slice(&acc.acc);
    }

    C
}
