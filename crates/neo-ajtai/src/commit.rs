use crate::error::{AjtaiError, AjtaiResult};
use crate::types::{Commitment, PP};
use neo_ccs::Mat;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Fq;
use rayon::prelude::*;
use rand::{CryptoRng, RngCore};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{cf, cf_inv as cf_unmap, Rq as RqEl, D, ETA};
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
        if x < Q {
            return Fq::from_u64(x);
        }
    }
}

/// Sample a uniform element from R_q by sampling D uniform coefficients in F_q and mapping with `cf^{-1}`.
#[doc(hidden)]
#[inline]
pub fn sample_uniform_rq<R: RngCore + CryptoRng>(rng: &mut R) -> RqEl {
    let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(rng));
    cf_unmap(coeffs)
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
    next[0] -= last; // -1 * last
    next[27] -= last; // -X^27 * last
}

/// Rotation step for internal use by commit implementations
/// This implementation is specialized for η=81 (D=54) as enforced by compile-time assertions.
#[inline]
#[cfg(not(feature = "testing"))]
pub(crate) fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    // Note: ETA == 81 is guaranteed at compile-time by const assertions at module top
    rot_step_phi_81(cur, next)
}

/// Rotation step for internal use by commit implementations
/// This implementation is specialized for η=81 (D=54) as enforced by compile-time assertions.
#[inline]
#[cfg(feature = "testing")]
pub fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    // Note: ETA == 81 is guaranteed at compile-time by const assertions at module top
    rot_step_phi_81(cur, next)
}

#[inline]
fn acc_mul_add_inplace(acc: &mut [Fq; D], col: &[Fq; D], scalar: Fq) {
    // Fast paths for the common balanced-digit case (b ∈ {2,3} ⇒ scalar ∈ {-1,0,1}).
    //
    // NOTE: This is intentionally variable-time w.r.t. `scalar`. It is only used in the
    // seeded PP row-major commitment path, which is a prover-only performance hot loop.
    if scalar == Fq::ZERO {
        return;
    }
    if scalar == Fq::ONE {
        // Unrolled to encourage LLVM auto-vectorization on platforms that support it.
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
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            // sample ring element uniformly by sampling d random coefficients in F_q and mapping via cf^{-1}
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

/// Parallel version of [`setup`], primarily intended for large `m` where setup dominates runtime.
///
/// Implementation notes:
/// - Uses the provided `rng` only to generate one 32-byte seed per row.
/// - Each row is generated independently in parallel using `ChaCha8Rng` seeded from that seed.
/// - Output is deterministic given the input `rng` state, but will not match the sequential `setup`
///   output for the same RNG because the RNG stream is partitioned.
pub fn setup_par<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions(
            "d parameter must match ring dimension D".to_string(),
        ));
    }

    if m == 0 {
        return Ok(PP {
            kappa,
            m,
            d,
            m_rows: vec![Vec::new(); kappa],
        });
    }

    let mut row_seeds = vec![[0u8; 32]; kappa];
    for seed in row_seeds.iter_mut() {
        rng.fill_bytes(seed);
    }

    // Deterministic chunking: must NOT depend on runtime thread count, so a verifier can
    // re-derive the same PP from the same seed across environments.
    let chunk_size = core::cmp::min(m, 1 << 15).max(1024);
    let num_chunks = (m + chunk_size - 1) / chunk_size;

    let mut rows = Vec::with_capacity(kappa);
    for row_seed in row_seeds {
        // Derive per-chunk seeds deterministically from the row seed.
        let mut seed_rng = ChaCha8Rng::from_seed(row_seed);
        let mut chunk_seeds = vec![[0u8; 32]; num_chunks];
        for seed in chunk_seeds.iter_mut() {
            seed_rng.fill_bytes(seed);
        }

        // Fill the row in place in parallel. This avoids extra copies of multi-GB buffers.
        let mut row = vec![RqEl::zero(); m];
        row.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut chunk_rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                for el in chunk.iter_mut() {
                    let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(&mut chunk_rng));
                    *el = cf_unmap(coeffs);
                }
            });

        rows.push(row);
    }

    Ok(PP {
        kappa,
        m,
        d,
        m_rows: rows,
    })
}

#[inline]
fn seeded_pp_chunking(m: usize) -> (usize, usize) {
    let chunk_size = core::cmp::min(m, 1 << 15).max(1024);
    let num_chunks = (m + chunk_size - 1) / chunk_size;
    (chunk_size, num_chunks)
}

#[inline]
fn seeded_pp_row_seeds(master_seed: [u8; 32], kappa: usize) -> Vec<[u8; 32]> {
    let mut rng = ChaCha8Rng::from_seed(master_seed);
    let mut row_seeds = vec![[0u8; 32]; kappa];
    for seed in row_seeds.iter_mut() {
        rng.fill_bytes(seed);
    }
    row_seeds
}

#[inline]
fn seeded_pp_chunk_seeds_for_row(row_seed: [u8; 32], num_chunks: usize) -> Vec<[u8; 32]> {
    let mut seed_rng = ChaCha8Rng::from_seed(row_seed);
    let mut chunk_seeds = vec![[0u8; 32]; num_chunks];
    for seed in chunk_seeds.iter_mut() {
        seed_rng.fill_bytes(seed);
    }
    chunk_seeds
}

/// Deterministically derive PP chunk seeds for a seeded PP (the same partitioning used by [`setup_par`]).
///
/// Returns `(chunk_size, chunk_seeds_by_row)`, where `chunk_seeds_by_row[row][chunk]` seeds the
/// ChaCha stream used to generate the PP ring elements for that chunk.
#[doc(hidden)]
pub fn seeded_pp_chunk_seeds(master_seed: [u8; 32], kappa: usize, m: usize) -> (usize, Vec<Vec<[u8; 32]>>) {
    let (chunk_size, num_chunks) = seeded_pp_chunking(m);
    let row_seeds = seeded_pp_row_seeds(master_seed, kappa);
    let chunk_seeds = row_seeds
        .into_iter()
        .map(|rs| seeded_pp_chunk_seeds_for_row(rs, num_chunks))
        .collect();
    (chunk_size, chunk_seeds)
}

/// Commit to a **row-major** `Mat<Fq>` using a *seeded PP* without materializing the multi-GB PP matrix.
///
/// This produces the same commitment as:
/// - `setup_par(ChaCha8Rng::from_seed(seed), d, kappa, m)` followed by
/// - [`commit_row_major`].
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn commit_row_major_seeded(seed: [u8; 32], d: usize, kappa: usize, m: usize, Z: &Mat<Fq>) -> Commitment {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);
    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self { acc: [Fq::ZERO; D] }
        }
    }

    // Fast row slices.
    let z_rows: Vec<&[Fq]> = (0..d).map(|r| Z.row(r)).collect();
    let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, m);

    for i in 0..kappa {
        let chunk_seeds = &chunk_seeds_by_row[i];
        let acc = (0..chunk_seeds.len())
            .into_par_iter()
            .fold(Acc::new, |mut st, chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = core::cmp::min(m, start + chunk_size);
                let mut rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                let mut nxt = [Fq::ZERO; D];
                for col_idx in start..end {
                    let a_ij = sample_uniform_rq(&mut rng);
                    let mut rot_col = cf(a_ij);
                    for t in 0..d {
                        let mask = z_rows[t][col_idx];
                        acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                        rot_step(&rot_col, &mut nxt);
                        core::mem::swap(&mut rot_col, &mut nxt);
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
            .unwrap_or_else(Acc::new);

        C.col_mut(i).copy_from_slice(&acc.acc);
    }

    C
}

// Variable-time optimization removed for security and simplicity

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
/// Uses constant-time dense computation for all inputs (audit-ready).
/// Returns error if Z dimensions don't match expected d×m.
#[allow(non_snake_case)]
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

    // Mirror `try_commit`'s dispatch choice, but operate directly on the row-major `Mat`.
    const PRECOMP_THRESHOLD: usize = 256;
    Ok(if m >= PRECOMP_THRESHOLD {
        commit_precomp_ct_row_major(pp, Z)
    } else {
        commit_masked_ct_row_major(pp, Z)
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
    &commit(pp, Z) == c
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    if k != Z_is.len() {
        return false;
    }
    // Check shapes
    for ci in c_is {
        if ci.d != c.d || ci.kappa != c.kappa {
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
/// Since we don't have direct access to SMatrix, we use SAction to operate on the commitment data.
pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d;
    let kappa = c.kappa;
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
                let mask = Z.row(t)[j];
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

        let acc = row
            .par_iter()
            .zip(Z.par_chunks_exact(d))
            .fold(
                Acc::new,
                |mut st, (&a_ij, z_col)| {
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
                },
            )
            .reduce_with(|mut a, b| {
                for r in 0..d {
                    a.acc[r] += b.acc[r];
                }
                a
            })
            .unwrap_or_else(Acc::new);

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

        let acc = row
            .par_iter()
            .enumerate()
            .fold(
                Acc::new,
                |mut st, (j, &a_ij)| {
                    precompute_rot_columns(a_ij, &mut st.cols);
                    for t in 0..d {
                        let mask = z_rows[t][j];
                        let col_t = &st.cols[t];
                        for r in 0..d {
                            st.acc[r] += col_t[r] * mask;
                        }
                    }
                    st
                },
            )
            .reduce_with(|mut a, b| {
                for r in 0..d {
                    a.acc[r] += b.acc[r];
                }
                a
            })
            .unwrap_or_else(Acc::new);

        C.col_mut(i).copy_from_slice(&acc.acc);
    }

    C
}
