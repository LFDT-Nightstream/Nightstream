//! Common utilities and helper functions shared across engines.
//!
//! This module contains:
//! - Balanced base-b digit splitting for DEC operations
//! - RLC sampling (diagonal ρ matrices)
//! - ME relation helpers (compute y from Z and r)
//! - Matrix arithmetic helpers
//! - Extension field formatting utilities

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, Mat};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::error::PiCcsError;

// ---------------------------------------------------------------------------
// Balanced Base-b Digit Splitting
// ---------------------------------------------------------------------------

/// Helper: returns (r, q) with r in balanced range around zero.
///
/// Matches the Ajtai decomp_b balanced style (Definition 11):
/// Digits in approximately [-(b-1)/2, (b-1)/2], choosing residue with smallest absolute value.
///
/// This ensures termination for both positive and negative values.
fn balanced_divrem(v: i128, b: i128) -> (i128, i128) {
    debug_assert!(b >= 2);

    // Start with standard division
    let mut r = v % b;
    let mut q = (v - r) / b;

    // Shift remainder to balanced range (minimize |r|)
    let half = b / 2; // floor(b/2)

    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }

    (r, q)
}

#[inline]
fn balanced_divrem_i64(v: i64, b: i64) -> (i64, i64) {
    debug_assert!(b >= 2);

    let mut r = v % b;
    let mut q = (v - r) / b;

    let half = b / 2; // floor(b/2)
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }

    (r, q)
}

/// Split Z into **balanced base-b digits** Z = Σ_{i=0}^{k-1} b^i · Z_i, entrywise.
/// Each digit lies in [-floor(b/2), +floor(b/2)] for even b (inclusive upper bound),
/// and the analogous balanced range for odd b.
/// Returns an error if an entry cannot be represented within k digits (i.e., if |value| ≥ b^k)
/// — this indicates a bad RLC sample or overflow.
pub fn split_b_matrix_k_with_nonzero_flags(
    Z: &Mat<F>,
    k: usize,
    b: u32,
) -> Result<(Vec<Mat<F>>, Vec<bool>), PiCcsError> {
    let Z_rows = Z.rows();
    let Z_cols = Z.cols();

    let mut outs = (0..k)
        .map(|_| Mat::zero(Z_rows, Z_cols, F::ZERO))
        .collect::<Vec<_>>();
    let mut digit_nonzero = vec![false; k];

    let b_i = b as i128;
    let mut B: i128 = 1;
    for _ in 0..k {
        B = B.saturating_mul(b_i);
    } // b^k
    let neg_one = F::ZERO - F::ONE;

    // Helpers to interpret field element as a small signed integer in (-(B-1), B-1)
    let p: u128 = F::ORDER_U64 as u128; // Goldilocks prime fits in u64
    let B_u: u128 = B as u128;

    let z_data = Z.as_slice();
    {
        let mut out_slices: Vec<&mut [F]> = outs.iter_mut().map(|m| m.as_mut_slice()).collect();
        let total = z_data.len();
        debug_assert_eq!(total, Z_rows * Z_cols);

        if B_u <= i64::MAX as u128 {
            let b_i64 = b as i64;
            for idx in 0..total {
                let u = z_data[idx].as_canonical_u64() as u128;
                // Map to a small signed integer if within the DEC budget.
                let val_opt: Option<i64> = if u < B_u {
                    Some(u as i64)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    // negative representative
                    Some(-((p - u) as i64))
                } else {
                    None
                };

                let mut v = match val_opt {
                    Some(v) => v,
                    None => {
                        let r = idx / Z_cols;
                        let c = idx % Z_cols;
                        let B_signed = B_u as i128;
                        return Err(PiCcsError::ProtocolError(format!(
                            "DEC split: Z[{},{}] = {} (0x{:X}) is out of range for k_rho={}, b={}\n\
                             Matrix Z is {}×{}\n\
                             Balanced range: [{}, {}), where B = b^k_rho = {}^{} = {}\n\
                             This typically means witness values grew too large during RLC (expansion factor T=216 for rotation matrices)",
                            r, c, u, u, k, b, Z_rows, Z_cols, -B_signed, B_signed, b, k, B_u
                        )));
                    }
                };

                // Balanced digit extraction: r_i ∈ [-floor(b/2), ..., ceil(b/2)-1], v ← q
                for i in 0..k {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem_i64(v, b_i64);
                    if r_i != 0 {
                        let digit_f = match r_i {
                            1 => F::ONE,
                            -1 => neg_one,
                            _ => {
                                if r_i >= 0 {
                                    F::from_u64(r_i as u64)
                                } else {
                                    let abs = (-r_i) as u64;
                                    F::ZERO - F::from_u64(abs)
                                }
                            }
                        };
                        out_slices[i][idx] = digit_f;
                        digit_nonzero[i] = true;
                    }
                    v = q;
                }

                if v != 0 {
                    let r = idx / Z_cols;
                    let c = idx % Z_cols;
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split: Z[{},{}] needs more than k_rho={} digits in base b={}\n\
                         Matrix Z is {}×{}\n\
                         After extracting {} digits, remainder v={} (should be 0)\n\
                         Original value exceeded the range [{}, {}) for B = {}^{} = {}\n\
                         This typically means witness values grew too large during RLC (expansion factor T=216 for rotation matrices)",
                        r,
                        c,
                        k,
                        b,
                        Z_rows,
                        Z_cols,
                        k,
                        v,
                        -(B_u as i128),
                        B_u as i128,
                        b,
                        k,
                        B_u
                    )));
                }
            }
        } else {
            for idx in 0..total {
                let u = z_data[idx].as_canonical_u64() as u128;
                // Map to a small signed integer if within the DEC budget.
                let val_opt: Option<i128> = if u < B_u {
                    Some(u as i128)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    // negative representative
                    Some(-((p - u) as i128))
                } else {
                    None
                };

                let mut v = match val_opt {
                    Some(v) => v,
                    None => {
                        let r = idx / Z_cols;
                        let c = idx % Z_cols;
                        let B_signed = B_u as i128;
                        return Err(PiCcsError::ProtocolError(format!(
                            "DEC split: Z[{},{}] = {} (0x{:X}) is out of range for k_rho={}, b={}\n\
                             Matrix Z is {}×{}\n\
                             Balanced range: [{}, {}), where B = b^k_rho = {}^{} = {}\n\
                             This typically means witness values grew too large during RLC (expansion factor T=216 for rotation matrices)",
                            r, c, u, u, k, b, Z_rows, Z_cols, -B_signed, B_signed, b, k, B_u
                        )));
                    }
                };

                // Balanced digit extraction: r_i ∈ [-floor(b/2), ..., ceil(b/2)-1], v ← q
                for i in 0..k {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem(v, b_i);
                    if r_i != 0 {
                        let digit_f = match r_i {
                            1 => F::ONE,
                            -1 => neg_one,
                            _ => {
                                if r_i >= 0 {
                                    F::from_u64(r_i as u64)
                                } else {
                                    let abs = (-r_i) as u64;
                                    F::ZERO - F::from_u64(abs)
                                }
                            }
                        };
                        out_slices[i][idx] = digit_f;
                        digit_nonzero[i] = true;
                    }
                    v = q;
                }

                if v != 0 {
                    let r = idx / Z_cols;
                    let c = idx % Z_cols;
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split: Z[{},{}] needs more than k_rho={} digits in base b={}\n\
                         Matrix Z is {}×{}\n\
                         After extracting {} digits, remainder v={} (should be 0)\n\
                         Original value exceeded the range [{}, {}) for B = {}^{} = {}\n\
                         This typically means witness values grew too large during RLC (expansion factor T=216 for rotation matrices)",
                        r,
                        c,
                        k,
                        b,
                        Z_rows,
                        Z_cols,
                        k,
                        v,
                        -(B_u as i128),
                        B_u as i128,
                        b,
                        k,
                        B_u
                    )));
                }
            }
        }
    }

    Ok((outs, digit_nonzero))
}

pub fn split_b_matrix_k(Z: &Mat<F>, k: usize, b: u32) -> Result<Vec<Mat<F>>, PiCcsError> {
    split_b_matrix_k_with_nonzero_flags(Z, k, b).map(|(digits, _nonzero)| digits)
}

// ---------------------------------------------------------------------------
// RLC Sampling - Rotation Matrices (Paper-Compliant)
// ---------------------------------------------------------------------------

/// Ring metadata for ΠRLC rotation-matrix challenges (Section 3.4, Definition 14).
///
/// Specifies the cyclotomic polynomial Φ_η and the coefficient alphabet A
/// used to construct the strong sampling set C = {rot(a) : a ∈ C_R}, where
/// Module-level statics for Goldilocks ring parameters.
/// Φ₈₁(X) = X^54 + X^27 + 1
pub static PHI_GL: [i32; D] = {
    let mut a = [0i32; D];
    a[0] = 1; // constant term
    a[27] = 1; // X^27 coefficient
    a
};

/// Goldilocks alphabet: [-2,-1,0,1,2]
pub static A5_GL: [i8; 5] = [-2, -1, 0, 1, 2];

/// Module-level statics for Almost-Goldilocks ring parameters.
/// Φ(X) = X^64 + 1
pub static PHI_AGL: [i32; D] = {
    let mut a = [0i32; D];
    a[0] = 1; // X^64 + 1 => c_0 = 1
    a
};

/// Almost-Goldilocks alphabet: [-1,0,1]
pub static A3_AGL: [i8; 3] = [-1, 0, 1];

/// C_R = {a ∈ R_q : all coeffs of a lie in A}.
pub struct RotRing {
    /// Coefficients [c_0, c_1, ..., c_{d-1}] of Φ_η(X) = X^d + c_{d-1}·X^{d-1} + ... + c_0.
    /// Must have length D (the ring dimension).
    pub phi_coeffs: &'static [i32],

    /// Small coefficient alphabet A ⊂ ℤ (e.g., [-2,-1,0,1,2] or [-1,0,1]).
    /// The strong sampling set is C_R = {polynomials with coeffs in A}.
    pub alphabet: &'static [i8],

    /// Optional: lower bound on b_inv from Theorem 1 (invertibility threshold).
    /// If provided, enforces Δ_A < b_inv where Δ_A = max(A) - min(A).
    pub binv_floor: Option<u64>,
}

impl RotRing {
    /// Goldilocks (Section 6.2): Φ_η = X^54 + X^27 + 1, alphabet = [-2,-1,0,1,2].
    /// Yields T=216, b_inv ≈ 2.5×10^9.
    pub const fn goldilocks() -> Self {
        Self {
            phi_coeffs: &PHI_GL,
            alphabet: &A5_GL,
            binv_floor: Some(2_500_000_000), // ≈ 2.5×10^9 from paper
        }
    }

    /// Almost-Goldilocks (Section 6.1): Φ_η = X^64 + 1, alphabet = [-1,0,1].
    /// Yields T=128, b_inv > 4 (sufficient for small alphabets).
    pub const fn almost_goldilocks() -> Self {
        Self {
            phi_coeffs: &PHI_AGL,
            alphabet: &A3_AGL,
            binv_floor: None, // Known safe for this choice
        }
    }
}

/// Compute expansion factor T per Theorem 3: T ≤ 2·φ(η)·max|coeff|.
/// For prime-power cyclotomics, φ(η) = d (the degree).
#[inline]
fn expansion_factor_T(alphabet: &[i8]) -> u128 {
    let c_max = alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap_or(0) as u128;
    2u128 * (D as u128) * c_max
}

/// Convert signed small integer to field element F.
#[inline]
fn f_from_i64(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

/// Build rotation matrix rot(a) given coefficients of a and Φ_η coefficients.
///
/// Uses the shift recurrence (Definition 7, Remark 1):
///   col_0 = cf(a)
///   col_{j+1} = F_shift · col_j
/// where F_shift implements the reduction X·a ≡ (X·a) mod Φ_η.
fn rot_from_coeffs(a_coeffs: &[F], phi_coeffs: &[i32]) -> Mat<F> {
    debug_assert_eq!(a_coeffs.len(), D);
    debug_assert_eq!(phi_coeffs.len(), D);

    // Precompute -c_r for shift matrix F
    let neg_c: Vec<F> = phi_coeffs
        .iter()
        .map(|&cr| f_from_i64(-(cr as i64)))
        .collect();

    // Build columns: col_j = F^j · cf(a)
    // F_shift(v)[0] = v[d-1]·(-c_0)
    // F_shift(v)[r] = v[r-1] + v[d-1]·(-c_r) for r ≥ 1
    let mut rho = Mat::zero(D, D, F::ZERO);
    let mut col = a_coeffs.to_vec();

    for j in 0..D {
        // Write column j
        for r in 0..D {
            rho[(r, j)] = col[r];
        }

        // Compute next column: col ← F_shift(col)
        let last = col[D - 1];
        let mut next = vec![F::ZERO; D];
        next[0] = last * neg_c[0];
        for r in 1..D {
            next[r] = col[r - 1] + last * neg_c[r];
        }
        col = next;
    }

    rho
}

/// Draw `need` samples uniformly from `alphabet` using transcript randomness (rejection sampling).
///
/// Uses 16-bit chunks from the transcript digest to achieve unbiased sampling:
/// - Accept chunk if it falls in [0, largest_multiple_of_|alphabet|)
/// - Reject and retry otherwise
fn draw_alphabet_vector(
    tr: &mut Poseidon2Transcript,
    need: usize,
    alphabet: &[i8],
    label: &'static [u8],
    seed: u64,
) -> Vec<i8> {
    let m = alphabet.len() as u32;
    let bucket = (1u32 << 16) / m * m; // Largest multiple of m below 2^16

    let mut out = Vec::with_capacity(need);
    let mut ctr = seed;

    while out.len() < need {
        tr.append_message(label, &ctr.to_le_bytes());
        let dig = tr.digest32();

        for w in dig.chunks_exact(2) {
            let x = u16::from_le_bytes([w[0], w[1]]) as u32;
            if x < bucket {
                let idx = (x % m) as usize;
                out.push(alphabet[idx]);
                if out.len() == need {
                    break;
                }
            }
        }
        ctr = ctr.wrapping_add(1);
    }

    out
}

/// Sample `count` rotation matrices ρ_i = rot(a_i) for ΠRLC with a_i having small coefficients.
///
/// This is the **paper-compliant** ΠRLC sampler (Section 4.5, Definition 14).
///
/// ## Key Insight: Decoupling `count` from `k_rho`
///
/// - `k_rho` controls the **DEC exponent** (accumulator width, B = b^{k_rho})
/// - `count` is the **number of ME claims being RLC'd** (can be different from k_rho+1)
///
/// The soundness constraint is: `count · T · (b-1) < b^{k_rho}`
/// - If this fails, you need to increase `k_rho` or reduce `count` (e.g., hierarchical merging)
///
/// ## Properties
/// - Strong sampling set: differences (ρ_i - ρ_j) are invertible for distinct i,j (Theorem 1)
/// - Expansion factor T: Computed from ring/alphabet via Theorem 3: T ≤ 2·φ(η)·max|coeff|
///
/// # Arguments
/// * `tr` - Fiat-Shamir transcript for deterministic randomness
/// * `params` - Neo parameters (k_rho determines norm bound B = b^{k_rho})
/// * `ring` - Ring metadata (cyclotomic polynomial and coefficient alphabet)
/// * `count` - Number of rhos to sample (= number of ME claims being RLC'd)
///
/// # Returns
/// `count` rotation matrices ρ_i ∈ S ⊆ F^{D×D}, or error if soundness checks fail.
pub fn sample_rot_rhos_n(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    ring: &RotRing,
    count: usize,
) -> Result<Vec<Mat<F>>, PiCcsError> {
    // ---- Sanity checks ----
    if ring.phi_coeffs.len() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "phi_coeffs length {} != D={}",
            ring.phi_coeffs.len(),
            D
        )));
    }
    if ring.alphabet.is_empty() {
        return Err(PiCcsError::InvalidInput("alphabet is empty".into()));
    }
    if count == 0 {
        return Err(PiCcsError::InvalidInput("count must be > 0".into()));
    }

    // ---- Strong sampling set check (Definition 14 + Theorem 1) ----
    if let Some(binv) = ring.binv_floor {
        let min = *ring.alphabet.iter().min().unwrap() as i64;
        let max = *ring.alphabet.iter().max().unwrap() as i64;
        let delta_a = (max - min).unsigned_abs();
        if delta_a >= binv {
            return Err(PiCcsError::InvalidInput(format!(
                "Strong-set check failed: Δ_A = {} must be < b_inv = {} (Theorem 1)",
                delta_a, binv
            )));
        }
    }

    // ---- ΠRLC norm bound check (Section 4.3) ----
    // The REAL constraint: count · T · (b-1) < b^{k_rho}
    // This ensures the combined witness after RLC stays within norm bound B = b^{k_rho}
    let T = expansion_factor_T(ring.alphabet);
    let b = params.b as u128;
    let k_rho = params.k_rho;

    // Compute b^{k_rho} carefully to avoid overflow
    let b_pow_k: u128 = if k_rho >= 64 {
        // For k_rho >= 64 with b=2, b^k would overflow u128
        // But we also need B < q/2, so this is already invalid
        return Err(PiCcsError::InvalidInput(format!(
            "k_rho={} is too large (b^k_rho would overflow); max is ~62 for b=2",
            k_rho
        )));
    } else {
        (b as u128)
            .checked_pow(k_rho)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("b^k_rho overflow: b={}, k_rho={}", b, k_rho)))?
    };

    let lhs = (count as u128) * T * (b.saturating_sub(1));

    if lhs >= b_pow_k {
        return Err(PiCcsError::InvalidInput(format!(
            "ΠRLC norm bound violated: count·T·(b-1) = {}·{}·{} = {} must be < b^{{k_rho}} = {} (Section 4.3)\n\
             count={} is the number of ME claims being RLC'd\n\
             k_rho={} controls the norm bound B = b^k_rho = {}\n\
             T={} is the expansion factor (Theorem 3)\n\
             \n\
             Solutions:\n\
             1. Increase k_rho to allow more claims (increases accumulator size)\n\
             2. Use hierarchical merging to reduce count\n\
             3. Reduce the number of memory ME claims",
            count,
            T,
            b - 1,
            lhs,
            b_pow_k,
            count,
            k_rho,
            b_pow_k,
            T
        )));
    }

    // ---- Sample ρ_i = rot(a_i) ----
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        // Domain-separate each ρ_i
        tr.append_message(b"rlc/rot/index", &(i as u64).to_le_bytes());

        // Draw D coefficients from the small alphabet (unbiased rejection sampling)
        let coeffs_i8 = draw_alphabet_vector(tr, D, ring.alphabet, b"rlc/rot/chunk", i as u64);

        // For k=1 there is nothing to mix, so Π_RLC can be the identity without affecting soundness.
        // We still consume transcript randomness above to keep Fiat–Shamir behavior consistent.
        if count == 1 {
            out.push(Mat::identity(D));
            continue;
        }

        // Lift to field F
        let a_coeffs_f: Vec<F> = coeffs_i8.iter().map(|&c| f_from_i64(c as i64)).collect();

        // Build rotation matrix rot(a_i)
        let rho = rot_from_coeffs(&a_coeffs_f, ring.phi_coeffs);
        out.push(rho);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// ME Relation Helpers
// ---------------------------------------------------------------------------

/// Compute y from Z and r according to the ME relation: y_j := Z · (M_j^T · r^b).
///
/// Returns (y, y_scalars) where:
/// - y[j] is padded to 2^{ell_d} and contains the first D digits
/// - y_scalars[j] = Σ_{d=0}^{D-1} b^d · y[j][d] (base-b recomposition)
pub fn compute_y_from_Z_and_r(s: &CcsStructure<F>, Z: &Mat<F>, r: &[K], ell_d: usize, b: u32) -> (Vec<Vec<K>>, Vec<K>) {
    use neo_ccs::{utils::mat_vec_mul_fk, CcsMatrix};
    let d_pad = 1usize << ell_d;
    let mut y_new: Vec<Vec<K>> = Vec::with_capacity(s.t());
    // Build r^b over rows
    let rb = neo_ccs::utils::tensor_point::<K>(r);
    // v_j = M_j^T · r^b ∈ K^m
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        let n_eff = core::cmp::min(s.n, rb.len());

        match &s.matrices[j] {
            CcsMatrix::Identity { n } => {
                let cap = core::cmp::min(n_eff, *n);
                for i in 0..cap {
                    vj[i] += rb[i];
                }
            }
            CcsMatrix::Csc(csc) => {
                for c in 0..csc.ncols {
                    let s0 = csc.col_ptr[c];
                    let e0 = csc.col_ptr[c + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        if row >= n_eff {
                            continue;
                        }
                        let wr = rb[row];
                        if wr == K::ZERO {
                            continue;
                        }
                        vj[c] += wr.scale_base_k(K::from(csc.vals[k]));
                    }
                }
            }
        }
        vjs.push(vj);
    }
    // y_j = Z · v_j
    for j in 0..s.t() {
        let yj_digits = mat_vec_mul_fk::<F, K>(Z.as_slice(), Z.rows(), Z.cols(), &vjs[j]);
        let mut yj_pad = yj_digits;
        let cur = yj_pad.len();
        if d_pad > cur {
            yj_pad.resize(d_pad, K::ZERO);
        }
        y_new.push(yj_pad);
    }
    // y_scalars: base-b recomposition from digits
    let bK = K::from(F::from_u64(b as u64));
    let mut y_scalars = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut acc = K::ZERO;
        let mut pw = K::ONE;
        for d in 0..D {
            acc += pw * y_new[j][d];
            pw *= bK;
        }
        y_scalars.push(acc);
    }
    (y_new, y_scalars)
}

// ---------------------------------------------------------------------------
// Matrix Arithmetic
// ---------------------------------------------------------------------------

/// Left-multiply accumulator by rho: `acc += rho * a`.
pub fn left_mul_acc(acc: &mut Mat<F>, rho: &Mat<F>, a: &Mat<F>) {
    debug_assert_eq!(rho.rows(), rho.cols());
    debug_assert_eq!(rho.rows(), acc.rows());
    debug_assert_eq!(a.rows(), acc.rows());
    debug_assert_eq!(a.cols(), acc.cols());
    let d = acc.rows();
    let m = acc.cols();
    for r in 0..d {
        for c in 0..m {
            let mut sum = F::ZERO;
            for k in 0..d {
                sum += rho[(r, k)] * a[(k, c)];
            }
            acc[(r, c)] += sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting Utilities
// ---------------------------------------------------------------------------

/// Helper formatting for extension field elements used in debug logs.
pub fn format_ext(x: K) -> String {
    let coeffs = x.as_coeffs();
    format!("({}, {})", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
}
