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
use neo_math::{balanced::to_balanced_i128, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

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

#[inline]
fn balanced_divrem_i64_base2(v: i64) -> (i64, i64) {
    if (v & 1) == 0 {
        (0, v >> 1)
    } else if v > 0 {
        (1, (v - 1) >> 1)
    } else {
        (-1, (v + 1) >> 1)
    }
}

#[inline]
fn build_balanced_digit_lut(b: u32) -> (i64, Vec<F>) {
    let half = (b as i64) / 2;
    let mut lut = Vec::with_capacity((2 * half + 1) as usize);
    for d in -half..=half {
        let f = if d >= 0 {
            F::from_u64(d as u64)
        } else {
            F::ZERO - F::from_u64((-d) as u64)
        };
        lut.push(f);
    }
    (half, lut)
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
    let (digit_half, digit_lut) = build_balanced_digit_lut(b);

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
            let fast_base2 = b == 2;
            for idx in 0..total {
                let z_entry = z_data[idx];
                if z_entry == F::ZERO {
                    continue;
                }
                let u = z_entry.as_canonical_u64() as u128;
                // Map to a small signed integer if within the DEC budget.
                let val_opt: Option<i64> = {
                    let neg_mag = p.saturating_sub(u);
                    let pos_ok = u < B_u;
                    let neg_ok = neg_mag < B_u;
                    match (pos_ok, neg_ok) {
                        (false, false) => None,
                        (true, false) => Some(u as i64),
                        (false, true) => Some(-(neg_mag as i64)),
                        (true, true) => {
                            // Choose the smaller-magnitude balanced representative.
                            if u <= neg_mag {
                                Some(u as i64)
                            } else {
                                Some(-(neg_mag as i64))
                            }
                        }
                    }
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
                    let (r_i, q) = if fast_base2 {
                        balanced_divrem_i64_base2(v)
                    } else {
                        balanced_divrem_i64(v, b_i64)
                    };
                    if r_i != 0 {
                        debug_assert!(r_i >= -digit_half && r_i <= digit_half);
                        let digit_f = digit_lut[(r_i + digit_half) as usize];
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
            let b_i64 = b as i64;
            let fast_base2 = b == 2;
            for idx in 0..total {
                let z_entry = z_data[idx];
                if z_entry == F::ZERO {
                    continue;
                }
                let u = z_entry.as_canonical_u64() as u128;
                // Map to a small signed integer if within the DEC budget.
                let val_opt: Option<i128> = {
                    let neg_mag = p.saturating_sub(u);
                    let pos_ok = u < B_u;
                    let neg_ok = neg_mag < B_u;
                    match (pos_ok, neg_ok) {
                        (false, false) => None,
                        (true, false) => Some(u as i128),
                        (false, true) => Some(-(neg_mag as i128)),
                        (true, true) => {
                            // Choose the smaller-magnitude balanced representative.
                            if u <= neg_mag {
                                Some(u as i128)
                            } else {
                                Some(-(neg_mag as i128))
                            }
                        }
                    }
                };

                let v = match val_opt {
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

                // Even when B is large, the selected balanced representative is often small.
                // Prefer i64 extraction in that common case to avoid expensive i128 division.
                if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                    let mut v64 = v as i64;
                    for i in 0..k {
                        if v64 == 0 {
                            break;
                        }
                        let (r_i, q) = if fast_base2 {
                            balanced_divrem_i64_base2(v64)
                        } else {
                            balanced_divrem_i64(v64, b_i64)
                        };
                        if r_i != 0 {
                            debug_assert!(r_i >= -digit_half && r_i <= digit_half);
                            let digit_f = digit_lut[(r_i + digit_half) as usize];
                            out_slices[i][idx] = digit_f;
                            digit_nonzero[i] = true;
                        }
                        v64 = q;
                    }

                    if v64 != 0 {
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
                            v64,
                            -(B_u as i128),
                            B_u as i128,
                            b,
                            k,
                            B_u
                        )));
                    }
                    continue;
                }

                let mut v = v;
                // Balanced digit extraction: r_i ∈ [-floor(b/2), ..., ceil(b/2)-1], v ← q
                for i in 0..k {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem(v, b_i);
                    if r_i != 0 {
                        let r_i64 = r_i as i64;
                        debug_assert!(r_i64 >= -digit_half && r_i64 <= digit_half);
                        let digit_f = digit_lut[(r_i64 + digit_half) as usize];
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

/// C_R = {a ∈ R_q : all coeffs of a lie in A}.
pub struct RotRing {
    /// Coefficients [c_0, c_1, ..., c_{d-1}] of Φ_η(X) = X^d + c_{d-1}·X^{d-1} + ... + c_0.
    /// Must have length D (the ring dimension).
    pub phi_coeffs: &'static [i32],

    /// Small coefficient alphabet A ⊂ ℤ (e.g., [-2,-1,0,1,2]).
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

/// Convert signed small integer to a field element.
#[inline]
fn ff_from_i64<Ff: Field + PrimeCharacteristicRing>(x: i64) -> Ff {
    if x >= 0 {
        Ff::from_u64(x as u64)
    } else {
        Ff::ZERO - Ff::from_u64((-x) as u64)
    }
}

/// Convert signed small integer to field element `F`.
#[inline]
fn f_from_i64(x: i64) -> F {
    ff_from_i64::<F>(x)
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

/// Resolve the supported cyclotomic polynomial coefficients from parameters.
///
/// We keep this strict to prevent accepting arbitrary linear operators in Π_RLC.
pub fn phi_coeffs_from_params(params: &NeoParams) -> Result<&'static [i32], PiCcsError> {
    if params.d as usize != D {
        return Err(PiCcsError::InvalidInput(format!(
            "Π_RLC: params.d={} must equal D={}",
            params.d, D
        )));
    }
    match params.eta {
        81 => Ok(&PHI_GL),
        128 => Err(PiCcsError::InvalidInput(
            "Π_RLC: eta=128 (Almost-Goldilocks) is disabled while D=54; enable only with a full D=64 migration".into(),
        )),
        _ => Err(PiCcsError::InvalidInput(format!(
            "Π_RLC: unsupported cyclotomic eta={} for strict rotation-matrix validation",
            params.eta
        ))),
    }
}

/// Validate that a matrix is a ring-scalar rotation matrix `rot(a)` over the given cyclotomic ring.
///
/// This enforces the shift recurrence:
/// - col_{j+1}[0] = col_j[d-1] * (-c_0)
/// - col_{j+1}[r] = col_j[r-1] + col_j[d-1] * (-c_r), r >= 1
/// where `phi_coeffs = [c_0, ..., c_{d-1}]`.
pub fn validate_rho_is_rotation_matrix<Ff>(rho: &Mat<Ff>, phi_coeffs: &[i32], label: &str) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if rho.rows() != D || rho.cols() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: rho shape {}x{} must be {}x{}",
            rho.rows(),
            rho.cols(),
            D,
            D
        )));
    }
    if phi_coeffs.len() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: phi coeff length {} must equal D={}",
            phi_coeffs.len(),
            D
        )));
    }

    let neg_c: Vec<Ff> = phi_coeffs
        .iter()
        .map(|&cr| ff_from_i64::<Ff>(-(cr as i64)))
        .collect();

    for j in 0..(D - 1) {
        let last = rho[(D - 1, j)];

        let want0 = last * neg_c[0];
        if rho[(0, j + 1)] != want0 {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: rho fails rotation recurrence at col={}, row=0",
                j + 1
            )));
        }
        for r in 1..D {
            let want = rho[(r - 1, j)] + last * neg_c[r];
            if rho[(r, j + 1)] != want {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: rho fails rotation recurrence at col={}, row={}",
                    j + 1,
                    r
                )));
            }
        }
    }

    Ok(())
}

/// Validate that all `rhos` are strict ring-scalar rotation matrices for the current params.
pub fn validate_rhos_are_rotation_matrices<Ff>(
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let phi = phi_coeffs_from_params(params)?;
    for (idx, rho) in rhos.iter().enumerate() {
        validate_rho_is_rotation_matrix(rho, phi, &format!("{label}[{idx}]"))?;
    }
    Ok(())
}

/// Typed Π_RLC challenge: a validated ring-scalar rotation matrix.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RotRho(pub(crate) Mat<F>);

impl RotRho {
    /// Construct a typed rho after strict rotation-matrix validation.
    pub fn new_checked(params: &NeoParams, rho: Mat<F>) -> Result<Self, PiCcsError> {
        let phi = phi_coeffs_from_params(params)?;
        validate_rho_is_rotation_matrix(&rho, phi, "RotRho::new_checked")?;
        Ok(Self(rho))
    }

    #[inline]
    pub(crate) fn new_unchecked(rho: Mat<F>) -> Self {
        Self(rho)
    }

    #[inline]
    pub fn as_mat(&self) -> &Mat<F> {
        &self.0
    }

    #[inline]
    pub fn into_mat(self) -> Mat<F> {
        self.0
    }
}

impl AsRef<Mat<F>> for RotRho {
    #[inline]
    fn as_ref(&self) -> &Mat<F> {
        self.as_mat()
    }
}

/// Validate and convert raw rho matrices into typed rotation-matrix challenges.
pub fn rot_rhos_from_mats(params: &NeoParams, rhos: &[Mat<F>], label: &str) -> Result<Vec<RotRho>, PiCcsError> {
    validate_rhos_are_rotation_matrices(params, rhos, label)?;
    Ok(rhos.iter().cloned().map(RotRho::new_unchecked).collect())
}

/// Materialize typed rho challenges as raw matrices.
pub fn rot_rhos_to_mats(rhos: &[RotRho]) -> Vec<Mat<F>> {
    rhos.iter().map(|rho| rho.as_mat().clone()).collect()
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
    let lhs = (count as u128) * T * (b.saturating_sub(1));
    let k_required = min_k_rho_for_rlc_count(params, ring, count)?;
    let b_pow_k = (b as u128)
        .checked_pow(params.k_rho)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("b^k_rho overflow: b={}, k_rho={}", b, params.k_rho)))?;

    if params.k_rho < k_required {
        return Err(PiCcsError::InvalidInput(format!(
            "ΠRLC norm bound violated: count·T·(b-1) = {}·{}·{} = {} must be < b^{{k_rho}} = {} (Section 4.3)\n\
             count={} is the number of ME claims being RLC'd\n\
             k_rho={} controls the norm bound B = b^k_rho = {}\n\
             minimum required k_rho for this count is {}\n\
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
            params.k_rho,
            b_pow_k,
            k_required,
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

        // Lift to field F
        let a_coeffs_f: Vec<F> = coeffs_i8.iter().map(|&c| f_from_i64(c as i64)).collect();

        // Build rotation matrix rot(a_i)
        let rho = rot_from_coeffs(&a_coeffs_f, ring.phi_coeffs);
        out.push(rho);
    }

    Ok(out)
}

/// Typed variant of `sample_rot_rhos_n` returning validated `RotRho` values.
pub fn sample_rot_rhos_n_typed(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    ring: &RotRing,
    count: usize,
) -> Result<Vec<RotRho>, PiCcsError> {
    let mats = sample_rot_rhos_n(tr, params, ring, count)?;
    Ok(mats.into_iter().map(RotRho::new_unchecked).collect())
}

/// Minimum `k_rho` satisfying the ΠRLC norm bound for a given batch count.
///
/// Finds the smallest `k` such that:
/// `count · T · (b - 1) < b^k`
/// where `T` is derived from the strong-set alphabet (Theorem 3).
pub fn min_k_rho_for_rlc_count(params: &NeoParams, ring: &RotRing, count: usize) -> Result<u32, PiCcsError> {
    if count == 0 {
        return Err(PiCcsError::InvalidInput("count must be > 0".into()));
    }
    let b = params.b as u128;
    if b < 2 {
        return Err(PiCcsError::InvalidInput(format!("invalid base b={}", params.b)));
    }
    let lhs = (count as u128) * expansion_factor_T(ring.alphabet) * (b.saturating_sub(1));

    let mut k: u32 = 0;
    let mut pow: u128 = 1;
    while lhs >= pow {
        k = k
            .checked_add(1)
            .ok_or_else(|| PiCcsError::InvalidInput("k_rho overflow while computing ΠRLC bound".into()))?;
        pow = pow
            .checked_mul(b)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("b^k overflow while computing ΠRLC bound: b={b}")))?;
    }
    Ok(k)
}

// ---------------------------------------------------------------------------
// ME Relation Helpers
// ---------------------------------------------------------------------------

/// Concrete witness-matrix layouts currently accepted by reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessMatLayout {
    /// SuperNeo packed layout: `Z ∈ F^{D×(m/D)}`.
    SuperneoPacked,
    /// Compatibility layout: `Z ∈ F^{D×m}` (one logical column per matrix column).
    DenseUnpacked,
}

/// Classify a witness matrix shape against the expected CCS width.
pub fn witness_mat_layout<Ff>(Z: &Mat<Ff>, expected_m: usize) -> Result<WitnessMatLayout, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if Z.rows() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "witness_mat_layout: expected Z.rows()={}, got {}",
            D,
            Z.rows()
        )));
    }
    if expected_m == 0 {
        return Err(PiCcsError::InvalidInput(
            "witness_mat_layout: expected_m must be > 0".into(),
        ));
    }
    let want_cols = expected_m.div_ceil(D);
    if Z.cols() == want_cols {
        // NOTE: mixed witnesses (e.g. after Π_RLC) can legitimately carry non-zero values in
        // padded tail lanes. We therefore classify layout by shape only here.
        return Ok(WitnessMatLayout::SuperneoPacked);
    }
    if Z.cols() == expected_m {
        return Ok(WitnessMatLayout::DenseUnpacked);
    }
    Err(PiCcsError::InvalidInput(format!(
        "witness_mat_layout: expected packed {}x{} or dense {}x{} witness for expected_m={expected_m}, got {}x{}",
        D,
        want_cols,
        D,
        expected_m,
        Z.rows(),
        Z.cols(),
    )))
}

/// Layout-aware `Z[rho, col]` access in the logical `D×expected_m` view.
#[inline]
pub fn witness_mat_get_f<Ff>(Z: &Mat<Ff>, layout: WitnessMatLayout, expected_m: usize, rho: usize, col: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if rho >= D || col >= expected_m {
        return Ff::ZERO;
    }
    match layout {
        WitnessMatLayout::SuperneoPacked => {
            let blk = col / D;
            let off = col % D;
            if off == rho {
                Z[(rho, blk)]
            } else {
                Ff::ZERO
            }
        }
        WitnessMatLayout::DenseUnpacked => Z[(rho, col)],
    }
}

/// Layout-aware `Z[rho, col]` lifted to `K`.
#[inline]
pub fn witness_mat_get_k<Ff>(Z: &Mat<Ff>, layout: WitnessMatLayout, expected_m: usize, rho: usize, col: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    K::from(witness_mat_get_f(Z, layout, expected_m, rho, col))
}

/// Layout-aware projection of the first `m_in` logical columns of `Z` into `X ∈ F^{D×m_in}`.
pub fn project_x_from_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize, m_in: usize) -> Result<Mat<Ff>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for rho in 0..D {
        for c in 0..m_in {
            X[(rho, c)] = witness_mat_get_f(Z, layout, expected_m, rho, c);
        }
    }
    Ok(X)
}

/// Build `X ∈ F^{D×m_in}` directly from public inputs `x` under SuperNeo packed semantics.
///
/// Column `c` stores `x[c]` at row `c % D`; all off-lane rows are zero.
pub fn project_x_from_public_inputs<Ff>(x: &[Ff], m_in: usize) -> Result<Mat<Ff>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if x.len() != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "project_x_from_public_inputs: x.len()={} does not match m_in={m_in}",
            x.len()
        )));
    }
    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for c in 0..m_in {
        X[(c % D, c)] = x[c];
    }
    Ok(X)
}

/// Decode a witness matrix into a field vector `z` under a known CCS width.
///
/// SuperNeo-only layout:
/// - packed layout `Z ∈ F^{D×(m/D)}` where `m == expected_m`.
pub fn decode_z_from_witness_mat<Ff>(_params: &NeoParams, Z: &Mat<Ff>, expected_m: usize) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    // SuperNeo packed layout: each column is one D-coefficient block.
    let mut z = vec![K::ZERO; expected_m];
    for c in 0..expected_m {
        let rho = c % D;
        z[c] = witness_mat_get_k(Z, layout, expected_m, rho, c);
    }
    Ok(z)
}

/// Decode packed witness coefficients including padded tail lanes.
///
/// Returns a vector of length `ceil(expected_m / D) * D`, so ring-linear operations
/// stay closed inside each `D`-coefficient block even when `expected_m % D != 0`.
pub fn decode_superneo_coeffs_from_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    let m_eff = expected_m.div_ceil(D) * D;
    let mut z = vec![K::ZERO; m_eff];
    match layout {
        WitnessMatLayout::SuperneoPacked => {
            // Keep all packed lanes (including padded tail) so RLC/DEC remain closed in block space.
            for (c, zc) in z.iter_mut().enumerate() {
                let blk = c / D;
                let off = c % D;
                if blk < Z.cols() {
                    *zc = K::from(Z[(off, blk)]);
                }
            }
        }
        WitnessMatLayout::DenseUnpacked => {
            for (c, zc) in z.iter_mut().enumerate().take(expected_m) {
                *zc = witness_mat_get_k(Z, layout, expected_m, c % D, c);
            }
        }
    }
    Ok(z)
}

#[inline]
fn i128_to_field_f<Ff>(v: i128) -> Ff
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
{
    if v >= 0 {
        Ff::from_u64(v as u64)
    } else {
        Ff::ZERO - Ff::from_u64((-v) as u64)
    }
}

/// Check whether one value is representable in exactly `D` balanced base-`b` digits.
#[inline]
fn is_representable_balanced_fixed_d_digits<Ff>(val: Ff, b: u32) -> Result<bool, PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
{
    if b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "is_representable_balanced_fixed_d_digits: invalid base b={b}"
        )));
    }

    let b_i = b as i128;
    let mut rem = to_balanced_i128(val);
    if rem >= i64::MIN as i128 && rem <= i64::MAX as i128 {
        let mut rem64 = rem as i64;
        let b_i64 = b as i64;
        for _ in 0..D {
            if rem64 == 0 {
                return Ok(true);
            }
            let (_, q) = balanced_divrem_i64(rem64, b_i64);
            rem64 = q;
        }
        return Ok(rem64 == 0);
    }

    for _ in 0..D {
        if rem == 0 {
            return Ok(true);
        }
        let (_, q) = balanced_divrem(rem, b_i);
        rem = q;
    }
    Ok(rem == 0)
}

/// Balanced base-`b` decomposition of one field value into exactly `D` digits.
///
/// Returns an error when the value is not representable with `D` balanced digits for this base.
pub fn decompose_balanced_fixed_d_digits_k<Ff>(val: Ff, b: u32) -> Result<[K; D], PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "decompose_balanced_fixed_d_digits_k: invalid base b={b}"
        )));
    }

    let mut rem = to_balanced_i128(val);
    let b_i = b as i128;
    let mut digits_f = [Ff::ZERO; D];
    for d in digits_f.iter_mut().take(D) {
        let (r_i, q) = balanced_divrem(rem, b_i);
        *d = i128_to_field_f(r_i);
        rem = q;
    }
    if rem != 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "value {} is not representable in D={} balanced digits for base b={}",
            to_balanced_i128(val),
            D,
            b
        )));
    }

    let mut out = [K::ZERO; D];
    for rho in 0..D {
        out[rho] = K::from(digits_f[rho]);
    }
    Ok(out)
}

/// Build NC digit rows (`D` digits per logical column) for a witness matrix.
///
/// SuperNeo packed layout: decomposes each packed logical value into `D` balanced base-`b` digits.
pub fn build_witness_nc_digit_table<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
) -> Result<Vec<[K; D]>, PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    let mut out = vec![[K::ZERO; D]; expected_m];
    for (col, dst) in out.iter_mut().enumerate().take(expected_m) {
        let raw = witness_mat_get_f(Z, layout, expected_m, col % D, col);
        *dst = decompose_balanced_fixed_d_digits_k(raw, params.b)
            .map_err(|e| PiCcsError::InvalidInput(format!("witness logical_col={col} decomposition failed: {e}")))?;
    }

    Ok(out)
}

/// Compute NC channel opening `y_zcol := Z_digits · χ_s`, padded to `d_pad`.
///
/// `Z_digits` is the balanced decomposition rows for SuperNeo packed layout.
pub fn compute_y_zcol_from_witness_digits<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    chi_s: &[K],
    d_pad: usize,
) -> Result<Vec<K>, PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let digits_by_col = build_witness_nc_digit_table(params, Z, expected_m)?;
    let mut yz = vec![K::ZERO; d_pad.max(D)];
    for col in 0..expected_m {
        let w = chi_s.get(col).copied().unwrap_or(K::ZERO);
        if w == K::ZERO {
            continue;
        }
        for rho in 0..D {
            yz[rho] += digits_by_col[col][rho] * w;
        }
    }
    yz.truncate(d_pad);
    Ok(yz)
}

/// Compute linear channel opening `y_zcol := Z · χ_s`, padded to `d_pad`.
///
/// This projection is linear in `Z`, so it composes directly under Π_RLC/Π_DEC.
pub fn compute_y_zcol_from_witness<Ff>(
    _params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    chi_s: &[K],
    d_pad: usize,
) -> Result<Vec<K>, PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    let mut yz = vec![K::ZERO; d_pad.max(D)];
    for col in 0..expected_m {
        let w = chi_s.get(col).copied().unwrap_or(K::ZERO);
        if w == K::ZERO {
            continue;
        }
        let off = col % D;
        yz[off] += witness_mat_get_k(Z, layout, expected_m, off, col) * w;
    }
    yz.truncate(d_pad);
    Ok(yz)
}

/// Enforce NC-range compatibility for SuperNeo packed witnesses.
pub fn validate_packed_witness_nc_range<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let layout = witness_mat_layout(Z, expected_m)?;
    if params.b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: invalid b={} (must be >= 2)",
            params.b
        )));
    }
    for col in 0..expected_m {
        let off = col % D;
        let v = witness_mat_get_f(Z, layout, expected_m, off, col);
        if !is_representable_balanced_fixed_d_digits(v, params.b)? {
            let x = to_balanced_i128(v);
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: witness logical_col={col} is not representable in D={} balanced base-{} digits (centered value {})",
                D,
                params.b,
                x,
            )));
        }
    }
    Ok(())
}

/// Compute one scalar opening `ct` from a ring-digit row under SuperNeo semantics.
///
/// SuperNeo semantics: `ct` is the constant coefficient.
#[inline]
pub fn ct_from_y_digits(y_digits: &[K]) -> K {
    y_digits.first().copied().unwrap_or(K::ZERO)
}

/// Compute one scalar opening `ct` from a ring-digit row for a concrete CCS width.
#[inline]
pub fn ct_from_y_digits_for_ccs_m(y_digits: &[K], _params: &NeoParams, expected_m: usize) -> K {
    debug_assert!(expected_m > 0);
    ct_from_y_digits(y_digits)
}

#[inline]
pub fn ct_from_y_ring(y_ring: &[Vec<K>]) -> Vec<K> {
    y_ring.iter().map(|row| ct_from_y_digits(row)).collect()
}

/// Compute scalar openings `ct` from all ring-digit rows for a concrete CCS width.
#[inline]
pub fn ct_from_y_ring_for_ccs_m(y_ring: &[Vec<K>], params: &NeoParams, expected_m: usize) -> Vec<K> {
    y_ring
        .iter()
        .map(|row| ct_from_y_digits_for_ccs_m(row, params, expected_m))
        .collect()
}

/// Compute y from Z and r according to the ME relation: y_j := Z · (M_j^T · r^b).
///
/// Returns (y, y_scalars) where:
/// - y[j] is padded to 2^{ell_d} and contains the first D digits
/// - y_scalars[j] is the SuperNeo constant term
pub fn compute_y_from_Z_and_r<Ff>(
    s: &CcsStructure<Ff>,
    Z: &Mat<Ff>,
    r: &[K],
    ell_d: usize,
    _b: u32,
) -> (Vec<Vec<K>>, Vec<K>)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    use neo_ccs::CcsMatrix;
    let d_pad = 1usize << ell_d;
    let mut y_new: Vec<Vec<K>> = Vec::with_capacity(s.t());
    let z_layout = witness_mat_layout(Z, s.m)
        .unwrap_or_else(|e| panic!("compute_y_from_Z_and_r: invalid witness shape for m={}: {e}", s.m));
    // Build r^b over rows
    let rb = neo_ccs::utils::tensor_point::<K>(r);
    if let Some(cache) = crate::superneo_eval::build_superneo_eval_cache(s) {
        // SuperNeo fast path: evaluate cached transformed rows against decoded packed witness.
        let n_eff = core::cmp::min(s.n, rb.len());
        let z_vec = decode_superneo_coeffs_from_witness_mat(Z, s.m)
            .unwrap_or_else(|e| panic!("compute_y_from_Z_and_r: failed to decode packed witness coefficients: {e}"));
        let y_ring = crate::superneo_eval::eval_all_mats_ring_cached(&cache, &z_vec, &rb, n_eff);
        for coeffs in y_ring.into_iter().take(s.t()) {
            let mut yj_pad = coeffs.to_vec();
            if d_pad > yj_pad.len() {
                yj_pad.resize(d_pad, K::ZERO);
            }
            y_new.push(yj_pad);
        }
    } else {
        // Fallback path: explicitly assemble v_j = M_j^T · r^b and then y_j = Z · v_j.
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

        for j in 0..s.t() {
            let mut yj_pad = vec![K::ZERO; D];
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += witness_mat_get_k(Z, z_layout, s.m, rho, c) * vjs[j][c];
                }
                yj_pad[rho] = acc;
            }
            if d_pad > yj_pad.len() {
                yj_pad.resize(d_pad, K::ZERO);
            }
            y_new.push(yj_pad);
        }
    }
    let y_scalars = ct_from_y_ring(&y_new);
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
