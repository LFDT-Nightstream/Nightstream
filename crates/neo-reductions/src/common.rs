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
use neo_math::{
    balanced::to_balanced_i128, balanced::within_nc_bound, superneo_bar_block, Fq, KExtensions, Rq, D, F, K,
};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use crate::error::PiCcsError;

// ---------------------------------------------------------------------------
// Balanced Base-b Digit Splitting
// ---------------------------------------------------------------------------

/// Helper: returns (r, q) with r in the SuperNeo digit range `|r| < b`.
///
/// Matches the Ajtai decomp_b balanced style (Definition 11):
/// digits are signed, but the norm bound is the full `{-(b-1), ..., +(b-1)}`
/// alphabet required by `split_b`.
///
/// This ensures termination for both positive and negative values.
fn balanced_divrem(v: i128, b: i128) -> (i128, i128) {
    debug_assert!(b >= 2);

    let r = v % b;
    let q = (v - r) / b;
    (r, q)
}

#[inline]
fn balanced_divrem_i64(v: i64, b: i64) -> (i64, i64) {
    debug_assert!(b >= 2);

    let r = v % b;
    let q = (v - r) / b;
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
    let bound = (b as i64) - 1;
    let mut lut = Vec::with_capacity((2 * bound + 1) as usize);
    for d in -bound..=bound {
        let f = if d >= 0 {
            F::from_u64(d as u64)
        } else {
            F::ZERO - F::from_u64((-d) as u64)
        };
        lut.push(f);
    }
    (bound, lut)
}

/// Split Z into **balanced base-b digits** Z = Σ_{i=0}^{k-1} b^i · Z_i, entrywise.
/// Each digit lies in `{-(b-1), ..., +(b-1)}`, so `||Z_i||_∞ < b`.
/// Returns an error if an entry cannot be represented within k digits (i.e., if |value| ≥ b^k)
/// — this indicates a bad RLC sample or overflow.
pub fn split_b_matrix_k_with_nonzero_flags(
    Z: &Mat<F>,
    k: usize,
    b: u32,
) -> Result<(Vec<Mat<F>>, Vec<bool>), PiCcsError> {
    let Z_rows = Z.rows();
    let Z_cols = Z.cols();

    if Z.virtual_constant_value()
        .is_some_and(|value| *value == F::ZERO)
    {
        return Ok((
            (0..k)
                .map(|_| Mat::virtual_constant(Z_rows, Z_cols, F::ZERO))
                .collect(),
            vec![false; k],
        ));
    }

    let mut out_data = (0..k).map(|_| None::<Vec<F>>).collect::<Vec<_>>();
    let mut digit_nonzero = vec![false; k];

    let b_i = b as i128;
    let mut B: i128 = 1;
    for _ in 0..k {
        B = B.saturating_mul(b_i);
    } // b^k
    let (digit_bound, digit_lut) = build_balanced_digit_lut(b);

    // Helpers to interpret field element as a small signed integer in (-(B-1), B-1)
    let p: u128 = F::ORDER_U64 as u128; // Goldilocks prime fits in u64
    let B_u: u128 = B as u128;

    let z_data = Z.as_slice();
    {
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
                             This typically means witness values grew too large during RLC for the configured rotation challenge set",
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
                        debug_assert!(r_i >= -digit_bound && r_i <= digit_bound);
                        let digit_f = digit_lut[(r_i + digit_bound) as usize];
                        out_data[i].get_or_insert_with(|| vec![F::ZERO; total])[idx] = digit_f;
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
                         This typically means witness values grew too large during RLC for the configured rotation challenge set",
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
                             This typically means witness values grew too large during RLC for the configured rotation challenge set",
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
                            debug_assert!(r_i >= -digit_bound && r_i <= digit_bound);
                            let digit_f = digit_lut[(r_i + digit_bound) as usize];
                            out_data[i].get_or_insert_with(|| vec![F::ZERO; total])[idx] = digit_f;
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
                             This typically means witness values grew too large during RLC for the configured rotation challenge set",
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
                        debug_assert!(r_i64 >= -digit_bound && r_i64 <= digit_bound);
                        let digit_f = digit_lut[(r_i64 + digit_bound) as usize];
                        out_data[i].get_or_insert_with(|| vec![F::ZERO; total])[idx] = digit_f;
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
                         This typically means witness values grew too large during RLC for the configured rotation challenge set",
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

    let outs = out_data
        .into_iter()
        .map(|data| {
            data.map_or_else(
                || Mat::virtual_constant(Z_rows, Z_cols, F::ZERO),
                |data| {
                    if b == 2 {
                        Mat::compact_signed_unit(Z_rows, Z_cols, data)
                    } else {
                        Mat::from_row_major(Z_rows, Z_cols, data)
                    }
                },
            )
        })
        .collect();
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
/// used to construct the strong sampling set C = {rot(a) : a ∈ C_R}.
/// C_R = {a ∈ R_q : all coeffs of a lie in A}.
pub struct RotRing {
    /// Coefficients [c_0, c_1, ..., c_{d-1}] of Φ_η(X) = X^d + c_{d-1}·X^{d-1} + ... + c_0.
    /// Must have length D (the ring dimension).
    pub phi_coeffs: &'static [i32],

    /// Small coefficient alphabet A ⊂ ℤ.
    /// The strong sampling set is C_R = {polynomials with coeffs in A}.
    pub alphabet: &'static [i8],

    /// Optional: lower bound on b_inv from Theorem 1 (invertibility threshold).
    /// If provided, enforces Δ_A < b_inv where Δ_A = max(A) - min(A).
    pub binv_floor: Option<u64>,
}

impl RotRing {
    /// Goldilocks Appendix B.2 profile, sourced from `neo_params::goldilocks_paper_b2`.
    pub const fn goldilocks() -> Self {
        Self {
            phi_coeffs: &goldilocks_paper_b2::PHI_COEFFS,
            alphabet: &goldilocks_paper_b2::CHALLENGE_ALPHABET,
            binv_floor: Some(goldilocks_paper_b2::B_INV_FLOOR),
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
        eta if eta as usize == goldilocks_paper_b2::ETA => Ok(&goldilocks_paper_b2::PHI_COEFFS),
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
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RotRho(pub(crate) Mat<F>);

impl RotRho {
    /// Construct a typed rho after strict ring and strong-set validation.
    pub fn new_checked(params: &NeoParams, rho: Mat<F>) -> Result<Self, PiCcsError> {
        let phi = phi_coeffs_from_params(params)?;
        validate_rho_is_rotation_matrix(&rho, phi, "RotRho::new_checked")?;
        validate_rho_is_in_selected_strong_set(params, &rho, "RotRho::new_checked")?;
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

impl<'de> serde::Deserialize<'de> for RotRho {
    fn deserialize<DeserializerT>(deserializer: DeserializerT) -> Result<Self, DeserializerT::Error>
    where
        DeserializerT: serde::Deserializer<'de>,
    {
        let rho = <Mat<F> as serde::Deserialize>::deserialize(deserializer)?;
        Self::new_checked(&NeoParams::goldilocks_paper_b2(), rho).map_err(serde::de::Error::custom)
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
    rhos.iter()
        .cloned()
        .enumerate()
        .map(|(index, rho)| {
            let phi = phi_coeffs_from_params(params)?;
            let item_label = format!("{label}[{index}]");
            validate_rho_is_rotation_matrix(&rho, phi, &item_label)?;
            validate_rho_is_in_selected_strong_set(params, &rho, &item_label)?;
            Ok(RotRho::new_unchecked(rho))
        })
        .collect()
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
fn draw_alphabet_vector(tr: &mut Poseidon2Transcript, need: usize, alphabet: &[i8], seed: u64) -> Vec<i8> {
    let m = alphabet.len() as u32;
    let bucket = (1u32 << 16) / m * m; // Largest multiple of m below 2^16

    let mut out = Vec::with_capacity(need);
    let mut ctr = seed;

    while out.len() < need {
        tr.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
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

fn draw_alphabet_vector_pow2(tr: &mut Poseidon2Transcript, need: usize, alphabet: &[i8], seed: u64) -> Vec<i8> {
    debug_assert!(alphabet.len().is_power_of_two());
    let bits_per_symbol = alphabet.len().trailing_zeros() as usize;
    let mask = (1u64 << bits_per_symbol) - 1;

    let mut out = Vec::with_capacity(need);
    let mut ctr = seed;
    while out.len() < need {
        tr.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
        let dig = tr.digest32();
        for limb in dig.chunks_exact(8) {
            let value = u64::from_le_bytes(limb.try_into().expect("digest32 limbs are 8 bytes"));
            let symbols = 64 / bits_per_symbol;
            for symbol_idx in 0..symbols {
                let idx = ((value >> (bits_per_symbol * symbol_idx)) & mask) as usize;
                out.push(alphabet[idx]);
                if out.len() == need {
                    break;
                }
            }
            if out.len() == need {
                break;
            }
        }
        ctr = ctr.wrapping_add(1);
    }
    out
}

fn validate_sampling_alphabet(alphabet: &[i8]) -> Result<(), PiCcsError> {
    if alphabet.len() < 2 {
        return Err(PiCcsError::InvalidInput(
            "strong-set alphabet must contain at least two distinct values".into(),
        ));
    }
    for (index, value) in alphabet.iter().enumerate() {
        if alphabet[..index].contains(value) {
            return Err(PiCcsError::InvalidInput(
                "strong-set alphabet contains a duplicate value".into(),
            ));
        }
    }
    Ok(())
}

fn validate_rho_is_in_selected_strong_set(params: &NeoParams, rho: &Mat<F>, label: &str) -> Result<(), PiCcsError> {
    let alphabet = &goldilocks_paper_b2::CHALLENGE_ALPHABET;
    let required_expansion = expansion_factor_T(alphabet);
    if (params.T as u128) < required_expansion {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: params.T={} is smaller than the selected strong-set expansion bound {required_expansion}",
            params.T
        )));
    }
    for coefficient in 0..D {
        let value = rho[(coefficient, 0)];
        if !alphabet
            .iter()
            .any(|&candidate| value == f_from_i64(candidate as i64))
        {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: first-column coefficient {coefficient} is outside the selected strong-set alphabet"
            )));
        }
    }
    Ok(())
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
    validate_sampling_alphabet(ring.alphabet)?;
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
        tr.append_fields_raw(&[F::from_u64(0), F::from_u64(i as u64)]);

        // Draw D coefficients from the strong-set alphabet.
        let coeffs_i8 = if ring.alphabet.len().is_power_of_two() {
            draw_alphabet_vector_pow2(tr, D, ring.alphabet, i as u64)
        } else {
            draw_alphabet_vector(tr, D, ring.alphabet, i as u64)
        };

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
    mats.into_iter()
        .map(|rho| RotRho::new_checked(params, rho))
        .collect()
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

/// Number of physical coefficients in the complete packed carrier.
#[inline]
pub fn superneo_carrier_width(logical_width: usize) -> usize {
    logical_width.div_ceil(D) * D
}

/// Validate the packed SuperNeo witness shape against the expected CCS width.
pub fn validate_superneo_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if Z.rows() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "validate_superneo_witness_mat: expected Z.rows()={}, got {}",
            D,
            Z.rows()
        )));
    }
    if expected_m == 0 {
        return Err(PiCcsError::InvalidInput(
            "validate_superneo_witness_mat: expected_m must be > 0".into(),
        ));
    }
    let want_cols = expected_m.div_ceil(D);
    if Z.cols() == want_cols {
        // NOTE: mixed witnesses (e.g. after Π_RLC) can legitimately carry non-zero values in
        // padded tail lanes. Shape is the only invariant checked here.
        return Ok(());
    }
    Err(PiCcsError::InvalidInput(format!(
        "validate_superneo_witness_mat: expected packed SuperNeo {}x{} witness for expected_m={expected_m}, got {}x{}",
        D,
        want_cols,
        Z.rows(),
        Z.cols(),
    )))
}

/// Fresh sources do not own the completed carrier tail.
pub fn validate_fresh_witness_tail_zero<Ff>(Z: &Mat<Ff>, logical_width: usize, label: &str) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_mat(Z, logical_width)?;
    for carrier_col in logical_width..superneo_carrier_width(logical_width) {
        if Z[(carrier_col % D, carrier_col / D)] != Ff::ZERO {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: fresh carrier_col={carrier_col} must be zero"
            )));
        }
    }
    Ok(())
}

/// Read `Z[rho, col]` in the logical `D×expected_m` view of a packed SuperNeo witness.
#[inline]
pub fn witness_mat_get_f<Ff>(Z: &Mat<Ff>, expected_m: usize, rho: usize, col: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if rho >= D || col >= expected_m {
        return Ff::ZERO;
    }
    let blk = col / D;
    let off = col % D;
    if off == rho {
        Z[(rho, blk)]
    } else {
        Ff::ZERO
    }
}

/// Read a packed SuperNeo witness coefficient and lift it to `K`.
#[inline]
pub fn witness_mat_get_k<Ff>(Z: &Mat<Ff>, expected_m: usize, rho: usize, col: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    K::from(witness_mat_get_f(Z, expected_m, rho, col))
}

/// Project the SuperNeo public-input ring slots of packed witness matrix `Z`
/// into the compact coefficient embedding `X ∈ F^{D×ceil(m_in/D)}`.
///
/// `m_in` counts public field elements, but SuperNeo carries public inputs
/// in packed ring columns. Rows in the final column that do not correspond
/// to a scalar public input are part of the ring slot: after RLC they can be
/// nonzero, and DEC must split and recombine them.
pub fn project_x_from_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize, m_in: usize) -> Result<Mat<Ff>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_mat(Z, expected_m)?;
    if m_in > expected_m {
        return Err(PiCcsError::InvalidInput(format!(
            "project_x_from_witness_mat: m_in={m_in} exceeds expected_m={expected_m}"
        )));
    }
    let required_cols = m_in.div_ceil(D);
    if required_cols > Z.cols() {
        return Err(PiCcsError::InvalidInput(format!(
            "project_x_from_witness_mat: m_in={m_in} needs {required_cols} packed columns, but Z has {}",
            Z.cols()
        )));
    }

    let mut X = Mat::zero(D, required_cols, Ff::ZERO);
    for col in 0..required_cols {
        for row in 0..D {
            X[(row, col)] = Z[(row, col)];
        }
    }
    Ok(X)
}

/// Decode a witness matrix into a field vector `z` under a known CCS width.
///
/// SuperNeo-only layout:
/// - packed layout `Z ∈ F^{D×ceil(m/D)}` where `m == expected_m`.
pub fn decode_z_from_witness_mat<Ff>(_params: &NeoParams, Z: &Mat<Ff>, expected_m: usize) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    validate_superneo_witness_mat(Z, expected_m)?;
    let mut z = vec![K::ZERO; expected_m];
    for c in 0..expected_m {
        let rho = c % D;
        z[c] = witness_mat_get_k(Z, expected_m, rho, c);
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
    validate_superneo_witness_mat(Z, expected_m)?;
    let m_eff = superneo_carrier_width(expected_m);
    let mut z = vec![K::ZERO; m_eff];
    // Keep all packed lanes, including padded tail lanes, so RLC/DEC stay closed in block space.
    for (c, zc) in z.iter_mut().enumerate() {
        let blk = c / D;
        let off = c % D;
        if blk < Z.cols() {
            *zc = K::from(Z[(off, blk)]);
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
    if val == Ff::ZERO {
        return Ok([K::ZERO; D]);
    }
    if val == Ff::ONE {
        let mut out = [K::ZERO; D];
        out[0] = K::ONE;
        return Ok(out);
    }

    let mut rem = to_balanced_i128(val);
    let b_i = b as i128;
    let mut digits_f = [Ff::ZERO; D];
    if rem >= i64::MIN as i128 && rem <= i64::MAX as i128 {
        let mut rem64 = rem as i64;
        let b_i64 = b as i64;
        for d in digits_f.iter_mut().take(D) {
            let (r_i, q) = if b == 2 {
                balanced_divrem_i64_base2(rem64)
            } else {
                balanced_divrem_i64(rem64, b_i64)
            };
            *d = i128_to_field_f(r_i as i128);
            rem64 = q;
        }
        rem = rem64 as i128;
    } else {
        for d in digits_f.iter_mut().take(D) {
            let (r_i, q) = balanced_divrem(rem, b_i);
            *d = i128_to_field_f(r_i);
            rem = q;
        }
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
            let (_, q) = if b == 2 {
                balanced_divrem_i64_base2(rem64)
            } else {
                balanced_divrem_i64(rem64, b_i64)
            };
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

/// Build NC rows (`D` coefficient lanes per logical column) for a witness matrix.
///
/// SuperNeo's NC polynomial checks the witness coefficient itself:
/// `range_product(z_i[col], b)`. It must not check a balanced base-`b`
/// decomposition of that coefficient; otherwise a high-norm value such as
/// `2` at `b=2` decomposes into low-norm digits and evades the NC check.
pub fn build_witness_nc_digit_table<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
) -> Result<Vec<[K; D]>, PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    build_witness_nc_digit_table_with_masks(params, Z, expected_m).map(|(digits, _masks)| digits)
}

pub fn build_witness_nc_digit_table_with_masks<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
) -> Result<(Vec<[K; D]>, Vec<u64>), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    validate_superneo_witness_mat(Z, expected_m)?;
    if params.b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "NC witness table: invalid b={} (must be >= 2)",
            params.b
        )));
    }
    let carrier_width = superneo_carrier_width(expected_m);
    let mut out = vec![[K::ZERO; D]; carrier_width];
    let mut masks = vec![0u64; carrier_width];
    let active_cols = expected_m.div_ceil(D);
    // Process column blocks in parallel; each block writes to disjoint
    // slices of `out` and `masks` (D contiguous columns per block).
    let process_block = |blk: usize, out_chunk: &mut [[K; D]], mask_chunk: &mut [u64]| -> Result<(), PiCcsError> {
        if blk >= active_cols {
            return Ok(());
        }
        for (rho, (dst, mask_slot)) in out_chunk.iter_mut().zip(mask_chunk.iter_mut()).enumerate() {
            let col = blk * D + rho;
            if col >= carrier_width {
                break;
            }
            let raw = Z[(rho, blk)];
            if raw == Ff::ZERO {
                continue;
            }
            dst[rho] = K::from(raw);
            *mask_slot = 1u64 << rho;
        }
        Ok(())
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_thread_index().is_none() {
            out.par_chunks_mut(D)
                .zip(masks.par_chunks_mut(D))
                .enumerate()
                .try_for_each(|(blk, (out_chunk, mask_chunk))| process_block(blk, out_chunk, mask_chunk))?;
        } else {
            for (blk, (out_chunk, mask_chunk)) in out.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
                process_block(blk, out_chunk, mask_chunk)?;
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for (blk, (out_chunk, mask_chunk)) in out.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
            process_block(blk, out_chunk, mask_chunk)?;
        }
    }

    Ok((out, masks))
}

/// Enforce DEC/RLC packed-witness representability.
///
/// This is not the Π_CCS low-norm predicate. It only checks that each entry
/// can be represented by the fixed `D`-digit balanced base-`b` machinery used
/// by split/decomposition paths.
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
    validate_superneo_witness_mat(Z, expected_m)?;
    if params.b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: invalid b={} (must be >= 2)",
            params.b
        )));
    }
    if Z.is_packed_signed_unit() {
        return Ok(());
    }
    if let Some(&value) = Z.virtual_constant_value() {
        if is_representable_balanced_fixed_d_digits(value, params.b)? {
            return Ok(());
        }
        let x = to_balanced_i128(value);
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: constant witness is not representable in D={} balanced base-{} digits (centered value {})",
            D, params.b, x,
        )));
    }
    for col in 0..superneo_carrier_width(expected_m) {
        let off = col % D;
        let v = Z[(off, col / D)];
        if !is_representable_balanced_fixed_d_digits(v, params.b)? {
            let x = to_balanced_i128(v);
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: witness carrier_col={col} is not representable in D={} balanced base-{} digits (centered value {})",
                D, params.b, x,
            )));
        }
    }
    Ok(())
}

/// Enforce the SuperNeo Π_CCS NC alphabet `|x| < b` on packed witness coefficients.
pub fn validate_packed_witness_nc_alphabet<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_mat(Z, expected_m)?;
    if params.b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: invalid b={} (must be >= 2)",
            params.b
        )));
    }
    for col in 0..superneo_carrier_width(expected_m) {
        let off = col % D;
        let v = Z[(off, col / D)];
        if !within_nc_bound(v, params.b) {
            let x = to_balanced_i128(v);
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: witness carrier_col={col} violates NC alphabet |x| < b={} (centered value {})",
                params.b, x,
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

/// Compute the selected identity-first CE images from `Z` and `r`.
///
/// Returns (y, y_scalars) where:
/// - y[0] is the padded identity image;
/// - y[j + 1] is the image of application matrix `M_j`;
/// - every row is padded to 2^{ell_d} and contains the first D digits;
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
    let rb = neo_ccs::utils::tensor_point_parallel::<K>(r);
    let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s);
    compute_y_from_Z_and_rb_with_cache(s, Z, &rb, ell_d, superneo_cache.as_ref())
}

/// Compute identity-first y from Z and a precomputed row tensor point `r^b`.
///
/// This variant enables callers to amortize the tensor-point and SuperNeo matrix-cache
/// construction across many ME claims that share `(s, r)`.
pub fn compute_y_from_Z_and_rb_with_cache<Ff>(
    s: &CcsStructure<Ff>,
    Z: &Mat<Ff>,
    rb: &[K],
    ell_d: usize,
    superneo_cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
) -> (Vec<Vec<K>>, Vec<K>)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let local_cache;
    let cache = if let Some(cache) = superneo_cache {
        cache
    } else {
        local_cache = crate::superneo_eval::build_superneo_eval_cache(s)
            .expect("compute_y_from_Z_and_r: SuperNeo evaluator cache must build for valid CCS width");
        &local_cache
    };
    let z_vec = decode_superneo_coeffs_from_witness_mat(Z, s.m)
        .unwrap_or_else(|e| panic!("compute_y_from_Z_and_r: failed to decode packed witness coefficients: {e}"));
    let z_blocks = crate::superneo_eval::SuperneoZBlocks::from_z(&z_vec);
    let d_pad = 1usize << ell_d;
    let mut y_new = Vec::with_capacity(s.t() + 1);
    let mut identity = identity_ring_mle(&z_vec, rb).to_vec();
    identity.resize(d_pad, K::ZERO);
    y_new.push(identity);

    let n_eff = core::cmp::min(s.n, rb.len());
    let application_images = crate::superneo_eval::eval_all_mats_ring_cached_with_blocks(cache, &z_blocks, rb, n_eff);
    for coefficients in application_images.into_iter().take(s.t()) {
        let mut row = coefficients.to_vec();
        row.resize(d_pad, K::ZERO);
        y_new.push(row);
    }
    let y_scalars = ct_from_y_ring(&y_new);
    (y_new, y_scalars)
}

fn identity_ring_mle(assignment: &[K], weights: &[K]) -> [K; D] {
    let mut output = [K::ZERO; D];
    for (row, &weight) in weights.iter().take(assignment.len()).enumerate() {
        let block = row / D;
        let mut basis = [Fq::ZERO; D];
        basis[row % D] = Fq::ONE;
        let transformed = Rq(superneo_bar_block(basis));
        let mut real = [Fq::ZERO; D];
        let mut imaginary = [Fq::ZERO; D];
        for lane in 0..D {
            let [low, high] = assignment[block * D + lane].as_coeffs();
            real[lane] = low;
            imaginary[lane] = high;
        }
        let real_product = transformed.mul(&Rq(real));
        let imaginary_product = transformed.mul(&Rq(imaginary));
        for coefficient in 0..D {
            output[coefficient] +=
                weight * K::from_coeffs([real_product.0[coefficient], imaginary_product.0[coefficient]]);
        }
    }
    output
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
