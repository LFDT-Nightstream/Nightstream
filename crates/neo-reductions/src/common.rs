//! Common utilities and helper functions shared across engines.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, Mat};
use neo_math::{
    balanced::to_balanced_i128, balanced::within_nc_bound, superneo_bar_block, Fq, KExtensions, Rq, D, F, K,
};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::error::PiCcsError;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
mod packed_split;
mod pi_rlc_sampler;
pub use pi_rlc_sampler::decode_pi_rlc_coefficients;

#[inline]
fn signed_divrem_base2(v: i64) -> (i64, i64) {
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

#[derive(Clone, Debug, PartialEq)]
pub struct SplitDigits {
    planes: Vec<Mat<F>>,
    nonzero: Vec<bool>,
}

impl SplitDigits {
    pub fn from_device_split(planes: Vec<Mat<F>>, nonzero: Vec<bool>) -> Self {
        assert_eq!(
            planes.len(),
            nonzero.len(),
            "split digit flag count must match the plane count"
        );
        Self { planes, nonzero }
    }

    pub fn planes(&self) -> &[Mat<F>] {
        &self.planes
    }

    pub fn nonzero(&self) -> &[bool] {
        &self.nonzero
    }

    pub fn into_planes(self) -> Vec<Mat<F>> {
        self.planes
    }

    pub fn into_parts(self) -> (Vec<Mat<F>>, Vec<bool>) {
        (self.planes, self.nonzero)
    }
}

/// SuperNeo v1.2 Section 4 signed `split_b`.
pub fn split_b_matrix_k_with_nonzero_flags(Z: &Mat<F>, k: usize, b: u32) -> Result<SplitDigits, PiCcsError> {
    let Z_rows = Z.rows();
    let Z_cols = Z.cols();

    if Z.virtual_constant_value()
        .is_some_and(|value| *value == F::ZERO)
    {
        return Ok(SplitDigits {
            planes: (0..k)
                .map(|_| Mat::virtual_constant(Z_rows, Z_cols, F::ZERO))
                .collect(),
            nonzero: vec![false; k],
        });
    }

    let b_i = b as i128;
    let mut B: i128 = 1;
    for _ in 0..k {
        B = B.saturating_mul(b_i);
    }
    let B_u: u128 = B as u128;
    let z_data = Z.as_slice();
    let out_of_range = |idx: usize| {
        let u = z_data[idx].as_canonical_u64();
        PiCcsError::ProtocolError(format!(
            "DEC split: Z[{},{}] = {} (0x{:X}) is out of range for k_rho={}, b={}\n\
             Matrix Z is {}×{}\n\
             Balanced range: [{}, {}), where B = b^k_rho = {}^{} = {}\n\
             This typically means witness values grew too large during RLC for the configured rotation challenge set",
            idx / Z_cols,
            idx % Z_cols,
            u,
            u,
            k,
            b,
            Z_rows,
            Z_cols,
            -B,
            B,
            b,
            k,
            B_u
        ))
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if b == 2 && Z_rows <= u64::BITS as usize && (1..i64::BITS as usize - 1).contains(&k) {
        return packed_split::split_base2_packed(Z, k)
            .map(|(planes, nonzero)| SplitDigits { planes, nonzero })
            .map_err(out_of_range);
    }

    let mut out_data = (0..k).map(|_| None::<Vec<F>>).collect::<Vec<_>>();
    let packed_binary = b == 2 && Z_rows <= u64::BITS as usize;
    let mut out_masks = if packed_binary {
        (0..k).map(|_| None::<(Vec<u64>, Vec<u64>)>).collect()
    } else {
        Vec::new()
    };
    let mut digit_nonzero = vec![false; k];
    let (digit_bound, digit_lut) = build_balanced_digit_lut(b);
    let p: u128 = F::ORDER_U64 as u128;
    let total = z_data.len();
    debug_assert_eq!(total, Z_rows * Z_cols);

    for idx in 0..total {
        let z_entry = z_data[idx];
        if z_entry == F::ZERO {
            continue;
        }
        let u = z_entry.as_canonical_u64() as u128;
        let neg_mag = p.saturating_sub(u);
        let mut v = match (u < B_u, neg_mag < B_u) {
            (false, false) => return Err(out_of_range(idx)),
            (true, false) => u as i128,
            (false, true) => -(neg_mag as i128),
            (true, true) if u <= neg_mag => u as i128,
            (true, true) => -(neg_mag as i128),
        };
        for plane in 0..k {
            if v == 0 {
                break;
            }
            let (r_i, q) = (v % b_i, v / b_i);
            if r_i != 0 {
                let r_i = r_i as i64;
                debug_assert!(r_i >= -digit_bound && r_i <= digit_bound);
                let digit = digit_lut[(r_i + digit_bound) as usize];
                if packed_binary {
                    let (positive, negative) =
                        out_masks[plane].get_or_insert_with(|| (vec![0; Z_cols], vec![0; Z_cols]));
                    let mask = if digit == F::ONE { positive } else { negative };
                    mask[idx % Z_cols] |= 1u64 << (idx / Z_cols);
                } else {
                    out_data[plane].get_or_insert_with(|| vec![F::ZERO; total])[idx] = digit;
                }
                digit_nonzero[plane] = true;
            }
            v = q;
        }
        if v != 0 {
            return Err(out_of_range(idx));
        }
    }

    if packed_binary {
        let digits = out_masks
            .into_iter()
            .map(|masks| match masks {
                None => Mat::virtual_constant(Z_rows, Z_cols, F::ZERO),
                Some((positive, negative)) => {
                    Mat::compact_signed_unit_from_column_masks(Z_rows, Z_cols, &positive, &negative)
                        .expect("binary split writes disjoint in-range row masks")
                }
            })
            .collect();
        return Ok(SplitDigits {
            planes: digits,
            nonzero: digit_nonzero,
        });
    }

    let outs = out_data
        .into_iter()
        .map(|data| {
            data.map_or_else(
                || Mat::virtual_constant(Z_rows, Z_cols, F::ZERO),
                |data| Mat::from_row_major(Z_rows, Z_cols, data),
            )
        })
        .collect();
    Ok(SplitDigits {
        planes: outs,
        nonzero: digit_nonzero,
    })
}

pub fn split_b_matrix_k(Z: &Mat<F>, k: usize, b: u32) -> Result<Vec<Mat<F>>, PiCcsError> {
    split_b_matrix_k_with_nonzero_flags(Z, k, b).map(SplitDigits::into_planes)
}

pub struct RotRing {
    pub phi_low_coeffs: &'static [i32; D],

    pub alphabet: &'static [i8],

    /// Optional: lower bound on b_inv from SuperNeo v1.2 Theorem 4 (invertibility threshold).
    /// If provided, enforces Δ_A < b_inv where Δ_A = max(A) - min(A).
    pub binv_floor: Option<u64>,
}

impl RotRing {
    /// SuperNeo v1.2 Section 8.2 ring and challenge set.
    pub const fn goldilocks() -> Self {
        Self {
            phi_low_coeffs: &goldilocks_paper_b2::PHI_COEFFS,
            alphabet: &goldilocks_paper_b2::CHALLENGE_ALPHABET,
            binv_floor: Some(goldilocks_paper_b2::B_INV_FLOOR),
        }
    }
}

/// Compute expansion factor T per SuperNeo v1.2 Theorem 5: T ≤ 2·φ(η)·max|coeff|.
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

#[inline]
fn ff_from_i64<Ff: Field + PrimeCharacteristicRing>(x: i64) -> Ff {
    if x >= 0 {
        Ff::from_u64(x as u64)
    } else {
        Ff::ZERO - Ff::from_u64((-x) as u64)
    }
}

#[inline]
fn f_from_i64(x: i64) -> F {
    ff_from_i64::<F>(x)
}

fn rot_from_coeffs(a_coeffs: &[F], phi_low_coeffs: &[i32; D]) -> Mat<F> {
    debug_assert_eq!(a_coeffs.len(), D);

    let neg_c: Vec<F> = phi_low_coeffs
        .iter()
        .map(|&cr| f_from_i64(-(cr as i64)))
        .collect();

    let mut rho = Mat::zero(D, D, F::ZERO);
    let mut col = a_coeffs.to_vec();

    for j in 0..D {
        for r in 0..D {
            rho[(r, j)] = col[r];
        }

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

pub fn phi_low_coeffs_from_params(params: &NeoParams) -> Result<&'static [i32; D], PiCcsError> {
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

pub fn validate_rho_is_rotation_matrix<Ff>(
    rho: &Mat<Ff>,
    phi_low_coeffs: &[i32; D],
    label: &str,
) -> Result<(), PiCcsError>
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
    let neg_c: Vec<Ff> = phi_low_coeffs
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

pub fn validate_rhos_are_rotation_matrices<Ff>(
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let phi = phi_low_coeffs_from_params(params)?;
    for (idx, rho) in rhos.iter().enumerate() {
        validate_rho_is_rotation_matrix(rho, phi, &format!("{label}[{idx}]"))?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RotRho(pub(crate) Mat<F>);

impl RotRho {
    pub fn new_checked(params: &NeoParams, rho: Mat<F>) -> Result<Self, PiCcsError> {
        let phi = phi_low_coeffs_from_params(params)?;
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

pub fn rot_rhos_from_mats(params: &NeoParams, rhos: &[Mat<F>], label: &str) -> Result<Vec<RotRho>, PiCcsError> {
    rhos.iter()
        .cloned()
        .enumerate()
        .map(|(index, rho)| {
            let phi = phi_low_coeffs_from_params(params)?;
            let item_label = format!("{label}[{index}]");
            validate_rho_is_rotation_matrix(&rho, phi, &item_label)?;
            validate_rho_is_in_selected_strong_set(params, &rho, &item_label)?;
            Ok(RotRho::new_unchecked(rho))
        })
        .collect()
}

pub fn rot_rhos_to_mats(rhos: &[RotRho]) -> Vec<Mat<F>> {
    rhos.iter().map(|rho| rho.as_mat().clone()).collect()
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

fn sample_rot_rhos_n(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    ring: &RotRing,
    count: usize,
) -> Result<Vec<Mat<F>>, PiCcsError> {
    validate_sampling_alphabet(ring.alphabet)?;
    if count == 0 {
        return Err(PiCcsError::InvalidInput("count must be > 0".into()));
    }
    if tr.absorbed() != 0 {
        return Err(PiCcsError::InvalidInput(
            "PiRLC v1_2 sampler requires a zero transcript absorb cursor".into(),
        ));
    }

    if let Some(binv) = ring.binv_floor {
        let min = *ring.alphabet.iter().min().unwrap() as i64;
        let max = *ring.alphabet.iter().max().unwrap() as i64;
        let delta_a = (max - min).unsigned_abs();
        if delta_a >= binv {
            return Err(PiCcsError::InvalidInput(format!(
                "Strong-set check failed: Δ_A = {} must be < b_inv = {} (SuperNeo v1.2 Theorem 4)",
                delta_a, binv
            )));
        }
    }
    if ring.alphabet != goldilocks_paper_b2::CHALLENGE_ALPHABET.as_slice() {
        return Err(PiCcsError::InvalidInput(
            "PiRLC sampler requires the fixed alphabet [-2, -1, 0, 1, 2]".into(),
        ));
    }

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
             T={} is the expansion factor (SuperNeo v1.2 Theorem 5)\n\
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

    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let coordinate =
            u64::try_from(i).map_err(|_| PiCcsError::InvalidInput("PiRLC challenge coordinate exceeds u64".into()))?;
        tr.absorb_v1_2(&[F::from_u64(4), F::from_u64(coordinate)]);
        let digest = tr.squeeze_digest_v1_2();
        let coeffs_i8 = decode_pi_rlc_coefficients(&digest);

        let a_coeffs_f: Vec<F> = coeffs_i8.iter().map(|&c| f_from_i64(c as i64)).collect();

        let rho = rot_from_coeffs(&a_coeffs_f, ring.phi_low_coeffs);
        out.push(rho);
    }

    Ok(out)
}

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

#[inline]
pub fn superneo_carrier_width(logical_width: usize) -> usize {
    logical_width.div_ceil(D) * D
}

pub fn validate_superneo_witness_shape<Ff>(Z: &Mat<Ff>, expected_m: usize) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if Z.rows() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "validate_superneo_witness_shape: expected Z.rows()={}, got {}",
            D,
            Z.rows()
        )));
    }
    if expected_m == 0 {
        return Err(PiCcsError::InvalidInput(
            "validate_superneo_witness_shape: expected_m must be > 0".into(),
        ));
    }
    let want_cols = expected_m.div_ceil(D);
    if Z.cols() == want_cols {
        return Ok(());
    }
    Err(PiCcsError::InvalidInput(format!(
        "validate_superneo_witness_shape: expected packed SuperNeo {}x{} witness for expected_m={expected_m}, got {}x{}",
        D,
        want_cols,
        Z.rows(),
        Z.cols(),
    )))
}

pub fn validate_fresh_witness_tail_zero<Ff>(Z: &Mat<Ff>, logical_width: usize, label: &str) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_shape(Z, logical_width)?;
    for carrier_col in logical_width..superneo_carrier_width(logical_width) {
        if Z[(carrier_col % D, carrier_col / D)] != Ff::ZERO {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: fresh carrier_col={carrier_col} must be zero"
            )));
        }
    }
    Ok(())
}

pub fn project_x_from_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize, m_in: usize) -> Result<Mat<Ff>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_shape(Z, expected_m)?;
    if m_in > expected_m {
        return Err(PiCcsError::InvalidInput(format!(
            "project_x_from_witness_mat: m_in={m_in} exceeds expected_m={expected_m}"
        )));
    }
    if m_in % D != 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "project_x_from_witness_mat: m_in={m_in} is not a whole number of degree-{D} ring elements"
        )));
    }
    let required_cols = m_in / D;
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

pub fn decode_z_from_witness_mat<Ff>(_params: &NeoParams, Z: &Mat<Ff>, expected_m: usize) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    validate_superneo_witness_shape(Z, expected_m)?;
    let mut z = vec![K::ZERO; expected_m];
    for c in 0..expected_m {
        z[c] = K::from(Z[(c % D, c / D)]);
    }
    Ok(z)
}

pub fn decode_superneo_coeffs_from_witness_mat<Ff>(Z: &Mat<Ff>, expected_m: usize) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    validate_superneo_witness_shape(Z, expected_m)?;
    let m_eff = superneo_carrier_width(expected_m);
    let mut z = vec![K::ZERO; m_eff];
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
                signed_divrem_base2(rem64)
            } else {
                (rem64 % b_i64, rem64 / b_i64)
            };
            *d = i128_to_field_f(r_i as i128);
            rem64 = q;
        }
        rem = rem64 as i128;
    } else {
        for d in digits_f.iter_mut().take(D) {
            *d = i128_to_field_f(rem % b_i);
            rem /= b_i;
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
            rem64 = if b == 2 {
                signed_divrem_base2(rem64).1
            } else {
                rem64 / b_i64
            };
        }
        return Ok(rem64 == 0);
    }

    for _ in 0..D {
        if rem == 0 {
            return Ok(true);
        }
        rem /= b_i;
    }
    Ok(rem == 0)
}

pub fn validate_packed_witness_dec_representable<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    validate_superneo_witness_shape(Z, expected_m)?;
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

pub fn validate_packed_witness_nc_alphabet<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
    label: &str,
) -> Result<(), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy,
{
    validate_superneo_witness_shape(Z, expected_m)?;
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

pub fn compute_v1_2_evaluations_from_z_and_r<Ff>(
    s: &CcsStructure<Ff>,
    z: &Mat<Ff>,
    r: &[K],
    ell_d: usize,
) -> neo_ccs::V1_2Evaluations<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let rb = neo_ccs::utils::tensor_point_parallel::<K>(r);
    let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s);
    compute_v1_2_evaluations_from_z_and_rb_with_cache(s, z, &rb, ell_d, superneo_cache.as_ref())
}

pub fn compute_v1_2_evaluations_from_z_and_rb_with_cache<Ff>(
    s: &CcsStructure<Ff>,
    Z: &Mat<Ff>,
    rb: &[K],
    ell_d: usize,
    superneo_cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
) -> neo_ccs::V1_2Evaluations<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let local_cache;
    let cache = if let Some(cache) = superneo_cache {
        cache
    } else {
        local_cache = crate::superneo_eval::build_superneo_eval_cache(s)
            .expect("v1_2 evaluation cache must build for valid CCS width");
        &local_cache
    };
    let z_vec = decode_superneo_coeffs_from_witness_mat(Z, s.m)
        .unwrap_or_else(|e| panic!("v1_2 evaluation failed to decode packed witness coefficients: {e}"));
    let z_blocks = crate::superneo_eval::SuperneoZBlocks::from_z(&z_vec);
    let d_pad = 1usize << ell_d;
    let mut eval_k = identity_ring_mle(&z_vec, rb).to_vec();
    eval_k.resize(d_pad, K::ZERO);

    let n_eff = core::cmp::min(s.n, rb.len());
    let application_images = crate::superneo_eval::eval_all_mats_ring_cached_with_blocks(cache, &z_blocks, rb, n_eff);
    let eval_a = application_images
        .into_iter()
        .take(s.t())
        .map(|coefficients| {
            let mut row = coefficients.to_vec();
            row.resize(d_pad, K::ZERO);
            row
        })
        .collect();
    neo_ccs::V1_2Evaluations { eval_k, eval_a }
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

pub fn format_ext(x: K) -> String {
    let coeffs = x.as_coeffs();
    format!("({}, {})", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
}
