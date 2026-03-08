use neo_ajtai::{decomp_b_row_major, decomp_b_row_major_into, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_math::balanced::to_balanced_i128;
use neo_math::{D, F as BaseField};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField};

#[inline]
fn balanced_divrem(v: i128, b: i128) -> (i128, i128) {
    debug_assert!(b >= 2);
    let mut r = v % b;
    let mut q = (v - r) / b;
    let half = b / 2;
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }
    (r, q)
}

/// Encode a logical witness vector `z ∈ F^m` into the canonical SuperNeo packed matrix.
///
/// This convenience helper enforces SuperNeo packed layout `D×ceil(m/D)`.
pub fn encode_vector_balanced_to_mat(params: &NeoParams, z: &[BaseField]) -> Mat<BaseField> {
    encode_vector_for_ccs_m(params, z.len(), z).expect("encode_vector_balanced_to_mat: packed witness encoding")
}

/// Encode a vector `z ∈ F^m` into its Ajtai digit matrix `Z ∈ F^{d×m}` using **balanced** digits
/// and an explicit decomposition base.
pub fn encode_vector_balanced_to_mat_with_base(params: &NeoParams, z: &[BaseField], base: u32) -> Mat<BaseField> {
    let d = params.d as usize;
    debug_assert_eq!(
        d,
        neo_math::D,
        "Ajtai d mismatch: params.d={}, neo_math::D={}",
        params.d,
        neo_math::D
    );
    let m = z.len();

    // Row-major digits of shape d×m, balanced so recomposition equals z mod p.
    let row_major = decomp_b_row_major(z, base, d, DecompStyle::Balanced);
    Mat::from_row_major(d, m, row_major)
}

/// Reuse-capable row-major encoding for callers that immediately consume the digit slice.
pub fn encode_vector_balanced_to_row_major_with_base_into(
    params: &NeoParams,
    z: &[BaseField],
    base: u32,
    out: &mut Vec<BaseField>,
) {
    let d = params.d as usize;
    debug_assert_eq!(
        d,
        neo_math::D,
        "Ajtai d mismatch: params.d={}, neo_math::D={}",
        params.d,
        neo_math::D
    );
    decomp_b_row_major_into(z, base, d, DecompStyle::Balanced, out);
}

/// Returns true when the CCS width can be represented in SuperNeo packed layout.
#[inline]
pub fn uses_superneo_packed_layout(ccs_m: usize) -> bool {
    ccs_m > 0
}

/// Number of Ajtai witness columns implied by the active embedding layout.
#[inline]
pub fn commit_cols_for_ccs_m(ccs_m: usize) -> usize {
    ccs_m.div_ceil(D)
}

/// Encode a witness vector for a CCS width in SuperNeo-only mode.
///
/// Supports any `ccs_m > 0` and returns packed `D×ceil(ccs_m/D)` layout.
/// Coefficients beyond logical width `ccs_m` in the last block are zero-padded.
pub fn encode_vector_for_ccs_m(params: &NeoParams, ccs_m: usize, z: &[BaseField]) -> Result<Mat<BaseField>, String> {
    if z.len() != ccs_m {
        return Err(format!(
            "encode_vector_for_ccs_m: witness length {} != ccs_m {}",
            z.len(),
            ccs_m
        ));
    }
    if ccs_m == 0 {
        return Err("encode_vector_for_ccs_m: ccs_m must be > 0".into());
    }
    validate_packed_vector_nc_range(params, ccs_m, z, "encode_vector_for_ccs_m")?;
    let cols = ccs_m.div_ceil(D);
    let mut out = Mat::zero(D, cols, BaseField::ZERO);
    for c in 0..ccs_m {
        let blk = c / D;
        let rho = c % D;
        out[(rho, blk)] = z[c];
    }
    Ok(out)
}

/// Validate packed coefficients are representable in `D` balanced base-`b` digits.
pub fn validate_packed_vector_nc_range(
    params: &NeoParams,
    ccs_m: usize,
    z: &[BaseField],
    label: &str,
) -> Result<(), String> {
    if z.len() != ccs_m {
        return Err(format!("{label}: witness length {} != ccs_m {}", z.len(), ccs_m));
    }
    if ccs_m == 0 {
        return Err(format!("{label}: ccs_m must be > 0"));
    }
    if params.b < 2 {
        return Err(format!("{label}: invalid b={} (must be >= 2)", params.b));
    }

    let b_i = params.b as i128;
    for (idx, &v) in z.iter().enumerate() {
        let centered = to_balanced_i128(v);
        let mut rem = centered;
        for _ in 0..D {
            let (_r, q) = balanced_divrem(rem, b_i);
            rem = q;
        }
        if rem != 0 {
            return Err(format!(
                "{label}: packed coefficient at index {idx} is not representable in D={} balanced base-{} digits (centered value {})",
                D, params.b, centered
            ));
        }
    }
    Ok(())
}

/// Encode `z ∈ F^m` into SuperNeo packed layout `Z ∈ F^{D×(m/D)}`.
///
/// Each consecutive block `z[blk*D .. (blk+1)*D)` is placed as one matrix column,
/// preserving coefficient order by row index.
pub fn encode_vector_superneo_packed_to_mat(z: &[BaseField]) -> Result<Mat<BaseField>, String> {
    if !z.len().is_multiple_of(D) {
        return Err(format!(
            "encode_vector_superneo_packed_to_mat: z.len()={} is not divisible by D={D}",
            z.len()
        ));
    }
    let cols = z.len() / D;
    let mut out = Mat::zero(D, cols, BaseField::ZERO);
    for blk in 0..cols {
        let base = blk * D;
        for rho in 0..D {
            out[(rho, blk)] = z[base + rho];
        }
    }
    Ok(out)
}

/// Decode SuperNeo packed layout `Z ∈ F^{D×n}` back into `z ∈ F^{D*n}`.
pub fn decode_vector_superneo_packed_from_mat(mat: &Mat<BaseField>) -> Result<Vec<BaseField>, String> {
    if mat.rows() != D {
        return Err(format!(
            "decode_vector_superneo_packed_from_mat: mat.rows()={} expected D={D}",
            mat.rows()
        ));
    }
    let mut out = Vec::with_capacity(mat.cols() * D);
    for blk in 0..mat.cols() {
        for rho in 0..D {
            out.push(mat[(rho, blk)]);
        }
    }
    Ok(out)
}

/// Decode an Ajtai witness matrix for a known logical CCS width in SuperNeo-only mode.
///
/// Expects packed `D×ceil(ccs_m/D)` and flattens/truncates to `ccs_m`.
pub fn decode_vector_for_ccs_m<F: PrimeField>(
    _params: &NeoParams,
    ccs_m: usize,
    mat: &Mat<F>,
) -> Result<Vec<F>, String> {
    if mat.rows() != D {
        return Err(format!(
            "decode_vector_for_ccs_m: mat.rows()={} expected D={D}",
            mat.rows()
        ));
    }
    if ccs_m == 0 {
        return Err("decode_vector_for_ccs_m: ccs_m must be > 0".into());
    }
    let expected_cols = ccs_m.div_ceil(D);
    if mat.cols() != expected_cols {
        return Err(format!(
            "decode_vector_for_ccs_m: packed layout expects cols={} for ccs_m={}, got {}",
            expected_cols,
            ccs_m,
            mat.cols()
        ));
    }

    // Canonical packed layout requires padded tail coefficients to be zero.
    let pad_start = ccs_m;
    let pad_end = expected_cols
        .checked_mul(D)
        .ok_or_else(|| "decode_vector_for_ccs_m: expected_cols*D overflow".to_string())?;
    for c in pad_start..pad_end {
        let blk = c / D;
        let rho = c % D;
        if mat[(rho, blk)] != F::ZERO {
            return Err(format!(
                "decode_vector_for_ccs_m: non-zero padded coefficient at logical index {} (blk={}, rho={})",
                c, blk, rho
            ));
        }
    }

    let mut out = Vec::with_capacity(ccs_m);
    for c in 0..ccs_m {
        let blk = c / D;
        let rho = c % D;
        out.push(mat[(rho, blk)]);
    }
    Ok(out)
}
