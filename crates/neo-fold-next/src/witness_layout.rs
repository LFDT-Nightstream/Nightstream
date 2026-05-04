//! Owns the local packed witness layout helpers used by CHIP-8 and RV32IM kernels.
//! It does not own deprecated memory-sidecar logic or witness builders.

use neo_ccs::matrix::Mat;
use neo_math::balanced::to_balanced_i128;
use neo_math::{D, F as BaseField};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField};

#[inline]
fn balanced_divrem(value: i128, base: i128) -> (i128, i128) {
    debug_assert!(base >= 2);
    let mut remainder = value % base;
    let mut quotient = (value - remainder) / base;
    let half = base / 2;
    if remainder > half {
        remainder -= base;
        quotient += 1;
    } else if remainder < -half {
        remainder += base;
        quotient -= 1;
    }
    (remainder, quotient)
}

/// Returns the number of packed Ajtai commitment columns for a witness of
/// length `full_width` (= word_count * 4 + 1, including the leading ONE).
pub fn commit_cols_for_full_width(full_width: usize) -> usize {
    full_width.div_ceil(D)
}

/// Packs a witness vector of length `full_width` into a D×cols ring-element
/// matrix for Ajtai commitment.
pub fn encode_vector_for_full_width(
    params: &NeoParams,
    full_width: usize,
    witness: &[BaseField],
) -> Result<Mat<BaseField>, String> {
    if witness.len() != full_width {
        return Err(format!(
            "encode_vector_for_full_width: witness length {} != full_width {}",
            witness.len(),
            full_width
        ));
    }
    if full_width == 0 {
        return Err("encode_vector_for_full_width: full_width must be > 0".into());
    }
    validate_packed_vector_nc_range(params, full_width, witness, "encode_vector_for_full_width")?;
    let cols = full_width.div_ceil(D);
    let mut out = Mat::zero(D, cols, BaseField::ZERO);
    for column in 0..full_width {
        let block = column / D;
        let rho = column % D;
        out[(rho, block)] = witness[column];
    }
    Ok(out)
}

/// Unpacks a D×cols ring-element matrix back to a witness vector of length
/// `full_width`. Validates that all padding positions are zero.
pub fn decode_vector_for_full_width<F: PrimeField>(
    _params: &NeoParams,
    full_width: usize,
    mat: &Mat<F>,
) -> Result<Vec<F>, String> {
    if mat.rows() != D {
        return Err(format!(
            "decode_vector_for_full_width: mat.rows()={} expected D={D}",
            mat.rows()
        ));
    }
    if full_width == 0 {
        return Err("decode_vector_for_full_width: full_width must be > 0".into());
    }
    let expected_cols = full_width.div_ceil(D);
    if mat.cols() != expected_cols {
        return Err(format!(
            "decode_vector_for_full_width: packed layout expects cols={} for full_width={}, got {}",
            expected_cols,
            full_width,
            mat.cols()
        ));
    }

    let pad_start = full_width;
    let pad_end = expected_cols
        .checked_mul(D)
        .ok_or_else(|| "decode_vector_for_full_width: expected_cols*D overflow".to_string())?;
    for column in pad_start..pad_end {
        let block = column / D;
        let rho = column % D;
        if mat[(rho, block)] != F::ZERO {
            return Err(format!(
                "decode_vector_for_full_width: non-zero padded coefficient at logical index {} (blk={}, rho={})",
                column, block, rho
            ));
        }
    }

    let mut out = Vec::with_capacity(full_width);
    for column in 0..full_width {
        let block = column / D;
        let rho = column % D;
        out.push(mat[(rho, block)]);
    }
    Ok(out)
}

fn validate_packed_vector_nc_range(
    params: &NeoParams,
    full_width: usize,
    witness: &[BaseField],
    label: &str,
) -> Result<(), String> {
    if witness.len() != full_width {
        return Err(format!(
            "{label}: witness length {} != full_width {}",
            witness.len(),
            full_width
        ));
    }
    if full_width == 0 {
        return Err(format!("{label}: full_width must be > 0"));
    }
    if params.b < 2 {
        return Err(format!("{label}: invalid b={} (must be >= 2)", params.b));
    }

    let base = params.b as i128;
    for (index, &value) in witness.iter().enumerate() {
        let mut remainder = to_balanced_i128(value);
        for _ in 0..D {
            let (_digit, quotient) = balanced_divrem(remainder, base);
            remainder = quotient;
        }
        if remainder != 0 {
            return Err(format!(
                "{label}: packed coefficient at index {index} is not representable in D={} balanced base-{} digits (centered value {})",
                D, params.b, to_balanced_i128(value)
            ));
        }
    }
    Ok(())
}
