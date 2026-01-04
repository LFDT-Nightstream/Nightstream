use neo_ajtai::{decomp_b_row_major, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_math::F as BaseField;
use neo_params::NeoParams;
use p3_field::PrimeField;

/// Encode a vector `z ∈ F^m` into its Ajtai digit matrix `Z ∈ F^{d×m}` using **balanced** digits.
///
/// The returned matrix is **row-major** with shape `d×m`, matching `neo_ccs::matrix::Mat`.
///
/// This is the canonical helper for building Ajtai commitments to a witness vector `z`.
pub fn encode_vector_balanced_to_mat(params: &NeoParams, z: &[BaseField]) -> Mat<BaseField> {
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
    let row_major = decomp_b_row_major(z, params.b, d, DecompStyle::Balanced);
    Mat::from_row_major(d, m, row_major)
}

/// Decode an Ajtai-encoded digit matrix back into the original vector.
///
/// Each column encodes a base-`b` digit expansion across `params.d` rows.
pub fn decode_vector<F: PrimeField>(params: &NeoParams, mat: &Mat<F>) -> Vec<F> {
    let d = mat.rows();
    let m = mat.cols();
    assert_eq!(
        d, params.d as usize,
        "Ajtai d mismatch: mat has {} rows, params.d={}",
        d, params.d
    );

    let b = F::from_u64(params.b as u64);

    let mut pow = vec![F::ONE; d];
    for i in 1..d {
        pow[i] = pow[i - 1] * b;
    }

    let mut out = Vec::with_capacity(m);
    for col in 0..m {
        let mut acc = F::ZERO;
        for row in 0..d {
            acc += mat[(row, col)] * pow[row];
        }
        out.push(acc);
    }
    out
}
