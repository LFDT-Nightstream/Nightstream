use neo_ccs::matrix::Mat;
use neo_params::NeoParams;
use p3_field::PrimeField;

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
