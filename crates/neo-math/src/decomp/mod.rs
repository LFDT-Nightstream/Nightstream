//! Vector-to-matrix decomposition as described in the Neo paper.

use crate::F;
use crate::modint::Coeff;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

/// Decompose each entry of `z` in base `b` with `d` digits.
/// Returns a `d x m` matrix where columns reconstruct the original vector.
pub fn decomp_b(z: &[F], b: u64, d: usize) -> RowMajorMatrix<F> {
    let m = z.len();
    let mut data = vec![F::ZERO; d * m];
    for (col, &val) in z.iter().enumerate() {
        let mut x = val;
        for row in 0..d {
            let digit = F::from_u64(x.as_canonical_u64() % b);
            data[row * m + col] = digit;
            x = F::from_u64(x.as_canonical_u64() / b);
        }
    }
    RowMajorMatrix::new(data, m)
}

/// Signed decomposition of a vector into `k` layers in base `b`, returned as a
/// `k x m` [`RowMajorMatrix`]. Also returns the gadget vector `g = [1, b, b^2,
/// ..., b^{k-1}]` such that `z = g * digits`.
pub fn signed_decomp_b<C: Coeff + Into<i128> + From<i128> + Send + Sync>(
    z: &[C],
    b: u64,
    k: usize,
) -> (RowMajorMatrix<C>, Vec<C>) {
    let base = b as i128;
    let r = (base - 1) / 2;
    let m = z.len();
    let mut data = vec![C::zero(); k * m];
    let mut g = vec![C::zero(); k];
    let mut pow = C::one();
    for g_item in g.iter_mut() {
        *g_item = pow;
        pow *= C::from(b as i128);
    }

    for (col, &val) in z.iter().enumerate() {
        let q: i128 = C::modulus() as i128;
        let mut x = val.into();
        if x > q / 2 {
            x -= q;
        }
        for row in 0..k {
            let mut rem = x % base;
            let mut quot = x / base;
            if rem > r {
                rem -= base;
                quot += 1;
            } else if rem < -r {
                rem += base;
                quot -= 1;
            }
            data[row * m + col] = C::from(rem);
            x = quot;
        }
        assert_eq!(
            x,
            0,
            "Decomposition incomplete: k={} too small for value {} with b={}",
            k,
            val.into(),
            b
        );
    }

    (RowMajorMatrix::new(data, m), g)
}

/// Reconstruct the original vector from the decomposed matrix and gadget
/// vector `g` where `z = g * matrix`.
pub fn reconstruct_decomp<C: Coeff + Send + Sync>(matrix: &RowMajorMatrix<C>, g: &[C]) -> Vec<C> {
    let m = matrix.width();
    let mut z = vec![C::zero(); m];
    for (col, z_item) in z.iter_mut().enumerate() {
        let mut acc = C::zero();
        for (row, &g_val) in g.iter().enumerate() {
            acc += g_val * matrix.get(row, col).unwrap();
        }
        *z_item = acc;
    }
    z
}


