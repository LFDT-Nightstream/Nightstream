//! Owns small multilinear-extension helpers shared by active proof frontends.

use neo_math::K;
use p3_field::PrimeCharacteristicRing;

pub(crate) fn build_eq_table(point_le: &[K]) -> Vec<K> {
    let ell = point_le.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];
    for (i, &ri) in point_le.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = K::ONE - ri;
        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                let value = out[idx + j];
                out[idx + j] = value * one_minus;
            }
            for j in 0..stride {
                let value = out[idx + stride + j];
                out[idx + stride + j] = value * ri;
            }
            idx += 2 * stride;
        }
    }
    out
}

pub(crate) fn eq_eval_le(point_a_le: &[K], point_b_le: &[K]) -> K {
    point_a_le
        .iter()
        .zip(point_b_le.iter())
        .fold(K::ONE, |acc, (&a, &b)| acc * ((K::ONE - a) * (K::ONE - b) + a * b))
}
