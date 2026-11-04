//! Sumcheck protocol interface

use neo_math::{KExtensions, K};
use p3_field::PrimeCharacteristicRing;

pub trait RoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K>;

    fn num_rounds(&self) -> usize;

    fn degree_bound(&self) -> usize;

    fn fold(&mut self, r: K);
}

pub fn poly_eval_k(coeffs: &[K], x: K) -> K {
    if coeffs.is_empty() {
        return K::ZERO;
    }
    let mut result = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        result = result * x + c;
    }
    result
}

pub fn interpolate_from_evals(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len(), "xs/ys length mismatch");
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];

    for i in 0..n {
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] += -xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }

        let mut denom = K::ONE;
        for j in 0..n {
            if i != j {
                denom *= xs[i] - xs[j];
            }
        }
        let scale = ys[i] * denom.inv();
        for d in 0..=cur_deg {
            coeffs[d] += scale * numer[d];
        }
    }

    coeffs
}
