//! Small math helpers for WHIR sumcheck-style protocols.

#![forbid(unsafe_code)]

use super::F;
use p3_field::{Field as _, PrimeCharacteristicRing as _};

pub(crate) fn eval_lagrange_0_to_deg(evals: &[F], r: F) -> F {
    // Evaluate the unique degree-(evals.len()-1) polynomial matching:
    //   p(i) = evals[i] for i=0..deg
    // at the point r.
    let Some(deg) = evals.len().checked_sub(1) else {
        return F::ZERO;
    };

    let mut out = F::ZERO;
    for i in 0..=deg {
        let mut num = F::ONE;
        let mut den = F::ONE;
        let xi = F::from_u64(i as u64);
        for j in 0..=deg {
            if i == j {
                continue;
            }
            let xj = F::from_u64(j as u64);
            num *= r - xj;
            den *= xi - xj;
        }
        out += evals[i] * num * den.inverse();
    }
    out
}
