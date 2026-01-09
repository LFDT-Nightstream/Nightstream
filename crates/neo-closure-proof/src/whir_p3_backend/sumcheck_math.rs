//! Small math helpers for WHIR sumcheck-style protocols.

#![forbid(unsafe_code)]

use super::F;
use p3_field::{Field as _, PrimeCharacteristicRing as _};

pub(super) fn eval_quad(evals: [F; 3], r: F) -> F {
    // Evaluate the unique degree-2 polynomial matching:
    //   p(0)=evals[0], p(1)=evals[1], p(2)=evals[2]
    // at the point r.
    //
    // Lagrange basis at x=0,1,2:
    //   L0(x) = (x-1)(x-2)/((0-1)(0-2)) = (x-1)(x-2)/2
    //   L1(x) = (x-0)(x-2)/((1-0)(1-2)) = -x(x-2)
    //   L2(x) = (x-0)(x-1)/((2-0)(2-1)) = x(x-1)/2
    //
    // Division by 2 uses 2^{-1} in Goldilocks (exists since modulus is odd).
    let inv2 = F::from_u64(2).inverse();
    let r1 = r - F::ONE;
    let r2 = r - F::from_u64(2);

    let l0 = r1 * r2 * inv2;
    let l1 = (F::ZERO - r) * r2;
    let l2 = r * r1 * inv2;
    evals[0] * l0 + evals[1] * l1 + evals[2] * l2
}

pub(super) fn eval_lagrange_0_to_deg(evals: &[F], r: F) -> F {
    // Evaluate the unique degree-(evals.len()-1) polynomial matching:
    //   p(i) = evals[i] for i=0..deg
    // at the point r.
    let deg = evals
        .len()
        .checked_sub(1)
        .expect("eval_lagrange_0_to_deg: empty evals");

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
