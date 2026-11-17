//! The lookup that bounds a witness (SPEC.md section 6).

use neo_math::{F, K};
use p3_field::{batch_multiplicative_inverse, PrimeCharacteristicRing, PrimeField64};

use crate::layout::LOOKUPS;
use crate::Error;

pub(crate) const OUTSIDE: u16 = u16::MAX;

pub(crate) fn code(value: F, bound: u64) -> u16 {
    let value = value.as_canonical_u64();
    if value <= bound {
        (value + bound) as u16
    } else if value >= F::ORDER_U64 - bound {
        (value - (F::ORDER_U64 - bound)) as u16
    } else {
        OUTSIDE
    }
}

pub(crate) struct Inverses([Vec<F>; LOOKUPS]);

impl Inverses {
    pub(crate) fn new(bound: u64, taus: &[F; LOOKUPS]) -> Result<Self, Error> {
        if taus.iter().any(|&tau| code(tau, bound) != OUTSIDE) {
            return Err(Error::Challenge);
        }
        Ok(Self(taus.map(|tau| {
            let differences: Vec<F> = (0..=2 * bound)
                .map(|index| tau + F::from_u64(bound) - F::from_u64(index))
                .collect();
            batch_multiplicative_inverse(&differences)
        })))
    }

    pub(crate) fn values(&self) -> usize {
        self.0[0].len()
    }

    pub(crate) fn at(&self, k: usize, code: u16) -> F {
        self.0[k].get(usize::from(code)).copied().unwrap_or(F::ZERO)
    }

    pub(crate) fn combine(&self, weights: &[K; LOOKUPS]) -> Vec<K> {
        (0..self.0[0].len())
            .map(|index| (0..LOOKUPS).map(|k| weights[k] * self.0[k][index]).sum())
            .collect()
    }
}
