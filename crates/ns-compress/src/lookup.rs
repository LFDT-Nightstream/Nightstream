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

    pub(crate) fn combine(&self, weights: &[K; LOOKUPS]) -> Vec<K> {
        (0..self.0[0].len())
            .map(|index| (0..LOOKUPS).map(|k| weights[k] * self.0[k][index]).sum())
            .collect()
    }
}

pub(crate) struct LookupPolynomial {
    denominator: [F; 5],
    numerator: [K; 4],
}

impl LookupPolynomial {
    pub(crate) fn new(poles: &[F; LOOKUPS], weights: &[K; LOOKUPS]) -> Self {
        let mut denominator = [F::ZERO; 5];
        let mut numerator = [K::ZERO; 4];
        denominator[0] = F::ONE;
        for (k, (&pole, &weight)) in poles.iter().zip(weights).enumerate() {
            for i in (0..=k).rev() {
                numerator[i] =
                    numerator[i] * pole + weight * denominator[i] - if i > 0 { numerator[i - 1] } else { K::ZERO };
            }
            for i in (0..=k + 1).rev() {
                denominator[i] = denominator[i] * pole - if i > 0 { denominator[i - 1] } else { F::ZERO };
            }
        }
        Self { denominator, numerator }
    }
    #[inline]
    pub(crate) fn at(&self, z: K) -> (K, K) {
        let denominator = (((z + K::from(self.denominator[3])) * z + K::from(self.denominator[2])) * z
            + K::from(self.denominator[1]))
            * z
            + K::from(self.denominator[0]);
        let numerator = ((self.numerator[3] * z + self.numerator[2]) * z + self.numerator[1]) * z + self.numerator[0];
        (denominator, numerator)
    }
}
