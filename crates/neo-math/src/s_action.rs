//! S-action implementation: "left multiplication by a in R_q" as rot(a).

use crate::ring::{cf, cf_inv};
use crate::{Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

#[derive(Clone, Debug)]
pub struct SAction {
    a: Rq,
}

impl SAction {
    pub fn from_ring(a: Rq) -> Self {
        Self { a }
    }

    pub fn scalar(f: Fq) -> Self {
        Self::from_ring(Rq::from_field_scalar(f))
    }

    pub fn to_matrix(&self) -> DenseMatrix<Fq> {
        let mut values = vec![Fq::ZERO; D * D];
        let mut x_power = Rq::one();

        for j in 0..D {
            let col = cf(self.a.mul(&x_power));
            for i in 0..D {
                values[i * D + j] = col[i];
            }
            x_power = x_power.mul_by_monomial(1);
        }

        DenseMatrix::new(values, D)
    }

    #[inline]
    pub fn apply_vec(&self, v: &[Fq; D]) -> [Fq; D] {
        let prod = self.a.mul(&cf_inv(*v));
        cf(prod)
    }

    #[inline]
    pub fn compose(&self, other: &SAction) -> SAction {
        SAction {
            a: self.a.mul(&other.a),
        }
    }
}
