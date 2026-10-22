//! Field layer: Goldilocks F_q and K = F_{q^2}; two-adic hooks for NTT.

use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;

pub type Fq = Goldilocks;

pub type K = BinomialExtensionField<Fq, 2>;

pub trait KExtensions {
    fn conj(self) -> Self;
    fn inv(self) -> Self;
    fn as_coeffs(&self) -> [Fq; 2];
    fn from_coeffs(coefs: [Fq; 2]) -> Self;
    fn scale_base(self, s: Fq) -> Self;
    fn scale_base_k(self, s: Self) -> Self;
    fn real(&self) -> Fq {
        self.as_coeffs()[0]
    }
    fn imag(&self) -> Fq {
        self.as_coeffs()[1]
    }
    fn to_limbs_u64(&self) -> (u64, u64) {
        use p3_field::PrimeField64;
        let coeffs = self.as_coeffs();
        (coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
    }
}

impl KExtensions for K {
    #[inline]
    fn conj(self) -> Self {
        self.conjugate()
    }
    #[inline]
    fn inv(self) -> Self {
        self.inverse()
    }
    #[inline]
    fn as_coeffs(&self) -> [Fq; 2] {
        [self.real(), self.imag()]
    }
    #[inline]
    fn from_coeffs(coefs: [Fq; 2]) -> Self {
        BinomialExtensionField::new_complex(coefs[0], coefs[1])
    }

    #[inline]
    fn scale_base(self, s: Fq) -> Self {
        let [c0, c1] = self.as_coeffs();
        Self::from_coeffs([c0 * s, c1 * s])
    }

    #[inline]
    fn scale_base_k(self, s: Self) -> Self {
        let [s0, s1] = s.as_coeffs();
        if s1 == Fq::ZERO {
            self.scale_base(s0)
        } else {
            self * s
        }
    }
}

#[inline]
pub fn from_complex(real: Fq, imag: Fq) -> K {
    K::from_coeffs([real, imag])
}
