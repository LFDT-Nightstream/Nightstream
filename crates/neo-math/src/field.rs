//! Field layer: Goldilocks F_q and K = F_{q^2}; two-adic hooks for NTT.
//! MUST: implement F_q and K with conjugation/inversion; constant-time basic ops.
//! SHOULD: expose roots-of-unity hooks sized for ring ops.

use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_goldilocks::Goldilocks;

/// Goldilocks base field F_q, q = 2^64 - 2^32 + 1.
pub type Fq = Goldilocks;

/// Quadratic extension K = F_{q^2}. (This is the only extension used by Neo.)
pub type K = BinomialExtensionField<Fq, 2>;

/// Goldilocks modulus (internal constant).
pub(crate) const GOLDILOCKS_MODULUS: u128 = 18446744069414584321u128;

/// Constructor compatibility shim to handle potential upstream API changes
#[inline]
fn new_k_from_coeffs(coefs: [Fq; 2]) -> K {
    // If `new_complex` ever changes upstream, swap in the correct constructor here.
    // This provides a single point of maintenance for field construction compatibility.
    BinomialExtensionField::new_complex(coefs[0], coefs[1])
}

// Convenience shims for K - using extension trait instead of inherent impl
pub trait KExtensions {
    /// Conjugation in K (a + b * u) ↦ (a - b * u).
    fn conj(self) -> Self;
    /// Multiplicative inverse; panics if zero (same as Field::inverse().unwrap()).
    fn inv(self) -> Self;
    /// Extract coefficients as [real, imag] for K = F_{q^2}.
    fn as_coeffs(&self) -> [Fq; 2];
    /// Construct from coefficients [real, imag]
    fn from_coeffs(coefs: [Fq; 2]) -> Self;
    /// Real part (convenience)  
    fn real(&self) -> Fq {
        self.as_coeffs()[0]
    }
    /// Imaginary part (convenience)
    fn imag(&self) -> Fq {
        self.as_coeffs()[1]
    }
    /// Extract limbs as u64 tuple (c0, c1) for circuit gadgets
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
        new_k_from_coeffs(coefs)
    }
}

/// Create extension field element from real/imaginary parts  
#[inline]
pub fn from_complex(real: Fq, imag: Fq) -> K {
    K::from_coeffs([real, imag])
}
