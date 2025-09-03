//! Field layer: Goldilocks F_q and K = F_{q^2}; two-adic hooks for NTT.
//! MUST: implement F_q and K with conjugation/inversion; constant-time basic ops.
//! SHOULD: expose roots-of-unity hooks sized for ring ops.

use p3_field::{Field, TwoAdicField, PrimeCharacteristicRing};
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;

/// Goldilocks base field F_q, q = 2^64 - 2^32 + 1.
pub type Fq = Goldilocks;

/// Quadratic extension K = F_{q^2}. (This is the only extension used by Neo.)
pub type K = BinomialExtensionField<Fq, 2>;

/// Goldilocks modulus (public constant for audits & tests).
pub const GOLDILOCKS_MODULUS: u128 = 18446744069414584321u128;

/// Two-adicity of F_q^* (Goldilocks has 2^32 | q-1).
pub const TWO_ADICITY: usize = <Fq as TwoAdicField>::TWO_ADICITY;

/// A fixed quadratic non-residue for F_q; verified in tests via Euler's criterion.
/// We use 7, which is known to be a quadratic non-residue modulo the Goldilocks prime.
pub fn nonresidue() -> Fq {
    Fq::from_u64(7)
}

/// SHOULD: provide a two-adic generator (2^bits-th root of unity) for NTT hooks.
#[inline]
pub fn two_adic_generator(bits: usize) -> Fq {
    <Fq as TwoAdicField>::two_adic_generator(bits)
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
    fn real(&self) -> Fq { self.as_coeffs()[0] }
    /// Imaginary part (convenience)
    fn imag(&self) -> Fq { self.as_coeffs()[1] }
}

impl KExtensions for K {
    #[inline] fn conj(self) -> Self { self.conjugate() }
    #[inline] fn inv(self) -> Self { self.inverse() }
    #[inline] fn as_coeffs(&self) -> [Fq; 2] { [self.real(), self.imag()] }
    #[inline] fn from_coeffs(coefs: [Fq; 2]) -> Self {
        BinomialExtensionField::new_complex(coefs[0], coefs[1])
    }
}

/// Random K generator for testing
pub fn random_extf() -> K {
    use rand::Rng;
    let mut rng = rand::rng();
    let real = Fq::from_u64(rng.random::<u64>());
    let imag = Fq::from_u64(rng.random::<u64>());
    from_complex(real, imag)
}

/// Embed base field element into extension field
#[inline] pub fn from_base(x: Fq) -> K { K::new_real(x) }

/// Create extension field element from real/imaginary parts  
#[inline] pub fn from_complex(real: Fq, imag: Fq) -> K { K::from_coeffs([real, imag]) }

/// Embed base field into extension (alias for clarity)
#[inline] pub fn embed_base_to_ext(x: Fq) -> K { from_base(x) }

/// Project extension field element to base field (real part only)
#[inline] pub fn project_ext_to_base(x: K) -> Fq { x.real() }
