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
    fn real(&self) -> Fq { self.as_coeffs()[0] }
    /// Imaginary part (convenience)
    fn imag(&self) -> Fq { self.as_coeffs()[1] }
}

impl KExtensions for K {
    #[inline] fn conj(self) -> Self { self.conjugate() }
    #[inline] fn inv(self) -> Self { self.inverse() }
    #[inline] fn as_coeffs(&self) -> [Fq; 2] { [self.real(), self.imag()] }
    #[inline] fn from_coeffs(coefs: [Fq; 2]) -> Self {
        new_k_from_coeffs(coefs)
    }
}

/// Random K generator for testing only
/// Gated to avoid accidentally introducing a hard dependency on rand in neo-math
#[cfg(any(test, feature = "testing"))]
pub fn random_extf() -> K {
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
    use rand_chacha::rand_core::RngCore;
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    let real = Fq::from_u64(rng.next_u64());
    let imag = Fq::from_u64(rng.next_u64());
    from_complex(real, imag)
}

/// Embed base field element into extension field
#[inline] pub fn from_base(x: Fq) -> K { K::from_coeffs([x, Fq::ZERO]) }

/// Create extension field element from real/imaginary parts  
#[inline] pub fn from_complex(real: Fq, imag: Fq) -> K { K::from_coeffs([real, imag]) }

/// Embed base field into extension (alias for clarity)
#[inline] pub fn embed_base_to_ext(x: Fq) -> K { from_base(x) }

/// Returns Some(real part) iff imaginary part == 0; otherwise None.
/// **Preferred for correctness-critical paths** - prevents silent loss of imaginary components.
#[inline] pub fn try_project_ext_to_base(x: K) -> Option<Fq> {
    let [re, im] = x.as_coeffs();
    if im == Fq::ZERO { Some(re) } else { None }
}

/// Always returns the real part, even if imaginary part != 0.
/// **Use with caution** - silently discards imaginary components.
/// Prefer `try_project_ext_to_base` in correctness-critical paths.
#[must_use = "discarding the result defeats the purpose of this lossy projection"]
#[inline] pub fn project_ext_to_base_lossy(x: K) -> Fq { x.real() }

/// Backward compatibility alias - routes to the SAFE version
#[inline] pub fn project_ext_to_base(x: K) -> Option<Fq> { try_project_ext_to_base(x) }
