//! Field utilities wrapping the Goldilocks prime field.

use p3_field::{extension::BinomialExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use subtle::{ConditionallySelectable, ConstantTimeLess};
pub use p3_goldilocks::Goldilocks as F;

/// Quadratic extension F[u] / (u^2 - β) with β=7 (non-residue mod p for Goldilocks).
/// Note: BinomialExtensionField<F, 2> implements the Complex API for degree 2.
pub type ExtF = BinomialExtensionField<F, 2>;

/// Extension field elements can be viewed as vectors over the base field.
/// This helper trait exposes a simple "norm" used only for bounding blinded
/// evaluations in tests. It returns the maximum absolute value of the
/// components when represented canonically in the base field.
pub trait ExtFieldNorm {
    fn abs_norm(&self) -> u64;

    /// Alias for `abs_norm` used in some verifier code.
    fn norm(&self) -> u64 {
        self.abs_norm()
    }
}

impl ExtFieldNorm for ExtF {
    fn abs_norm(&self) -> u64 {
        let q = F::ORDER_U64;
        let half = q / 2;
        [self.real(), self.imag()]
            .iter()
            .map(|f| {
                let val = f.as_canonical_u64();
                let gt = half.ct_lt(&val);
                let neg = val.wrapping_sub(q);
                let selected = u64::conditional_select(&val, &neg, gt);
                let signed = selected as i64;
                let mask = signed >> 63;
                ((signed ^ mask) - mask) as u64
            })
            .max()
            .unwrap_or(0)
    }
}

/// Maximum allowed norm when projecting extension-field elements back to the base.
/// Derived as `\sigma \sqrt{n k}` times a tail factor (here 6 for \(<2^{-128}\) failure)
/// with parameters \(\sigma=3.2, n=64, k=16\), giving roughly 614.
pub const MAX_BLIND_NORM: u64 = (3.2_f64 * 32.0 * 6.0) as u64;

/// Convert a base field element into the extension field.
pub fn from_base(f: F) -> ExtF {
    ExtF::new_real(f)
}

/// Construct an extension field element from its base components.
pub fn from_base_pair(a0: u64, a1: u64) -> ExtF {
    ExtF::new_complex(F::from_u64(a0), F::from_u64(a1))
}

/// Sample a random element of the extension field using the given RNG.
/// If real_only=true, set second component=0 for base-field compatibility (e.g., tests).
pub fn random_extf_with_flag(rng: &mut impl rand::Rng, real_only: bool) -> ExtF {
    let a0 = F::from_u64(rng.random());
    let a1 = if real_only { F::ZERO } else { F::from_u64(rng.random()) };
    ExtF::new_complex(a0, a1)
}

/// Sample a random element of the extension field using the given RNG.
/// Defaults to full random (real_only=false).
pub fn random_extf(rng: &mut impl rand::Rng) -> ExtF {
    random_extf_with_flag(rng, false)
}

/// Embed a base field element into the extension field as a purely base element.
pub fn embed_base_to_ext(f: F) -> ExtF {
    ExtF::new_real(f)
}

/// Project an extension field element back to the base field if it has zero second component.
pub fn project_ext_to_base(e: ExtF) -> Option<F> {
    if e.imag() == F::ZERO && e.abs_norm() <= MAX_BLIND_NORM {
        Some(e.real())
    } else {
        None
    }
}

/// Return the multiplicative inverse of `x` in the field.
pub fn inverse(x: F) -> F {
    x.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_quadratic_non_residue() {
        // Verify that 7 is a non-residue in Goldilocks field (p = 2^64 - 2^32 + 1)
        // This ensures our extension x^2 - 7 is irreducible and forms a proper field
        let seven = F::from_u64(7);
        let p_minus_1_div_2 = (F::ORDER_U64 - 1) / 2;
        
        // Compute 7^((p-1)/2) mod p. If result is -1, then 7 is a non-residue
        let mut result = F::ONE;
        let mut base = seven;
        let mut exp = p_minus_1_div_2;
        
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
        
        // Should be -1 (which is p-1 in the field)
        assert_eq!(result, F::from_u64(F::ORDER_U64 - 1));
        println!("✓ Verified: 7 is a quadratic non-residue in Goldilocks field");
    }

    #[test]
    fn test_inverse_roundtrip() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut val = F::from_u64(rng.random());
            if val == F::ZERO {
                val = F::ONE;
            }
            let inv = inverse(val);
            assert_eq!(val * inv, F::ONE);
        }
    }

    #[test]
    fn test_extf_ops() {
        let mut rng = rand::rng();
        let x = random_extf(&mut rng);
        let y = random_extf(&mut rng);
        let z = x * y;
        assert_eq!(z * y.inverse(), x);
    }

    #[test]
    fn test_extf_traits() {
        let x = from_base(F::from_u64(2));
        let y = from_base(F::from_u64(3));
        assert_eq!(x + y, from_base(F::from_u64(5)));
        assert_eq!(x.inverse() * x, ExtF::ONE);
    }

    #[test]
    fn test_extension_field_structure() {
        // Test that our extension field has the correct structure
        let _a = from_base(F::from_u64(2));
        let b = from_base_pair(0, 1); // This represents the primitive element α where α² = 7
        
        // Verify α² = 7 in our extension
        let alpha_squared = b * b;
        let expected = from_base(F::from_u64(7));
        assert_eq!(alpha_squared, expected);
    }
}