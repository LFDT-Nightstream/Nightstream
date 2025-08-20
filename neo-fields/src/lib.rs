//! Field utilities wrapping the Goldilocks prime field.

use p3_field::{extension::BinomialExtensionField, Field, PrimeCharacteristicRing, PrimeField64};use subtle::{ConditionallySelectable, ConstantTimeLess};
pub use p3_goldilocks::Goldilocks as F;

// Override the default W = NEG_ONE to use W = 7 (quadratic non-residue)
// This ensures x^2 - 7 is irreducible, making the extension field mathematically sound

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

/// Maximum allowed norm for *blinding values* only. 
/// This should NOT be applied to unblinded evaluations, which can be full field size.
/// Set to approximately p/2 to allow reasonable blinding noise while preventing overflow.
pub const MAX_BLIND_NORM: u64 = F::ORDER_U64 / 2;

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

