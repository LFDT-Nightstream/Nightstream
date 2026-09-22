//! Exact shifted centered-ternary encoder and linear decoder.
//!
//! Owns: arithmetic for one ordinary-private field word.
//!
//! Does not own: slot placement, source-role classification, constraint
//! emission, or any claim that local alphabet rows may be removed.
//!
//! Emits constraints: no.
//!
//! Authority boundary: every accepted digit is checked locally to be one of
//! `-1, 0, 1`; decoding returns the source-field value represented by the word.
//!
//! | Function | Mathematical obligation | Multiplicity | Emitted rows | Lean theorem |
//! |---|---|---:|---:|---|
//! | `encode` | deterministic shifted 41-trit representative | one per ordinary field | 0 | `encodeDigit_represents` |
//! | `decode` | radix-three linear reconstruction modulo Goldilocks | every source use by substitution | 0 | `decode_encodeDigit` |

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::GadgetNativeError;

pub const ORDINARY_PRIVATE_DIGITS: usize = 41;
pub const ORDINARY_PRIVATE_RADIX_40: u128 = 12_157_665_459_056_928_801;
pub const ORDINARY_PRIVATE_RADIX_41: u128 = 36_472_996_377_170_786_403;
pub const ORDINARY_PRIVATE_SHIFT: u128 = 18_236_498_188_585_393_201;

const GOLDILOCKS_MODULUS: u128 = F::ORDER_U64 as u128;

const _: () = assert!(ORDINARY_PRIVATE_RADIX_40 < GOLDILOCKS_MODULUS);
const _: () = assert!(GOLDILOCKS_MODULUS < ORDINARY_PRIVATE_RADIX_41);
const _: () = assert!(ORDINARY_PRIVATE_SHIFT == (ORDINARY_PRIVATE_RADIX_41 - 1) / 2);

/// Public only for exact Rust/Lean fixtures; production calls this through the
/// ordinary-private slot constructor.
#[doc(hidden)]
pub fn encode_ordinary_private_field(value: F) -> [F; ORDINARY_PRIVATE_DIGITS] {
    encode(value)
}

pub(super) fn encode(value: F) -> [F; ORDINARY_PRIVATE_DIGITS] {
    // `x + shift` exceeds `u64` near p-1. Keeping the full operation in u128
    // is part of the executable correspondence with Lean's Nat definition.
    let mut target = (u128::from(value.as_canonical_u64()) + ORDINARY_PRIVATE_SHIFT) % GOLDILOCKS_MODULUS;
    let digits = std::array::from_fn(|_| {
        let trit = target % 3;
        target /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("remainder modulo three"),
        }
    });
    debug_assert_eq!(target, 0, "41 trits cover every canonical Goldilocks value");
    digits
}

pub(super) fn decode(digits: &[F], column: usize) -> Result<F, GadgetNativeError> {
    if digits.len() != ORDINARY_PRIVATE_DIGITS {
        return Err(GadgetNativeError::OrdinaryPrivateWidth {
            column,
            expected: ORDINARY_PRIVATE_DIGITS,
            got: digits.len(),
        });
    }
    let mut value = F::ZERO;
    let mut power = F::ONE;
    for &digit in digits {
        if digit != -F::ONE && digit != F::ZERO && digit != F::ONE {
            return Err(GadgetNativeError::NonCenteredDigit { column });
        }
        value += digit * power;
        power *= F::from_u64(3);
    }
    Ok(value)
}
