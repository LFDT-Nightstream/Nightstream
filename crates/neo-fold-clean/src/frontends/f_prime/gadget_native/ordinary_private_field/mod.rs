//! Ordinary-private Goldilocks field materialization.
//!
//! Owns: the exact 41-coordinate shifted centered-ternary representation used
//! only by `CanonicalFieldKind::OrdinaryPrivate`.
//!
//! Does not own: SIS balanced openings, direct canonical-u64 fields, synthetic
//! canonical fields, CE/norm authority, or row removal.
//!
//! Emits constraints: no. `coordinate_gates` emits one local centered-unit
//! obligation per coordinate and keeps those rows until CE authority is proved.
//!
//! Authority boundary: the decoded source field supplies the exact local value
//! represented inside the current source R1CS. For the committed
//! representation, this module owns the 41-coordinate word and its inverse;
//! callers may not substitute a digest or alternate word encoding. This is an
//! encoding-refinement claim, not paper-level semantic authority.
//!
//! | Obligation | Formula | Rust owner | Lean owner |
//! |---|---|---|---|
//! | width | `3^40 < p < 3^41` | `encoding` constants | `CenteredTernaryField.width_floor` |
//! | materialize | `t=(x+(3^41-1)/2) mod p`, little-endian trits, then subtract one | `encoding::encode` | `CenteredTernaryField.encodeDigit` |
//! | decode substitution | `x=sum_i d_i 3^i mod p` | `encoding::decode` / `slots::slot_terms` | `CenteredTernaryField.decode_encodeDigit` |
//! | local alphabet | `d_i^3-d_i=0` for all 41 coordinates | `coordinate_gates` | `CenteredTernaryField.gateWord_iff_alphabetWord` |

mod encoding;

pub(super) fn encode(value: neo_math::F) -> [neo_math::F; encoding::ORDINARY_PRIVATE_DIGITS] {
    encoding::encode(value)
}

pub(super) fn decode(digits: &[neo_math::F], column: usize) -> Result<neo_math::F, super::GadgetNativeError> {
    encoding::decode(digits, column)
}

#[doc(hidden)]
pub use encoding::{
    encode_ordinary_private_field, ORDINARY_PRIVATE_DIGITS, ORDINARY_PRIVATE_RADIX_40, ORDINARY_PRIVATE_RADIX_41,
    ORDINARY_PRIVATE_SHIFT,
};
