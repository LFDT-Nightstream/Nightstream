//! Low-norm coordinate slots and exact source-value reconstruction.
//!
//! Owns: Boolean, canonical-binary, ordinary-private centered-ternary,
//! centered-unit alias, and SIS balanced-ternary assignment encodings.
//!
//! Does not own: source-row selection or gadget trace validation.
//!
//! Emits constraints: canonical Goldilocks prefix relations only; the parent
//! emits the common Boolean/centered alphabet rows.
//!
//! Authority boundary: slots are representations, never authority. Decoding
//! must reproduce the source assignment, and every alias must reference a
//! validated structural coordinate owned by the source trace.
//!
//! | Slot kind | Coordinates | Decode equation | Extra rows |
//! |---|---:|---|---:|
//! | Boolean | 1 | `b` | common bitness |
//! | Centered unit alias | 1 | `d` | owned by parent balanced field |
//! | Ordinary private field | 41 | `sum d_i 3^i` | local centered residual pair/tail |
//! | SIS balanced field | 41 | `sum d_i 3^i` | source canonical-opening rows |
//! | Canonical field | 64 + 31 | `sum b_i 2^i` | 32 prefix/canonicity relations in 16 residual-pair rows |

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{BalancedTernaryOpeningTraceEntry, R1csSnapshot};

use super::gates::OneProductResidualTerms;
use super::ordinary_private_field;
use super::{GadgetNativeError, TraceGateBuilder, CANONICAL_PREFIX_AUX, FIELD_BITS, HIGH_BITS_START};

pub(super) const GOLDILOCKS_CANONICALITY_RELATIONS: usize = 32;
pub(super) const GOLDILOCKS_CANONICALITY_PAIR_ROWS: usize = GOLDILOCKS_CANONICALITY_RELATIONS / 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ValueEncoding {
    Boolean,
    CanonicalBinary { auxiliary_start: usize },
    CenteredUnit,
    OrdinaryCenteredTernary,
    BalancedTernary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueSlot {
    pub(super) start: usize,
    pub(super) width: usize,
    pub(super) encoding: ValueEncoding,
}

impl ValueSlot {
    pub(super) fn centered_alias(parent: Self, digit: usize) -> Self {
        debug_assert_eq!(parent.encoding, ValueEncoding::BalancedTernary);
        debug_assert!(digit < parent.width);
        Self {
            start: parent.start + digit,
            width: 1,
            encoding: ValueEncoding::CenteredUnit,
        }
    }
}

pub(super) fn push_boolean_slot(out: &mut Vec<F>, value: F, column: usize) -> Result<ValueSlot, GadgetNativeError> {
    if value != F::ZERO && value != F::ONE {
        return Err(GadgetNativeError::BooleanWitness { column });
    }
    let start = out.len();
    out.push(value);
    Ok(ValueSlot {
        start,
        width: 1,
        encoding: ValueEncoding::Boolean,
    })
}

pub(super) fn push_field_slot(out: &mut Vec<F>, value: F) -> ValueSlot {
    let start = out.len();
    let value = value.as_canonical_u64();
    out.extend((0..FIELD_BITS).map(|bit| F::from_u64((value >> bit) & 1)));
    let canonical_aux_start = out.len();
    let aux = canonical_prefix_values(&out[start..start + FIELD_BITS]);
    out.extend(aux);
    ValueSlot {
        start,
        width: FIELD_BITS,
        encoding: ValueEncoding::CanonicalBinary {
            auxiliary_start: canonical_aux_start,
        },
    }
}

pub(super) fn push_ordinary_private_field_slot(out: &mut Vec<F>, value: F) -> ValueSlot {
    let start = out.len();
    out.extend(ordinary_private_field::encode(value));
    ValueSlot {
        start,
        width: ordinary_private_field::ORDINARY_PRIVATE_DIGITS,
        encoding: ValueEncoding::OrdinaryCenteredTernary,
    }
}

pub(super) fn push_balanced_ternary_slot(
    out: &mut Vec<F>,
    source: &R1csSnapshot,
    opening: &BalancedTernaryOpeningTraceEntry,
) -> Result<ValueSlot, GadgetNativeError> {
    let start = out.len();
    let mut value = F::ZERO;
    let mut power = F::ONE;
    for &column in &opening.digit_cols {
        let digit = source.witness()[column];
        if digit != -F::ONE && digit != F::ZERO && digit != F::ONE {
            return Err(GadgetNativeError::CenteredWitness { column });
        }
        out.push(digit);
        value += digit * power;
        power *= F::from_u64(3);
    }
    if value != source.witness()[opening.field_col] {
        return Err(GadgetNativeError::BalancedTernaryWitness {
            column: opening.field_col,
        });
    }
    Ok(ValueSlot {
        start,
        width: BALANCED_TERNARY_DIGITS,
        encoding: ValueEncoding::BalancedTernary,
    })
}

pub(super) fn decode_slot(slot: ValueSlot, column: usize, encoded: &[F]) -> Result<F, GadgetNativeError> {
    match slot.encoding {
        ValueEncoding::Boolean => {
            let value = encoded[slot.start];
            if value != F::ZERO && value != F::ONE {
                return Err(GadgetNativeError::NonBinaryDigit { column });
            }
            Ok(value)
        }
        ValueEncoding::CenteredUnit => {
            let value = encoded[slot.start];
            if value != -F::ONE && value != F::ZERO && value != F::ONE {
                return Err(GadgetNativeError::NonCenteredDigit { column });
            }
            Ok(value)
        }
        ValueEncoding::OrdinaryCenteredTernary => {
            ordinary_private_field::decode(&encoded[slot.start..slot.start + slot.width], column)
        }
        ValueEncoding::BalancedTernary => {
            let mut value = F::ZERO;
            let mut power = F::ONE;
            for digit in &encoded[slot.start..slot.start + slot.width] {
                if *digit != -F::ONE && *digit != F::ZERO && *digit != F::ONE {
                    return Err(GadgetNativeError::NonCenteredDigit { column });
                }
                value += *digit * power;
                power *= F::from_u64(3);
            }
            Ok(value)
        }
        ValueEncoding::CanonicalBinary { auxiliary_start } => {
            let mut value = 0u64;
            for bit in 0..slot.width {
                let digit = encoded[slot.start + bit];
                if digit != F::ZERO && digit != F::ONE {
                    return Err(GadgetNativeError::NonBinaryDigit { column });
                }
                if digit == F::ONE {
                    value |= 1u64 << bit;
                }
            }
            if value >= F::ORDER_U64 {
                return Err(GadgetNativeError::NonCanonicalField { column, value });
            }
            let expected = canonical_prefix_values(&encoded[slot.start..slot.start + FIELD_BITS]);
            for (offset, expected_bit) in expected.into_iter().enumerate() {
                if encoded[auxiliary_start + offset] != expected_bit {
                    return Err(GadgetNativeError::CanonicalAuxMismatch { column, offset });
                }
            }
            Ok(F::from_u64(value))
        }
    }
}

pub(super) fn slot_terms(slot: ValueSlot) -> Vec<(usize, F)> {
    match slot.encoding {
        ValueEncoding::Boolean | ValueEncoding::CenteredUnit => vec![(slot.start, F::ONE)],
        ValueEncoding::CanonicalBinary { .. } => (0..slot.width)
            .map(|bit| (slot.start + bit, F::from_u64(1u64 << bit)))
            .collect(),
        ValueEncoding::OrdinaryCenteredTernary | ValueEncoding::BalancedTernary => {
            let mut power = F::ONE;
            (0..slot.width)
                .map(|digit| {
                    let term = (slot.start + digit, power);
                    power *= F::from_u64(3);
                    term
                })
                .collect()
        }
    }
}

pub(super) fn emit_goldilocks_canonicality(gates: &mut TraceGateBuilder, slot: ValueSlot) {
    let ValueEncoding::CanonicalBinary {
        auxiliary_start: aux_start,
    } = slot.encoding
    else {
        unreachable!("only canonical binary slots have Goldilocks prefix auxiliaries")
    };
    let mut relations = Vec::with_capacity(GOLDILOCKS_CANONICALITY_RELATIONS);
    relations.push(OneProductResidualTerms {
        a: vec![(slot.start + HIGH_BITS_START, F::ONE)],
        b: vec![(slot.start + HIGH_BITS_START + 1, F::ONE)],
        c: vec![(aux_start, F::ONE)],
    });
    for high_offset in 2..32 {
        relations.push(OneProductResidualTerms {
            a: vec![(aux_start + high_offset - 2, F::ONE)],
            b: vec![(slot.start + HIGH_BITS_START + high_offset, F::ONE)],
            c: vec![(aux_start + high_offset - 1, F::ONE)],
        });
    }
    relations.push(OneProductResidualTerms {
        a: vec![(aux_start + CANONICAL_PREFIX_AUX - 1, F::ONE)],
        b: (0..32)
            .map(|bit| (slot.start + bit, F::from_u64(1u64 << bit)))
            .collect(),
        c: Vec::new(),
    });
    assert_eq!(relations.len(), GOLDILOCKS_CANONICALITY_RELATIONS);
    let mut relations = relations.into_iter();
    for _ in 0..GOLDILOCKS_CANONICALITY_PAIR_ROWS {
        let left = relations.next().expect("left canonicality relation");
        let right = relations.next().expect("right canonicality relation");
        gates.one_product_residual_pair(left, right);
    }
    assert!(relations.next().is_none(), "canonical slot pairing has no tail");
}

fn canonical_prefix_values(bits: &[F]) -> Vec<F> {
    let mut out = Vec::with_capacity(CANONICAL_PREFIX_AUX);
    let mut prefix = bits[HIGH_BITS_START] * bits[HIGH_BITS_START + 1];
    out.push(prefix);
    for &bit in &bits[HIGH_BITS_START + 2..] {
        prefix *= bit;
        out.push(prefix);
    }
    out
}
