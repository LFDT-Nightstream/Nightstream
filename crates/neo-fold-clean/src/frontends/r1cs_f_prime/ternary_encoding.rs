//! Exact balanced-ternary witness encoding for retained field values.
//!
//! Owns: the fixed 41-digit representation, chunk lookup table, sign selection,
//! and witness digit materialization.
//!
//! Does not own: canonicality constraints, slot allocation, source-field
//! binding, or selective CCS construction.
//!
//! Emits constraints: no.
//!
//! Authority boundary: generated digits are witness data until the selective
//! canonicality rows bind them to the retained source value.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Chunk decomposition | `balanced_ternary_chunks` | no | Fixed radix-three arithmetic |
//! | Field encoding | [`balanced_ternary_digits`] | no | Canonical field representative |

use neo_math::F;
use p3_field::PrimeField64;

use super::lowering::LowNormR1csError;

pub(super) const BALANCED_TERNARY_FIELD_WIDTH: usize = 41;
pub(super) const BALANCED_SEPTENARY_FIELD_WIDTH: usize = 23;

const CHUNK_DIGITS: usize = 5;
const CHUNK_RADIX: u64 = 243;
const CHUNKS: [([i8; CHUNK_DIGITS], u8); CHUNK_RADIX as usize] = balanced_ternary_chunks();

const fn balanced_ternary_chunks() -> [([i8; CHUNK_DIGITS], u8); CHUNK_RADIX as usize] {
    let mut chunks = [([0; CHUNK_DIGITS], 0); CHUNK_RADIX as usize];
    let mut value = 0usize;
    while value < CHUNK_RADIX as usize {
        let mut remaining = value as u64;
        let mut digit = 0usize;
        while digit < CHUNK_DIGITS {
            let residue = remaining % 3;
            chunks[value].0[digit] = if residue == 2 { -1 } else { residue as i8 };
            remaining = remaining / 3 + (residue == 2) as u64;
            digit += 1;
        }
        chunks[value].1 = remaining as u8;
        value += 1;
    }
    chunks
}

pub(super) fn balanced_ternary_digits(
    value: F,
    field_col: usize,
) -> Result<[i8; BALANCED_TERNARY_FIELD_WIDTH], LowNormR1csError> {
    let modulus = F::ORDER_U64;
    let canonical = value.as_canonical_u64();
    let negative = canonical > modulus / 2;
    let mut remaining = if negative { modulus - canonical } else { canonical };
    let mut digit_index = 0usize;
    let mut out = [0i8; BALANCED_TERNARY_FIELD_WIDTH];

    while digit_index < BALANCED_TERNARY_FIELD_WIDTH {
        let residue = (remaining % CHUNK_RADIX) as usize;
        remaining /= CHUNK_RADIX;
        let (digits, carry) = CHUNKS[residue];
        remaining += u64::from(carry);
        for digit in digits {
            if digit_index == BALANCED_TERNARY_FIELD_WIDTH {
                if digit != 0 {
                    return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
                }
                continue;
            }
            out[digit_index] = if negative { -digit } else { digit };
            digit_index += 1;
        }
    }
    if remaining != 0 {
        return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
    }
    Ok(out)
}

pub(super) fn balanced_septenary_digits(
    value: F,
    field_col: usize,
) -> Result<[i8; BALANCED_SEPTENARY_FIELD_WIDTH], LowNormR1csError> {
    const RADIX_POWER: u128 = 27_368_747_340_080_916_343;
    const SHIFT: u128 = (RADIX_POWER - 1) / 2;
    let modulus = u128::from(F::ORDER_U64);
    let mut remaining = (u128::from(value.as_canonical_u64()) + SHIFT) % modulus;
    let mut out = [0i8; BALANCED_SEPTENARY_FIELD_WIDTH];

    for digit in &mut out {
        let residue = remaining % 7;
        remaining /= 7;
        *digit = residue as i8 - 3;
    }
    if remaining != 0 {
        return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
    }
    Ok(out)
}
