//! Exact balanced-ternary witness encoding for full field values.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::lowering::LowNormR1csError;

pub(super) const BALANCED_TERNARY_FIELD_WIDTH: usize = 41;

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

pub(super) fn write_balanced_ternary(
    assignment: &mut [F],
    start: usize,
    value: F,
    field_col: usize,
) -> Result<(), LowNormR1csError> {
    let modulus = F::ORDER_U64;
    let canonical = value.as_canonical_u64();
    let negative = canonical > modulus / 2;
    let mut remaining = if negative { modulus - canonical } else { canonical };
    let mut digit_index = 0usize;

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
            let positive_digit = match digit {
                -1 => -F::ONE,
                0 => F::ZERO,
                1 => F::ONE,
                _ => unreachable!("balanced ternary chunk digit"),
            };
            assignment[start + digit_index] = if negative { -positive_digit } else { positive_digit };
            digit_index += 1;
        }
    }
    if remaining != 0 {
        return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
    }
    Ok(())
}
