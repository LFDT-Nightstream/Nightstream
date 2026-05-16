//! Owns low-norm source encoding checks for terminal Construction-2 witnesses.

use neo_math::{balanced::to_balanced_i128, D, F};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::types::{TerminalPrivateColumnEncoding, U32_BIT_WIDTH, U64_BIT_WIDTH};

pub(crate) fn committed_nc_range_error(
    params: &NeoParams,
    full_vector: &[F],
    mut committed_index_label: impl FnMut(usize) -> String,
    context: &str,
) -> Option<String> {
    for (idx, value) in full_vector.iter().copied().enumerate() {
        if is_superneo_digit_representable(value, params.b) {
            continue;
        }
        return Some(format!(
            "{context} committed value at {} is not representable in D={} balanced base-{} digits (centered value {})",
            committed_index_label(idx),
            D,
            params.b,
            to_balanced_i128(value),
        ));
    }
    None
}

pub(crate) fn low_norm_encoded_values(
    value: F,
    encoding: TerminalPrivateColumnEncoding,
    context: &str,
) -> Result<Vec<F>, String> {
    match encoding {
        TerminalPrivateColumnEncoding::UnusedPadding => {
            if value != F::ZERO {
                return Err(format!(
                    "{context} padded witness value is non-zero: {}",
                    value.as_canonical_u64()
                ));
            }
            Ok(Vec::new())
        }
        TerminalPrivateColumnEncoding::Bit => {
            let canonical = value.as_canonical_u64();
            if canonical > 1 {
                return Err(format!("{context} boolean witness value is not binary: {canonical}"));
            }
            Ok(vec![value])
        }
        TerminalPrivateColumnEncoding::U32 => low_norm_bit_values(value, U32_BIT_WIDTH, context),
        TerminalPrivateColumnEncoding::U64 => low_norm_bit_values(value, U64_BIT_WIDTH, context),
    }
}

fn low_norm_bit_values(value: F, bit_width: usize, context: &str) -> Result<Vec<F>, String> {
    let canonical = value.as_canonical_u64();
    if bit_width < U64_BIT_WIDTH && (canonical >> bit_width) != 0 {
        return Err(format!(
            "{context} witness value {canonical} does not fit in {bit_width} base-2 digits"
        ));
    }
    Ok((0..bit_width)
        .map(|bit_idx| F::from_u64((canonical >> bit_idx) & 1))
        .collect())
}

fn is_superneo_digit_representable(value: F, base: u32) -> bool {
    if base < 2 {
        return false;
    }
    let mut remainder = to_balanced_i128(value);
    let base = base as i128;
    for _ in 0..D {
        let (_, quotient) = balanced_divrem(remainder, base);
        remainder = quotient;
    }
    remainder == 0
}

fn balanced_divrem(value: i128, base: i128) -> (i128, i128) {
    debug_assert!(base >= 2);
    let mut remainder = value % base;
    let mut quotient = (value - remainder) / base;
    let half = base / 2;
    if remainder > half {
        remainder -= base;
        quotient += 1;
    } else if remainder < -half {
        remainder += base;
        quotient -= 1;
    }
    (remainder, quotient)
}
