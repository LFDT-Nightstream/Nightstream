//! Balanced base-3 digits of a centered field word (`encode_field`).

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

const CENTERED_HALF_MODULUS: u64 = (Goldilocks::ORDER_U64 - 1) / 2;

pub(crate) fn encode_field(value: u64, output: &mut [i8]) {
    let negative = value > CENTERED_HALF_MODULUS;
    let mut magnitude = if negative { Goldilocks::ORDER_U64 - value } else { value };
    for coordinate in output {
        let remainder = magnitude % 3;
        let unsigned = match remainder {
            0 => 0,
            1 => 1,
            _ => -1,
        };
        *coordinate = if negative { -unsigned } else { unsigned };
        magnitude = magnitude / 3 + u64::from(remainder == 2);
    }
    debug_assert_eq!(magnitude, 0);
}
