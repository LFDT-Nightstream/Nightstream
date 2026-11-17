//! The Π_RLC draw decoder of SPEC.md section 5.3
//! (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample`).

use neo_math::{D, F};
use p3_field::PrimeField64;

/// Within `2^−173` of the uniform law on the challenge set (`PiRlcSampler.distance_lt`).
pub const PI_RLC_DRAW_LANES: usize = 5;

pub fn decode_pi_rlc_coefficients(draw: &[F; PI_RLC_DRAW_LANES]) -> [i8; D] {
    let modulus = u128::from(F::ORDER_U64);
    let mut integer = [0u64; PI_RLC_DRAW_LANES];

    for lane in draw.iter().rev() {
        let mut carry = u128::from(lane.as_canonical_u64());
        for word in &mut integer {
            let product = u128::from(*word) * modulus + carry;
            *word = product as u64;
            carry = product >> 64;
        }
        debug_assert_eq!(carry, 0);
    }

    let mut coefficients = [0i8; D];
    for coefficient in &mut coefficients {
        let mut remainder = 0u128;
        for word in integer.iter_mut().rev() {
            let dividend = (remainder << 64) | u128::from(*word);
            *word = (dividend / 5) as u64;
            remainder = dividend % 5;
        }
        *coefficient = remainder as i8 - 2;
    }
    coefficients
}
