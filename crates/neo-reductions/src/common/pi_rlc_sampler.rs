//! Whole-vector PiRLC decoding. Input lanes are canonical Goldilocks values;
//! output coefficients are the first 54 base-five digits of X, minus two.

use neo_math::{D, F};
use p3_field::PrimeField64;

pub fn decode_pi_rlc_coefficients(digest: &[F; 4]) -> [i8; D] {
    const MODULUS: u128 = 0xffff_ffff_0000_0001;
    let mut integer = [0u64; 4];

    for lane in digest.iter().rev() {
        let mut carry = u128::from(lane.as_canonical_u64());
        for word in &mut integer {
            let product = u128::from(*word) * MODULUS + carry;
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
