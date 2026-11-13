//! The Π_RLC draw decoder of SPEC.md section 5.3
//! (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample`).

use neo_math::{D, F};
use p3_field::PrimeField64;

/// The field lanes of one draw. `q^5 / 5^64 > 2^171`, so the law of a challenge is within
/// `2^−132` of the uniform law on the challenge set. Four lanes give only about `2^−109`.
pub const PI_RLC_DRAW_LANES: usize = 5;

/// `N = Σ_k draw[k] · q^k`, lane 0 least significant. Coefficient `h` of the challenge is
/// digit `h` of `N mod 5^D` in base five, minus two.
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

    // The first `D` base-five digits of `N` are the digits of `N mod 5^D`.
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
