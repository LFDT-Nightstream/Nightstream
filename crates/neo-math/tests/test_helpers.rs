use neo_math::Fq;
use p3_field::TwoAdicField;

#[allow(dead_code)]
pub const GOLDILOCKS_MODULUS: u128 = 18446744069414584321u128;

#[allow(dead_code)]
pub const TWO_ADICITY: usize = <Fq as TwoAdicField>::TWO_ADICITY;

#[inline]
#[allow(dead_code)]
pub fn two_adic_generator(bits: usize) -> Fq {
    <Fq as TwoAdicField>::two_adic_generator(bits)
}
