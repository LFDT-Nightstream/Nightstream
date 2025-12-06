mod test_helpers;
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use test_helpers::{two_adic_generator, TWO_ADICITY};

#[test]
fn two_adic_generator_orders() {
    let bits = 10.min(TWO_ADICITY);
    let g = two_adic_generator(bits);
    let mut x = g;
    for _ in 0..bits {
        x = x.square();
    } // g^{2^bits} = 1
    assert_eq!(x, Fq::ONE);
}
