mod test_helpers;
use neo_math::{Fq, KExtensions, K};
use p3_field::PrimeCharacteristicRing;
use test_helpers::{two_adic_generator, GOLDILOCKS_MODULUS, TWO_ADICITY};

#[test]
fn goldilocks_two_adicity_is_32() {
    assert_eq!(TWO_ADICITY, 32);
    let g = two_adic_generator(32);
    // g has order 2^32 ⇒ g^(2^32) = 1 and g^(2^31) != 1.
    let mut x = g;
    for _ in 0..32 {
        x = x.square();
    }
    assert_eq!(x, Fq::ONE);
}

#[test]
fn nonresidue_7_is_non_square_mod_q() {
    // Euler's criterion: a^((p-1)/2) ≡ -1 mod p for non-residue a.
    let p = GOLDILOCKS_MODULUS;
    let a = 7u128;
    let e = (p - 1) / 2;
    let r = modexp(a, e, p);
    assert_eq!(r, p - 1);
}

fn modexp(mut b: u128, mut e: u128, m: u128) -> u128 {
    let mut acc = 1u128;
    b %= m;
    while e > 0 {
        if (e & 1) == 1 {
            acc = acc * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    acc
}

#[test]
fn k_conjugation_and_inverse() {
    let a = K::new_complex(Fq::from_u64(5), Fq::from_u64(9));
    let inv = a.inv();
    assert_eq!(a * inv, K::ONE);
    assert_eq!(a.conj().conj(), a);
}
