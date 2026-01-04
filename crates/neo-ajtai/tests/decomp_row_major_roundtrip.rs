#![allow(non_snake_case)]

use neo_ajtai::{assert_range_b, decomp_b_row_major, DecompStyle};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

#[test]
fn decomp_b_row_major_roundtrips_balanced_b2() {
    let d = 54usize;
    let m = 256usize;
    let b = 2u32;

    // Mix positive/negative and odd/even values to exercise the b=2 fast path.
    let z: Vec<Fq> = (0..m)
        .map(|i| {
            let v = (i as u64).wrapping_mul(17).wrapping_add(5);
            match i % 4 {
                0 => Fq::from_u64(v),
                1 => Fq::ZERO - Fq::from_u64(v),
                2 => Fq::from_u64(v.wrapping_add(1)),
                _ => Fq::ZERO - Fq::from_u64(v.wrapping_add(1)),
            }
        })
        .collect();

    let Z = decomp_b_row_major(&z, b, d, DecompStyle::Balanced);
    assert_range_b(&Z, b).expect("digits must satisfy ||Z||_∞ < b");

    // Recompose each column using the row-major layout: Z[row*m + col].
    let mut z_back = vec![Fq::ZERO; m];
    for col in 0..m {
        let mut pow = Fq::ONE;
        for row in 0..d {
            z_back[col] += Z[row * m + col] * pow;
            pow = pow + pow; // b=2
        }
    }

    assert_eq!(z, z_back, "decomp_b_row_major(b=2,Balanced) must round-trip");
}

