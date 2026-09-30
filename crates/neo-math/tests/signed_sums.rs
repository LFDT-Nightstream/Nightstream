use neo_math::{
    signed_sums::{SignedShiftSums, SplitRing},
    Fq, Rq, D,
};
use p3_field::PrimeCharacteristicRing;

fn element(seed: u64) -> [Fq; D] {
    let mut x = seed;
    core::array::from_fn(|lane| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Include coefficients at and near the top of the field.
        match lane % 9 {
            0 => -Fq::ONE,
            4 => Fq::ZERO,
            _ => Fq::from_u64(x),
        }
    })
}

/// The ring element with +1 on positive lanes and -1 on negative lanes.
fn signed_unit(positive: u64, negative: u64) -> Rq {
    Rq(core::array::from_fn(|lane| {
        match (positive >> lane & 1, negative >> lane & 1) {
            (1, _) => Fq::ONE,
            (_, 1) => -Fq::ONE,
            _ => Fq::ZERO,
        }
    }))
}

#[test]
fn signed_shift_sums_match_ring_products_after_one_reduction() {
    let terms = [
        (element(1), 1u64, 0u64),
        (element(2), 1 << (D - 1), 0),
        (element(3), 0, (1 << D) - 1),
        (element(4), 0b1011 << 20, 0b0100 << 20 | 1 << (D - 1)),
        (element(5), 0, 0),
        (element(6), (1 << D) - 1, 0),
    ];
    let mut expected = Rq::zero();
    let mut first = SignedShiftSums::zero();
    let mut second = SignedShiftSums::zero();
    for (index, (coefficients, positive, negative)) in terms.iter().enumerate() {
        let product = Rq(*coefficients).mul(&signed_unit(*positive, *negative));
        for lane in 0..D {
            expected.0[lane] += product.0[lane];
        }
        let sums = if index % 2 == 0 { &mut first } else { &mut second };
        sums.add_signed_units(&SplitRing::new(coefficients), *positive, *negative);
    }
    first.add(&second);
    assert_eq!(first.reduce(), expected.0);
}

#[test]
fn repeated_negative_terms_stay_exact() {
    // Many subtractions of the largest coefficients drive raw sums far below zero.
    let coefficients = [-Fq::ONE; D];
    let mut sums = SignedShiftSums::zero();
    for _ in 0..10_000 {
        sums.add_signed_units(&SplitRing::new(&coefficients), 0, (1 << D) - 1);
    }
    let product = Rq(coefficients).mul(&signed_unit(0, (1 << D) - 1));
    let expected: [Fq; D] = core::array::from_fn(|lane| product.0[lane] * Fq::from_u64(10_000));
    assert_eq!(sums.reduce(), expected);
}
