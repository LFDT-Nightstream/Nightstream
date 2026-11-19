use neo_math::{
    field::Q,
    signed_sums::{SignedLimbs, SignedShiftSums, SmallDigits, SmallEventDigits, SplitRing, SMALL_DIGIT_BOUND},
    Fq, Rq, D,
};
use p3_field::PrimeCharacteristicRing;

fn element(seed: u64) -> [Fq; D] {
    let mut x = seed;
    core::array::from_fn(|lane| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        match lane % 9 {
            0 => -Fq::ONE,
            4 => Fq::ZERO,
            _ => Fq::from_u64(x),
        }
    })
}

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
        (element(3), 0, u64::MAX),
        (element(4), 0b1011 << 20, 0b0100 << 20 | 1 << (D - 1)),
        (element(5), 0, 0),
        (element(6), u64::MAX, 0),
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
    let coefficients = [-Fq::ONE; D];
    let mut sums = SignedShiftSums::zero();
    for _ in 0..10_000 {
        sums.add_signed_units(&SplitRing::new(&coefficients), 0, u64::MAX);
    }
    let product = Rq(coefficients).mul(&signed_unit(0, u64::MAX));
    let expected: [Fq; D] = core::array::from_fn(|lane| product.0[lane] * Fq::from_u64(10_000));
    assert_eq!(sums.reduce(), expected);
}

#[test]
fn centered_event_digits_match_the_canonical_range_check() {
    let real = element(5);
    let imaginary = element(13);
    for word in [0, 1, 2048, 2049, Q / 2, Q / 2 + 1, 1 << 63, Q - 2049, Q - 2048, Q - 1] {
        let mut digits = [Fq::ZERO; D];
        digits[1] = Fq::from_u64(Q - 2048);
        digits[D - 1] = Fq::from_u64(word);
        let mut reference = [SignedShiftSums::zero(), SignedShiftSums::zero()];
        let expected =
            SignedShiftSums::add_small_event_pair(reference.each_mut(), [&real, &imaginary], u64::MAX, &digits);
        let centered = SmallEventDigits::from_centered(&digits);
        assert_eq!(centered.is_some(), expected);
        if let Some(centered) = centered {
            let mut actual = [SignedShiftSums::zero(), SignedShiftSums::zero()];
            SignedShiftSums::add_prepared_event_pair(actual.each_mut(), [&real, &imaginary], u64::MAX, &centered);
            assert_eq!(actual.map(|sums| sums.reduce()), reference.map(|sums| sums.reduce()));
        }
    }
}

#[test]
fn signed_events_match_mixed_negacyclic_sums_at_the_term_bound() {
    let mut coefficients = [Fq::ZERO; D];
    coefficients[0] = -Fq::ONE;
    coefficients[D - 1] = -Fq::ONE;
    let positive = u64::MAX ^ (1 << (D - 1));
    let negative = (1 << 1) | (1 << (D - 1));
    let mut sums = SignedShiftSums::zero();
    let mut imaginary = SignedShiftSums::zero();
    let imaginary_coefficients = coefficients.map(|v| -v);
    SignedShiftSums::add_signed_event_pair(
        [&mut sums, &mut imaginary],
        [&coefficients, &imaginary_coefficients],
        1 | (1 << (D - 1)),
        positive,
        negative,
    );
    sums.add_signed_units(&SplitRing::new(&coefficients), 1, 0);
    let product = Rq(coefficients).mul(&signed_unit(positive & !negative, negative & !positive));
    assert_eq!(imaginary.reduce(), product.0.map(|v| -v));
    let terms = neo_math::signed_sums::MAX_EXACT_TERMS;
    for _ in 0..(terms.ilog2() - 1) {
        sums.add(&sums.clone());
    }
    assert_eq!(
        sums.reduce(),
        core::array::from_fn(|lane| (product.0[lane] + coefficients[lane]) * Fq::from_u64(terms / 2)),
    );
}

fn digits(seed: u64, bound: i64) -> [Fq; D] {
    let mut x = seed;
    core::array::from_fn(|lane| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let digit = match lane % 7 {
            0 => bound,
            1 => -bound,
            _ => (x >> 1) as i64 % (2 * bound + 1) - bound,
        };
        Fq::from_i64(digit)
    })
}

fn limb_edges(seed: u64) -> [Fq; D] {
    let edges = [
        0,
        1,
        (1 << 31) - 1,
        1 << 31,
        (1 << 63) - (1 << 31) - 1,
        (1 << 63) - (1 << 31),
        1 << 63,
        Q - (1 << 31),
        Q - 1,
    ];
    let mut values = element(seed);
    for (value, edge) in values.iter_mut().zip(edges) {
        *value = Fq::from_u64(edge);
    }
    values
}

#[test]
fn small_digit_products_match_ring_products() {
    let bound = SMALL_DIGIT_BOUND as i64;
    let largest_high_limb = [Fq::from_u64((1 << 63) - (1 << 31) - 1); D];
    let smallest_low_limb = [Fq::from_u64(Q - (1 << 31)); D];
    for seed in 0..64 {
        let all_digits = [
            digits(seed, bound),
            digits(seed, 16_256),
            digits(seed, 2),
            [Fq::from_i64(bound); D],
            [Fq::from_i64(-bound); D],
        ];
        for digits in all_digits {
            let small = SmallDigits::new(&digits).expect("digits within the bound");
            for form in [element(seed), limb_edges(seed), largest_high_limb, smallest_low_limb] {
                assert_eq!(small.mul(&SignedLimbs::new(&form)), Rq(form).mul(&Rq(digits)).0);
            }
        }
    }
}

#[test]
fn small_digits_reject_values_above_the_bound() {
    let bound = SMALL_DIGIT_BOUND as i64;
    for digit in [bound + 1, -bound - 1, 1 << 40] {
        let mut digits = digits(7, 3);
        digits[D - 1] = Fq::from_i64(digit);
        assert!(SmallDigits::new(&digits).is_none());
    }
}

#[test]
fn small_digits_report_signed_unit_masks() {
    let mut digits = [Fq::ZERO; D];
    digits[0] = Fq::ONE;
    digits[D - 1] = -Fq::ONE;
    assert_eq!(
        SmallDigits::new(&digits).unwrap().signed_unit_masks(),
        Some((1, 1 << (D - 1)))
    );
    digits[5] = Fq::TWO;
    assert_eq!(SmallDigits::new(&digits).unwrap().signed_unit_masks(), None);
}
