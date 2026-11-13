use neo_math::{
    signed_sums::{wide256_halves, SignedShiftSums, SplitRing, LIMB_BOUND},
    Fq, Rq, D,
};
use num_bigint::BigUint;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

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
fn wide_integer_limbs_match_division_and_signed_ring_products() {
    let mut random = 0x1234_5678_9abc_def0u64;
    let mut actual = SignedShiftSums::zero();
    let mut expected = Rq::zero();
    for case in 0..128 {
        let words: [[u32; 8]; D] = core::array::from_fn(|lane| match (case + lane) % 12 {
            0 => [0; 8],
            1 => [u32::MAX; 8],
            2 => [1, u32::MAX, 0, 0, 0, 0, 0, 0],
            3 => [2, u32::MAX, 0, 0, 0, 0, 0, 0],
            4 => [0, 0, 0, 1, 0, 0, 0, 0],
            5 => [0, 0, 0, 0, 0, 1, 0, 0],
            _ => core::array::from_fn(|_| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                (random >> 32) as u32
            }),
        });
        let coefficients = words.map(|words| {
            Fq::from_u64(words.iter().rev().fold(0u128, |value, &word| {
                ((value << 32) + u128::from(word)) % u128::from(Fq::ORDER_U64)
            }) as u64)
        });
        let positive = random.rotate_left(17);
        let negative = !positive;
        actual.add_signed_units(&SplitRing::from_wide256(words), positive, negative);
        actual.add_signed_units(&SplitRing::from_wide256(words), positive, positive);
        expected = expected.add(&Rq(coefficients).mul(&signed_unit(positive, negative)));
    }
    assert_eq!(actual.reduce(), expected.0);
}

fn wide_value(words: [u32; 8]) -> BigUint {
    words
        .iter()
        .rev()
        .fold(BigUint::ZERO, |value, &word| (value << 32) + word)
}

#[test]
fn wide_halves_match_the_big_integer_and_stay_below_the_limb_bound() {
    let q = BigUint::from(Fq::ORDER_U64);
    let mut random = 0x0dd_ba11u64;
    let mut cases = vec![
        [0; 8],
        [u32::MAX; 8],
        [0, 0, 0, 0, 0, 0, 0, u32::MAX],
        [u32::MAX, 0, 0, 0, 0, 0, 0, 0],
    ];
    cases.extend((0..8).map(|word| core::array::from_fn(|i| u32::from(i == word))));
    cases.extend((0..64).map(|_| {
        core::array::from_fn(|_| {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            (random >> 32) as u32
        })
    }));

    let weight_sum = 1 + 59 + 59 * 59 + 59 * 59 * 59;
    assert_eq!(LIMB_BOUND, (1 << 32) + 59 * weight_sum);
    for words in cases {
        let (a, b) = wide256_halves(words);
        assert!(u128::from(a.max(b)) <= weight_sum * u128::from(u32::MAX));
        assert_eq!(
            (BigUint::from(a) + (BigUint::from(b) << 32)) % &q,
            wide_value(words) % &q
        );
    }
    assert_eq!(
        wide256_halves([u32::MAX; 8]).0,
        (weight_sum * u128::from(u32::MAX)) as u64
    );
}

#[test]
fn extreme_wide_terms_stay_exact_over_many_additions() {
    let q = BigUint::from(Fq::ORDER_U64);
    let words = [[u32::MAX; 8]; D];
    let coefficient = Fq::from_u64((wide_value([u32::MAX; 8]) % &q).try_into().unwrap());
    let element = SplitRing::from_wide256(words);
    let mut sums = SignedShiftSums::zero();
    for _ in 0..50_000 {
        sums.add_signed_units(&element, u64::MAX, 0);
    }
    let product = Rq([coefficient; D]).mul(&Rq([Fq::ONE; D]));
    let expected: [Fq; D] = core::array::from_fn(|lane| product.0[lane] * Fq::from_u64(50_000));
    assert_eq!(sums.reduce(), expected);
}
