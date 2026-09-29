use super::*;
use neo_math::KExtensions;

#[test]
fn encoded_prefix_matches_dense_folding_through_the_representation_change() {
    // The last coefficient makes the first folded prefix odd. Both signs and
    // zeros occur in every block; the following positions are implicit zero.
    let mut positive = [0u64; 2];
    let mut negative = [0u64; 2];
    for index in 0..2 * D - 2 {
        match index % 3 {
            0 => positive[index / D] |= 1 << (index % D),
            1 => negative[index / D] |= 1 << (index % D),
            _ => {}
        }
    }
    let packed = Mat::compact_signed_unit_from_column_masks(D, 2, &positive, &negative).unwrap();
    let mut actual = Assignment::new(&packed, 2 * D);
    let mut expected: Vec<K> = (0..2 * D)
        .map(|i| K::from(packed[(i % D, i / D)]))
        .collect();
    let mut saw_encoded = false;
    let mut saw_dense = false;
    for round in 0..(2 * D).next_power_of_two().ilog2() {
        let challenge = K::from_coeffs([F::from_u64(u64::from(round) + 2), F::from_u64(u64::from(round) + 9)]);
        actual.fold(challenge);
        fold(&mut expected, challenge);
        saw_encoded |= matches!(actual, Assignment::Encoded { .. });
        saw_dense |= matches!(actual, Assignment::Folded(_));
        for (index, &expected) in expected.iter().enumerate() {
            assert_eq!(actual.get(index), expected, "round {round}, index {index}");
        }
        assert_eq!(actual.get(expected.len()), K::ZERO);
    }
    assert!(saw_encoded && saw_dense);
}

#[test]
fn early_norm_coefficients_match_the_per_pair_sum() {
    // Three columns with every sign, a zero column and an odd final length.
    let positive = [0b1001_0110u64, 0, 1 << 17];
    let negative = [0b0110_1000u64 | 1 << (D - 1), 0, 0b101];
    let packed = Mat::compact_signed_unit_from_column_masks(D, 3, &positive, &negative).unwrap();
    let mut assignment = Assignment::new(&packed, 3 * D);
    let point: Vec<K> = (0..8u64)
        .map(|index| K::from_coeffs([F::from_u64(11 * index + 5), F::from_u64(3 * index + 7)]))
        .collect();
    let weights = EqualityWeights::new(&point);
    let offset = 3;
    let mut early_rounds = 0;
    for round in 0..5u64 {
        let pairs = assignment.len().div_ceil(2);
        let expected = (0..pairs).fold([K::ZERO; 4], |total, index| {
            let (low, high) = assignment.pair(index);
            let norm = norm_pair(low, high);
            std::array::from_fn(|coefficient| total[coefficient] + norm[coefficient] * weights.at(offset + index))
        });
        if let Some(early) = assignment.early_pairs() {
            assert_eq!(
                early_norm_coefficients(&early, pairs, &weights, offset),
                expected,
                "round {round}"
            );
            early_rounds += 1;
        }
        assignment.fold(K::from_coeffs([F::from_u64(round + 4), F::from_u64(round + 13)]));
    }
    // Signed units, then 9 and 81 encoded values; 6,561 values are too many.
    assert_eq!(early_rounds, 3);
}
