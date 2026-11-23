use super::*;
use crate::engines::optimized_engine::prefix::early_norm_coefficients;
use neo_ccs::Mat;
use neo_math::{KExtensions, F};

#[test]
fn packed_prefix_matches_each_round_and_the_next_assignment() {
    let w_pos = [0b110011, u64::MAX, 0, 0b1010, 1];
    let w_neg = [0b001100, 0, u64::MAX, 0b101, 0];
    let witness = Mat::compact_signed_unit_from_column_masks(D, w_pos.len(), &w_pos, &w_neg).unwrap();
    let point = [K::ZERO, K::from_coeffs([F::from_u64(19), F::from_u64(7)]), K::ONE];
    let alpha = [
        K::ONE,
        K::ZERO,
        K::from_coeffs([F::from_u64(11), F::from_u64(5)]),
        K::ONE,
        K::ZERO,
        K::from_coeffs([F::from_u64(89), F::from_u64(173)]),
        K::ONE,
        K::ZERO,
        K::ONE,
    ];
    let mut serial = Assignment::new(&witness, D * w_pos.len());
    let mut actual = [Assignment::new(&witness, D * w_pos.len())];
    let packed = PackedNorm::new(&actual, &alpha, K::ONE).unwrap();
    for round in 0..PACKED_NORM_ROUNDS {
        let weights = EqualityWeights::new(&alpha[round + 1..]);
        let expected = early_norm_coefficients(&serial.early_pairs().unwrap(), serial.len().div_ceil(2), &weights);
        assert_eq!(
            packed.coefficients(&point[..round], &alpha[round + 1..PACKED_NORM_ROUNDS]),
            expected
        );
        serial.fold(point[round]);
    }

    let previous = crate::sumcheck::poly_eval_k(&packed.coefficients(&point[..2], &[]), point[2]);
    packed.bind(&mut actual, &point);
    let [actual] = actual;
    assert_eq!(actual.len(), serial.len());
    for slot in 0..actual.len() {
        assert_eq!(actual.get(slot), serial.get(slot));
    }
    let reference = |assignment: &Assignment<'_>, weights: &EqualityWeights| {
        (0..assignment.len().div_ceil(2)).fold([K::ZERO; 4], |total, index| {
            let (low, high) = assignment.pair(index);
            add_coefficients(total, norm_pair(low, high).map(|value| value * weights.at(index)))
        })
    };
    let Assignment::Encoded { codes, values, zero } = actual else {
        panic!("encoded result")
    };
    let weights = EqualityWeights::new(&alpha[4..]);
    let higher = encoded_moments(&codes, &values, zero, &weights);
    let third = super::super::recover_weighted_constant(previous, alpha[3], higher);
    assert_eq!(third, reference(&serial, &weights));
    let challenge = K::from_coeffs([F::from_u64(29), F::from_u64(53)]);
    let weights = EqualityWeights::new(&alpha[5..]);
    let (dense, higher) = bind_encoded(&codes, &values, zero, challenge, &weights);
    serial.fold(challenge);
    let Assignment::Folded(expected) = &serial else {
        panic!("expected dense assignment")
    };
    assert_eq!(&dense, expected);
    let next =
        super::super::recover_weighted_constant(crate::sumcheck::poly_eval_k(&third, challenge), alpha[4], higher);
    assert_eq!(next, reference(&serial, &weights));
}

#[test]
fn six_packed_challenges_match_dense_norm_rounds_and_assignments() {
    use crate::engines::optimized_engine::prefix::{fold, pair};

    let witnesses = [
        ([0xffff, 0, 0x31], [0, 0, 0x200]),
        ([0; 3], [0; 3]),
        ([0x5467, 0, 0], [0, 1, 0]),
        ([0, 1 << 29, 0], [0x53, 0, 1 << 63]),
    ]
    .map(|(positive, negative)| Mat::compact_signed_unit_from_column_masks(D, 3, &positive, &negative).unwrap());
    let alpha = [3, 7, 101, 11, 987, 47, 97, 37].map(|n| K::from_coeffs([F::from_u64(n), F::from_u64(n + 19)]));
    let challenges = [alpha[7], K::ZERO, alpha[6], alpha[1], alpha[2], K::ONE];

    for gamma in [K::ZERO, K::ONE, alpha[0]] {
        let mut assignments = witnesses
            .each_ref()
            .map(|witness| Assignment::new(witness, 3 * D));
        let mut dense = assignments.each_ref().map(|assignment| {
            (0..assignment.len())
                .map(|i| assignment.get(i))
                .collect::<Vec<_>>()
        });
        let packed = PackedNorm::new(&assignments, &alpha, gamma).unwrap();
        for (round, &challenge) in challenges.iter().enumerate() {
            for source in &mut dense {
                fold(source, challenge);
            }
            let point = &challenges[..=round];
            let weights = EqualityWeights::new(&alpha[round + 2..]);
            let actual = match point.len() {
                3 => packed.moments(point, &weights, gamma),
                4 => packed.face::<2>(point, &weights, gamma, &mut assignments),
                5 => packed.face::<4>(point, &weights, gamma, &mut assignments),
                6 => packed.face::<8>(point, &weights, gamma, &mut assignments),
                _ => continue,
            };
            let mut expected = [K::ZERO; 4];
            let mut source_weight = K::ONE;
            for source in &dense {
                for i in 0..source.len().div_ceil(2) {
                    let (low, high) = pair(source, i);
                    let weight = source_weight * weights.at(i);
                    expected = add_coefficients(expected, norm_pair(low, high).map(|coefficient| weight * coefficient));
                }
                source_weight *= gamma;
            }
            assert_eq!(actual, expected[1..], "after {} challenges", point.len());
        }
        for (source, actual) in dense.iter().zip(&assignments) {
            assert_eq!(actual.len(), source.len());
            for (i, &expected) in source.iter().enumerate() {
                assert_eq!(actual.get(i), expected);
            }
        }
    }
}
