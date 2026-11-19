use super::*;
use crate::engines::optimized_engine::prefix::early_norm_coefficients;
use neo_math::KExtensions;

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
    let packed = PackedNorm::new(&witness, serial.len(), &alpha).unwrap();
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
    let actual = packed.into_assignment(&point);
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
    let third = encoded_moments(&codes, &values, zero, &weights, previous, alpha[3]);
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
