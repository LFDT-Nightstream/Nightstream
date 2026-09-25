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
