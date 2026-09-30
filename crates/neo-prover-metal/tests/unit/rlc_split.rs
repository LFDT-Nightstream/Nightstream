use super::*;
use neo_reductions::{common::split_b_matrix_k_with_nonzero_flags, optimized_engine::rlc_mix_witnesses};

/// Deterministic signed-unit witness with disjoint positive and negative lanes.
fn witness(seed: u64, columns: usize) -> Mat<F> {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let lanes = (1u64 << D) - 1;
    let (positive, negative): (Vec<_>, Vec<_>) = (0..columns)
        .map(|_| {
            let (signs, nonzero) = (next(), next() & next() & lanes);
            (nonzero & signs, nonzero & !signs)
        })
        .unzip();
    Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap()
}

/// D x D matrix with entries in `-bound..=bound`.
fn rho(seed: u64, bound: i64) -> Mat<F> {
    let entries = (0..D * D)
        .map(|index| {
            let value = ((index as u64 * 2_654_435_761 + seed * 40_503) % (2 * bound as u64 + 1)) as i64 - bound;
            if value < 0 {
                -F::from_u64(value.unsigned_abs())
            } else {
                F::from_u64(value as u64)
            }
        })
        .collect();
    Mat::from_row_major(D, D, entries)
}

fn host_split(rhos: &[Mat<F>], witnesses: &[&Mat<F>], digits: usize) -> Result<(Vec<Mat<F>>, Vec<bool>), ()> {
    let parent = rlc_mix_witnesses(D * witnesses[0].cols(), rhos, witnesses);
    split_b_matrix_k_with_nonzero_flags(&parent, digits, 2).map_err(|_| ())
}

#[test]
fn device_rlc_split_matches_host_mix_and_split() {
    let columns = 1000;
    let session = MetalSession::new().unwrap();
    let zero = Mat::virtual_constant(D, columns, F::ZERO);
    let owned: Vec<_> = (1..=4).map(|seed| witness(seed, columns)).collect();
    let witnesses: Vec<&Mat<F>> = vec![&owned[0], &zero, &owned[1], &owned[2], &owned[3]];
    let rhos: Vec<_> = (0..witnesses.len() as u64)
        .map(|seed| rho(seed, 6))
        .collect();

    let expected = host_split(&rhos, &witnesses, 16).unwrap();
    assert!(expected.1.iter().any(|&used| used) && !expected.1.iter().all(|&used| used));
    let actual = session
        .split_rlc_witnesses(&rhos, &witnesses, 16, 2)
        .unwrap();
    assert_eq!(actual, expected);
    // The planes also keep the host storage: column masks or a zero constant.
    for (actual, expected) in actual.0.iter().zip(&expected.0) {
        assert_eq!(
            actual.packed_signed_unit_column_masks(),
            expected.packed_signed_unit_column_masks()
        );
        assert_eq!(actual.virtual_constant_value(), expected.virtual_constant_value());
    }

    // A parent outside the digit range is an error on both paths.
    assert!(host_split(&rhos, &witnesses, 3).is_err());
    assert!(session
        .split_rlc_witnesses(&rhos, &witnesses, 3, 2)
        .is_err());

    // Only zero witnesses give zero digits.
    let zeros = [&zero, &zero];
    assert_eq!(
        session
            .split_rlc_witnesses(&rhos[..2], &zeros, 16, 2)
            .unwrap(),
        host_split(&rhos[..2], &zeros, 16).unwrap()
    );
}
