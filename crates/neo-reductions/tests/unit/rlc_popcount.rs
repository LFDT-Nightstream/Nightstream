use super::*;
use neo_math::F;

fn rotation(coefficients: &[F; D]) -> Mat<F> {
    Mat::from_row_major(
        D,
        D,
        (0..D)
            .flat_map(|row| {
                (0..D).map(move |col| {
                    if row >= col {
                        coefficients[row - col]
                    } else {
                        -coefficients[row + D - col]
                    }
                })
            })
            .collect(),
    )
}

fn masks(positive: &[u64], negative: &[u64]) -> Mat<F> {
    Mat::compact_signed_unit_from_column_masks(D, positive.len(), positive, negative).unwrap()
}

#[test]
fn light_popcount_matches_general_mix_and_keeps_large_coefficient_fallback() {
    let fresh = masks(&[u64::MAX, 1 << 63, 101], &[0, 1, 152]);
    let running = Mat::from_row_major(D, 3, (0..D * 3).map(F::from_usize).collect());
    let zero = Mat::virtual_constant(D, 3, F::ZERO);
    let coefficients = std::array::from_fn(|lane| F::from_i32(lane as i32 % 5 - 2));
    let rhos = [rotation(&coefficients), Mat::identity(D)];
    for previous in [&running, &zero] {
        assert_eq!(
            light_popcount_mix(&rhos, &[&fresh, previous]).unwrap(),
            mix_validated(&rhos, &[&fresh, previous], 3)
        );
    }
    let mut large = coefficients;
    large[0] = F::from_u64(3);
    let rhos = [rotation(&large), Mat::identity(D)];
    assert!(light_popcount_mix(&rhos, &[&fresh, &running]).is_none());
    assert_eq!(
        rlc_mix_witnesses(D * 3, &rhos, &[&fresh, &running]),
        mix_validated(&rhos, &[&fresh, &running], 3)
    );
}
