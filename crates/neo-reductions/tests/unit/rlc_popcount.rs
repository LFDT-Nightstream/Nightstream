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

#[test]
fn refresh_popcounts_match_the_general_mix_for_all_supported_counts() {
    let columns = 257;
    let mut rhos = Vec::new();
    let mut witnesses = Vec::new();
    for source in 0..32 {
        let coefficients = std::array::from_fn(|lane| F::from_i32(((source + lane) % 5) as i32 - 2));
        rhos.push(rotation(&coefficients));
        let mut positive = Vec::with_capacity(columns);
        let mut negative = Vec::with_capacity(columns);
        for column in 0..columns {
            let shift = ((column + column / D + source) % D) as u32;
            let pos = 0x35cd_018e_7bf4_a290u64.rotate_left(shift);
            positive.push(pos);
            negative.push(0x8202_dc01_0003_154cu64.rotate_left(shift) & !pos);
        }
        witnesses.push(if source % 5 == 3 {
            Mat::virtual_constant(D, columns, F::ZERO)
        } else {
            masks(&positive, &negative)
        });
    }

    let refs: Vec<_> = witnesses.iter().collect();
    for count in 3..=32 {
        let rhos = &rhos[..count];
        let witnesses = &refs[..count];
        let expected = mix_validated(rhos, witnesses, columns);
        assert_eq!(packed_popcount_mix(rhos, witnesses, columns).unwrap(), expected);
        assert_eq!(rlc_mix_witnesses(D * columns, rhos, witnesses), expected);
    }
}

#[test]
fn refresh_mix_preserves_unsupported_inputs_and_large_signed_totals() {
    let positive = masks(&[u64::MAX, 0], &[0, u64::MAX]);
    let constant = Mat::virtual_constant(D, 2, F::from_u64(7));
    let dense = Mat::from_row_major(D, 2, (0..2 * D).map(F::from_usize).collect());
    let rho = rotation(&[F::TWO; D]);
    let check_fallback = |rhos: &[Mat<F>], witnesses: &[&Mat<F>]| {
        assert!(packed_popcount_mix(rhos, witnesses, 2).is_none());
        assert_eq!(
            rlc_mix_witnesses(D * 2, rhos, witnesses),
            mix_validated(rhos, witnesses, 2)
        );
    };

    let mut rhos = vec![rho; 33];
    let witnesses = [&positive; 33];
    let result = rlc_mix_witnesses(D * 2, &rhos[..32], &witnesses[..32]);
    assert_eq!(result[(D - 1, 0)], F::from_i32(4096));
    assert_eq!(result[(D - 1, 1)], F::from_i32(-4096));
    check_fallback(&rhos, &witnesses);
    for last in [&constant, &dense] {
        check_fallback(&rhos[..3], &[&positive, &positive, last]);
    }
    rhos[2] = rotation(&[-F::from_u64(3); D]);
    check_fallback(&rhos[..3], &witnesses[..3]);
}
