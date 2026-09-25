use neo_ccs::Mat;
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::common::{split_b_matrix_k_with_nonzero_flags, validate_superneo_witness_mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn field(value: i128) -> F {
    let magnitude = F::from_u64(value.unsigned_abs().try_into().unwrap());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn check_split(rows: usize, columns: usize, values: &[i128], count: usize) {
    let input = Mat::from_row_major(rows, columns, values.iter().copied().map(field).collect());
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&input, count, 2).unwrap();
    assert_eq!(digits.len(), count);
    assert_eq!(flags.len(), count);
    let mut expected_flags = vec![false; count];
    for (position, &original) in values.iter().enumerate() {
        let mut remainder = original;
        let mut reconstructed = 0i128;
        let mut weight = 1i128;
        for (plane, digit) in digits.iter().enumerate() {
            let expected = remainder % 2;
            remainder /= 2;
            assert_eq!((digit.rows(), digit.cols()), (rows, columns));
            assert_eq!(digit[(position / columns, position % columns)], field(expected));
            expected_flags[plane] |= expected != 0;
            reconstructed += expected * weight;
            weight *= 2;
        }
        assert_eq!(remainder, 0);
        assert_eq!(reconstructed, original);
    }
    assert_eq!(flags, expected_flags);
    for (digit, nonzero) in digits.iter().zip(flags) {
        if nonzero {
            assert!(digit.is_packed_signed_unit());
            // This assertion fails on the former dense-then-row-major path.
            assert_eq!(digit.packed_signed_unit_column_masks().is_some(), rows <= 64);
        } else {
            assert_eq!(digit.virtual_constant_value(), Some(&F::ZERO));
        }
    }
}

#[test]
fn base_two_split_matches_signed_radix_digits() {
    let count = NeoParams::nightstream_goldilocks_k16().k_rho as usize;
    let bound = 1i128 << count;
    let cases = [-(bound - 1), -(bound / 2), -3, -2, -1, 0, 1, 2, 3, bound / 2, bound - 1];
    let mut values = (0..D * 2)
        .map(|index| cases[index % cases.len()])
        .collect::<Vec<_>>();
    values[D * 2 - 1] = -(bound - 1);
    let input = Mat::from_row_major(D, 2, values.iter().copied().map(field).collect());
    // A mixed CE parent may have a nonzero lane after its logical width.
    // The full two-block carrier, including that lane, must be decomposed.
    validate_superneo_witness_mat(&input, D + 3).unwrap();
    assert_ne!(input[(D - 1, 1)], F::ZERO);
    check_split(D, 2, &values, count);

    for invalid in [bound, -bound, bound + 1, -(bound + 1)] {
        let mut invalid_input = input.clone();
        invalid_input[(D - 1, 1)] = field(invalid);
        assert!(split_b_matrix_k_with_nonzero_flags(&invalid_input, count, 2).is_err());
    }

    // Exercise the existing wide-bound arithmetic branch. These are primitive
    // integer cases, not a production profile with a different digit count.
    let midpoint = i128::from(F::ORDER_U64 / 2);
    check_split(2, 2, &[midpoint, -midpoint, 1, -1], u64::BITS as usize);
}

#[test]
fn base_two_split_keeps_zero_planes_and_tall_matrix_fallback() {
    let count = NeoParams::nightstream_goldilocks_k16().k_rho as usize;
    check_split(D, 2, &vec![0; D * 2], count);
    check_split(2, 2, &[0, 1, -1, 0], count);

    let zero = Mat::virtual_constant(D, 2, F::ZERO);
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&zero, count, 2).unwrap();
    assert_eq!(flags, vec![false; count]);
    assert_eq!(digits.len(), count);
    assert!(digits
        .iter()
        .all(|digit| digit == &zero && digit.is_virtual_constant()));

    // Column masks have an existing 64-row representation limit. Check its
    // boundary and retain the prior generic storage route above that limit.
    for rows in [u64::BITS as usize, u64::BITS as usize + 1] {
        let mut values = vec![0; rows * 2];
        values[rows * 2 - 2] = -3;
        values[rows * 2 - 1] = 2;
        check_split(rows, 2, &values, count);
    }
}
