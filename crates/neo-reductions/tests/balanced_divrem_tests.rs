use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::split_b_matrix_k;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn test_split_b_matrix_k_simple() {
    let b = 2u32;
    let k = 12usize;

    #[allow(non_snake_case)]
    let Z = Mat::from_row_major(
        2,
        2,
        vec![F::from_u64(4), F::from_u64(2), F::ZERO - F::from_u64(2), F::from_u64(1)],
    );

    let result = split_b_matrix_k(&Z, k, b);
    assert!(
        result.is_ok(),
        "Should successfully split simple matrix: {:?}",
        result.err()
    );

    let split_matrices = result.unwrap();
    assert_eq!(split_matrices.len(), k, "Should produce k={} output matrices", k);

    for (i, mat) in split_matrices.iter().enumerate() {
        assert_eq!(
            mat.rows(),
            Z.rows(),
            "Split matrix {} should have same number of rows",
            i
        );
        assert_eq!(
            mat.cols(),
            Z.cols(),
            "Split matrix {} should have same number of cols",
            i
        );
    }

    let mut reconstructed = Mat::zero(Z.rows(), Z.cols(), F::ZERO);
    let mut power = F::ONE;
    let b_field = F::from_u64(b as u64);

    for mat in &split_matrices {
        for r in 0..Z.rows() {
            for c in 0..Z.cols() {
                reconstructed[(r, c)] += mat[(r, c)] * power;
            }
        }
        power *= b_field;
    }

    for r in 0..Z.rows() {
        for c in 0..Z.cols() {
            assert_eq!(
                reconstructed[(r, c)].as_canonical_u64(),
                Z[(r, c)].as_canonical_u64(),
                "Reconstructed Z[{},{}] should match original",
                r,
                c
            );
        }
    }
}

#[test]
fn test_split_b_matrix_k_base3_accepts_full_norm_bound_digit() {
    let b = 3u32;
    let k = 1usize;
    #[allow(non_snake_case)]
    let Z = Mat::from_row_major(1, 2, vec![F::from_u64(2), F::ZERO - F::from_u64(2)]);

    let split = split_b_matrix_k(&Z, k, b).expect("values with |z| < b must split in one digit");

    assert_eq!(split.len(), 1);
    assert_eq!(split[0][(0, 0)], F::from_u64(2));
    assert_eq!(split[0][(0, 1)], F::ZERO - F::from_u64(2));
}

#[test]
fn split_b_matrix_round_trip_preserves_signed_values_and_digit_bound() {
    let base = 4u32;
    let digit_count = 6usize;
    let signed_values = [-100i64, -25, -4, -3, -1, 0, 1, 3, 4, 17, 63, 99, -77, 28, 2, -2];
    let values = signed_values
        .into_iter()
        .map(|value| {
            if value < 0 {
                F::ZERO - F::from_u64(value.unsigned_abs())
            } else {
                F::from_u64(value as u64)
            }
        })
        .collect();
    let input = Mat::from_row_major(2, 8, values);
    let digits = split_b_matrix_k(&input, digit_count, base).expect("signed values must decompose");

    let mut reconstructed = Mat::zero(input.rows(), input.cols(), F::ZERO);
    let mut power = F::ONE;
    let base_field = F::from_u64(base as u64);
    let bound = (base - 1) as u64;
    for digit in &digits {
        for row in 0..input.rows() {
            for column in 0..input.cols() {
                let canonical = digit[(row, column)].as_canonical_u64();
                assert!(
                    canonical <= bound || canonical >= F::ORDER_U64 - bound,
                    "digit at ({row},{column}) exceeds the strict SuperNeo bound"
                );
                reconstructed[(row, column)] += power * digit[(row, column)];
            }
        }
        power *= base_field;
    }
    assert_eq!(reconstructed, input);
}
