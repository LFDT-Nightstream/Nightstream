use super::*;

fn field(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        -F::from_u64(value.unsigned_abs())
    }
}

fn witness(rows: usize, columns: usize, bad: Option<usize>) -> (Mat<F>, Vec<i64>) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let values: Vec<i64> = (0..rows * columns)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match index % 6 {
                0 => 0,
                1 => (state % 65_536) as i64,
                2 => -((state % 65_536) as i64),
                3 => 65_535,
                4 => -65_535,
                _ => (state % 7) as i64 - 3,
            }
        })
        .collect();
    let mut entries: Vec<F> = values.iter().map(|&value| field(value)).collect();
    if let Some(index) = bad {
        entries[index] = F::from_u64(70_000);
    }
    (Mat::from_row_major(rows, columns, entries), values)
}

#[test]
fn packed_split_matches_scalar_balanced_digits_across_column_ranges() {
    let (rows, columns, k) = (54, 3 * CHUNK_COLUMNS + 5, 16);
    let (z, values) = witness(rows, columns, None);
    let (digits, flags) = split_base2_packed(&z, k).expect("every entry is in range");
    let mut used = vec![false; k];
    for row in 0..rows {
        for column in 0..columns {
            let mut value = values[row * columns + column];
            for plane in 0..k {
                let (digit, quotient) = balanced_divrem_i64_base2(value);
                used[plane] |= digit != 0;
                assert_eq!(
                    digits[plane][(row, column)],
                    field(digit),
                    "({row},{column}) plane {plane}"
                );
                value = quotient;
            }
            assert_eq!(value, 0);
        }
    }
    assert_eq!(flags, used);
}

#[test]
fn packed_split_leaves_out_of_range_entries_to_the_serial_error() {
    let (rows, columns) = (54, CHUNK_COLUMNS + 3);
    let (z, _) = witness(rows, columns, Some(7 * columns + CHUNK_COLUMNS + 1));
    assert!(split_base2_packed(&z, 16).is_none());
    let error = super::super::split_b_matrix_k_with_nonzero_flags(&z, 16, 2).unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&format!("Z[7,{}] = 70000", CHUNK_COLUMNS + 1)),
        "{error}"
    );
}
