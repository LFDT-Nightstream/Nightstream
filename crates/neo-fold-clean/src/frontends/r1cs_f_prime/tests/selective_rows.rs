use super::*;
use crate::engine::r1cs_circuit::builder::{ShiftedTernaryCanonicalTrace, BALANCED_TERNARY_DIGITS};

fn decomposition(index: usize) -> BalancedTernaryDecomposition {
    let digit_start = 1 + index * (BALANCED_TERNARY_DIGITS + 1);
    BalancedTernaryDecomposition {
        field_col: digit_start - 1,
        digit_cols: core::array::from_fn(|digit| digit_start + digit),
    }
}

#[test]
fn wasm_scale_balanced_ternary_index_is_complete() {
    const WASM_PROFILE_ENTRIES: usize = 190_576;

    let decompositions = (0..WASM_PROFILE_ENTRIES)
        .map(decomposition)
        .collect::<Vec<_>>();
    let index = balanced_ternary_decompositions_by_digit_start(&decompositions).expect("unique decomposition index");

    assert_eq!(index.len(), decompositions.len());
    for expected in &decompositions {
        let actual = index
            .get(&expected.digit_cols[0])
            .copied()
            .expect("indexed decomposition");
        assert!(core::ptr::eq(actual, expected));
    }
}

#[test]
fn balanced_ternary_index_rejects_duplicate_digit_starts() {
    let first = decomposition(0);
    let mut duplicate = decomposition(1);
    duplicate.digit_cols = first.digit_cols;

    let error = balanced_ternary_decompositions_by_digit_start(&[first, duplicate])
        .expect_err("duplicate digit starts must fail closed");
    assert!(
        error
            .to_string()
            .contains("balanced-ternary decompositions have duplicate digit starts"),
        "unexpected error: {error}"
    );
}

#[test]
fn balanced_ternary_index_does_not_trust_input_order() {
    let decompositions = [decomposition(2), decomposition(0), decomposition(1)];
    let index = balanced_ternary_decompositions_by_digit_start(&decompositions).expect("unordered unique index");

    for expected in &decompositions {
        assert!(core::ptr::eq(
            index
                .get(&expected.digit_cols[0])
                .copied()
                .expect("indexed decomposition"),
            expected
        ));
    }
}

#[test]
fn shifted_ternary_validation_keeps_exact_word_checks() {
    let decomposition = decomposition(0);
    let trace = ShiftedTernaryCanonicalTrace {
        field_column: decomposition.field_col,
        digit_columns_start: decomposition.digit_cols[0],
        negative_columns_start: decomposition.digit_cols[0] + BALANCED_TERNARY_DIGITS,
        borrow_columns_start: decomposition.digit_cols[0] + 2 * BALANCED_TERNARY_DIGITS,
        digit_rows_start: 0,
        reconstruction_row: 2 * BALANCED_FIELD_WIDTH,
        transition_rows_start: 2 * BALANCED_FIELD_WIDTH + 1,
    };
    validate_shifted_ternary_reconstruction_row(&decomposition, &trace).expect("exact shifted-ternary word");

    let mut wrong_field = trace;
    wrong_field.field_column += 1;
    assert!(validate_shifted_ternary_reconstruction_row(&decomposition, &wrong_field).is_err());

    let mut wrong_rows = trace;
    wrong_rows.reconstruction_row += 1;
    assert!(validate_shifted_ternary_reconstruction_row(&decomposition, &wrong_rows).is_err());
}
