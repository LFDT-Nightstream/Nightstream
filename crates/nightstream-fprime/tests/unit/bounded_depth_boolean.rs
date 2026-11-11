use super::*;

fn form(constant: u64, terms: &[(usize, i64)]) -> Linear {
    Linear {
        constant: F::from_u64(constant),
        terms: terms
            .iter()
            .map(|&(column, coefficient)| (column, F::from_i64(coefficient)))
            .collect(),
    }
}

fn bit(column: usize) -> Row {
    Row {
        a: form(0, &[(column, 1)]),
        b: form(0, &[(column, 1)]),
        c: form(0, &[(column, 1)]),
    }
}

fn and(left: usize, right: usize, output: usize) -> Row {
    Row {
        a: form(0, &[(left, 1)]),
        b: form(0, &[(right, 1)]),
        c: form(0, &[(output, 1)]),
    }
}

#[test]
fn follows_the_forward_boolean_dependency_order() {
    let rows = [bit(1), bit(2), and(1, 2, 3), and(3, 2, 4), and(4, 1, 5)];
    assert_eq!(
        inferred_columns(&rows, &[false, false, false, true, true, true]),
        vec![3, 4, 5]
    );
    let reversed = [bit(1), bit(2), and(4, 1, 5), and(3, 2, 4), and(1, 2, 3)];
    assert_eq!(inferred_columns(&reversed, &[true; 6]), vec![3]);
}

#[test]
fn recognizes_or_xor_negated_inputs_and_constant_outputs() {
    let or = Row {
        a: form(0, &[(1, 1)]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[(1, 1), (2, 1), (3, -1)]),
    };
    let xor = Row {
        a: form(0, &[(1, 2)]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[(1, 1), (2, 1), (3, -1)]),
    };
    let negated = Row {
        a: form(1, &[(1, -1)]),
        b: form(1, &[(2, -1)]),
        c: form(0, &[(3, 1)]),
    };
    let one = Row {
        a: form(1, &[]),
        b: form(1, &[]),
        c: form(0, &[(3, 1)]),
    };
    for row in [or, xor, negated, one] {
        assert_eq!(inferred_columns(&[bit(1), bit(2), row], &[true; 4]), vec![3]);
    }
}

#[test]
fn canonicalizes_duplicates_and_constant_column_terms() {
    let row = Row {
        a: form(0, &[(0, 1), (1, -2), (1, 1), (99, 3), (99, -3)]),
        b: form(0, &[(2, 2), (2, -1)]),
        c: form(0, &[(3, 4), (3, -3)]),
    };
    assert_eq!(inferred_columns(&[bit(1), bit(2), row], &[true; 4]), vec![3]);
}

#[test]
fn declines_unproved_and_non_boolean_rows() {
    let coefficient = Row {
        a: form(0, &[(1, 2)]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[(3, 1)]),
    };
    let self_read = and(3, 2, 3);
    let cancelled = Row {
        a: form(0, &[(1, 1)]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[(3, 1), (3, -1)]),
    };
    let unknown = and(1, 9, 3);
    let multiple_outputs = Row {
        a: form(0, &[(1, 1)]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[(3, 1), (4, 1)]),
    };
    let three_inputs = Row {
        a: form(0, &[(1, 1)]),
        b: form(0, &[(2, 1), (4, 1)]),
        c: form(0, &[(3, 1)]),
    };
    for row in [
        coefficient,
        self_read,
        cancelled,
        unknown,
        multiple_outputs,
        three_inputs,
    ] {
        assert!(inferred_columns(&[bit(1), bit(2), bit(4), row], &[true; 5]).is_empty());
    }
    assert!(inferred_columns(&[and(1, 2, 3), and(3, 2, 1)], &[true; 4]).is_empty());
    assert!(inferred_columns(&[bit(1), bit(2), and(1, 2, 99)], &[true; 4]).is_empty());
    assert!(inferred_columns(&[bit(1), bit(2), and(1, 2, 3)], &[false; 4]).is_empty());
}

#[test]
fn each_inferred_truth_table_has_only_boolean_outputs() {
    for a_scale in [-2, -1, 0, 1, 2] {
        for b_scale in [-2, -1, 0, 1, 2] {
            for c_scale in [-2, -1, 1, 2] {
                let row = Row {
                    a: form(0, &[(1, a_scale)]),
                    b: form(0, &[(2, b_scale)]),
                    c: form(0, &[(3, c_scale)]),
                };
                let inferred = inferred_columns(&[bit(1), bit(2), row], &[true; 4]);
                let value = F::from_i64(a_scale * b_scale) / F::from_i64(c_scale);
                assert_eq!(!inferred.is_empty(), value == F::ZERO || value == F::ONE);
            }
        }
    }
}
