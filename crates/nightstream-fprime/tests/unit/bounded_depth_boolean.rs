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

#[test]
fn affine_rows_infer_outputs_from_either_product_side() {
    let left = Row {
        a: form(0, &[(2, 1), (1, -1)]),
        b: form(1, &[]),
        c: form(0, &[]),
    };
    let right = Row {
        a: form(7, &[]),
        b: form(0, &[(2, 3), (1, -3)]),
        c: form(0, &[]),
    };
    for row in [left, right] {
        assert_eq!(inferred_columns(&[bit(1), row], &[true; 3]), vec![2]);
    }
}

#[test]
fn affine_cancellation_keeps_only_constrained_boolean_outputs() {
    let cancelled_output = Row {
        a: form(0, &[(2, 1)]),
        b: form(3, &[]),
        c: form(0, &[(2, 3)]),
    };
    let scaled_copy = Row {
        a: form(0, &[(2, 2), (1, 1)]),
        b: form(0, &[(0, 3)]),
        c: form(0, &[(2, 3), (1, 6)]),
    };
    let zero_output = Row {
        a: form(0, &[]),
        b: form(0, &[(3, 1)]),
        c: form(0, &[(2, 1)]),
    };
    let free_output = Row {
        a: form(0, &[]),
        b: form(0, &[(2, 1)]),
        c: form(0, &[]),
    };
    let constant_one = Row {
        a: form(0, &[(2, 1)]),
        b: form(7, &[]),
        c: form(7, &[]),
    };
    let doubled_bit = Row {
        a: form(0, &[(2, 1), (1, -2)]),
        b: form(1, &[]),
        c: form(0, &[]),
    };
    let two_unknowns = Row {
        a: form(0, &[(2, 1), (3, -1)]),
        b: form(1, &[]),
        c: form(0, &[]),
    };
    for row in [scaled_copy, zero_output, constant_one] {
        assert_eq!(inferred_columns(&[bit(1), row.clone()], &[true; 4]), vec![2]);
        assert!(inferred_columns(&[bit(1), row], &[false; 4]).is_empty());
    }
    for row in [cancelled_output, free_output, doubled_bit, two_unknowns.clone()] {
        assert!(inferred_columns(&[bit(1), row], &[true; 4]).is_empty());
    }
    assert!(inferred_columns(&[two_unknowns.clone(), two_unknowns], &[true; 4]).is_empty());
    let outside = Row {
        a: form(0, &[(9, 1), (1, -1)]),
        b: form(1, &[]),
        c: form(0, &[]),
    };
    assert!(inferred_columns(&[bit(1), outside], &[true; 4]).is_empty());
}

#[test]
fn affine_inference_matches_the_exact_field_solution() {
    for factor in [-2, -1, 0, 1, 2] {
        for output_left in [-2, -1, 0, 1, 2] {
            for input_left in [-2, -1, 0, 1, 2] {
                for output_right in [-2, -1, 0, 1, 2] {
                    for input_right in [-2, -1, 0, 1, 2] {
                        for constant in [-2, -1, 0, 1, 2] {
                            let variable = form(0, &[(2, output_left), (1, input_left), (0, constant)]);
                            let scalar = form(0, &[(0, factor)]);
                            let c = form(0, &[(2, output_right), (1, input_right)]);
                            let divisor = F::from_i64(factor * output_left - output_right);
                            let expected = divisor != F::ZERO
                                && [F::ZERO, F::ONE].into_iter().all(|input| {
                                    let output = (F::from_i64(input_right - factor * input_left) * input
                                        - F::from_i64(factor * constant))
                                        / divisor;
                                    output == F::ZERO || output == F::ONE
                                });
                            for (a, b) in [(variable.clone(), scalar.clone()), (scalar, variable)] {
                                let found = inferred_columns(&[bit(1), Row { a, b, c: c.clone() }], &[true; 3]);
                                assert_eq!(found == vec![2], expected);
                            }
                        }
                    }
                }
            }
        }
    }
}
