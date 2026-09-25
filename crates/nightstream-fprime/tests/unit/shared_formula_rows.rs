use super::*;
use crate::components::SparseForm;
use p3_field::PrimeField64;
use serde_json::json;
use std::ops::ControlFlow;

#[allow(dead_code)]
#[path = "../per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

#[path = "../support/poseidon_affine_fixture.rs"]
mod affine_fixture;

fn equal_row(actual: &RowForms, expected: &reference::RowForms, ordinal: usize) {
    for port in 0..MEANINGFUL_PORTS {
        let actual: Vec<_> = actual[port]
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
            .collect();
        let expected: Vec<_> = expected[port]
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.canonical()))
            .collect();
        assert_eq!(actual, expected, "row {ordinal}, matrix port {port}");
    }
    assert!(expected[MEANINGFUL_PORTS].entries().is_empty());
}

#[test]
fn shared_poseidon_templates_match_every_reference_row() {
    let logical_width = 12_000;
    let external_input = json!([[[0, 1, 0, 8], [2, [2, 8, 11_000], 0, 0]]]);
    let blocks = [
        json!([1, 0, [2, 86, 100], []]),
        json!([3, 0, [2, 3 * 86, 100], affine_fixture::program()]),
        json!([1, 0, [2, 86, 100], external_input]),
    ];
    for encoded in blocks {
        let actual = poseidon::Block::decode(&encoded).unwrap();
        let expected = reference::poseidon::Block::decode(&encoded, logical_width).unwrap();
        let count = actual.row_count().unwrap();
        assert_eq!(count, expected.row_count().unwrap());
        let mut visited = Vec::new();
        let flow = actual
            .visit_rows_until(logical_width, 0, count, |row| {
                visited.push(owned_row(row));
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(flow, ControlFlow::Continue(()));
        assert_eq!(visited.len(), count);
        for (ordinal, row) in visited.iter().enumerate() {
            equal_row(row, &expected.row(logical_width, ordinal).unwrap(), ordinal);
            assert_eq!(*row, actual.row(logical_width, ordinal).unwrap());
        }
        // A retained field remains one operator through formula substitution.
        let output = &visited[0][4];
        assert_eq!(output.terms().len(), 1);
        assert_eq!(output.terms()[0].column_count(), 41);
        assert_eq!(output.entries().len(), 41);
        let mut partial = Vec::new();
        let terminal_round_start = 4 * 8 + 22;
        let flow = actual
            .visit_rows_until(logical_width, terminal_round_start, count - 1, |row| {
                partial.push(owned_row(row));
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(flow, ControlFlow::Continue(()));
        assert_eq!(partial, visited[terminal_round_start..count - 1]);
        assert!(actual.row(logical_width, count).is_err());
        assert!(actual
            .visit_rows_until(logical_width, count, count - 1, |_| Ok(ControlFlow::Continue(())))
            .is_err());
    }
}

#[test]
fn shared_phi81_templates_match_all_points_sources_blocks_and_components() {
    // Ring order crosses sources, blocks, extension components, and families.
    let logical_width = 4_000;
    let encoded = json!([
        [[2, 2, 2], [1, 1, 1]],
        0,
        [0, 108, 20],
        0,
        54,
        [[[0, 486, [0, 486, 128], 0]], []],
        [0, 486, 614],
        [0, 486, 1100]
    ]);
    let actual = phi81::Block::decode(&encoded).unwrap();
    let expected = reference::phi81::Block::decode(&encoded, logical_width).unwrap();
    let count = actual.row_count().unwrap();
    assert_eq!(count, 9 * 108);
    assert_eq!(count, expected.row_count().unwrap());
    let mut visited = Vec::new();
    let flow = actual
        .visit_rows_until(logical_width, 0, count, |row| {
            visited.push(owned_row(row));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    assert_eq!(visited.len(), count);
    for (ordinal, row) in visited.iter().enumerate() {
        equal_row(row, &expected.row(logical_width, ordinal).unwrap(), ordinal);
        assert_eq!(*row, actual.row(logical_width, ordinal).unwrap());
    }
    let mut partial = Vec::new();
    let flow = actual
        .visit_rows_until(logical_width, 107, count - 1, |row| {
            partial.push(owned_row(row));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    assert_eq!(partial, visited[107..count - 1]);
    assert!(actual.row(logical_width, count).is_err());
    assert!(actual
        .visit_rows_until(logical_width, count, count - 1, |_| Ok(ControlFlow::Continue(())))
        .is_err());
}

#[test]
fn phi81_quotient_rows_accept_product_and_reject_omitted_node_attack() {
    let logical_width = 217;
    let block = phi81::Block::decode(&json!([
        [[1, 1, 1]],
        0,
        [0, 54, 1],
        0,
        54,
        [[[0, 54, [0, 54, 55], 0]], []],
        [0, 54, 109],
        [0, 54, 163]
    ]))
    .unwrap();
    let mut rows = Vec::new();
    let flow = block
        .visit_rows_until(logical_width, 0, 108, |row| {
            rows.push(owned_row(row));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    let residuals = |values: &[Goldilocks]| {
        rows.iter()
            .map(|row| {
                let evaluate = |port: usize| {
                    row[port]
                        .entries()
                        .iter()
                        .fold(Goldilocks::ZERO, |sum, entry| {
                            sum + entry.coefficient * values[entry.column]
                        })
                };
                evaluate(7) * (evaluate(0) * evaluate(2) - evaluate(4))
            })
            .collect::<Vec<_>>()
    };
    let mut values = vec![Goldilocks::ZERO; logical_width];
    values[0] = Goldilocks::ONE;
    values[1..55].fill(Goldilocks::from_u64(2));
    // X^53 * X^53 = X^25 + Phi81 * (X^52 - X^25).
    values[1 + 53] += Goldilocks::ONE;
    values[55 + 53] = Goldilocks::ONE;
    values[109 + 25] = Goldilocks::ONE;
    values[163 + 52] = Goldilocks::ONE;
    values[163 + 25] = -Goldilocks::ONE;
    assert!(residuals(&values)
        .iter()
        .all(|value| *value == Goldilocks::ZERO));
    values[109 + 25] += Goldilocks::ONE;
    assert!(residuals(&values)
        .iter()
        .any(|value| *value != Goldilocks::ZERO));

    let attack: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tools/recursive-constraint-minimizer/experiments/phi81_quotient.json"
    ))
    .unwrap();
    values[1..55].fill(Goldilocks::from_u64(2));
    values[55..109].fill(Goldilocks::ZERO);
    for (base, field) in [(109, "h_coefficients"), (163, "q_coefficients")] {
        for (degree, coefficient) in attack["attack_replay"][field]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            values[base + degree] = Goldilocks::from_u64(coefficient.as_u64().unwrap());
        }
    }
    let attack_residuals = residuals(&values);
    assert!(attack_residuals[..107]
        .iter()
        .all(|value| *value == Goldilocks::ZERO));
    assert_ne!(attack_residuals[107], Goldilocks::ZERO);
}

#[test]
fn shared_external_template_matches_reference_and_preserves_zero_forms() {
    let zero = vec![Form::default(); 8];
    assert_eq!(external_layer(&zero, 0).unwrap(), zero);
    let logical_width = 8 * RetainedKind::Field.width();
    let retained = RetainedBlock::decode(&json!([2, 8, 0])).unwrap();
    let state: Vec<_> = (0..8)
        .map(|lane| {
            retained
                .form(logical_width, lane)
                .unwrap()
                .scaled(Goldilocks::from_u64((lane + 1) as u64))
        })
        .collect();
    let input: Vec<_> = state
        .iter()
        .map(|form| {
            reference::Form::from_entries(
                form.entries()
                    .iter()
                    .map(|entry| reference::Entry {
                        column: entry.column,
                        coefficient: reference::Field::checked(entry.coefficient.as_canonical_u64(), "coefficient")
                            .unwrap(),
                    })
                    .collect(),
            )
        })
        .collect();
    let actual = external_layer(&state, logical_width).unwrap();
    let expected = reference::external_layer(&input).unwrap();
    for lane in 0..8 {
        assert!(actual[lane]
            .terms()
            .iter()
            .all(|term| term.column_count() == 41));
        let actual: Vec<_> = actual[lane]
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
            .collect();
        let expected: Vec<_> = expected[lane]
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.canonical()))
            .collect();
        assert_eq!(actual, expected, "external lane {lane}");
    }
    assert!(external_layer(&state[..7], logical_width).is_err());
    assert!(external_layer(&state, 7).is_err());
}

#[test]
fn retained_runs_add_overlapping_scalars_and_reject_partial_columns() {
    let retained = RetainedBlock::decode(&json!([2, 1, 5])).unwrap();
    let field = retained.form(46, 0).unwrap();
    assert!(retained.form(45, 0).is_err());
    assert!(retained.form(46, 1).is_err());
    assert!(validate_form(&field, 45).is_err());
    let scalar = Form::singleton(6, -Goldilocks::from_u64(3));
    let actual = field.clone().append(scalar.clone());
    let expected = reference::Form::from_entries(
        field
            .entries()
            .iter()
            .chain(scalar.entries().iter())
            .map(|entry| reference::Entry {
                column: entry.column,
                coefficient: reference::Field::checked(entry.coefficient.as_canonical_u64(), "coefficient").unwrap(),
            })
            .collect(),
    );
    assert_eq!(actual.terms().len(), 2, "the scalar does not expand the retained run");
    assert_eq!(
        actual.entries().len(),
        40,
        "the overlapping scalar cancels one coordinate"
    );
    assert_eq!(
        actual
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
            .collect::<Vec<_>>(),
        expected
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.canonical()))
            .collect::<Vec<_>>()
    );
    let cancelled = actual.append(field.scaled(-Goldilocks::ONE));
    assert_eq!(cancelled, scalar);
    assert!(cancelled.scaled(Goldilocks::ZERO).terms().is_empty());
}

#[test]
fn template_scratch_matches_append_normalization_and_reuses_empty_ports() {
    let field = Form::retained(5, 41);
    let positive = field
        .clone()
        .append(Form::singleton(1, Goldilocks::from_u64(7)));
    let inputs = [
        positive.clone(),
        positive.scaled(-Goldilocks::ONE),
        field
            .scaled(Goldilocks::from_u64(2))
            .append(Form::singleton(6, -Goldilocks::from_u64(6))),
        Form::retained(5, 2),
        Form::retained(7, 41),
        Form::default(),
        Form::singleton(5, Goldilocks::from_u64(11)),
    ];
    let cases = [
        vec![(0, 3), (1, 3), (2, 1), (3, 1), (4, 1), (5, 7), (6, 1)],
        vec![(2, 1)],
        vec![(5, 1)],
        vec![(2, GOLDILOCKS_MODULUS - 1)],
        vec![(2, 0)],
        vec![(0, 1), (1, 1)],
        vec![(0, 1), (1, 1), (2, 1)],
        vec![(0, 0), (1, 0)],
    ];
    let mut terms = Vec::new();
    let mut allocation = None;
    for (case, coefficients) in cases.into_iter().enumerate() {
        let sparse = SparseForm::new(coefficients).unwrap();
        let expected = sparse
            .entries()
            .fold(Form::default(), |sum, (input, coefficient)| {
                sum.append(
                    inputs[input]
                        .clone()
                        .scaled(Goldilocks::from_u64(coefficient)),
                )
            });
        template::substitute_into(&sparse, &inputs, &mut terms).unwrap();
        assert_eq!(terms, expected.terms(), "canonical run sequence, case {case}");
        assert_eq!(form::entries(&terms), expected.entries(), "scalar values, case {case}");
        if case == 0 {
            assert_eq!(terms.len(), 5, "different overlapping keys stay separate");
            allocation = Some((terms.as_ptr(), terms.capacity()));
        } else {
            assert_eq!(Some((terms.as_ptr(), terms.capacity())), allocation);
        }
    }
    assert!(terms.is_empty(), "the last zero port clears the preceding row");
}

#[test]
fn template_visitors_stop_before_invalid_later_invocations() {
    let logical_width = 12_000;
    let poseidon = json!([2, [2, 0, [2, 2 * 86, 100], [[[1, 1, 0, 1], [0, [0, 0, 0], 0, 0, 0]]]]]);
    // Two rings (one source, two blocks, one cell); retained blocks cover only the first.
    let phi81 = json!([
        3,
        [
            [[1, 2, 1]],
            0,
            [0, 54, 20],
            0,
            54,
            [[[0, 54, [0, 54, 74], 0]], []],
            [0, 54, 128],
            [0, 54, 182]
        ]
    ]);
    for (encoded, invalid_row) in [(poseidon, 86), (phi81, 108)] {
        let program = MatrixProgram::decode(&json!([encoded])).unwrap();
        let source = |_| panic!("template rows do not read ordinary source rows");
        let expected = program.row(logical_width, 0, &source).unwrap();
        assert!(program.row(logical_width, invalid_row, &source).is_err());
        let count = program.row_count().unwrap();
        let mut visited = 0;
        let flow = program
            .visit_rows_until(logical_width, 0, count, &source, |ordinal, row| {
                assert_eq!(ordinal, 0);
                assert_eq!(row, borrowed_row(&expected));
                visited += 1;
                Ok(ControlFlow::Break(()))
            })
            .unwrap();
        assert_eq!(flow, ControlFlow::Break(()));
        assert_eq!(visited, 1);

        let failure = program.visit_rows_until(logical_width, 0, count, &source, |_, _| {
            Err(PackageError::Invalid("test visitor stop"))
        });
        assert!(matches!(failure, Err(PackageError::Invalid("test visitor stop"))));
    }
}

#[test]
fn shared_templates_preserve_empty_invocation_ranges() {
    let poseidon = poseidon::Block::decode(&json!([0, 0, [2, 0, 0], []])).unwrap();
    let flow = poseidon
        .visit_rows_until(0, 0, 0, |_| panic!("no rows"))
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    assert!(poseidon.row(0, 0).is_err());
    let phi81 = phi81::Block::decode(&json!([[], 0, [0, 0, 0], 0, 54, [[], []], [0, 0, 0], [0, 0, 0]])).unwrap();
    let flow = phi81
        .visit_rows_until(0, 0, 0, |_| panic!("no rows"))
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    assert!(phi81.row(0, 0).is_err());
}

#[test]
fn shared_poseidon_templates_reject_malformed_input_programs() {
    for (name, program) in affine_fixture::malformed() {
        let encoded = json!([3, 0, [2, 3 * 86, 100], program]);
        let result = poseidon::Block::decode(&encoded).and_then(|block| block.row(12_000, 0));
        assert!(result.is_err(), "accepted malformed Poseidon input: {name}");
    }
}
