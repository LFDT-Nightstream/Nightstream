use super::*;
use p3_field::PrimeField64;
use serde_json::json;

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
        actual
            .visit_rows(logical_width, 0, count, |row| {
                visited.push(row);
                Ok(())
            })
            .unwrap();
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
        actual
            .visit_rows(logical_width, 86, count - 1, |row| {
                partial.push(row);
                Ok(())
            })
            .unwrap();
        assert_eq!(partial, visited[86..count - 1]);
        assert!(actual.row(logical_width, count).is_err());
        assert!(actual
            .visit_rows(logical_width, count, count - 1, |_| Ok(()))
            .is_err());
    }
}

#[test]
fn shared_phi81_templates_match_all_lanes_and_prior_source_cases() {
    // Two sources cover both a zero prior and a carried prior for all 54 lanes.
    let logical_width = 4_000;
    let encoded = json!([
        [[2, 1, 1]],
        0,
        [0, 108, 20],
        0,
        54,
        [[[0, 108, [0, 108, 128], 0]], []],
        [0, 108, 236],
        [0, 108 * 33, 344]
    ]);
    let actual = phi81::Block::decode(&encoded).unwrap();
    let expected = reference::phi81::Block::decode(&encoded, logical_width).unwrap();
    let count = actual.row_count().unwrap();
    assert_eq!(count, expected.row_count().unwrap());
    let mut visited = Vec::new();
    actual
        .visit_rows(logical_width, 0, count, |row| {
            visited.push(row);
            Ok(())
        })
        .unwrap();
    assert_eq!(visited.len(), count);
    for (ordinal, row) in visited.iter().enumerate() {
        equal_row(row, &expected.row(logical_width, ordinal).unwrap(), ordinal);
        assert_eq!(*row, actual.row(logical_width, ordinal).unwrap());
    }
    let mut partial = Vec::new();
    actual
        .visit_rows(logical_width, 33, count - 1, |row| {
            partial.push(row);
            Ok(())
        })
        .unwrap();
    assert_eq!(partial, visited[33..count - 1]);
    assert!(actual.row(logical_width, count).is_err());
    assert!(actual
        .visit_rows(logical_width, count, count - 1, |_| Ok(()))
        .is_err());
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
fn shared_templates_preserve_empty_invocation_ranges() {
    let poseidon = poseidon::Block::decode(&json!([0, 0, [2, 0, 0], []])).unwrap();
    poseidon.visit_rows(0, 0, 0, |_| panic!("no rows")).unwrap();
    assert!(poseidon.row(0, 0).is_err());
    let phi81 = phi81::Block::decode(&json!([[], 0, [0, 0, 0], 0, 54, [[], []], [0, 0, 0], [0, 0, 0]])).unwrap();
    phi81.visit_rows(0, 0, 0, |_| panic!("no rows")).unwrap();
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
