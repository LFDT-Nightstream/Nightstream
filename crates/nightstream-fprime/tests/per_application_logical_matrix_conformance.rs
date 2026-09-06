//! Exact final 14-matrix comparison from the sealed Lean package alone.
//!
//! The expected path has its own raw package decoder, source-row custody,
//! Goldilocks sparse arithmetic, and interpreter for every matrix-program
//! opcode. It does not call the production interpreter for expected rows.
//!
//! Run the full ignored gate with:
//!
//! ```text
//! RUSTC_WRAPPER="" cargo test -p nightstream-fprime --release --test per_application_logical_matrix_conformance -- --ignored --nocapture
//! ```

use std::{fs, path::PathBuf};

use nightstream_fprime::{
    load_per_application_package, load_poseidon2_hash_chain_v1_package, CcsMatrixSource, LoadedPerApplicationPackage,
    LogicalMatrixRow, PackageError, PI_CCS_V1_1_MATRIX_COUNT, POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER,
};
use rayon::prelude::*;
use serde_json::Value;

#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

#[path = "support/poseidon_affine_fixture.rs"]
mod affine_fixture;

#[test]
fn affine_poseidon_reference_preserves_values_indices_and_tag_selection() {
    let program = reference::poseidon_input::Program::decode(&affine_fixture::program(), 32)
        .expect("independent affine input decoder");
    let first = program.state(32, 31, 0).expect("first independent input");
    let second = program.state(32, 31, 1).expect("second independent input");
    for (form, expected) in [
        (&first[0], vec![(4, 1), (5, 3), (31, 7)]),
        (&first[1], vec![(5, 4), (31, 11)]),
        (&second[0], vec![(31, 13)]),
        (&second[1], vec![(4, 5), (31, 17)]),
    ] {
        let entries: Vec<_> = form
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.canonical()))
            .collect();
        assert_eq!(entries, expected);
    }
    assert!(first[2..].iter().all(|form| form.entries().is_empty()));
    assert!(second[2..].iter().all(|form| form.entries().is_empty()));
    assert!(program
        .state(32, 31, 2)
        .expect("inactive tag")
        .iter()
        .all(|form| form.entries().is_empty()));
    assert!(program.state(32, 32, 0).is_err(), "independent constant-column bound");
}

#[test]
fn affine_poseidon_reference_rejects_malformed_words_and_sources() {
    for (label, input) in affine_fixture::malformed() {
        let result =
            reference::poseidon_input::Program::decode(&input, 32).and_then(|program| program.state(32, 31, 0));
        assert!(result.is_err(), "independent decoder accepted {label}");
    }
}

use reference::{
    empty_row,
    matrix::MatrixProgram,
    source::{Artifact, SourcePackage},
    RowForms, GOLDILOCKS_MODULUS,
};

const EXPECTED_ACTIVE_ROWS: usize = 6_377_559;
const EXPECTED_LOGICAL_COLUMNS: usize = 254_260_583;
const EXPECTED_CUBE_VARIABLES: usize = 28;
const EXPECTED_PADDED_ROWS: usize = 268_435_456;
const EXPECTED_PHYSICAL_ROWS: usize = 29_225_729;
const EXPECTED_PHYSICAL_COLUMNS: usize = 29_344_425;
const EXPECTED_PUBLIC_COLUMNS: usize = 278;
const EXPECTED_LOGICAL_PUBLIC_INPUTS: usize = 270;
const MAX_OPCODE_ROWS_PER_INVOCATION: usize = 94;

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-fprime/artifacts/\
         nightstream-fprime-stage1-poseidon2-hash-chain-v1.json",
    )
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("canonical package mutation");
    bytes.push(b'\n');
    bytes
}

fn matrix_program_value(package: &Value) -> &Value {
    package
        .as_array()
        .and_then(|sealed| sealed.get(2))
        .expect("sealed final matrix program")
}

fn canonical_pin_entry(package: &mut Value) -> &mut Vec<Value> {
    package
        .as_array_mut()
        .and_then(|sealed| sealed.get_mut(2))
        .and_then(Value::as_array_mut)
        .and_then(|blocks| blocks.get_mut(3))
        .and_then(Value::as_array_mut)
        .and_then(|block| block.get_mut(1))
        .and_then(Value::as_array_mut)
        .and_then(|pin| pin.get_mut(1))
        .and_then(Value::as_array_mut)
        .and_then(|rows| rows.get_mut(760))
        .and_then(Value::as_array_mut)
        .and_then(|row| {
            row.iter_mut().find(|entry| {
                entry
                    .get(1)
                    .and_then(Value::as_u64)
                    .is_some_and(|coefficient| coefficient != 0)
            })
        })
        .and_then(Value::as_array_mut)
        .expect("canonical final matrix pin entry")
}

fn assert_independent_decode_then_identity_rejection(
    label: &str,
    package: &Value,
    artifact: &Artifact,
    expected_identity: [u64; 4],
    expected_row: (usize, &RowForms),
) {
    let changed = MatrixProgram::decode(
        matrix_program_value(package),
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .unwrap_or_else(|error| panic!("independent decoder rejected valid {label}: {error}"));
    assert_ne!(
        &changed
            .row(expected_row.0, &artifact.sources)
            .expect("changed canonical row"),
        expected_row.1,
        "{label} must change an expanded matrix row"
    );
    drop(changed);

    match load_per_application_package(&canonical_bytes(package), expected_identity) {
        Err(PackageError::ExpectedIdentityMismatch { expected, computed }) => {
            assert_eq!(expected, expected_identity);
            assert_ne!(computed, expected, "{label} did not change the structural identity");
        }
        Err(error) => panic!("{label} failed for a reason other than the selected structural identity: {error}"),
        Ok(_) => panic!("production loader accepted {label} under the selected identity"),
    }
    println!(
        "{label}: changed expanded row {}, selected identity rejected",
        expected_row.0
    );
}

fn compare_row(ordinal: usize, expected: &RowForms, actual: &LogicalMatrixRow) -> [u64; PI_CCS_V1_1_MATRIX_COUNT] {
    let mut counts = [0u64; PI_CCS_V1_1_MATRIX_COUNT];
    for (matrix, expected_form) in expected.iter().enumerate() {
        let actual_form = actual
            .matrix(matrix)
            .unwrap_or_else(|| panic!("missing production matrix {matrix} at row {ordinal}"));
        assert_eq!(
            actual_form.len(),
            expected_form.entries().len(),
            "entry count at row {ordinal}, matrix {matrix}"
        );
        for (entry_index, (expected_entry, actual_entry)) in expected_form.entries().iter().zip(actual_form).enumerate()
        {
            assert_eq!(
                [
                    u64::try_from(actual_entry.column()).expect("production matrix column fits u64"),
                    actual_entry.coefficient(),
                ],
                [
                    u64::try_from(expected_entry.column).expect("independent matrix column fits u64"),
                    expected_entry.coefficient.canonical(),
                ],
                "entry {entry_index} at row {ordinal}, matrix {matrix}"
            );
        }
        counts[matrix] = u64::try_from(expected_form.entries().len()).expect("row nonzero count fits u64");
    }
    counts
}

#[test]
fn valid_matrix_mutations_decode_but_fail_the_pinned_structural_identity() {
    let sealed_bytes = fs::read(artifact_path()).expect("Lean-emitted sealed package");
    let current = load_poseidon2_hash_chain_v1_package(&sealed_bytes).expect("current pinned production package");
    assert_eq!(
        current.structural_identifier(),
        POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER
    );
    check_matrix_mutations(current, sealed_bytes);
}

/// Apply the matrix mutations to a package that passed the selected identity
/// check. Every mutation must decode and then fail that same identity check.
pub fn check_matrix_mutations(current: LoadedPerApplicationPackage, sealed_bytes: Vec<u8>) {
    let expected_identity = current.structural_identifier();
    drop(current);

    let artifact = SourcePackage::decode(&sealed_bytes).expect("independent sealed package decoder");
    let value: Value = serde_json::from_slice(&sealed_bytes).expect("sealed package JSON");
    let original = MatrixProgram::decode(
        matrix_program_value(&value),
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent original matrix program");
    let first_row = original
        .row(0, &artifact.sources)
        .expect("original first row");
    let pin_ordinal = original.block_ends().nth(2).expect("start of fourth block") + 760;
    let pin_row = original
        .row(pin_ordinal, &artifact.sources)
        .expect("original selected pin row");
    drop(original);

    let mut changed_block_order = value.clone();
    let blocks = changed_block_order
        .as_array_mut()
        .and_then(|sealed| sealed.get_mut(2))
        .and_then(Value::as_array_mut)
        .expect("final matrix blocks");
    assert_ne!(blocks[0], blocks[1], "distinct final matrix blocks");
    blocks.swap(0, 1);
    assert_independent_decode_then_identity_rejection(
        "matrix block-order mutation",
        &changed_block_order,
        &artifact,
        expected_identity,
        (0, &first_row),
    );
    drop(changed_block_order);

    let mut changed_column = value.clone();
    let entry = canonical_pin_entry(&mut changed_column);
    let original_column = entry[0].as_u64().expect("canonical pin column");
    let column_count = u64::try_from(artifact.logical_columns).expect("logical columns fit u64");
    let replacement_column = original_column.checked_add(1).expect("next pin column");
    assert!(replacement_column < column_count);
    entry[0] = Value::from(replacement_column);
    assert_independent_decode_then_identity_rejection(
        "in-range matrix-column mutation",
        &changed_column,
        &artifact,
        expected_identity,
        (pin_ordinal, &pin_row),
    );
    drop(changed_column);

    let mut changed_coefficient = value;
    let entry = canonical_pin_entry(&mut changed_coefficient);
    let original_coefficient = entry[1].as_u64().expect("selected nonzero coefficient");
    assert!(original_coefficient != 0 && original_coefficient < GOLDILOCKS_MODULUS);
    let replacement_coefficient = if original_coefficient == GOLDILOCKS_MODULUS - 1 {
        1
    } else {
        original_coefficient + 1
    };
    entry[1] = Value::from(replacement_coefficient);
    assert_independent_decode_then_identity_rejection(
        "canonical nonzero matrix-coefficient mutation",
        &changed_coefficient,
        &artifact,
        expected_identity,
        (pin_ordinal, &pin_row),
    );
    println!("matrix block-order, in-range column, and canonical coefficient mutations rejected");
}

#[test]
#[ignore = "full independent 6,377,559-row matrix interpretation; run the documented release target under the 300-second cap"]
fn final_fourteen_matrices_equal_the_independent_sealed_interpretation() {
    let sealed_bytes = fs::read(artifact_path()).expect("Lean-emitted sealed package");
    let package = load_poseidon2_hash_chain_v1_package(&sealed_bytes).expect("production package decoder");
    check_logical_matrices(package, sealed_bytes);
}

/// Compare the actual final matrix rows with the independent sealed
/// interpretation for an already identity-checked package.
pub fn check_logical_matrices(package: LoadedPerApplicationPackage, sealed_bytes: Vec<u8>) {
    let artifact = SourcePackage::decode(&sealed_bytes).expect("independent sealed package decoder");
    drop(sealed_bytes);

    assert_eq!(artifact.sealed_schema, 6);
    assert_eq!(artifact.logical_rows, EXPECTED_ACTIVE_ROWS);
    assert_eq!(artifact.logical_columns, EXPECTED_LOGICAL_COLUMNS);
    assert_eq!(artifact.cube_variables, EXPECTED_CUBE_VARIABLES);
    assert_eq!(artifact.logical_public_inputs, EXPECTED_LOGICAL_PUBLIC_INPUTS);
    assert_eq!(artifact.sources.layout.row_count, EXPECTED_PHYSICAL_ROWS);
    assert_eq!(artifact.sources.layout.total_columns, EXPECTED_PHYSICAL_COLUMNS);
    assert_eq!(artifact.sources.layout.public_columns, EXPECTED_PUBLIC_COLUMNS);
    assert_eq!(package.row_count(), EXPECTED_ACTIVE_ROWS);
    assert_eq!(package.logical_column_count(), EXPECTED_LOGICAL_COLUMNS);
    assert_eq!(package.physical_row_count(), EXPECTED_PHYSICAL_ROWS);
    assert_eq!(package.total_column_count(), EXPECTED_PHYSICAL_COLUMNS);
    assert_eq!(package.public_input_count(), EXPECTED_PUBLIC_COLUMNS);
    assert_eq!(package.logical_public_input_count(), EXPECTED_LOGICAL_PUBLIC_INPUTS);
    let relation = package.ccs_relation();
    assert_eq!(relation.row_count(), EXPECTED_ACTIVE_ROWS);
    assert_eq!(relation.column_count(), EXPECTED_LOGICAL_COLUMNS);
    assert_eq!(relation.cube_variables(), EXPECTED_CUBE_VARIABLES);
    assert_eq!(relation.matrix_sources().len(), PI_CCS_V1_1_MATRIX_COUNT);
    assert_eq!(
        relation.matrix_sources()[PI_CCS_V1_1_MATRIX_COUNT - 1],
        CcsMatrixSource::Zero
    );
    assert_eq!(
        1usize << u32::try_from(artifact.cube_variables).expect("cube variables fit u32"),
        EXPECTED_PADDED_ROWS
    );

    let program = MatrixProgram::decode(
        &artifact.matrix_program,
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent matrix-program decoder");

    let range_starts = (0..artifact.logical_rows)
        .step_by(MAX_OPCODE_ROWS_PER_INVOCATION)
        .collect::<Vec<_>>();
    let nonzeros = range_starts
        .into_par_iter()
        .map(|start| {
            let end = start
                .checked_add(MAX_OPCODE_ROWS_PER_INVOCATION)
                .expect("matrix range end")
                .min(artifact.logical_rows);
            let mut expected_rows = Vec::with_capacity(end - start);
            let mut expected_next = start;
            program
                .visit_rows(start, end, &artifact.sources, |ordinal, row| {
                    assert_eq!(ordinal, expected_next, "independent row order");
                    assert!(
                        row[PI_CCS_V1_1_MATRIX_COUNT - 1].entries().is_empty(),
                        "independent zero slot at row {ordinal}"
                    );
                    expected_rows.push(row);
                    expected_next += 1;
                    Ok(())
                })
                .unwrap_or_else(|error| panic!("independent logical rows {start}..{end}: {error}"));
            assert_eq!(expected_next, end, "independent range coverage");
            assert_eq!(expected_rows.len(), end - start, "independent row count");

            let mut counts = [0u64; PI_CCS_V1_1_MATRIX_COUNT];
            let mut actual_next = start;
            package
                .visit_matrix_rows(start..end, |ordinal, actual| {
                    assert_eq!(ordinal, actual_next, "production row order");
                    let row_counts = compare_row(ordinal, &expected_rows[ordinal - start], &actual);
                    for (count, row_count) in counts.iter_mut().zip(row_counts) {
                        *count = count
                            .checked_add(row_count)
                            .expect("matrix nonzero count overflow");
                    }
                    actual_next += 1;
                    Ok(())
                })
                .unwrap_or_else(|error| panic!("production logical rows {start}..{end}: {error}"));
            assert_eq!(actual_next, end, "production range coverage");
            counts
        })
        .reduce(
            || [0u64; PI_CCS_V1_1_MATRIX_COUNT],
            |mut total, row| {
                for (total, row) in total.iter_mut().zip(row) {
                    *total = total
                        .checked_add(row)
                        .expect("matrix nonzero count overflow");
                }
                total
            },
        );

    let implicit_padding = empty_row();
    assert!(implicit_padding
        .iter()
        .all(|form| form.entries().is_empty()));
    for ordinal in [artifact.logical_rows, EXPECTED_PADDED_ROWS - 1] {
        assert!(ordinal < EXPECTED_PADDED_ROWS);
        assert!(
            program.row(ordinal, &artifact.sources).is_err(),
            "independent padding row {ordinal} must not decode as active data"
        );
        let end = ordinal.checked_add(1).expect("padding row end");
        assert!(
            package
                .visit_matrix_rows(ordinal..end, |_, _| Ok(()))
                .is_err(),
            "production padding row {ordinal} must not decode as active data"
        );
    }
    assert!(program
        .row(EXPECTED_PADDED_ROWS, &artifact.sources)
        .is_err());
    assert!(package
        .visit_matrix_rows(EXPECTED_PADDED_ROWS..EXPECTED_PADDED_ROWS + 1, |_, _| Ok(()))
        .is_err());
    assert_eq!(nonzeros[PI_CCS_V1_1_MATRIX_COUNT - 1], 0);
    eprintln!("independent_final_logical_matrix_nonzeros={nonzeros:?}");
}
