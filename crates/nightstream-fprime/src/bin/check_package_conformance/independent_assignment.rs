//! Independent evaluation of one sealed-package assignment.

use rayon::prelude::*;

use super::*;

fn reference_layout(raw: &RawPackage) -> ReferenceLayout {
    let domain_size = 1usize << 28;
    ReferenceLayout {
        unpadded_rows: word(raw.3 .0),
        unpadded_constant: word(raw.3 .2),
        public_columns: word(raw.3 .3),
        domain_size,
        final_columns: domain_size + 1 + word(raw.3 .3),
    }
}

fn sealed_raw(bytes: &[u8]) -> (RawPackage, ReferenceLayout) {
    let sealed: Value = serde_json::from_slice(bytes).expect("independent sealed-package decode");
    let raw: RawPackage = serde_json::from_value(sealed[1].clone()).expect("independent raw-package decode");
    assert_eq!(raw.0, 8, "independent raw-package schema");
    let layout = reference_layout(&raw);
    (raw, layout)
}

fn expanded_raw(bytes: &[u8]) -> (RawPackage, ReferenceLayout) {
    let raw: RawPackage = serde_json::from_slice(bytes).expect("independent Lean final-package decode");
    assert_eq!(raw.0, 8, "independent Lean final-package schema");
    let layout = reference_layout(&raw);
    (raw, layout)
}

/// Evaluate every physical row directly from the Lean-emitted raw package.
/// This path does not call witness generation, matrix expansion, or the
/// package constraint evaluator.
pub fn evaluate_sealed_assignment(bytes: &[u8], assignment: &WitnessAssignment) -> usize {
    let (raw, layout) = sealed_raw(bytes);
    assert_eq!(assignment.private_values().len(), layout.unpadded_constant);
    assert_eq!(assignment.public_values().len(), layout.public_columns);

    let schedule = events(&raw);
    let mut row_cursor = 0usize;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "independent row schedule");
        row_cursor += event_row_count(event, &raw);
    }
    assert_eq!(row_cursor, layout.unpadded_rows);
    schedule.par_iter().for_each(|&event| {
        for ordinal in 0..event_row_count(event, &raw) {
            let row_index = event.row_start() + ordinal;
            let value = |side| {
                evaluate_reference_combination(
                    &expected_row(event, &raw.5, ordinal, side, &layout),
                    &layout,
                    assignment,
                )
            };
            assert_eq!(
                mul_mod(value(MatrixSide::A), value(MatrixSide::B)),
                value(MatrixSide::C),
                "independent assignment row {row_index}",
            );
        }
    });
    let zero = evaluate_reference_combination(&[], &layout, assignment);
    assert_eq!(mul_mod(zero, zero), zero, "independent padded zero rows");
    layout.unpadded_rows
}

fn compare_raw_matrices(
    raw: &RawPackage,
    layout: &ReferenceLayout,
    matrices: &nightstream_fprime::PackageR1cs,
) -> [usize; 3] {
    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        assert_eq!(matrix.rows(), layout.domain_size);
        assert_eq!(matrix.columns(), layout.final_columns);
    }
    let schedule = events(raw);
    schedule.par_iter().for_each(|&event| {
        for ordinal in 0..event_row_count(event, raw) {
            let row_index = event.row_start() + ordinal;
            for (name, side, matrix) in [
                ("A", MatrixSide::A, matrices.a()),
                ("B", MatrixSide::B, matrices.b()),
                ("C", MatrixSide::C, matrices.c()),
            ] {
                compare_row(
                    matrix,
                    row_index,
                    &expected_row(event, &raw.5, ordinal, side, layout),
                    name,
                );
            }
        }
    });
    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        let end = matrix.nonzero_count();
        assert!(matrix.row_offsets()[layout.unpadded_rows..]
            .iter()
            .all(|offset| *offset == end));
    }
    [
        matrices.a().nonzero_count(),
        matrices.b().nonzero_count(),
        matrices.c().nonzero_count(),
    ]
}

/// Compare Rust's final padded matrix objects with every raw Lean row carried
/// inside the sealed package.
pub fn compare_sealed_matrices(bytes: &[u8], matrices: &nightstream_fprime::PackageR1cs) -> [usize; 3] {
    let (raw, layout) = sealed_raw(bytes);
    compare_raw_matrices(&raw, &layout, matrices)
}

/// Compare Rust's final padded matrix objects with the separately emitted
/// Lean final-package reference, then run owner-family mutations on those
/// exact matrices.
pub fn compare_lean_expanded_matrices(
    bytes: &[u8],
    matrices: &nightstream_fprime::PackageR1cs,
) -> ([usize; 3], usize, usize) {
    let (raw, layout) = expanded_raw(bytes);
    let nonzeros = compare_raw_matrices(&raw, &layout, matrices);
    let sides = [("A", matrices.a()), ("B", matrices.b()), ("C", matrices.c())];
    let row_mutations = owner_mutations::row_owner_mutation_checks(&sides, layout.unpadded_rows);
    let column_mutations = owner_mutations::column_owner_mutation_checks(&sides, &layout);
    (nonzeros, row_mutations, column_mutations)
}
