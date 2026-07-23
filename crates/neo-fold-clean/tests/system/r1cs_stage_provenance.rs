//! Physical row/allocation-stage provenance across field-R1CS lowering.
//!
//! | Test | Property |
//! |---|---|
//! | Reordered public outputs | Column permutation leaves row ownership unchanged and remaps allocation boundaries |
//! | Empty/repeated stages | Zero-cost occurrences remain visible and ordered |
//! | Implicit close | The final named stage closes at the relation row count |
//! | Late nested marker | A partial ownership tree is not invented |
//! | Malformed schedule | Empty paths and reserved terminator misuse fail closed |
//! | No stages | Missing provenance remains explicitly empty |

use neo_fold_clean::engine::r1cs_circuit::{Lc, PhysicalStageError, PhysicalStageRange, R1csBuilder, Var};
use neo_fold_clean::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn bind(builder: &mut R1csBuilder, variable: Var, value: F) {
    builder.enforce_eq(&Lc::from_var(variable), &Lc::from_const(value));
}

fn ranges(ranges: &[PhysicalStageRange]) -> Vec<(&'static str, usize, usize)> {
    ranges
        .iter()
        .map(|range| (range.path(), range.row_start(), range.row_end()))
        .collect()
}

fn column_ranges(ranges: &[PhysicalStageRange]) -> Vec<(&'static str, usize, usize)> {
    ranges
        .iter()
        .map(|range| (range.path(), range.column_start(), range.column_end()))
        .collect()
}

#[test]
fn reversed_public_output_allocation_preserves_physical_row_ranges() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.stage.first");
    let first = builder.alloc(F::from_u64(11));
    bind(&mut builder, first, F::from_u64(11));

    builder.begin_encoding_stage("test.stage.organization");
    builder.begin_encoding_stage("test.stage.second");
    let second = builder.alloc(F::from_u64(29));
    bind(&mut builder, second, F::from_u64(29));
    builder.begin_encoding_stage("complete");

    assert!(
        builder.encoding_trace().stages().is_empty(),
        "lightweight row stages must not enable detailed row/column tracing"
    );
    let source_rows = builder.rows();
    let lowered = lower_field_r1cs(builder, &[second, first]).expect("lower staged relation");

    assert_eq!(lowered.shape().n, source_rows);
    assert_eq!(lowered.assignment(), &[F::ONE, F::from_u64(29), F::from_u64(11)]);
    assert_eq!(
        ranges(lowered.shape().physical_stage_ranges()),
        vec![
            ("test.stage.first", 0, 1),
            ("test.stage.organization", 1, 1),
            ("test.stage.second", 1, 2),
        ]
    );
    assert_eq!(
        column_ranges(lowered.shape().physical_stage_ranges()),
        vec![
            ("test.stage.first", 3, 3),
            ("test.stage.organization", 3, 3),
            ("test.stage.second", 3, 3),
        ]
    );
    lowered
        .shape()
        .is_satisfied_by(lowered.assignment())
        .expect("reordered relation remains satisfied");
}

#[test]
fn mixed_public_extraction_preserves_one_complete_private_partition() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.mixed.first");
    let private_first = builder.alloc(F::from_u64(3));
    bind(&mut builder, private_first, F::from_u64(3));
    let public_first = builder.alloc(F::from_u64(5));
    bind(&mut builder, public_first, F::from_u64(5));

    builder.begin_encoding_stage("test.mixed.second");
    let private_second = builder.alloc(F::from_u64(7));
    bind(&mut builder, private_second, F::from_u64(7));
    let public_second = builder.alloc(F::from_u64(11));
    bind(&mut builder, public_second, F::from_u64(11));
    builder.begin_encoding_stage("complete");

    let lowered = lower_field_r1cs(builder, &[public_second, public_first]).expect("lower mixed staged relation");
    assert_eq!(
        lowered.assignment(),
        &[F::ONE, F::from_u64(11), F::from_u64(5), F::from_u64(3), F::from_u64(7),]
    );
    assert_eq!(
        column_ranges(lowered.shape().physical_stage_ranges()),
        vec![("test.mixed.first", 3, 4), ("test.mixed.second", 4, 5)]
    );
}

#[test]
fn zero_cost_and_repeated_stages_survive_an_implicit_final_close() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.root");
    builder.begin_encoding_stage("test.repeated");
    let first = builder.alloc(F::from_u64(3));
    bind(&mut builder, first, F::from_u64(3));
    builder.begin_encoding_stage("test.repeated");
    builder.begin_encoding_stage("test.tail");
    let second = builder.alloc(F::from_u64(5));
    bind(&mut builder, second, F::from_u64(5));

    let lowered = lower_field_r1cs(builder, &[]).expect("implicitly close final stage");
    assert_eq!(
        ranges(lowered.shape().physical_stage_ranges()),
        vec![
            ("test.root", 0, 0),
            ("test.repeated", 0, 1),
            ("test.repeated", 1, 1),
            ("test.tail", 1, 2),
        ]
    );
    assert_eq!(
        column_ranges(lowered.shape().physical_stage_ranges()),
        vec![
            ("test.root", 1, 1),
            ("test.repeated", 1, 2),
            ("test.repeated", 2, 2),
            ("test.tail", 2, 3),
        ]
    );
    assert!(lowered.shape().physical_stage_ranges()[1].contains_row(0));
    assert!(lowered.shape().physical_stage_ranges()[3].contains_row(1));
}

#[test]
fn a_late_nested_stage_does_not_invent_partial_provenance() {
    let mut builder = R1csBuilder::new();
    let value = builder.alloc(F::from_u64(7));
    bind(&mut builder, value, F::from_u64(7));
    builder.begin_encoding_stage("test.too_late");

    let lowered = lower_field_r1cs(builder, &[]).expect("lower relation without partial provenance");
    assert!(lowered.shape().physical_stage_ranges().is_empty());
}

#[test]
fn a_row_zero_stage_after_private_allocation_cannot_claim_complete_column_provenance() {
    let mut builder = R1csBuilder::new();
    let _unowned = builder.alloc(F::from_u64(7));
    builder.begin_encoding_stage("test.too_late_for_columns");
    let value = builder.alloc(F::from_u64(13));
    bind(&mut builder, value, F::from_u64(13));

    assert!(matches!(
        lower_field_r1cs(builder, &[]),
        Err(FieldR1csLoweringError::PhysicalStage(PhysicalStageError::FirstColumn {
            column: 2
        }))
    ));
}

#[test]
fn an_empty_stage_path_is_rejected() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("");
    let value = builder.alloc(F::from_u64(13));
    bind(&mut builder, value, F::from_u64(13));

    assert!(matches!(
        lower_field_r1cs(builder, &[]),
        Err(FieldR1csLoweringError::PhysicalStage(PhysicalStageError::EmptyPath {
            index: 0
        }))
    ));
}

#[test]
fn a_premature_complete_marker_is_rejected() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.root");
    let first = builder.alloc(F::from_u64(13));
    bind(&mut builder, first, F::from_u64(13));
    builder.begin_encoding_stage("complete");
    let second = builder.alloc(F::from_u64(17));
    bind(&mut builder, second, F::from_u64(17));

    assert!(matches!(
        lower_field_r1cs(builder, &[]),
        Err(FieldR1csLoweringError::PhysicalStage(
            PhysicalStageError::ReservedComplete {
                index: 1,
                row: 1,
                rows: 2,
            }
        ))
    ));
}

#[test]
fn a_doubled_complete_marker_is_rejected() {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.root");
    let value = builder.alloc(F::from_u64(19));
    bind(&mut builder, value, F::from_u64(19));
    builder.begin_encoding_stage("complete");
    builder.begin_encoding_stage("complete");

    assert!(matches!(
        lower_field_r1cs(builder, &[]),
        Err(FieldR1csLoweringError::PhysicalStage(
            PhysicalStageError::ReservedComplete {
                index: 1,
                row: 1,
                rows: 1,
            }
        ))
    ));
}

#[test]
fn a_relation_without_stage_markers_keeps_empty_provenance() {
    let mut builder = R1csBuilder::new();
    let value = builder.alloc(F::from_u64(17));
    bind(&mut builder, value, F::from_u64(17));

    let lowered = lower_field_r1cs(builder, &[]).expect("lower unstaged relation");
    assert!(lowered.shape().physical_stage_ranges().is_empty());
    lowered
        .shape()
        .is_satisfied_by(lowered.assignment())
        .expect("unstaged relation remains satisfied");
}
