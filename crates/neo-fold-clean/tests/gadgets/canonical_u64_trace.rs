//! Fail-closed tests for canonical-u64 source provenance.
//!
//! | Test family | Assurance boundary |
//! |---|---|
//! | Exact trace | One unique call records every role and all 69 rows |
//! | Role mutation | Field, low/high bit, flag, and inverse drift is rejected |
//! | Row mutation | Start, end, overlap, duplicate, and bounds drift is rejected |
//! | Census | Direct, equality-linked, and linear ownership stays explicit |
//! | Source allocation | Direct words consume 64 value bits plus 31 hidden prefix auxiliaries |

use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::encoding_trace::CanonicalU64TraceTestMutation;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::{alloc_u64_bits, decompose_var_to_u64_bits};
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, R1csEncodingTrace, R1csSnapshot, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_r1cs_gadget_native_canonical_u64, audit_r1cs_gadget_native_ordinary_placement, encode_r1cs_gadget_native,
    CanonicalU64Classification, GadgetNativeError,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

struct Fixture {
    source: R1csSnapshot,
    trace: R1csEncodingTrace,
    field: Var,
}

fn direct_fixture() -> Fixture {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.canonical_u64");
    let prefix = builder.alloc(F::ZERO);
    enforce_bit(&mut builder, prefix);
    let field = builder.alloc(F::from_u64(0x1234_5678_9abc_def0));
    let _ = decompose_var_to_u64_bits(&mut builder, field);
    let suffix = builder.alloc(F::ONE);
    enforce_bit(&mut builder, suffix);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied());
    Fixture {
        source: builder.snapshot(),
        trace: builder.encoding_trace().clone(),
        field,
    }
}

fn assert_rejected(fixture: &Fixture, mutation: CanonicalU64TraceTestMutation) {
    let mut trace = fixture.trace.clone();
    trace.apply_canonical_u64_trace_test_mutation(0, mutation);
    let error = audit_r1cs_gadget_native_canonical_u64(&fixture.source, &trace, &[])
        .expect_err("mutated canonical-u64 provenance must fail closed");
    assert!(
        matches!(
            error,
            GadgetNativeError::CanonicalU64Geometry { .. }
                | GadgetNativeError::CanonicalU64StageSchedule { .. }
                | GadgetNativeError::TraceRowMismatch { .. }
        ),
        "unexpected rejection: {error}"
    );
}

#[test]
fn trace_records_every_role_once_after_the_complete_69_row_gadget() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.canonical_u64");
    let field = builder.alloc(F::from_u64(37));
    let rows_before = builder.rows();
    let cols_before = builder.cols();
    let bits = decompose_var_to_u64_bits(&mut builder, field);
    let rows_after = builder.rows();
    let cols_after = builder.cols();
    let cached = decompose_var_to_u64_bits(&mut builder, field);
    builder.begin_encoding_stage("complete");

    assert_eq!(cached, bits);
    assert_eq!(builder.rows(), rows_after, "cache hit must emit no rows");
    assert_eq!(builder.cols(), cols_after, "cache hit must allocate no columns");
    let [entry] = builder.encoding_trace().canonical_u64_decompositions() else {
        panic!("one unique decomposition must produce one trace entry");
    };
    assert_eq!(entry.field, field);
    assert_eq!(entry.bits, bits);
    assert_eq!(entry.source_rows, rows_before..rows_after);
    assert_eq!(entry.source_rows.len(), 69);
    assert_eq!(cols_after - cols_before, 66);
    assert_eq!(
        entry.bits.map(Var::col),
        std::array::from_fn(|index| cols_before + index)
    );
    assert_eq!(entry.high_is_max.col(), cols_before + 64);
    assert_eq!(entry.inverse.col(), cols_before + 65);

    let source = builder.snapshot();
    let report = audit_r1cs_gadget_native_canonical_u64(&source, builder.encoding_trace(), &[])
        .expect("exact trace must validate");
    assert_eq!(report.census.total, 1);
}

#[test]
fn direct_word_consumes_95_source_loop_coordinates_but_returns_64_value_bits() {
    let fixture = direct_fixture();
    let placement = audit_r1cs_gadget_native_ordinary_placement(&fixture.source, &fixture.trace, &[])
        .expect("exact source allocation");
    let allocated = placement
        .source_loop_allocation_range_for_column(fixture.field.col())
        .expect("direct canonical-u64 source allocation");
    let materialized = encode_r1cs_gadget_native(&fixture.source, &fixture.trace, &[])
        .expect("bounded canonical-u64 fixture materializes");
    let returned = materialized
        .plan
        .encoded_range_for_source_column(fixture.field.col())
        .expect("canonical-u64 value slot");

    assert_eq!(allocated.start, returned.start);
    assert_eq!(allocated.len(), 95, "64 raw bits plus 31 prefix auxiliaries");
    assert_eq!(returned.len(), 64, "decoder-visible canonical value bits");
}

#[test]
fn validator_rejects_every_column_role_mutation() {
    let fixture = direct_fixture();
    let entry = &fixture.trace.canonical_u64_decompositions()[0];
    let prefix_column = fixture.field.col() - 1;
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::FieldColumn { column: prefix_column },
    );
    for index in [0, 31, 32, 63] {
        assert_rejected(
            &fixture,
            CanonicalU64TraceTestMutation::BitColumn {
                index,
                column: entry.bits[index].col() + 1,
            },
        );
    }
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::HighIsMaxColumn {
            column: entry.inverse.col(),
        },
    );
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::InverseColumn {
            column: entry.high_is_max.col(),
        },
    );
}

#[test]
fn validator_rejects_row_range_boundaries_duplicates_and_overlap() {
    let fixture = direct_fixture();
    let rows = fixture.trace.canonical_u64_decompositions()[0]
        .source_rows
        .clone();
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::SourceRows {
            rows: rows.start - 1..rows.end - 1,
        },
    );
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::SourceRows {
            rows: rows.start + 1..rows.end + 1,
        },
    );
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::SourceRows {
            rows: rows.start..rows.end - 1,
        },
    );
    assert_rejected(
        &fixture,
        CanonicalU64TraceTestMutation::SourceRows {
            rows: rows.start..fixture.source.rows() + 1,
        },
    );

    let mut duplicate = fixture.trace.clone();
    duplicate.duplicate_canonical_u64_trace_for_test(0);
    assert!(matches!(
        audit_r1cs_gadget_native_canonical_u64(&fixture.source, &duplicate, &[]),
        Err(GadgetNativeError::CanonicalU64Geometry { .. })
    ));

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.canonical_u64");
    let first = builder.alloc(F::from_u64(5));
    let _ = decompose_var_to_u64_bits(&mut builder, first);
    let second = builder.alloc(F::from_u64(7));
    let _ = decompose_var_to_u64_bits(&mut builder, second);
    builder.begin_encoding_stage("complete");
    let source = builder.snapshot();
    let mut overlap = builder.encoding_trace().clone();
    let first_rows = overlap.canonical_u64_decompositions()[0]
        .source_rows
        .clone();
    overlap.apply_canonical_u64_trace_test_mutation(
        1,
        CanonicalU64TraceTestMutation::SourceRows {
            rows: first_rows.start + 1..first_rows.end + 1,
        },
    );
    assert!(matches!(
        audit_r1cs_gadget_native_canonical_u64(&source, &overlap, &[]),
        Err(GadgetNativeError::CanonicalU64Geometry { .. })
    ));
}

#[test]
fn census_keeps_direct_linear_and_equality_linked_facts_explicit() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();

    builder.begin_encoding_stage("test.direct");
    let direct = builder.alloc(F::from_u64(11));
    let _ = decompose_var_to_u64_bits(&mut builder, direct);

    builder.begin_encoding_stage("test.linear_and_linked");
    let input = builder.alloc(F::from_u64(13));
    let linear = builder.alloc(F::from_u64(13));
    builder.enforce_eq(&Lc::from_var(linear), &Lc::from_var(input));
    let bits = decompose_var_to_u64_bits(&mut builder, linear);
    let mirror = alloc_u64_bits(&mut builder, 13);
    for (&bit, &copy) in bits.iter().zip(&mirror) {
        builder.enforce_eq(&Lc::from_var(bit), &Lc::from_var(copy));
    }
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let report =
        audit_r1cs_gadget_native_canonical_u64(&source, builder.encoding_trace(), &[]).expect("exact mixed census");
    assert_eq!(report.census.total, 2);
    assert_eq!(report.census.direct, 1);
    assert_eq!(report.census.equality_linked, 1);
    assert_eq!(report.census.linear, 0);
    assert_eq!(report.census.field_linearly_derived, 1);
    assert_eq!(report.entries[0].classification, CanonicalU64Classification::Direct);
    assert_eq!(report.entries[0].equality_linked_bits, 0);
    assert!(!report.entries[0].field_linearly_derived);
    assert_eq!(
        report.entries[1].classification,
        CanonicalU64Classification::EqualityLinked
    );
    assert_eq!(report.entries[1].equality_linked_bits, 64);
    assert!(report.entries[1].field_linearly_derived);
    assert_eq!(report.stages.len(), 2);
    assert_eq!(report.stages[0].stage, "test.direct");
    assert_eq!(report.stages[0].census.direct, 1);
    assert_eq!(report.stages[1].stage, "test.linear_and_linked");
    assert_eq!(report.stages[1].census.equality_linked, 1);
}

#[test]
fn partial_bit_linkage_is_reported_without_becoming_word_authority() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.partial");
    let field = builder.alloc(F::from_u64(1));
    let bits = decompose_var_to_u64_bits(&mut builder, field);
    let copy = builder.alloc(F::ONE);
    enforce_bit(&mut builder, copy);
    builder.enforce_eq(&Lc::from_var(bits[0]), &Lc::from_var(copy));
    builder.begin_encoding_stage("complete");

    let source = builder.snapshot();
    let report = audit_r1cs_gadget_native_canonical_u64(&source, builder.encoding_trace(), &[])
        .expect("partial equality is diagnostic only");
    assert_eq!(report.entries[0].classification, CanonicalU64Classification::Direct);
    assert_eq!(report.entries[0].equality_linked_bits, 1);
}
