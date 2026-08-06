//! Exact conformance tests for aggregate chunk-acceptance lowering.
//!
//! | Surface | Checked property |
//! |---|---|
//! | Source trace | Four rows, two ordered columns, and sixteen exact inputs |
//! | Lowered shape | Fifteen coordinates and nine rows per chunk by named family |
//! | Canonical inverse | Zero on rejection and `difference^-1` on acceptance |
//! | Decoder | Satisfied low-norm witness reconstructs the exact source witness |
//! | Outer image | Singleton/linear expansions, Boolean owners, and source/physical row images are exact |
//! | Fail-closed boundary | Every coordinate tamper and every trace-role mutation rejects |

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_alphabet_sample_5_d;
use neo_fold_clean::engine::r1cs_circuit::{AcceptanceTraceTestMutation, Lc, R1csBuilder, TranscriptGadget, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_r1cs_gadget_native_aggregate_acceptance_outer_image, encode_r1cs_gadget_native, estimate_r1cs_gadget_native,
    AggregateAcceptanceBooleanRowOwner, AggregateAcceptanceDecodedImage, GadgetNativeError,
};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

const APP: &[u8] = b"neo.test.alphabet_sampling/v1";
const REJECTION_SEED: u64 = 0xb72;
const CHUNKS: usize = 64;
const SOURCE_ROWS_PER_CHUNK: usize = 4;
const SOURCE_COLUMNS_PER_CHUNK: usize = 2;
const ENCODED_COORDINATES_PER_CHUNK: usize = 15;
const TREE_OUTPUTS_PER_CHUNK: usize = 14;
const ENCODED_ROWS_PER_CHUNK: usize = 9;

fn sampler_builder(seed: u64) -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.aggregate_acceptance");
    let mut transcript = TranscriptGadget::new(&mut builder, APP);
    let _symbols = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, seed);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied(), "production sampler source relation");
    builder
}

fn chunk_inputs(builder: &R1csBuilder) -> Vec<usize> {
    builder
        .encoding_trace()
        .acceptance_chunks()
        .iter()
        .flat_map(|event| event.chunk_bits)
        .map(Var::col)
        .collect()
}

#[test]
fn aggregate_acceptance_is_exactly_four_source_rows_and_nine_lowered_rows() {
    let builder = sampler_builder(7);
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    assert_eq!(trace.acceptance_chunks().len(), CHUNKS);
    for event in trace.acceptance_chunks() {
        assert_eq!(event.source_rows.len(), SOURCE_ROWS_PER_CHUNK);
        assert_eq!(event.allocated_columns.len(), SOURCE_COLUMNS_PER_CHUNK);
        assert_eq!(event.accept.col(), event.allocated_columns.start);
        assert_eq!(event.inverse.col(), event.allocated_columns.start + 1);
        assert!(event
            .chunk_bits
            .iter()
            .all(|bit| bit.col() > 0 && bit.col() < event.allocated_columns.start));
    }

    let inputs = chunk_inputs(&builder);
    let estimate = estimate_r1cs_gadget_native(&source, trace, &inputs).expect("validated acceptance estimate");
    assert_eq!(estimate.acceptance_chunks, CHUNKS);
    assert_eq!(estimate.acceptance_encoded_cols, CHUNKS * ENCODED_COORDINATES_PER_CHUNK);
    assert_eq!(estimate.acceptance_tree_output_cols, CHUNKS * TREE_OUTPUTS_PER_CHUNK);
    assert_eq!(estimate.acceptance_tree_bit_pair_rows, CHUNKS * 7);
    assert_eq!(estimate.acceptance_product_aggregate_rows, CHUNKS);
    assert_eq!(estimate.acceptance_root_binding_rows, CHUNKS);
    assert_eq!(
        estimate.acceptance_tree_bit_pair_rows
            + estimate.acceptance_product_aggregate_rows
            + estimate.acceptance_root_binding_rows,
        CHUNKS * ENCODED_ROWS_PER_CHUNK
    );

    let encoded = encode_r1cs_gadget_native(&source, trace, &inputs).expect("exact aggregate acceptance lowering");
    assert!(encoded.is_satisfied());
    assert_eq!(
        encoded.decode_source().expect("exact inverse decoder"),
        source.witness()
    );
    for chunk in 0..CHUNKS {
        let audit = encoded
            .plan
            .aggregate_acceptance_audit(chunk)
            .expect("acceptance role schedule");
        assert_eq!(audit.encoded_outputs.len(), TREE_OUTPUTS_PER_CHUNK);
        assert_eq!(audit.radix_weights[0], F::ONE);
        for pair in audit.radix_weights.windows(2) {
            assert_eq!(pair[1], pair[0] * F::from_u64(3));
        }
    }

    let outer = encoded
        .aggregate_acceptance_outer_image_audit(&source, trace)
        .expect("exact singleton outer image");
    assert_eq!(outer.chunks.len(), CHUNKS);
    assert!(outer.linear_definitions.is_empty());
    assert_eq!(outer.encoded_columns, encoded.structure.m);
    assert_eq!(outer.encoded_rows, encoded.structure.n);
    assert_eq!(outer.matrix_arity, encoded.structure.matrices.len());
    for chunk in &outer.chunks {
        assert_eq!(chunk.source_rows.len(), SOURCE_ROWS_PER_CHUNK);
        assert_eq!(chunk.active_rows.len(), ENCODED_ROWS_PER_CHUNK);
        assert!(chunk.bits.iter().all(|bit| {
            matches!(bit.decoded, AggregateAcceptanceDecodedImage::Singleton { .. })
                && matches!(
                    bit.boolean_owner,
                    AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { .. }
                        | AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { .. }
                        | AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. }
                )
                && bit.linear_definition_columns.is_empty()
        }));
    }
}

#[test]
fn aggregate_acceptance_outer_image_records_derived_terminal_bits_and_exact_rows() {
    let builder = sampler_builder(7);
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let encoded = encode_r1cs_gadget_native(&source, trace, &[]).expect("private sampler outer-image lowering");
    let audit = encoded
        .aggregate_acceptance_outer_image_audit(&source, trace)
        .expect("private sampler outer-image audit");
    let planned = audit_r1cs_gadget_native_aggregate_acceptance_outer_image(&source, trace, &[])
        .expect("sparse planned outer-image audit");
    assert_eq!(planned, audit, "sparse plan must reproduce materialized CCS");

    let mut singleton = 0usize;
    let mut linear = 0usize;
    let source_rows = audit
        .source_rows
        .iter()
        .map(|row| row.row)
        .collect::<std::collections::BTreeSet<_>>();
    let physical_rows = audit
        .physical_rows
        .iter()
        .map(|row| row.row)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(source_rows.len(), audit.source_rows.len());
    assert_eq!(physical_rows.len(), audit.physical_rows.len());

    for chunk in &audit.chunks {
        assert_eq!(chunk.source_rows.len(), SOURCE_ROWS_PER_CHUNK);
        assert_eq!(chunk.active_rows.len(), ENCODED_ROWS_PER_CHUNK);
        assert!(chunk
            .source_rows
            .clone()
            .all(|row| source_rows.contains(&row)));
        assert!(chunk
            .active_rows
            .clone()
            .all(|row| physical_rows.contains(&row)));
        for bit in &chunk.bits {
            match &bit.decoded {
                AggregateAcceptanceDecodedImage::Singleton { .. } => {
                    singleton += 1;
                    assert!(bit.linear_definition_columns.is_empty());
                    assert!(matches!(
                        bit.boolean_owner,
                        AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { .. }
                            | AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { .. }
                            | AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. }
                    ));
                }
                AggregateAcceptanceDecodedImage::SparseLinear { terms } => {
                    linear += 1;
                    assert!(!terms.is_empty());
                    assert!(terms.iter().all(|(_, coefficient)| *coefficient != F::ZERO));
                    assert!(terms.windows(2).all(|pair| pair[0].0 < pair[1].0));
                    assert!(!bit.linear_definition_columns.is_empty());
                    let AggregateAcceptanceBooleanRowOwner::TranslatedSource {
                        source_row,
                        encoded_row,
                    } = bit.boolean_owner
                    else {
                        panic!("linear bit must retain its translated source Boolean row")
                    };
                    assert!(source_rows.contains(&source_row));
                    assert!(physical_rows.contains(&encoded_row));
                }
            }
            assert!(physical_rows.contains(&bit.boolean_owner.encoded_row()));
        }
    }

    assert_eq!(linear, CHUNKS / 4);
    assert_eq!(singleton + linear, CHUNKS * 16);
    assert_eq!(audit.linear_definitions.len(), 3 * linear);
    for definition in &audit.linear_definitions {
        assert!(source_rows.contains(&definition.source_row));
        assert!(definition
            .terms
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0));
    }
    assert!(audit
        .physical_rows
        .iter()
        .all(|row| !row.matrices.is_empty()));
}

#[test]
fn aggregate_acceptance_materializes_both_canonical_inverse_branches() {
    let builder = sampler_builder(REJECTION_SEED);
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let inputs = chunk_inputs(&builder);
    let encoded = encode_r1cs_gadget_native(&source, trace, &inputs).expect("forced-rejection lowering");
    assert!(encoded.is_satisfied());

    let mut rejected = 0usize;
    let mut accepted = 0usize;
    for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
        let difference = event
            .chunk_bits
            .iter()
            .enumerate()
            .fold(-F::from_u64(65_535), |value, (index, bit)| {
                value + F::from_u64(1u64 << index) * source.witness()[bit.col()]
            });
        let audit = encoded.plan.aggregate_acceptance_audit(chunk).unwrap();
        let accept = source.witness()[event.accept.col()];
        assert_eq!(encoded.assignment[audit.encoded_accept], accept);
        if difference == F::ZERO {
            rejected += 1;
            assert_eq!(accept, F::ZERO);
            assert_eq!(source.witness()[event.inverse.col()], F::ZERO);
            assert!(audit
                .encoded_outputs
                .clone()
                .all(|column| encoded.assignment[column] == F::ONE));
        } else {
            accepted += 1;
            assert_eq!(accept, F::ONE);
            assert_eq!(source.witness()[event.inverse.col()], difference.inverse());
        }
    }
    assert!(rejected >= 1, "hard-coded seed must exercise canonical rejection");
    assert!(accepted >= 1, "fixture must also exercise ordinary acceptance");
    assert_eq!(
        encoded.decode_source().expect("branch-complete decoder"),
        source.witness()
    );
}

#[test]
fn every_aggregate_acceptance_coordinate_is_load_bearing() {
    let builder = sampler_builder(7);
    let source = builder.snapshot();
    let inputs = chunk_inputs(&builder);
    let encoded = encode_r1cs_gadget_native(&source, builder.encoding_trace(), &inputs).unwrap();
    let audit = encoded.plan.aggregate_acceptance_audit(0).unwrap();
    let columns = std::iter::once(audit.encoded_accept)
        .chain(audit.encoded_outputs)
        .collect::<Vec<_>>();
    assert_eq!(columns.len(), ENCODED_COORDINATES_PER_CHUNK);
    for column in columns {
        let mut tampered = encoded.clone();
        tampered.assignment[column] += F::ONE;
        assert!(
            !tampered.is_satisfied(),
            "aggregate acceptance coordinate {column} must be constrained"
        );
    }
}

#[test]
fn all_acceptance_trace_role_mutations_fail_closed() {
    let builder = sampler_builder(7);
    let source = builder.snapshot();
    let inputs = chunk_inputs(&builder);
    let first = &builder.encoding_trace().acceptance_chunks()[0];
    let mutations = [
        AcceptanceTraceTestMutation::SourceRowEnd {
            row_end: first.source_rows.end - 1,
        },
        AcceptanceTraceTestMutation::AllocatedColumnEnd {
            column_end: first.allocated_columns.end - 1,
        },
        AcceptanceTraceTestMutation::ChunkBitColumn {
            index: 0,
            column: first.accept.col(),
        },
        AcceptanceTraceTestMutation::AcceptColumn {
            column: first.inverse.col(),
        },
        AcceptanceTraceTestMutation::InverseColumn {
            column: first.accept.col(),
        },
    ];
    for mutation in mutations {
        let mut corrupted = builder.encoding_trace().clone();
        corrupted.apply_acceptance_trace_test_mutation(0, mutation);
        assert!(
            matches!(
                encode_r1cs_gadget_native(&source, &corrupted, &inputs),
                Err(GadgetNativeError::AcceptanceGeometry { .. }) | Err(GadgetNativeError::TraceRowMismatch { .. })
            ),
            "acceptance trace mutation must reject before row removal"
        );
    }
}

#[test]
fn canonical_inverse_may_not_escape_its_exact_source_block() {
    let mut builder = sampler_builder(7);
    let inverse = builder.encoding_trace().acceptance_chunks()[0].inverse;
    builder.enforce(&Lc::from_var(inverse), &Lc::zero(), &Lc::zero());
    builder.begin_encoding_stage("complete.after_inverse_escape");
    assert!(builder.is_satisfied());
    let source = builder.snapshot();
    let inputs = chunk_inputs(&builder);
    assert!(matches!(
        encode_r1cs_gadget_native(&source, builder.encoding_trace(), &inputs),
        Err(GadgetNativeError::GadgetTemporaryEscapes { column }) if column == inverse.col()
    ));
}
