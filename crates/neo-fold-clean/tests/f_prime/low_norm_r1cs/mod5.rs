//! Exact Rust conformance tests for the Lean-authorized packed mod-5 block.
//!
//! | Boundary | Evidence owned here |
//! |---|---|
//! | Source provenance | every production chunk records 20 rows and 19 ordered columns |
//! | Packed lowering | 13 low bits plus two centered residues emit exactly 6 + 1 + 1 rows |
//! | Inversion | a satisfied packed assignment reconstructs the complete source witness |
//! | Fail-closed behavior | provenance drift, coordinate tamper, and temporary escape are rejected |

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_alphabet_sample_5_d;
use neo_fold_clean::engine::r1cs_circuit::encoding_trace::Mod5TraceTestMutation;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, GadgetNativeError,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const APP: &[u8] = b"packed-mod5-gadget-native-test";
const CHUNKS: usize = 64;

fn sampler_builder() -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.packed_mod5");
    let mut transcript = TranscriptGadget::new(&mut builder, APP);
    let _symbols = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, 7);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied(), "production sampler source relation");
    builder
}

#[test]
fn packed_mod5_is_exactly_eight_rows_and_fifteen_coordinates_per_chunk() {
    let builder = sampler_builder();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    assert_eq!(trace.mod5_chunks().len(), CHUNKS);
    for chunk in trace.mod5_chunks() {
        assert_eq!(chunk.source_rows.len(), 20);
        assert_eq!(chunk.allocated_columns.len(), 19);
        assert_eq!(chunk.index.col(), chunk.allocated_columns.start);
        assert_eq!(chunk.quotient.col(), chunk.allocated_columns.start + 1);
        assert_eq!(chunk.index_products[0].col(), chunk.allocated_columns.start + 2);
        assert_eq!(chunk.index_products[1].col(), chunk.allocated_columns.start + 3);
        assert_eq!(chunk.index_products[2].col(), chunk.allocated_columns.start + 4);
        for (offset, bit) in chunk.quotient_bits.iter().enumerate() {
            assert_eq!(bit.col(), chunk.allocated_columns.start + 5 + offset);
        }
    }

    let estimate = estimate_r1cs_gadget_native(&source, trace, &[]).expect("packed mod-5 estimate");
    assert_eq!(estimate.packed_mod5_chunks, CHUNKS);
    assert_eq!(estimate.packed_mod5_encoded_cols, CHUNKS * 15);
    assert_eq!(estimate.packed_mod5_low_bit_pair_rows, CHUNKS * 6);
    assert_eq!(estimate.packed_mod5_high_bit_pair_rows, CHUNKS);
    assert_eq!(estimate.packed_mod5_residue_pair_rows, CHUNKS);

    let mut encoded = encode_r1cs_gadget_native(&source, trace, &[]).expect("packed mod-5 lowering");
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.decode_source().expect("exact mod-5 inverse"), source.witness());

    let mut residues_seen = [false; 5];
    let mut high_bits_seen = [false; 2];
    for (chunk_index, chunk) in trace.mod5_chunks().iter().enumerate() {
        let residue = source.witness()[chunk.index.col()].as_canonical_u64() as usize;
        assert!(residue < 5);
        residues_seen[residue] = true;
        let high = source.witness()[chunk.quotient_bits[13].col()].as_canonical_u64() as usize;
        assert!(high < 2);
        high_bits_seen[high] = true;

        let residue_range = encoded
            .plan
            .packed_mod5_residue_range(chunk_index)
            .expect("two centered residues");
        assert_eq!(
            &encoded.assignment[residue_range],
            &canonical_centered_pair(residue),
            "canonical centered encoding for residue {residue}"
        );

        let raw_value = chunk
            .chunk_bits
            .iter()
            .enumerate()
            .fold(0u64, |value, (bit, variable)| {
                value + (source.witness()[variable.col()].as_canonical_u64() << bit)
            });
        let chunk_value = 65_535 - raw_value;
        let low = chunk.quotient_bits[..13]
            .iter()
            .enumerate()
            .fold(0u64, |value, (bit, variable)| {
                value + (source.witness()[variable.col()].as_canonical_u64() << bit)
            });
        assert_eq!(chunk_value, 5 * (low + 8192 * high as u64) + residue as u64);
    }
    assert!(
        residues_seen.into_iter().all(|seen| seen),
        "seed must cover all five residues"
    );
    assert!(
        high_bits_seen.into_iter().all(|seen| seen),
        "seed must cover both derived high bits"
    );

    let first = &trace.mod5_chunks()[0];
    for variable in [first.index, first.quotient, first.quotient_bits[13]] {
        assert!(!encoded.plan.is_gadget_derived(variable.col()));
        assert!(encoded
            .plan
            .encoded_range_for_source_column(variable.col())
            .is_none());
    }
    for product in first.index_products {
        assert!(encoded.plan.is_gadget_derived(product.col()));
        assert!(encoded
            .plan
            .encoded_range_for_source_column(product.col())
            .is_none());
    }

    let low = encoded
        .plan
        .packed_mod5_low_bit_range(0)
        .expect("thirteen low bits");
    let residues = encoded
        .plan
        .packed_mod5_residue_range(0)
        .expect("two centered residues");
    assert_eq!(low.len(), 13);
    assert_eq!(residues.len(), 2);

    let honest_low = encoded.assignment[low.start];
    encoded.assignment[low.start] = -F::ONE;
    assert!(
        encoded.first_unsatisfied_row().is_some(),
        "packed bit pair must reject -1"
    );
    encoded.assignment[low.start] = honest_low;
    assert!(encoded.is_satisfied());

    let honest_left = encoded.assignment[residues.start];
    encoded.assignment[residues.start] = F::from_u64(2);
    assert!(
        encoded.first_unsatisfied_row().is_some(),
        "packed residue row must reject a non-centered left coordinate"
    );
    encoded.assignment[residues.start] = honest_left;
    assert!(encoded.is_satisfied());

    let honest_right = encoded.assignment[residues.start + 1];
    encoded.assignment[residues.start + 1] = F::from_u64(2);
    assert!(
        encoded.first_unsatisfied_row().is_some(),
        "packed residue row must reject a non-centered right coordinate"
    );
    encoded.assignment[residues.start + 1] = honest_right;
    assert!(encoded.is_satisfied());
}

fn canonical_centered_pair(index: usize) -> [F; 2] {
    match index {
        0 => [-F::ONE, -F::ONE],
        1 => [-F::ONE, F::ZERO],
        2 => [F::ZERO, F::ZERO],
        3 => [F::ONE, F::ZERO],
        4 => [F::ONE, F::ONE],
        _ => panic!("residue index outside 0..5"),
    }
}

#[test]
fn packed_mod5_trace_mutations_fail_closed() {
    let builder = sampler_builder();
    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();
    let first = &trace.mod5_chunks()[0];
    let row_end = first.source_rows.end;
    let column_end = first.allocated_columns.end;
    let index_column = first.index.col();

    let rejected = |mutation| {
        let mut corrupted = trace.clone();
        corrupted.apply_mod5_trace_test_mutation(0, mutation);
        estimate_r1cs_gadget_native(&source, &corrupted, &[]).expect_err("corrupted mod-5 provenance")
    };

    assert!(matches!(
        rejected(Mod5TraceTestMutation::SourceRowEnd { row_end: row_end - 1 }),
        GadgetNativeError::PackedMod5Geometry { chunk: 0, .. }
    ));
    assert!(matches!(
        rejected(Mod5TraceTestMutation::AllocatedColumnEnd {
            column_end: column_end - 1,
        }),
        GadgetNativeError::PackedMod5Geometry { chunk: 0, .. }
    ));
    assert!(matches!(
        rejected(Mod5TraceTestMutation::QuotientBitColumn {
            index: 0,
            column: index_column,
        }),
        GadgetNativeError::PackedMod5Geometry { chunk: 0, .. }
    ));
    assert!(matches!(
        rejected(Mod5TraceTestMutation::ChunkBitColumn {
            index: 0,
            column: index_column,
        }),
        GadgetNativeError::PackedMod5Geometry { chunk: 0, .. }
    ));
}

#[test]
fn packed_mod5_rejects_product_temporary_escape_but_allows_checked_index_use() {
    let mut builder = sampler_builder();
    let baseline = builder.snapshot();
    estimate_r1cs_gadget_native(&baseline, builder.encoding_trace(), &[])
        .expect("the production index may feed its following symbol row");
    let product = builder.encoding_trace().mod5_chunks()[0].index_products[0];
    let value = builder.witness()[product.col()];
    builder.enforce_eq(&Lc::from_var(product), &Lc::from_const(value));
    assert!(
        builder.is_satisfied(),
        "escape probe remains a satisfied source relation"
    );

    let source = builder.snapshot();
    let error = estimate_r1cs_gadget_native(&source, builder.encoding_trace(), &[])
        .expect_err("mod-5 product temporary must not escape");
    assert!(matches!(
        error,
        GadgetNativeError::GadgetTemporaryEscapes { column } if column == product.col()
    ));
}
