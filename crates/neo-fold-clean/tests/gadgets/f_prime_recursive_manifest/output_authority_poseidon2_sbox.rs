//! External release checks for the output-authority Poseidon2 S-box census.
//!
//! Owns: exact production census assertions and fail-closed mutations for the
//! trace, source matrix, sponge schedule, and protected authority boundaries.
//!
//! Does not own: a lowered S-box encoding or generated Lean evidence.
//!
//! Emits constraints: no; it audits the production recursive relation.
//!
//! | Test branch | Mathematical obligation | Corruption class | Expected result |
//! |---|---|---|---|
//! | `census` | 17 permutations contain exactly 1,462 `x^7` outputs | count drift | reject |
//! | `replay` | every call is one exact affine renaming of production Poseidon2 | row/column/range drift | reject |
//! | `prehash_boundary` | prehash rows and fresh columns have the expected ownership shape | boundary geometry drift | reject |
//! | `whole_matrix` | each candidate has one definition and eight consumers, all inside its call | extra/missing/coefficient drift | reject |
//! | `authority_aliases` | candidates are disjoint from public, digest, and authority boundaries | alias | reject |

use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::{
    PoseidonHashTraceTestMutation, PoseidonPermutationTraceTestMutation, R1csEncodingTrace, R1csSnapshot,
    Sbox7TraceTestMutation,
};
use neo_fold_clean::frontends::f_prime::output_authority::audit_output_authority_poseidon2_sboxes;

use super::*;

fn expect_trace_rejected(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    mutate: impl FnOnce(&mut R1csEncodingTrace),
) {
    let mut corrupted = trace.clone();
    mutate(&mut corrupted);
    assert!(
        audit_output_authority_poseidon2_sboxes(source, &corrupted, &[]).is_err(),
        "corrupted output-authority provenance must fail closed",
    );
}

fn first_a_use(source: &R1csSnapshot, rows: Range<usize>, column: usize) -> (usize, F) {
    for row in rows {
        if let Some((_, coefficient)) = source
            .a_row(row)
            .iter()
            .find(|&&(candidate, _)| candidate == column)
        {
            return (row, *coefficient);
        }
    }
    panic!("candidate {column} has no A-matrix use in its permutation")
}

#[test]
#[ignore = "the production whole-matrix S-box audit exceeds the 24 GiB audit limit"]
fn output_authority_sbox_manifest_replays_exact_program_and_rejects_all_drift() {
    let builder = build_recursive_program();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let manifest = audit_output_authority_poseidon2_sboxes(&source, trace, &[])
        .expect("exact output-authority Poseidon2 S-box manifest");

    assert_eq!(manifest.stage_rows.len(), 10_278);
    assert_eq!(manifest.stage_columns.len(), 10_278);
    assert_eq!(manifest.prehash_rows.len(), 8);
    assert_eq!(manifest.prehash_columns.len(), 8);
    assert_eq!(manifest.census.prehash_binding_rows, 8);
    assert_eq!(manifest.census.prehash_fresh_columns, 8);
    assert_eq!(manifest.hash_input_columns.len(), 64);
    assert_eq!(manifest.permutation_trace_range.len(), 17);
    assert_eq!(manifest.sbox_trace_range.len(), 1_462);
    assert_eq!(manifest.calls.len(), 17);
    assert!(manifest
        .calls
        .iter()
        .all(|call| call.source_rows.len() == 600 && call.allocated_column_count == 600));
    assert_eq!(manifest.census.full_absorb_rounds, 16);
    assert_eq!(manifest.census.partial_absorb_fields, 0);
    assert_eq!(manifest.census.pad_rounds, 1);
    assert_eq!(manifest.census.initial_external_sboxes, 544);
    assert_eq!(manifest.census.partial_sboxes, 374);
    assert_eq!(manifest.census.terminal_external_sboxes, 544);
    assert_eq!(manifest.census.candidate_sbox_outputs, 1_462);
    assert_eq!(manifest.census.definition_uses, 1_462);
    assert_eq!(manifest.census.linear_consumer_uses, 11_696);
    assert_eq!(manifest.census.total_matrix_uses, 13_158);
    assert_eq!(manifest.isolated_sbox_output_offsets().len(), 86);
    assert_eq!(manifest.family_layout.initial_external, 0..32);
    assert_eq!(manifest.family_layout.partial, 32..54);
    assert_eq!(manifest.family_layout.terminal_external, 54..86);

    let hash = manifest.hash_index;
    let permutation = manifest.permutation_trace_range.start;
    let sbox = manifest.sbox_trace_range.start;
    let candidate = manifest.first_candidate_column();
    let first_call = &manifest.calls[0];

    // Hash input/output/range/escape drift.
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted
            .apply_poseidon_hash_trace_test_mutation(hash, PoseidonHashTraceTestMutation::InputLen { input_len: 63 });
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_hash_trace_test_mutation(
            hash,
            PoseidonHashTraceTestMutation::InputColumn {
                offset: 0,
                column: manifest.hash_input_columns[0] + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_hash_trace_test_mutation(
            hash,
            PoseidonHashTraceTestMutation::OutputColumn {
                lane: 0,
                column: manifest.hash_output_columns[0] + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_hash_trace_test_mutation(
            hash,
            PoseidonHashTraceTestMutation::PermutationRange {
                range: manifest.permutation_trace_range.start..manifest.permutation_trace_range.end - 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_hash_trace_test_mutation(
            hash,
            PoseidonHashTraceTestMutation::SourceRows {
                rows: manifest.hash_rows.start..manifest.hash_rows.end + 1,
            },
        );
    });

    // Per-call input/output/fresh-column/range drift.
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_permutation_trace_test_mutation(
            permutation,
            PoseidonPermutationTraceTestMutation::InputColumn {
                lane: 0,
                column: first_call.input_columns[0] + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_permutation_trace_test_mutation(
            permutation,
            PoseidonPermutationTraceTestMutation::OutputColumn {
                lane: 0,
                column: first_call.output_columns[0] + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_permutation_trace_test_mutation(
            permutation,
            PoseidonPermutationTraceTestMutation::AllocatedColumns {
                columns: first_call.first_allocated_column..first_call.first_allocated_column + 599,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_poseidon_permutation_trace_test_mutation(
            permutation,
            PoseidonPermutationTraceTestMutation::SourceRows {
                rows: first_call.source_rows.start..first_call.source_rows.end + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.duplicate_poseidon_permutation_trace_for_test(permutation);
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.remove_poseidon_permutation_trace_for_test(permutation);
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.swap_poseidon_permutation_traces_for_test(permutation, permutation + 1);
    });

    // S-box provenance and the exact 1,462-entry census.
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_sbox7_trace_test_mutation(
            sbox,
            Sbox7TraceTestMutation::InputColumn {
                offset: 0,
                column: trace.sbox7()[sbox].input.terms[0].0 + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_sbox7_trace_test_mutation(
            sbox,
            Sbox7TraceTestMutation::IntermediateColumn {
                index: 0,
                column: candidate + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_sbox7_trace_test_mutation(
            sbox,
            Sbox7TraceTestMutation::OutputColumn {
                column: trace.sbox7()[sbox].intermediates[2].col(),
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.apply_sbox7_trace_test_mutation(
            sbox,
            Sbox7TraceTestMutation::SourceRows {
                rows: trace.sbox7()[sbox].source_rows.start..trace.sbox7()[sbox].source_rows.end + 1,
            },
        );
    });
    expect_trace_rejected(&source, trace, |corrupted| {
        corrupted.duplicate_sbox7_trace_for_test(sbox)
    });
    expect_trace_rejected(&source, trace, |corrupted| corrupted.remove_sbox7_trace_for_test(sbox));

    // Public/digest/authority aliases all fail before any candidate is usable.
    assert!(audit_output_authority_poseidon2_sboxes(&source, trace, &[candidate]).is_err());
    for protected in [
        manifest.hash_output_columns[0],
        manifest.claimed_digest_columns[0],
        manifest.semantic_state_output_columns[0],
    ] {
        expect_trace_rejected(&source, trace, |corrupted| {
            corrupted.apply_sbox7_trace_test_mutation(sbox, Sbox7TraceTestMutation::OutputColumn { column: protected });
        });
    }

    // Missing and coefficient-drifted consumers are rejected by exact replay.
    let (consumer_row, coefficient) = first_a_use(&source, first_call.source_rows.clone(), candidate);
    let mut missing = source.clone();
    missing.apply_a_row_test_mutation(consumer_row, candidate, -coefficient);
    assert!(audit_output_authority_poseidon2_sboxes(&missing, trace, &[]).is_err());
    let mut coefficient_drift = source.clone();
    coefficient_drift.apply_a_row_test_mutation(consumer_row, candidate, F::ONE);
    assert!(audit_output_authority_poseidon2_sboxes(&coefficient_drift, trace, &[]).is_err());

    let definition_row = trace.sbox7()[sbox].source_rows.end - 1;
    assert_eq!(source.c_row(definition_row), [(candidate, F::ONE)]);
    let mut definition_drift = source.clone();
    definition_drift.apply_c_row_test_mutation(definition_row, candidate, F::ONE);
    assert!(audit_output_authority_poseidon2_sboxes(&definition_drift, trace, &[]).is_err());

    // Extra A/B/C uses outside all 17 calls pass local row replay but must
    // fail the whole-matrix escape scan.
    let escape_row = manifest.stage_rows.end;
    let mut escaped_a = source.clone();
    escaped_a.apply_a_row_test_mutation(escape_row, candidate, F::ONE);
    assert!(audit_output_authority_poseidon2_sboxes(&escaped_a, trace, &[]).is_err());
    let mut escaped_b = source.clone();
    escaped_b.apply_b_row_test_mutation(escape_row, candidate, F::ONE);
    assert!(audit_output_authority_poseidon2_sboxes(&escaped_b, trace, &[]).is_err());
    let mut escaped_c = source.clone();
    escaped_c.apply_c_row_test_mutation(escape_row, candidate, F::ONE);
    assert!(audit_output_authority_poseidon2_sboxes(&escaped_c, trace, &[]).is_err());
}
