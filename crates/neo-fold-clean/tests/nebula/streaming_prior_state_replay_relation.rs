//! Exact source tests for the bounded prior-state replay arms.

use std::collections::BTreeMap;

use neo_fold_clean::frontends::nebula::f_prime::{
    production_prior_state_replay_final_source_arm, production_prior_state_replay_full_source_arm,
    NebulaFPrimePriorStateReplayArmKind, NebulaFPrimePriorStateReplaySynthesis, NebulaFPrimeStreamingCircuitKind,
    NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit, NebulaFPrimeStreamingPublicLayout,
    PRIOR_STATE_REPLAY_AFTER_LAST_PROGRAM_CURSOR, PRIOR_STATE_REPLAY_CHUNKS, PRIOR_STATE_REPLAY_CHUNK_FIELDS,
    PRIOR_STATE_REPLAY_FINAL_FIELDS, PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMN_LAYOUT, PRIOR_STATE_REPLAY_FINAL_SOURCE_POSEIDON2_PERMUTATIONS,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS, PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_STAGE_SCHEDULE, PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS,
    PRIOR_STATE_REPLAY_FIRST_PROGRAM_CURSOR, PRIOR_STATE_REPLAY_FRAME_FIELDS, PRIOR_STATE_REPLAY_FULL_CHUNKS,
    PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS, PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMN_LAYOUT,
    PRIOR_STATE_REPLAY_FULL_SOURCE_POSEIDON2_PERMUTATIONS, PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS,
    PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256, PRIOR_STATE_REPLAY_FULL_SOURCE_STAGE_SCHEDULE,
    PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE, PRIOR_STATE_REPLAY_PROFILE_ID, PRIOR_STATE_REPLAY_SOURCE_ARTIFACT_ID,
    PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS, STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
    STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY,
    STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_AFTER_STATE_FAMILY, STREAMING_PRIOR_STATE_REPLAY_BEFORE_STATE_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_CHUNK_FAMILY, STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_LIFECYCLE_CARRY_FAMILY, STREAMING_PRIOR_STATE_REPLAY_STATE_TRANSITION_FAMILY,
};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

fn exact_column_family(
    synthesis: &NebulaFPrimePriorStateReplaySynthesis,
    name: &'static str,
) -> std::ops::Range<usize> {
    let matches = synthesis
        .builder_for_artifact()
        .column_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    let [range] = matches.as_slice() else {
        panic!("expected one column family {name}")
    };
    range.column_start..range.column_end
}

fn exact_row_family(synthesis: &NebulaFPrimePriorStateReplaySynthesis, name: &'static str) -> std::ops::Range<usize> {
    let matches = synthesis
        .builder_for_artifact()
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    let [range] = matches.as_slice() else {
        panic!("expected one row family {name}")
    };
    range.row_start..range.row_end
}

fn source_rows_sha256(source: &SparseR1cs) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream-normalized-sparse-r1cs-csc-v1\0");
    hasher.update((source.n as u64).to_le_bytes());
    hasher.update((source.m as u64).to_le_bytes());
    hasher.update((source.m_in as u64).to_le_bytes());
    for (matrix_index, matrix) in [&source.a, &source.b, &source.c].into_iter().enumerate() {
        assert!(matrix.seeded_phi81_blocks().is_empty());
        assert!(matrix.geometric_runs().is_empty());
        let csc = matrix
            .sparse_component()
            .expect("prior-state replay source uses canonical CSC matrices");
        assert!(csc.is_canonical());
        hasher.update([matrix_index as u8]);
        hasher.update((csc.nrows as u64).to_le_bytes());
        hasher.update((csc.ncols as u64).to_le_bytes());
        hasher.update((csc.col_ptr.len() as u64).to_le_bytes());
        for &pointer in &csc.col_ptr {
            hasher.update(pointer.to_le_bytes());
        }
        hasher.update((csc.row_idx.len() as u64).to_le_bytes());
        for (&row, coefficient) in csc.row_idx.iter().zip(&csc.vals) {
            hasher.update(row.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn assert_mutation_rejected(mut synthesis: NebulaFPrimePriorStateReplaySynthesis, column: usize) {
    let changed = synthesis.witness_value(column).expect("mutation column") + F::ONE;
    synthesis.tamper_witness_for_test(column, changed);
    assert!(
        synthesis.first_unsatisfied_row().is_some(),
        "column {column} mutation must fail"
    );
}

#[test]
fn prior_state_replay_owns_exact_full_and_final_sources() {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    assert_eq!(PRIOR_STATE_REPLAY_CHUNK_FIELDS, 1_024);
    assert_eq!(PRIOR_STATE_REPLAY_FINAL_FIELDS, 522);
    assert_eq!(PRIOR_STATE_REPLAY_FULL_CHUNKS, 93);
    assert_eq!(PRIOR_STATE_REPLAY_CHUNKS, 94);
    assert_eq!(PRIOR_STATE_REPLAY_FRAME_FIELDS, 95_754);
    assert_eq!(PRIOR_STATE_REPLAY_FIRST_PROGRAM_CURSOR, 1);
    assert_eq!(PRIOR_STATE_REPLAY_AFTER_LAST_PROGRAM_CURSOR, 95);
    assert_eq!(program.prior_state_frame_fields(), PRIOR_STATE_REPLAY_FRAME_FIELDS);
    assert_eq!(program.prior_state_chunks(), PRIOR_STATE_REPLAY_CHUNKS);
    assert_eq!(
        PRIOR_STATE_REPLAY_PROFILE_ID,
        "nightstream/goldilocks/b2-k16/streaming-prior-state-replay/v1"
    );
    assert_eq!(
        PRIOR_STATE_REPLAY_SOURCE_ARTIFACT_ID,
        "rust:streaming-prior-state-replay/source-b2-k16-v1"
    );
    assert_eq!(
        PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE,
        "recursive transition: prior-state replay indices 0..94"
    );
    assert!(PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS.starts_with("pending final selective link"));

    let items = program
        .work_items()
        .iter()
        .enumerate()
        .filter(|(_, item)| item.phase() == NebulaFPrimeStreamingPhase::PriorStateReplay)
        .collect::<Vec<_>>();
    assert_eq!(items.len(), PRIOR_STATE_REPLAY_CHUNKS);
    assert_eq!(items.first().map(|(_, item)| item.index()), Some(0));
    assert_eq!(items.last().map(|(_, item)| item.index()), Some(93));
    let kinds = program.circuit_kind_map();
    assert!(items[..PRIOR_STATE_REPLAY_FULL_CHUNKS]
        .iter()
        .all(|(arm, _)| kinds[*arm] == NebulaFPrimeStreamingCircuitKind::PriorStateReplayFull.code() as usize));
    assert_eq!(
        kinds[items[PRIOR_STATE_REPLAY_FULL_CHUNKS].0],
        NebulaFPrimeStreamingCircuitKind::PriorStateReplayFinal.code() as usize
    );

    for synthesis in [
        NebulaFPrimePriorStateReplaySynthesis::production_full(),
        NebulaFPrimePriorStateReplaySynthesis::production_final(),
    ] {
        assert!(synthesis.is_satisfied());
        assert_eq!(synthesis.first_unsatisfied_row(), None);
        assert!(synthesis.unconstrained_columns().is_empty());
        assert_eq!(synthesis.public_columns(), public.logical_columns());
        assert_eq!(
            exact_column_family(&synthesis, STREAMING_PRIOR_STATE_REPLAY_BEFORE_STATE_FAMILY).len(),
            10
        );
        assert_eq!(
            exact_column_family(&synthesis, STREAMING_PRIOR_STATE_REPLAY_AFTER_STATE_FAMILY).len(),
            10
        );
        assert_eq!(
            exact_column_family(&synthesis, STREAMING_PRIOR_STATE_REPLAY_CHUNK_FAMILY).len(),
            PRIOR_STATE_REPLAY_CHUNK_FIELDS
        );
        assert!(!exact_row_family(&synthesis, STREAMING_PRIOR_STATE_REPLAY_STATE_TRANSITION_FAMILY).is_empty());
        assert_eq!(
            exact_row_family(&synthesis, STREAMING_PRIOR_STATE_REPLAY_LIFECYCLE_CARRY_FAMILY).len(),
            8
        );
        assert_eq!(
            exact_column_family(&synthesis, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY).len(),
            4
        );
        assert_eq!(
            exact_column_family(&synthesis, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY).len(),
            4
        );
        let before_payload = exact_column_family(&synthesis, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY);
        let after_payload = exact_column_family(&synthesis, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY);
        assert_eq!(before_payload, after_payload);
        assert_eq!(before_payload.len(), STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS);
    }

    let full = NebulaFPrimePriorStateReplaySynthesis::production_full();
    assert_eq!(full.kind(), NebulaFPrimePriorStateReplayArmKind::Full);
    assert_eq!(full.target_digest_columns(), None);
    assert_eq!(
        full.witness_value(full.before_program_cursor_column()),
        Some(F::from_u64(PRIOR_STATE_REPLAY_FIRST_PROGRAM_CURSOR as u64))
    );
    assert_eq!(full.witness_value(full.before_state_columns()[9]), Some(F::ZERO));
    assert_eq!(
        full.witness_value(full.after_state_columns()[9]),
        Some(F::from_u64(PRIOR_STATE_REPLAY_CHUNK_FIELDS as u64))
    );

    let final_arm = NebulaFPrimePriorStateReplaySynthesis::production_final();
    assert_eq!(final_arm.kind(), NebulaFPrimePriorStateReplayArmKind::Final);
    assert!(!exact_row_family(&final_arm, STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY).is_empty());
    assert_eq!(
        final_arm.witness_value(final_arm.before_state_columns()[9]),
        Some(F::from_u64(
            (PRIOR_STATE_REPLAY_FULL_CHUNKS * PRIOR_STATE_REPLAY_CHUNK_FIELDS) as u64
        ))
    );
    assert_eq!(
        final_arm.witness_value(final_arm.after_state_columns()[9]),
        Some(F::from_u64(PRIOR_STATE_REPLAY_FRAME_FIELDS as u64))
    );
    for &column in &final_arm.chunk_columns()[PRIOR_STATE_REPLAY_FINAL_FIELDS..] {
        assert_eq!(final_arm.witness_value(column), Some(F::ZERO));
    }
    assert!(final_arm.target_digest_columns().is_some());
}

#[test]
fn prior_state_replay_rejects_transition_target_and_public_mutations() {
    let full = NebulaFPrimePriorStateReplaySynthesis::production_full();
    for column in [
        full.before_state_columns()[0],
        full.after_state_columns()[7],
        full.chunk_columns()[511],
        full.before_phase_local_state_source_columns()[0],
        full.after_phase_local_state_source_columns()[3],
        full.phase_delayed_payload_columns()[0],
        full.before_x_out_preimage_columns()[19],
        full.after_x_out_preimage_columns()[19],
        full.after_boundary_columns()[2],
        full.after_accumulator_columns()[3],
        full.public_output_column(0).expect("first public output"),
    ] {
        assert_mutation_rejected(NebulaFPrimePriorStateReplaySynthesis::production_full(), column);
    }

    let final_arm = NebulaFPrimePriorStateReplaySynthesis::production_final();
    for column in [
        final_arm.chunk_columns()[PRIOR_STATE_REPLAY_FINAL_FIELDS],
        final_arm.target_digest_columns().expect("final target")[0],
        final_arm.after_state_columns()[9],
    ] {
        assert_mutation_rejected(NebulaFPrimePriorStateReplaySynthesis::production_final(), column);
    }
}

#[test]
fn prior_state_replay_reports_exact_source_geometry() {
    assert_eq!(
        NebulaFPrimePriorStateReplaySynthesis::production_full()
            .shape_audit()
            .poseidon2_permutations,
        PRIOR_STATE_REPLAY_FULL_SOURCE_POSEIDON2_PERMUTATIONS
    );
    assert_eq!(
        NebulaFPrimePriorStateReplaySynthesis::production_final()
            .shape_audit()
            .poseidon2_permutations,
        PRIOR_STATE_REPLAY_FINAL_SOURCE_POSEIDON2_PERMUTATIONS
    );
    let sources = [
        (
            "full",
            production_prior_state_replay_full_source_arm().expect("lower full source"),
            PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_POSEIDON2_PERMUTATIONS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256,
            PRIOR_STATE_REPLAY_FULL_SOURCE_STAGE_SCHEDULE.as_slice(),
            PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMN_LAYOUT,
        ),
        (
            "final",
            production_prior_state_replay_final_source_arm().expect("lower final source"),
            PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_POSEIDON2_PERMUTATIONS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_STAGE_SCHEDULE.as_slice(),
            PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMN_LAYOUT,
        ),
    ];
    for (name, source, rows, columns, poseidon2_permutations, sha256, frozen_stages, column_layout) in sources {
        assert_eq!(source.n, rows);
        assert_eq!(source.m, columns);
        assert_eq!(source.m_in, PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS);
        assert_eq!(
            source.m_in,
            NebulaFPrimeStreamingPublicLayout::production().logical_columns()
        );
        assert!(poseidon2_permutations > 0);
        assert_eq!(source_rows_sha256(&source), sha256);
        assert_eq!(column_layout.constant_one(), 0);
        assert_eq!(column_layout.after_x_out_bits(), (1, 257));
        assert_eq!(column_layout.before_x_out_bits(), (257, 513));
        assert_eq!(column_layout.before_cursor_bits(), (513, 577));
        assert_eq!(column_layout.after_cursor_bits(), (577, 641));
        assert_eq!(column_layout.common_public_padding(), (641, 648));
        assert_eq!(column_layout.private_columns(), (source.m_in, source.m));
        let stages = source
            .physical_stage_ranges()
            .iter()
            .map(|stage| (stage.path(), stage.rows(), stage.columns()))
            .collect::<Vec<_>>();
        let expected_stages = frozen_stages
            .iter()
            .map(|stage| {
                (
                    stage.path(),
                    stage.row_start()..stage.row_end(),
                    stage.column_start()..stage.column_end(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(stages, expected_stages);
        let mut row_owner = BTreeMap::new();
        for (path, rows, _) in &stages {
            for row in rows.clone() {
                assert_eq!(row_owner.insert(row, *path), None, "row {row} has two physical owners");
            }
        }
        assert_eq!(row_owner.len(), source.n);
        println!(
            "prior-state {name}: rows={}, columns={}, public={}",
            source.n, source.m, source.m_in
        );
    }
}
