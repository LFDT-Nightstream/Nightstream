//! Exact source and transition tests for the streaming PiCCS start body.

use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_ccs_start_source_arm, NebulaFPrimePiCcsStartSynthesis, NebulaFPrimeStreamingPublicLayout,
    PI_CCS_START_AFTER_PROGRAM_CURSOR, PI_CCS_START_BEFORE_PROGRAM_CURSOR, PI_CCS_START_FINAL_BINDING_STATUS,
    PI_CCS_START_FINAL_COMMON_PUBLIC_COLUMNS, PI_CCS_START_LIFECYCLE_SCOPE, PI_CCS_START_PROFILE_ID,
    PI_CCS_START_SOURCE_ARTIFACT_ID, PI_CCS_START_SOURCE_COLUMNS, PI_CCS_START_SOURCE_COLUMN_LAYOUT,
    PI_CCS_START_SOURCE_HASH_SCHEMA, PI_CCS_START_SOURCE_POSEIDON2_PERMUTATIONS, PI_CCS_START_SOURCE_PUBLIC_COLUMNS,
    PI_CCS_START_SOURCE_ROWS, PI_CCS_START_SOURCE_SHA256, PI_CCS_START_SOURCE_STAGE_SCHEDULE,
    STREAMING_PI_CCS_START_CONTEXT_FAMILY, STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY,
    STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY, STREAMING_PI_CCS_START_READY_FAMILY,
    STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY, STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY,
};
use neo_fold_clean::frontends::r1cs_f_prime::{selective_polynomial, SparseR1cs};
use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const POINTS: usize = 26;
const RUNNING: usize = 16;
const MATRICES: usize = 14;
const COEFFICIENTS: usize = 54;
const VARIABLE_FIELDS: usize = 24_244;
const STATEMENT_FRESH_FIELDS: usize = 28_672;
const GAMMA_POWERS: usize = 12_130;

type Pair = [F; 2];

fn exact_row_family(synthesis: &NebulaFPrimePiCcsStartSynthesis, name: &'static str) -> std::ops::Range<usize> {
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

fn pair_add(left: Pair, right: Pair) -> Pair {
    [left[0] + right[0], left[1] + right[1]]
}

fn pair_mul(left: Pair, right: Pair) -> Pair {
    [
        left[0] * right[0] + F::from_u64(7) * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

fn evaluation_index(running: usize, matrix: usize, coefficient: usize) -> usize {
    2 * POINTS + 2 * ((running * MATRICES + matrix) * COEFFICIENTS + coefficient)
}

fn carried_exponent(running: usize, matrix: usize, coefficient: usize) -> usize {
    2 + RUNNING + running + RUNNING * matrix + RUNNING * MATRICES * coefficient
}

fn native_initial_claim(synthesis: &NebulaFPrimePiCcsStartSynthesis, gamma: Pair) -> Pair {
    let mut powers = Vec::with_capacity(GAMMA_POWERS);
    powers.push([F::ONE, F::ZERO]);
    while powers.len() < GAMMA_POWERS {
        powers.push(pair_mul(*powers.last().expect("nonempty powers"), gamma));
    }
    let mut sum = [F::ZERO; 2];
    for running in 0..RUNNING {
        for matrix in 0..MATRICES {
            for coefficient in 0..COEFFICIENTS {
                let start = evaluation_index(running, matrix, coefficient);
                let value = [
                    synthesis
                        .witness_value(
                            synthesis
                                .variable_field_column(start)
                                .expect("evaluation c0"),
                        )
                        .expect("evaluation c0 value"),
                    synthesis
                        .witness_value(
                            synthesis
                                .variable_field_column(start + 1)
                                .expect("evaluation c1"),
                        )
                        .expect("evaluation c1 value"),
                ];
                sum = pair_add(
                    sum,
                    pair_mul(powers[carried_exponent(running, matrix, coefficient)], value),
                );
            }
        }
    }
    sum
}

fn native_pre_sumcheck(synthesis: &NebulaFPrimePiCcsStartSynthesis) -> ([Pair; POINTS], Pair, [F; 8]) {
    let before = synthesis.before_runtime_columns().map(|column| {
        synthesis
            .witness_value(column)
            .expect("complete before transcript state")
    });
    let absorbed = synthesis
        .witness_value(synthesis.before_runtime_absorbed_column())
        .expect("before absorbed value")
        .as_canonical_u64() as usize;
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(before, absorbed);
    let polynomial = selective_polynomial();
    let mut statement = vec![
        neo_reductions::engines::pi_ccs_joint::STATEMENT_TAG,
        POINTS as u64,
        1,
        RUNNING as u64,
        MATRICES as u64,
        COEFFICIENTS as u64,
        polynomial.max_degree() as u64,
        polynomial.terms().len() as u64,
    ]
    .into_iter()
    .map(F::from_u64)
    .collect::<Vec<_>>();
    for term in polynomial.terms() {
        statement.extend([term.coeff, F::ZERO, F::ZERO]);
        statement.extend(
            term.exps
                .iter()
                .map(|&exponent| F::from_u64(exponent as u64)),
        );
    }
    statement.push(F::from_u64(neo_reductions::engines::pi_ccs_joint::COMPACT_BINDING_TAG));
    transcript.append_fields_unframed(&statement);

    let alpha = std::array::from_fn(|index| {
        transcript.append_fields_unframed(&[
            F::from_u64(neo_reductions::engines::pi_ccs_joint::ALPHA_TAG),
            F::from_usize(index),
        ]);
        let value = transcript.challenge_fields_raw(2);
        [value[0], value[1]]
    });
    transcript.append_fields_unframed(&[F::from_u64(neo_reductions::engines::pi_ccs_joint::GAMMA_TAG)]);
    let gamma_fields = transcript.challenge_fields_raw(2);
    (alpha, [gamma_fields[0], gamma_fields[1]], transcript.state())
}

fn assert_mutation_rejected_and_restore(synthesis: &mut NebulaFPrimePiCcsStartSynthesis, column: usize) {
    let original = synthesis.witness_value(column).expect("mutation column");
    synthesis.tamper_witness_for_test(column, original + F::ONE);
    assert!(
        synthesis.first_unsatisfied_row().is_some(),
        "column {column} mutation must fail"
    );
    synthesis.tamper_witness_for_test(column, original);
    assert_eq!(synthesis.first_unsatisfied_row(), None, "column {column} restore");
}

fn source_rows_sha256(source: &SparseR1cs) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PI_CCS_START_SOURCE_HASH_SCHEMA.as_bytes());
    hasher.update([0]);
    hasher.update((source.n as u64).to_le_bytes());
    hasher.update((source.m as u64).to_le_bytes());
    hasher.update((source.m_in as u64).to_le_bytes());
    for (matrix_index, matrix) in [&source.a, &source.b, &source.c].into_iter().enumerate() {
        let csc = matrix
            .sparse_component()
            .expect("streaming PiCCS start source uses a canonical CSC base");
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

        let blocks = matrix.seeded_phi81_blocks();
        hasher.update((blocks.len() as u64).to_le_bytes());
        for block in blocks {
            hasher.update((block.row_start() as u64).to_le_bytes());
            hasher.update((block.word_starts().len() as u64).to_le_bytes());
            for &word_start in block.word_starts() {
                hasher.update((word_start as u64).to_le_bytes());
            }
            hasher.update((block.word_width() as u64).to_le_bytes());
            hasher.update((block.kappa() as u64).to_le_bytes());
            hasher.update((block.message_cols() as u64).to_le_bytes());
            hasher.update((block.chunk_size() as u64).to_le_bytes());
            hasher.update([u8::from(block.has_superneo_transformed_columns())]);
            hasher.update((block.chunk_seeds_by_row().len() as u64).to_le_bytes());
            for seed_row in block.chunk_seeds_by_row() {
                hasher.update((seed_row.len() as u64).to_le_bytes());
                for seed in seed_row {
                    hasher.update(seed);
                }
            }
        }

        let runs = matrix.geometric_runs();
        hasher.update((runs.len() as u64).to_le_bytes());
        for run in runs {
            hasher.update((run.row() as u64).to_le_bytes());
            hasher.update((run.column_start() as u64).to_le_bytes());
            hasher.update((run.len() as u64).to_le_bytes());
            hasher.update(run.initial().as_canonical_u64().to_le_bytes());
            hasher.update(run.ratio().as_canonical_u64().to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn streaming_pi_ccs_start_opens_the_complete_statement_and_initializes_round_zero() {
    let mut synthesis = NebulaFPrimePiCcsStartSynthesis::production().expect("exact PiCCS start source");
    let shape = synthesis.shape_audit();
    eprintln!("{shape:#?}");
    eprintln!(
        "ready={:?} variable_binding={:?} transcript={:?} initial_claim={:?} context={:?} lifecycle_carry={:?}",
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_READY_FAMILY),
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY),
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY),
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY),
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_CONTEXT_FAMILY),
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY),
    );
    assert!(synthesis.is_satisfied());
    assert_eq!(synthesis.first_unsatisfied_row(), None);
    assert!(synthesis.unconstrained_columns().is_empty());
    assert_eq!(shape.rows, PI_CCS_START_SOURCE_ROWS);
    assert_eq!(shape.columns, PI_CCS_START_SOURCE_COLUMNS);
    assert_eq!(shape.public_columns, PI_CCS_START_SOURCE_PUBLIC_COLUMNS);
    assert_eq!(shape.poseidon2_permutations, PI_CCS_START_SOURCE_POSEIDON2_PERMUTATIONS);
    assert_eq!(shape.variable_fields, VARIABLE_FIELDS);
    assert_eq!(shape.gamma_powers, GAMMA_POWERS);
    assert_eq!(
        shape.public_columns,
        NebulaFPrimeStreamingPublicLayout::production().logical_columns()
    );
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_START_READY_FAMILY).is_empty());
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY).is_empty());
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY).is_empty());
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY).is_empty());
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_START_CONTEXT_FAMILY).is_empty());
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY).len(),
        8
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_READY_FAMILY),
        69..82
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY),
        82..3_006_597
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY),
        3_006_597..3_203_470
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY),
        3_203_470..3_324_599
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_CONTEXT_FAMILY),
        3_324_599..3_376_867
    );
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY),
        4_104_069..4_104_077
    );

    assert_eq!(
        PI_CCS_START_PROFILE_ID,
        "nightstream/goldilocks/b2-k16/streaming-pi-ccs-start/v3"
    );
    assert_eq!(
        PI_CCS_START_SOURCE_ARTIFACT_ID,
        "rust:streaming-pi-ccs-start/source-b2-k16-v3"
    );
    assert_eq!(
        PI_CCS_START_LIFECYCLE_SCOPE,
        "recursive transition: claim replay to PiCCS round 0"
    );
    assert_eq!(PI_CCS_START_BEFORE_PROGRAM_CURSOR, 193);
    assert_eq!(PI_CCS_START_AFTER_PROGRAM_CURSOR, 194);
    assert_eq!(PI_CCS_START_FINAL_COMMON_PUBLIC_COLUMNS, 648);
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.constant_one(), 0);
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.after_x_out_bits(), (1, 257));
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.before_x_out_bits(), (257, 513));
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.before_cursor_bits(), (513, 577));
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.after_cursor_bits(), (577, 641));
    assert_eq!(PI_CCS_START_SOURCE_COLUMN_LAYOUT.common_public_padding(), (641, 648));
    assert_eq!(
        PI_CCS_START_SOURCE_COLUMN_LAYOUT.private_columns(),
        (PI_CCS_START_SOURCE_PUBLIC_COLUMNS, PI_CCS_START_SOURCE_COLUMNS)
    );
    assert_eq!(
        PI_CCS_START_FINAL_BINDING_STATUS,
        "pending complete 23-kind selective CCS schedule; no final row identity is claimed"
    );

    for lane in 0..108 {
        assert_eq!(
            synthesis.witness_value(synthesis.expected_statement_fresh_commitment_columns()[lane]),
            Some(
                synthesis
                    .witness_value(synthesis.computed_statement_commitment_columns()[lane])
                    .expect("computed statement commitment")
                    + synthesis
                        .witness_value(synthesis.fresh_metadata_residual_columns()[lane])
                        .expect("fresh-metadata residual")
            ),
        );
    }
    for point in synthesis.after_reverse_point_columns() {
        assert_eq!(synthesis.witness_value(point[0]), Some(F::ZERO));
        assert_eq!(synthesis.witness_value(point[1]), Some(F::ZERO));
    }
    assert_eq!(
        synthesis.witness_value(synthesis.after_round_cursor_column()),
        Some(F::ZERO)
    );
    assert_eq!(
        synthesis.witness_value(synthesis.before_program_cursor_column()),
        Some(F::from_u64(193))
    );
    assert_eq!(
        synthesis.witness_value(synthesis.after_program_cursor_column()),
        Some(F::from_u64(194))
    );

    let (native_alpha, native_gamma, native_state) = native_pre_sumcheck(&synthesis);
    for (expected, columns) in native_alpha.into_iter().zip(synthesis.alpha_columns()) {
        assert_eq!(synthesis.witness_value(columns[0]), Some(expected[0]));
        assert_eq!(synthesis.witness_value(columns[1]), Some(expected[1]));
    }
    let gamma_columns = synthesis.gamma_columns();
    assert_eq!(synthesis.witness_value(gamma_columns[0]), Some(native_gamma[0]));
    assert_eq!(synthesis.witness_value(gamma_columns[1]), Some(native_gamma[1]));
    for (expected, column) in native_state
        .into_iter()
        .zip(synthesis.after_transcript_columns())
    {
        assert_eq!(synthesis.witness_value(column), Some(expected));
    }
    let initial = native_initial_claim(&synthesis, native_gamma);
    let current = synthesis.after_current_columns();
    assert_eq!(synthesis.witness_value(current[0]), Some(initial[0]));
    assert_eq!(synthesis.witness_value(current[1]), Some(initial[1]));

    for lane in 0..4 {
        assert_eq!(
            synthesis.witness_value(synthesis.before_boundary_columns()[lane]),
            synthesis.witness_value(synthesis.after_boundary_columns()[lane]),
        );
        assert_eq!(
            synthesis.witness_value(synthesis.before_accumulator_columns()[lane]),
            synthesis.witness_value(synthesis.after_accumulator_columns()[lane]),
        );
    }

    let mutations = [
        synthesis
            .variable_field_column(17)
            .expect("statement field"),
        synthesis.expected_statement_fresh_commitment_columns()[9],
        synthesis.expected_running_commitments_binding_columns()[31],
        synthesis.expected_running_public_binding_columns()[47],
        synthesis.computed_statement_commitment_columns()[77],
        synthesis.fresh_metadata_residual_columns()[55],
        synthesis.alpha_columns()[3][1],
        synthesis.gamma_columns()[0],
        synthesis.after_current_columns()[1],
        synthesis.after_reverse_point_columns()[8][0],
        synthesis.context_digest_columns()[2],
        synthesis.after_boundary_columns()[1],
        synthesis.after_accumulator_columns()[3],
        synthesis.before_phase_local_state_source_columns()[0],
        synthesis.after_phase_local_state_source_columns()[3],
        synthesis.phase_delayed_payload_columns()[0],
        synthesis.before_x_out_preimage_columns()[19],
        synthesis.after_x_out_preimage_columns()[19],
        synthesis
            .public_output_column(0)
            .expect("first public output"),
    ];
    for column in mutations {
        assert_mutation_rejected_and_restore(&mut synthesis, column);
    }

    drop(synthesis);
    let source = production_pi_ccs_start_source_arm().expect("lower exact PiCCS start source");
    assert_eq!(source_rows_sha256(&source), PI_CCS_START_SOURCE_SHA256);
    assert_eq!(source.n, PI_CCS_START_SOURCE_ROWS);
    assert_eq!(source.m, PI_CCS_START_SOURCE_COLUMNS);
    assert_eq!(source.m_in, PI_CCS_START_SOURCE_PUBLIC_COLUMNS);
    assert_eq!(source.a.seeded_phi81_blocks().len(), 1);
    assert!(source.b.seeded_phi81_blocks().is_empty());
    assert!(source.c.seeded_phi81_blocks().is_empty());
    for matrix in [&source.a, &source.b, &source.c] {
        assert!(matrix.geometric_runs().is_empty());
    }
    let [variable_binding] = source.a.seeded_phi81_blocks() else {
        panic!("one exact PiCCS variable-binding block")
    };
    assert_eq!(variable_binding.word_starts().len(), STATEMENT_FRESH_FIELDS);
    assert_eq!(variable_binding.word_width(), 41);
    assert_eq!(variable_binding.kappa(), 2);
    assert_eq!(variable_binding.message_cols(), 21_770);
    assert!(!variable_binding.has_superneo_transformed_columns());

    let stage_schedule = source
        .physical_stage_ranges()
        .iter()
        .map(|stage| (stage.path(), stage.rows(), stage.columns()))
        .collect::<Vec<_>>();
    let frozen_stage_schedule = PI_CCS_START_SOURCE_STAGE_SCHEDULE
        .iter()
        .map(|stage| {
            (
                stage.path(),
                stage.row_start()..stage.row_end(),
                stage.column_start()..stage.column_end(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(stage_schedule, frozen_stage_schedule);
    assert_eq!(
        stage_schedule,
        vec![
            ("nebula.streaming.pi_ccs.start.state_words", 0..69, 641..25_235),
            ("nebula.streaming.pi_ccs.start.ready", 69..82, 25_235..25_235),
            (
                "nebula.streaming.pi_ccs.start.variable_binding",
                82..3_006_597,
                25_235..2_983_262
            ),
            (
                "nebula.streaming.pi_ccs.start.transcript",
                3_006_597..3_203_470,
                2_983_262..3_180_135
            ),
            (
                "nebula.streaming.pi_ccs.start.initial_claim",
                3_203_470..3_324_599,
                3_180_135..3_301_264
            ),
            (
                "nebula.streaming.pi_ccs.start.context",
                3_324_599..3_376_867,
                3_301_264..3_353_532
            ),
            (
                "nebula.streaming.pi_ccs.start.state_digest",
                3_376_867..3_441_097,
                3_353_532..3_417_762
            ),
            (
                "nebula.streaming.pi_ccs.start.phase_envelope",
                3_441_097..4_104_068,
                3_417_762..4_080_733
            ),
            (
                "nebula.streaming.pi_ccs.start.state_x_out",
                4_104_068..4_115_653,
                4_080_733..4_091_727
            ),
        ]
    );
}
