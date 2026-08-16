//! Exact source and transition tests for the streaming PiCCS round body.

use std::collections::BTreeMap;

use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_ccs_round_source_arm, NebulaFPrimePiCcsRoundSynthesis, NebulaFPrimeStreamingPublicLayout,
    PI_CCS_ROUND_AFTER_LAST_PROGRAM_CURSOR, PI_CCS_ROUND_ARITHMETIC_BINDING,
    PI_CCS_ROUND_COMPACT_ARITHMETIC_ARTIFACT_ID, PI_CCS_ROUND_FINAL_COMMON_PUBLIC_COLUMNS,
    PI_CCS_ROUND_FIRST_PROGRAM_CURSOR, PI_CCS_ROUND_LIFECYCLE_SCOPE, PI_CCS_ROUND_PROFILE_ID,
    PI_CCS_ROUND_SOURCE_ARTIFACT_ID, PI_CCS_ROUND_SOURCE_COLUMNS, PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT,
    PI_CCS_ROUND_SOURCE_POSEIDON2_PERMUTATIONS, PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS, PI_CCS_ROUND_SOURCE_ROWS,
    PI_CCS_ROUND_SOURCE_SHA256, PI_CCS_ROUND_SOURCE_STAGE_SCHEDULE, STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
    STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY,
    STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
    STREAMING_PI_CCS_ROUND_AFTER_STATE_FAMILY, STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY,
    STREAMING_PI_CCS_ROUND_BEFORE_STATE_FAMILY, STREAMING_PI_CCS_ROUND_COEFFICIENT_FAMILY,
    STREAMING_PI_CCS_ROUND_LIFECYCLE_CARRY_FAMILY, STREAMING_PI_CCS_ROUND_STATE_DIGEST_FAMILY,
    STREAMING_PI_CCS_ROUND_STATE_TRANSITION_FAMILY, STREAMING_PI_CCS_ROUND_TRANSCRIPT_FAMILY,
};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const COEFFICIENT_COUNT: usize = 10;
const DEGREE: usize = COEFFICIENT_COUNT - 1;
const ARITHMETIC_ROWS: usize = 31;
const FIRST_ROUND_PROGRAM_CURSOR: u64 = 170;
const FIXTURE_ROUND: u64 = 7;

type Term = (usize, F);
type SourceRow = [Vec<Term>; 3];

fn singleton(column: usize) -> Vec<Term> {
    vec![(column, F::ONE)]
}

fn canonical_terms(terms: Vec<Term>) -> Vec<Term> {
    let mut combined = BTreeMap::new();
    for (column, coefficient) in terms {
        *combined.entry(column).or_insert(F::ZERO) += coefficient;
    }
    combined.retain(|_, coefficient| *coefficient != F::ZERO);
    combined.into_iter().collect()
}

fn remap_terms(terms: Vec<Term>, columns: &BTreeMap<usize, usize>) -> Vec<Term> {
    canonical_terms(
        terms
            .into_iter()
            .map(|(column, coefficient)| {
                (
                    *columns
                        .get(&column)
                        .unwrap_or_else(|| panic!("unmapped normalized arithmetic column {column}")),
                    coefficient,
                )
            })
            .collect(),
    )
}

fn carried_output(
    coefficient_columns: &[[usize; 2]; COEFFICIENT_COUNT],
    frame_columns: &[[usize; 3]; DEGREE],
    index: usize,
) -> [Vec<Term>; 2] {
    if index == DEGREE {
        return [
            singleton(coefficient_columns[index][0]),
            singleton(coefficient_columns[index][1]),
        ];
    }
    let frame = frame_columns[index];
    [
        vec![
            (coefficient_columns[index][0], F::ONE),
            (frame[0], F::ONE),
            (frame[1], F::from_u64(7)),
        ],
        vec![
            (coefficient_columns[index][1], F::ONE),
            (frame[2], F::ONE),
            (frame[0], -F::ONE),
            (frame[1], -F::ONE),
        ],
    ]
}

fn expected_arithmetic_row(
    row: usize,
    coefficient_columns: &[[usize; 2]; COEFFICIENT_COUNT],
    challenge_columns: [usize; 2],
    current_columns: [usize; 2],
    next_columns: [usize; 2],
    frame_columns: &[[usize; 3]; DEGREE],
) -> SourceRow {
    if row < 2 {
        let limb = row;
        let mut initial = vec![(coefficient_columns[0][limb], F::from_u64(2))];
        initial.extend(
            coefficient_columns[1..]
                .iter()
                .map(|coefficient| (coefficient[limb], F::ONE)),
        );
        return [singleton(current_columns[limb]), singleton(0), initial];
    }

    if row < 2 + 3 * DEGREE {
        let within = row - 2;
        let step = within / 3;
        let kind = within % 3;
        let suffix = carried_output(coefficient_columns, frame_columns, step + 1);
        let frame = frame_columns[step];
        return match kind {
            0 => [singleton(challenge_columns[0]), suffix[0].clone(), singleton(frame[0])],
            1 => [singleton(challenge_columns[1]), suffix[1].clone(), singleton(frame[1])],
            2 => {
                let mut challenge_sum = singleton(challenge_columns[0]);
                challenge_sum.extend(singleton(challenge_columns[1]));
                let mut suffix_sum = suffix[0].clone();
                suffix_sum.extend(suffix[1].clone());
                [challenge_sum, suffix_sum, singleton(frame[2])]
            }
            _ => unreachable!(),
        };
    }

    let limb = row - (2 + 3 * DEGREE);
    [
        carried_output(coefficient_columns, frame_columns, 0)[limb].clone(),
        singleton(0),
        singleton(next_columns[limb]),
    ]
}

fn exact_column_family(synthesis: &NebulaFPrimePiCcsRoundSynthesis, name: &'static str) -> std::ops::Range<usize> {
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

fn exact_row_family(synthesis: &NebulaFPrimePiCcsRoundSynthesis, name: &'static str) -> std::ops::Range<usize> {
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

fn assert_mutation_rejected(column: usize) {
    let mut synthesis = NebulaFPrimePiCcsRoundSynthesis::production();
    let changed = synthesis.witness_value(column).expect("mutation column") + F::ONE;
    synthesis.tamper_witness_for_test(column, changed);
    assert!(
        synthesis.first_unsatisfied_row().is_some(),
        "column {column} mutation must fail"
    );
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
            .expect("streaming PiCCS round source uses canonical CSC matrices");
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

#[test]
fn streaming_pi_ccs_round_owns_the_exact_compact_rows_and_phase_envelope() {
    let synthesis = NebulaFPrimePiCcsRoundSynthesis::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let shape = synthesis.shape_audit();
    assert!(synthesis.is_satisfied());
    assert_eq!(synthesis.first_unsatisfied_row(), None);
    assert!(synthesis.unconstrained_columns().is_empty());
    assert_eq!(shape.rows, PI_CCS_ROUND_SOURCE_ROWS);
    assert_eq!(shape.columns, PI_CCS_ROUND_SOURCE_COLUMNS);
    assert_eq!(shape.public_columns, PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS);
    assert_eq!(shape.poseidon2_permutations, PI_CCS_ROUND_SOURCE_POSEIDON2_PERMUTATIONS);
    assert_eq!(shape.public_columns, public.logical_columns());
    assert_eq!(shape.arithmetic_rows, ARITHMETIC_ROWS);
    assert!(shape.poseidon2_permutations > 0);
    assert_eq!(
        PI_CCS_ROUND_PROFILE_ID,
        "nightstream/goldilocks/streaming-pi-ccs-round/v1"
    );
    assert_eq!(PI_CCS_ROUND_SOURCE_ARTIFACT_ID, "rust:streaming-pi-ccs-round/source-v1");
    assert_eq!(PI_CCS_ROUND_LIFECYCLE_SCOPE, "recursive carry: PiCCS rounds 0..26");
    assert_eq!(PI_CCS_ROUND_FIRST_PROGRAM_CURSOR, FIRST_ROUND_PROGRAM_CURSOR as usize);
    assert_eq!(PI_CCS_ROUND_AFTER_LAST_PROGRAM_CURSOR, 196);
    assert_eq!(PI_CCS_ROUND_FINAL_COMMON_PUBLIC_COLUMNS, public.columns());
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.constant_one(), 0);
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.after_x_out_bits(), (1, 257));
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.before_x_out_bits(), (257, 513));
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.before_cursor_bits(), (513, 577));
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.after_cursor_bits(), (577, 641));
    assert_eq!(PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.common_public_padding(), (641, 648));
    assert_eq!(
        PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT.private_columns(),
        (PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS, PI_CCS_ROUND_SOURCE_COLUMNS)
    );

    assert_eq!(
        exact_column_family(&synthesis, STREAMING_PI_CCS_ROUND_BEFORE_STATE_FAMILY).len(),
        67
    );
    assert_eq!(
        exact_column_family(&synthesis, STREAMING_PI_CCS_ROUND_AFTER_STATE_FAMILY).len(),
        67
    );
    assert_eq!(
        exact_column_family(&synthesis, STREAMING_PI_CCS_ROUND_COEFFICIENT_FAMILY).len(),
        2 * COEFFICIENT_COUNT
    );
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_TRANSCRIPT_FAMILY).is_empty());
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY).len(),
        ARITHMETIC_ROWS
    );
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_STATE_TRANSITION_FAMILY).is_empty());
    assert!(!exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_STATE_DIGEST_FAMILY).is_empty());
    assert_eq!(
        exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_LIFECYCLE_CARRY_FAMILY).len(),
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

    let [round] = synthesis.builder_for_artifact().sumcheck_round_audits() else {
        panic!("expected one compact PiCCS round audit")
    };
    let arithmetic = exact_row_family(&synthesis, STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY);
    assert_eq!(round.row_start..round.row_end, arithmetic.clone());
    assert_eq!(round.coefficient_cols, synthesis.coefficient_columns());
    assert_eq!(round.challenge_cols, synthesis.challenge_columns());
    assert_eq!(round.claim_in_cols, synthesis.before_current_columns());
    assert_eq!(round.claim_out_cols, synthesis.after_current_columns());
    let frame_columns: [[usize; 3]; DEGREE] = round
        .allocated_cols
        .chunks_exact(3)
        .map(|columns| [columns[0], columns[1], columns[2]])
        .collect::<Vec<_>>()
        .try_into()
        .expect("nine compact Horner frames");
    let snapshot = synthesis.builder_for_artifact().snapshot();
    for local_row in 0..ARITHMETIC_ROWS {
        let expected = expected_arithmetic_row(
            local_row,
            &synthesis.coefficient_columns(),
            synthesis.challenge_columns(),
            synthesis.before_current_columns(),
            synthesis.after_current_columns(),
            &frame_columns,
        )
        .map(canonical_terms);
        let row = arithmetic.start + local_row;
        assert_eq!(snapshot.a_row(row), expected[0], "A row {local_row}");
        assert_eq!(snapshot.b_row(row), expected[1], "B row {local_row}");
        assert_eq!(snapshot.c_row(row), expected[2], "C row {local_row}");
    }

    assert_eq!(
        synthesis.witness_value(synthesis.before_round_cursor_column()),
        Some(F::from_u64(FIXTURE_ROUND))
    );
    assert_eq!(
        synthesis.witness_value(synthesis.after_round_cursor_column()),
        Some(F::from_u64(FIXTURE_ROUND + 1))
    );
    assert_eq!(
        synthesis.witness_value(synthesis.before_program_cursor_column()),
        Some(F::from_u64(FIRST_ROUND_PROGRAM_CURSOR + FIXTURE_ROUND))
    );
    assert_eq!(
        synthesis.witness_value(synthesis.after_program_cursor_column()),
        Some(F::from_u64(FIRST_ROUND_PROGRAM_CURSOR + FIXTURE_ROUND + 1))
    );
    for limb in 0..2 {
        assert_eq!(
            synthesis.witness_value(synthesis.challenge_columns()[limb]),
            synthesis.witness_value(synthesis.after_transcript_columns()[limb]),
        );
        assert_eq!(
            synthesis.witness_value(synthesis.after_reverse_point_columns()[0][limb]),
            synthesis.witness_value(synthesis.challenge_columns()[limb]),
        );
    }
    for index in 1..26 {
        for limb in 0..2 {
            assert_eq!(
                synthesis.witness_value(synthesis.after_reverse_point_columns()[index][limb]),
                synthesis.witness_value(synthesis.before_reverse_point_columns()[index - 1][limb]),
            );
        }
    }
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

    let source = production_pi_ccs_round_source_arm().expect("lower exact PiCCS round source");
    assert_eq!(source_rows_sha256(&source), PI_CCS_ROUND_SOURCE_SHA256);
    let stage_schedule = source
        .physical_stage_ranges()
        .iter()
        .map(|stage| (stage.path(), stage.rows(), stage.columns()))
        .collect::<Vec<_>>();
    let frozen_stage_schedule = PI_CCS_ROUND_SOURCE_STAGE_SCHEDULE
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
            ("nebula.streaming.pi_ccs.round.state_words", 0..0, 641..795),
            ("nebula.streaming.pi_ccs.round.transcript", 0..4_212, 795..4_999),
            ("nebula.streaming.pi_ccs.round.arithmetic", 4_212..4_243, 4_999..5_026),
            (
                "nebula.streaming.pi_ccs.round.state_transition",
                4_243..4_442,
                5_026..5_032
            ),
            (
                "nebula.streaming.pi_ccs.round.state_digest",
                4_442..27_272,
                5_032..27_862
            ),
            (
                "nebula.streaming.pi_ccs.round.phase_envelope",
                27_272..690_243,
                27_862..690_833
            ),
            (
                "nebula.streaming.pi_ccs.round.state_x_out",
                690_243..701_757,
                690_833..701_828
            ),
        ]
    );
    assert_eq!(PI_CCS_ROUND_ARITHMETIC_BINDING.source_rows(), (4_212, 4_243));
    assert_eq!(
        PI_CCS_ROUND_ARITHMETIC_BINDING.phase_local_selective_rows(),
        (0, ARITHMETIC_ROWS)
    );
    assert_eq!(
        PI_CCS_ROUND_ARITHMETIC_BINDING.artifact_identity(),
        PI_CCS_ROUND_COMPACT_ARITHMETIC_ARTIFACT_ID
    );

    let [normalized_round] = source.sumcheck_round_audits() else {
        panic!("normalized source must retain one PiCCS round audit")
    };
    assert_eq!(
        normalized_round.row_start..normalized_round.row_end,
        PI_CCS_ROUND_ARITHMETIC_BINDING.source_rows().0..PI_CCS_ROUND_ARITHMETIC_BINDING.source_rows().1
    );
    let mut normalized_to_phase_local = BTreeMap::from([(0, 0)]);
    for (source_column, phase_local_column) in normalized_round.claim_in_cols.into_iter().zip(1..3) {
        assert_eq!(
            normalized_to_phase_local.insert(source_column, phase_local_column),
            None
        );
    }
    for (coefficient, source_columns) in normalized_round.coefficient_cols.iter().enumerate() {
        for (limb, &source_column) in source_columns.iter().enumerate() {
            assert_eq!(
                normalized_to_phase_local.insert(source_column, 3 + 2 * coefficient + limb),
                None
            );
        }
    }
    for (source_column, phase_local_column) in normalized_round.challenge_cols.into_iter().zip(23..25) {
        assert_eq!(
            normalized_to_phase_local.insert(source_column, phase_local_column),
            None
        );
    }
    for (source_column, phase_local_column) in normalized_round.claim_out_cols.into_iter().zip(25..27) {
        assert_eq!(
            normalized_to_phase_local.insert(source_column, phase_local_column),
            None
        );
    }
    for (&source_column, phase_local_column) in normalized_round.allocated_cols.iter().zip(27..54) {
        assert_eq!(
            normalized_to_phase_local.insert(source_column, phase_local_column),
            None
        );
    }
    assert_eq!(normalized_to_phase_local.len(), 54);

    let local_coefficients = std::array::from_fn(|index| [3 + 2 * index, 4 + 2 * index]);
    let local_frames = std::array::from_fn(|index| [27 + 3 * index, 28 + 3 * index, 29 + 3 * index]);
    for local_row in 0..ARITHMETIC_ROWS {
        let source_row = PI_CCS_ROUND_ARITHMETIC_BINDING.source_rows().0 + local_row;
        let actual = [&source.a, &source.b, &source.c].map(|matrix| {
            remap_terms(
                matrix
                    .materialize_row(source_row)
                    .expect("normalized arithmetic row"),
                &normalized_to_phase_local,
            )
        });
        let expected = expected_arithmetic_row(
            local_row,
            &local_coefficients,
            [23, 24],
            [1, 2],
            [25, 26],
            &local_frames,
        )
        .map(canonical_terms);
        assert_eq!(actual, expected, "normalized-to-phase-local row {local_row}");
    }
    assert_eq!(source.m_in, public.logical_columns());
    assert_eq!(source.n, shape.rows);
    assert_eq!(source.m, shape.columns);
    for family in [
        STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
        STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
        STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY,
        STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    ] {
        let ranges = source
            .column_family_ranges()
            .iter()
            .filter(|range| range.name == family)
            .collect::<Vec<_>>();
        let [range] = ranges.as_slice() else {
            panic!("lowered source must retain {family}")
        };
        assert!(range.column_start >= source.m_in);
        assert!(range.column_end <= source.m);
    }
}

#[test]
fn streaming_pi_ccs_round_rejects_each_authoritative_transition_mutation() {
    let synthesis = NebulaFPrimePiCcsRoundSynthesis::production();
    let columns = [
        synthesis.coefficient_columns()[3][1],
        synthesis.before_transcript_columns()[0],
        synthesis.after_transcript_columns()[7],
        synthesis.before_current_columns()[0],
        synthesis.after_current_columns()[1],
        synthesis.before_reverse_point_columns()[3][0],
        synthesis.after_reverse_point_columns()[4][1],
        synthesis.before_round_cursor_column(),
        synthesis.after_round_cursor_column(),
        synthesis.before_context_digest_columns()[2],
        synthesis.after_context_digest_columns()[1],
        synthesis.after_boundary_columns()[2],
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
    for column in columns {
        assert_mutation_rejected(column);
    }
}
