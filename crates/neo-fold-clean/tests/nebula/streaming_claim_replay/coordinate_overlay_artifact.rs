//! Golden Lean artifact for the 99-kind claim-coordinate overlay.

use super::*;

use neo_fold_clean::frontends::nebula::f_prime::{
    production_claim_coordinate_overlay_kind_count, production_claim_coordinate_overlay_link_runs,
};

const OVERLAY_SCHEMA_VERSION: usize = 1;
const OVERLAY_PROFILE_ID: &str = "nebula-f-prime-streaming-claim-coordinate-overlay-goldilocks-b2-k16-v1";
const STATE_LINKS_PER_OUTPUT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StateBases {
    before_statement_fresh: usize,
    after_statement_fresh: usize,
    before_running_commitments: usize,
    after_running_commitments: usize,
    before_running_public: usize,
    after_running_public: usize,
}

#[derive(Clone, Debug)]
struct ActiveArmArtifact {
    overlay_kind: usize,
    phase_kind: usize,
    chunk_index: usize,
    row_count: usize,
    column_count: usize,
    phase_state: StateBases,
    phase_chunk_base: usize,
    coordinate_calls: Vec<CoordinateCall>,
    link_call_indices: Vec<usize>,
}

fn map_index(kind: CoordinateMapKind) -> usize {
    match kind {
        CoordinateMapKind::StatementFresh => 0,
        CoordinateMapKind::RunningCommitments => 1,
        CoordinateMapKind::RunningPublic => 2,
    }
}

fn map_width(kind: CoordinateMapKind) -> usize {
    match kind {
        CoordinateMapKind::StatementFresh => PI_CCS_STATEMENT_FRESH_FIELDS,
        CoordinateMapKind::RunningCommitments => PI_CCS_RUNNING_COMMITMENT_FIELDS,
        CoordinateMapKind::RunningPublic => PI_CCS_RUNNING_PUBLIC_FIELDS,
    }
}

fn map_name(kind: CoordinateMapKind) -> &'static str {
    match kind {
        CoordinateMapKind::StatementFresh => ".statementFresh",
        CoordinateMapKind::RunningCommitments => ".runningCommitments",
        CoordinateMapKind::RunningPublic => ".runningPublic",
    }
}

fn schedule_name(kind: CoordinateMapKind) -> &'static str {
    match kind {
        CoordinateMapKind::StatementFresh => "statementFreshSchedule",
        CoordinateMapKind::RunningCommitments => "runningCommitmentsSchedule",
        CoordinateMapKind::RunningPublic => "runningPublicSchedule",
    }
}

fn map_positions<'a>(
    kind: CoordinateMapKind,
    chunk: usize,
    maps: &'a [Vec<Vec<(usize, usize)>>; 3],
) -> &'a [(usize, usize)] {
    &maps[map_index(kind)][chunk]
}

fn consecutive_base(columns: impl IntoIterator<Item = usize>, label: &str) -> usize {
    let columns = columns.into_iter().collect::<Vec<_>>();
    let base = *columns.first().expect("nonempty coordinate state range");
    assert_eq!(
        columns,
        (base..base + COORDINATE_OUTPUTS).collect::<Vec<_>>(),
        "{label} columns are consecutive",
    );
    base
}

fn overlay_state_bases(synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis) -> StateBases {
    StateBases {
        before_statement_fresh: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .before_statement_fresh_column(coordinate)
                    .expect("before statement/fresh")
            }),
            "before statement/fresh",
        ),
        after_statement_fresh: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .after_statement_fresh_column(coordinate)
                    .expect("after statement/fresh")
            }),
            "after statement/fresh",
        ),
        before_running_commitments: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .before_running_commitments_column(coordinate)
                    .expect("before running commitments")
            }),
            "before running commitments",
        ),
        after_running_commitments: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .after_running_commitments_column(coordinate)
                    .expect("after running commitments")
            }),
            "after running commitments",
        ),
        before_running_public: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .before_running_public_column(coordinate)
                    .expect("before running public")
            }),
            "before running public",
        ),
        after_running_public: consecutive_base(
            (0..COORDINATE_OUTPUTS).map(|coordinate| {
                synthesis
                    .after_running_public_column(coordinate)
                    .expect("after running public")
            }),
            "after running public",
        ),
    }
}

fn linear_row(output: usize, terms: &[(usize, F)]) -> SparseRow {
    SparseRow {
        a: normalize_terms(
            std::iter::once((output, F::ONE)).chain(
                terms
                    .iter()
                    .map(|&(column, coefficient)| (column, -coefficient)),
            ),
        ),
        b: vec![(0, F::ONE)],
        c: Vec::new(),
    }
}

fn assert_linear_row(rows: &[SparseRow], row: usize, output: usize, terms: &[(usize, F)], label: &str) {
    assert_eq!(rows[row], linear_row(output, terms), "{label} row {row}");
}

fn before_column(
    synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis,
    kind: CoordinateMapKind,
    output: usize,
) -> usize {
    match kind {
        CoordinateMapKind::StatementFresh => synthesis.before_statement_fresh_column(output),
        CoordinateMapKind::RunningCommitments => synthesis.before_running_commitments_column(output),
        CoordinateMapKind::RunningPublic => synthesis.before_running_public_column(output),
    }
    .expect("before state column")
}

fn after_column(
    synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis,
    kind: CoordinateMapKind,
    output: usize,
) -> usize {
    match kind {
        CoordinateMapKind::StatementFresh => synthesis.after_statement_fresh_column(output),
        CoordinateMapKind::RunningCommitments => synthesis.after_running_commitments_column(output),
        CoordinateMapKind::RunningPublic => synthesis.after_running_public_column(output),
    }
    .expect("after state column")
}

fn chunk_column(synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis, offset: usize) -> usize {
    synthesis
        .chunk_columns()
        .iter()
        .find_map(|&(candidate, column)| (candidate == offset).then_some(column))
        .expect("active chunk offset has one overlay column")
}

fn build_coordinate_call(
    synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis,
    rows: &[SparseRow],
    kind: CoordinateMapKind,
    chunk_index: usize,
    positions: &[(usize, usize)],
    block: &neo_ccs::SeededPhi81LinearBlock,
    opening_traces: &[neo_fold_clean::engine::r1cs_circuit::BalancedTernaryOpeningTraceEntry],
    opening_trace_cursor: &mut usize,
    row_cursor: usize,
) -> CoordinateCall {
    assert!(!positions.is_empty());
    assert_eq!(block.word_starts().len(), map_width(kind));
    assert_eq!(block.word_width(), COORDINATE_DIGITS);
    assert_eq!(block.kappa(), 2);
    assert_eq!(block.message_cols(), (map_width(kind) * COORDINATE_DIGITS).div_ceil(D));
    assert!(!block.has_superneo_transformed_columns());

    let first_column = chunk_column(synthesis, positions[0].1);
    let chunk_base = first_column
        .checked_sub(positions[0].1)
        .expect("virtual chunk base precedes the first active offset");
    let active_digit_base = block.word_starts()[positions[0].0];
    let active_fields = positions
        .iter()
        .map(|&(field, _)| field)
        .collect::<std::collections::BTreeSet<_>>();
    let zero_digit_start = (0..map_width(kind))
        .find(|field| !active_fields.contains(field))
        .map(|field| block.word_starts()[field])
        .expect("one inactive field owns the shared zero word");
    assert_eq!(active_digit_base, zero_digit_start + COORDINATE_DIGITS);

    for digit in 0..COORDINATE_DIGITS {
        assert_linear_row(
            rows,
            row_cursor + digit,
            zero_digit_start + digit,
            &[],
            "shared zero digit",
        );
    }

    for (rank, &(field, offset)) in positions.iter().enumerate() {
        assert_eq!(chunk_column(synthesis, offset), chunk_base + offset);
        let digit_start = active_digit_base + rank * COORDINATE_OPENING_COLUMNS;
        assert_eq!(block.word_starts()[field], digit_start);
        let trace = &opening_traces[*opening_trace_cursor];
        *opening_trace_cursor += 1;
        let opening_row = row_cursor + COORDINATE_DIGITS + rank * COORDINATE_OPENING_ROWS;
        assert_eq!(trace.field_col, chunk_base + offset);
        assert_eq!(trace.digit_cols, std::array::from_fn(|index| digit_start + index));
        assert_eq!(
            trace.negative_cols,
            std::array::from_fn(|index| digit_start + COORDINATE_DIGITS + index),
        );
        assert_eq!(
            trace.borrow_cols,
            std::array::from_fn(|index| digit_start + 2 * COORDINATE_DIGITS + index),
        );
        assert_eq!(trace.digit_rows, opening_row..opening_row + 2 * COORDINATE_DIGITS);
        assert_eq!(trace.reconstruction_row, opening_row + 2 * COORDINATE_DIGITS);
        assert_eq!(
            trace.transition_rows,
            opening_row + 2 * COORDINATE_DIGITS + 1..opening_row + COORDINATE_OPENING_ROWS,
        );
    }
    for field in 0..map_width(kind) {
        if !active_fields.contains(&field) {
            assert_eq!(block.word_starts()[field], zero_digit_start);
        }
    }

    let seeded_row_start = row_cursor + COORDINATE_DIGITS + positions.len() * COORDINATE_OPENING_ROWS + 2;
    assert_eq!(block.row_start(), seeded_row_start);
    let output_columns = (0..COORDINATE_OUTPUTS)
        .map(|output| {
            let row = &rows[block.row_start() + output];
            assert!(row.a.is_empty(), "seeded A terms stay in the compact block");
            assert_eq!(row.b, vec![(0, F::ONE)]);
            assert_eq!(row.c.len(), 1);
            assert_eq!(row.c[0].1, F::ONE);
            row.c[0].0
        })
        .collect::<Vec<_>>();
    let output_base = consecutive_base(output_columns, "coordinate outputs");
    let d_column = output_base
        .checked_sub(2)
        .expect("shape columns precede outputs");
    let kappa_column = output_base - 1;
    assert_linear_row(
        rows,
        seeded_row_start - 2,
        d_column,
        &[(0, F::from_u64(D as u64))],
        "dimension pin",
    );
    assert_linear_row(
        rows,
        seeded_row_start - 1,
        kappa_column,
        &[(0, F::from_u64(2))],
        "rank pin",
    );

    CoordinateCall {
        map_kind: kind,
        rows: row_cursor..block.row_end(),
        chunk_index,
        chunk_base,
        zero_digit_start,
        active_digit_base,
        d_column,
        kappa_column,
        output_base,
        seeded_row_start,
        chunk_size: block.chunk_size(),
        seeds_by_output: block.chunk_seeds_by_row().to_vec(),
    }
}

fn phase_and_chunk_links(
    contract: &neo_fold_clean::frontends::r1cs_f_prime::OverlayKindLinks,
    synthesis: &NebulaFPrimeClaimCoordinateOverlaySynthesis,
    calls: &[CoordinateCall],
    maps: &[Vec<Vec<(usize, usize)>>; 3],
    chunk_index: usize,
) -> (StateBases, usize, Vec<usize>) {
    let state_count = STATE_LINKS_PER_OUTPUT * COORDINATE_OUTPUTS;
    assert!(contract.fields.len() >= state_count);
    let state_base = |role: usize, phase: bool| {
        let first = &contract.fields[role];
        let base = if phase { first.phase_field } else { first.overlay_field };
        for output in 0..COORDINATE_OUTPUTS {
            let link = &contract.fields[output * STATE_LINKS_PER_OUTPUT + role];
            assert_eq!(if phase { link.phase_field } else { link.overlay_field }, base + output);
        }
        base
    };
    let phase_state = StateBases {
        before_statement_fresh: state_base(0, true),
        after_statement_fresh: state_base(1, true),
        before_running_commitments: state_base(2, true),
        after_running_commitments: state_base(3, true),
        before_running_public: state_base(4, true),
        after_running_public: state_base(5, true),
    };
    let linked_overlay_state = StateBases {
        before_statement_fresh: state_base(0, false),
        after_statement_fresh: state_base(1, false),
        before_running_commitments: state_base(2, false),
        after_running_commitments: state_base(3, false),
        before_running_public: state_base(4, false),
        after_running_public: state_base(5, false),
    };
    assert_eq!(linked_overlay_state, overlay_state_bases(synthesis));

    let mut segments = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let positions = map_positions(call.map_kind, chunk_index, maps);
            (positions[0].1, index)
        })
        .collect::<Vec<_>>();
    segments.sort_unstable();
    let link_call_indices = segments.iter().map(|&(_, index)| index).collect::<Vec<_>>();
    let reconstructed = link_call_indices
        .iter()
        .flat_map(|&index| {
            let call = &calls[index];
            map_positions(call.map_kind, chunk_index, maps)
                .iter()
                .map(move |&(_, offset)| (offset, call.chunk_base + offset))
        })
        .collect::<Vec<_>>();
    assert_eq!(reconstructed, synthesis.chunk_columns());
    let chunk_links = &contract.fields[state_count..];
    assert_eq!(chunk_links.len(), reconstructed.len());
    let phase_chunk_base = chunk_links[0]
        .phase_field
        .checked_sub(reconstructed[0].0)
        .expect("phase chunk base precedes active offset");
    for (link, &(offset, overlay_column)) in chunk_links.iter().zip(&reconstructed) {
        assert_eq!(link.phase_field, phase_chunk_base + offset);
        assert_eq!(link.overlay_field, overlay_column);
    }
    (phase_state, phase_chunk_base, link_call_indices)
}

fn build_active_arm(
    overlay_kind: usize,
    maps: &[Vec<Vec<(usize, usize)>>; 3],
    contracts: &[neo_fold_clean::frontends::r1cs_f_prime::OverlayKindLinks],
    schedules: &mut [Option<(usize, Vec<Vec<[u8; 32]>>)>; 3],
) -> ActiveArmArtifact {
    let chunk_index = overlay_kind - 1;
    let synthesis = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(overlay_kind)
        .expect("production active overlay kind");
    let builder = synthesis.builder_for_artifact();
    let rows = normalized_rows(builder);
    let blocks = builder.seeded_phi81_a_blocks();
    let openings = builder.encoding_trace().balanced_ternary_openings();
    let mut block_cursor = 0;
    let mut opening_cursor = 0;
    let mut row_cursor = 0;
    let mut calls = Vec::new();

    for kind in [
        CoordinateMapKind::StatementFresh,
        CoordinateMapKind::RunningCommitments,
        CoordinateMapKind::RunningPublic,
    ] {
        let positions = map_positions(kind, chunk_index, maps);
        if positions.is_empty() {
            for output in 0..COORDINATE_OUTPUTS {
                assert_linear_row(
                    &rows,
                    row_cursor + output,
                    after_column(&synthesis, kind, output),
                    &[(before_column(&synthesis, kind, output), F::ONE)],
                    "inactive carry",
                );
            }
            row_cursor += COORDINATE_OUTPUTS;
            continue;
        }

        let block = &blocks[block_cursor];
        block_cursor += 1;
        let call = build_coordinate_call(
            &synthesis,
            &rows,
            kind,
            chunk_index,
            positions,
            block,
            openings,
            &mut opening_cursor,
            row_cursor,
        );
        let schedule = (call.chunk_size, call.seeds_by_output.clone());
        match &schedules[map_index(kind)] {
            Some(expected) => assert_eq!(&schedule, expected, "one exact schedule per map"),
            None => schedules[map_index(kind)] = Some(schedule),
        }
        row_cursor = call.rows.end;
        for output in 0..COORDINATE_OUTPUTS {
            assert_linear_row(
                &rows,
                row_cursor + output,
                after_column(&synthesis, kind, output),
                &[
                    (before_column(&synthesis, kind, output), F::ONE),
                    (call.output_base + output, F::ONE),
                ],
                "active update",
            );
        }
        row_cursor += COORDINATE_OUTPUTS;
        calls.push(call);
    }

    assert_eq!(block_cursor, blocks.len(), "all compact blocks are owned");
    assert_eq!(opening_cursor, openings.len(), "all opening traces are owned");
    if chunk_index == 0 {
        for kind in [
            CoordinateMapKind::StatementFresh,
            CoordinateMapKind::RunningCommitments,
            CoordinateMapKind::RunningPublic,
        ] {
            for output in 0..COORDINATE_OUTPUTS {
                assert_linear_row(
                    &rows,
                    row_cursor,
                    before_column(&synthesis, kind, output),
                    &[],
                    "chunk-zero initial pin",
                );
                row_cursor += 1;
            }
        }
    }
    assert_eq!(
        row_cursor,
        builder.rows(),
        "every active source row has one structural owner"
    );

    let contract = &contracts[chunk_index];
    assert_eq!(contract.overlay_kind, overlay_kind);
    let (phase_state, phase_chunk_base, link_call_indices) =
        phase_and_chunk_links(contract, &synthesis, &calls, maps, chunk_index);
    ActiveArmArtifact {
        overlay_kind,
        phase_kind: contract.phase_kind,
        chunk_index,
        row_count: builder.rows(),
        column_count: builder.cols(),
        phase_state,
        phase_chunk_base,
        coordinate_calls: calls,
        link_call_indices,
    }
}

fn lean_state_bases(layout: StateBases) -> String {
    format!(
        "{{ beforeStatementFresh := {}, afterStatementFresh := {}, beforeRunningCommitments := {}, \
         afterRunningCommitments := {}, beforeRunningPublic := {}, afterRunningPublic := {} }}",
        layout.before_statement_fresh,
        layout.after_statement_fresh,
        layout.before_running_commitments,
        layout.after_running_commitments,
        layout.before_running_public,
        layout.after_running_public,
    )
}

fn lean_coordinate_call(call: &CoordinateCall) -> String {
    let schedule = schedule_name(call.map_kind);
    format!(
        "{{ mapKind := {}, rowStart := {}, rowEnd := {}, chunkIndex := {}, chunkBase := {}, \
         zeroDigitStart := {}, activeDigitBase := {}, dColumn := {}, kappaColumn := {}, outputBase := {}, \
         seededRowStart := {}, chunkSize := {schedule}.chunkSize, seedsByOutput := {schedule}.seedsByOutput }}",
        map_name(call.map_kind),
        call.rows.start,
        call.rows.end,
        call.chunk_index,
        call.chunk_base,
        call.zero_digit_start,
        call.active_digit_base,
        call.d_column,
        call.kappa_column,
        call.output_base,
        call.seeded_row_start,
    )
}

fn lean_active_arm(arm: &ActiveArmArtifact) -> String {
    format!(
        "{{ overlayKind := {}, phaseKind := {}, chunkIndex := {}, rowCount := {}, columnCount := {}, \
         phaseState := {}, phaseChunkBase := {}, linkCallIndices := {}, coordinateCalls := {} }}",
        arm.overlay_kind,
        arm.phase_kind,
        arm.chunk_index,
        arm.row_count,
        arm.column_count,
        lean_state_bases(arm.phase_state),
        arm.phase_chunk_base,
        lean_nat_list(arm.link_call_indices.iter().copied()),
        grouped_list(
            arm.coordinate_calls
                .iter()
                .map(lean_coordinate_call)
                .collect(),
            1
        ),
    )
}

fn render_overlay_artifact() -> String {
    let kind_count = production_claim_coordinate_overlay_kind_count();
    assert_eq!(kind_count, 99);
    let maps = [
        production_claim_statement_fresh_field_map(),
        production_claim_running_commitment_field_map(),
        production_claim_running_public_field_map(),
    ];
    assert!(maps.iter().all(|map| map.len() + 1 == kind_count));
    let contracts = production_claim_coordinate_overlay_links();
    assert_eq!(contracts.len() + 1, kind_count);
    let link_runs = production_claim_coordinate_overlay_link_runs();
    assert_eq!(link_runs.len(), contracts.len());

    let noop = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(0).expect("no-op overlay kind");
    let noop_rows = normalized_rows(noop.builder_for_artifact());
    assert_eq!(
        noop_rows,
        vec![SparseRow {
            a: vec![(0, F::ONE)],
            b: vec![(0, F::ONE)],
            c: vec![(0, F::ONE)],
        }]
    );

    let mut schedules: [Option<(usize, Vec<Vec<[u8; 32]>>)>; 3] = [None, None, None];
    let arms = (1..kind_count)
        .map(|kind| build_active_arm(kind, &maps, &contracts, &mut schedules))
        .collect::<Vec<_>>();
    assert!(schedules.iter().all(Option::is_some));
    let overlay_state = overlay_state_bases(
        &NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(1).expect("first active overlay"),
    );
    for kind in 1..kind_count {
        let synthesis = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(kind).expect("active overlay");
        assert_eq!(overlay_state_bases(&synthesis), overlay_state);
    }
    for (arm, run) in arms.iter().zip(&link_runs) {
        assert_eq!(arm.overlay_kind, run.overlay_kind());
        assert_eq!(arm.phase_kind, run.phase_kind());
        assert_eq!(arm.chunk_index, run.chunk_index());
        let active = maps
            .iter()
            .flat_map(|map| map[arm.chunk_index].iter().map(|&(_, offset)| offset))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(active.len(), run.active_field_count());
        assert_eq!(active.first().copied().unwrap_or(0), run.active_offset_start());
    }

    let arms = grouped_list(arms.iter().map(lean_active_arm).collect(), 1);
    let mut payload = String::new();
    writeln!(
        payload,
        "def overlayState : StateBases :=\n  {}",
        lean_state_bases(overlay_state)
    )
    .unwrap();
    writeln!(payload, "\ndef activeArms : List RawActiveArm :=\n  {arms}").unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {OVERLAY_SCHEMA_VERSION}, profileId := \"{OVERLAY_PROFILE_ID}\",\n    \
            noopRowCount := {}, noopColumnCount := {}, overlayState := overlayState, activeArms := activeArms }}",
        noop.rows(),
        noop.columns(),
    )
    .unwrap();
    let hash = sha256_hex(&payload);
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlaySchema\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay\n\n\
         /-! Generated file: exact compact Rust source artifact for all 99\n\
         claim-coordinate overlay kinds. The SHA-256 value is review metadata,\n\
         not proof or protocol authority. Do not hand-edit. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimCoordinateOverlay\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact\n\
         open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimCoordinateOverlay\n"
    );
    assert!(rendered.lines().count() < 1_500, "generated overlay artifact line cap");
    rendered
}

fn generated_overlay_artifact_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimCoordinateOverlay.lean",
    )
}

#[test]
fn production_claim_coordinate_overlay_lean_artifact_is_current() {
    let path = generated_overlay_artifact_path();
    let rendered = render_overlay_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "claim-coordinate overlay Lean artifact drifted; inspect {}",
            path.display()
        );
    }
}
