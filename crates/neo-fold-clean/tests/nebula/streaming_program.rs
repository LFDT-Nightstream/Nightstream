//! Drift check for the verifier-owned streaming F-prime work schedule.

use std::fmt::Write as _;

use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_combined_overlay_low_norm_r1cs, production_claim_coordinate_overlay_kind_count,
    production_claim_coordinate_overlay_kind_map, production_claim_coordinate_overlay_link_runs,
    production_combined_overlay_kind_count, production_combined_overlay_kind_map,
    production_pi_rlc_family_overlay_kind_map, production_pi_rlc_family_overlay_link_runs,
    NebulaFPrimeClaimCoordinateOverlayLinkRun, NebulaFPrimePiRlcFamilyOverlayLinkRun, NebulaFPrimeStreamingCircuitKind,
    NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit, NebulaFPrimeStreamingPublicLayout,
    PI_RLC_FAMILY_BODY_EVEN_COLUMNS, PI_RLC_FAMILY_BODY_EVEN_ROWS, PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS,
    PI_RLC_FAMILY_BODY_ODD_COLUMNS, PI_RLC_FAMILY_BODY_ODD_ROWS, PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS,
    PI_RLC_FAMILY_BODY_SOURCE_ROWS, PI_RLC_FAMILY_COUNT, PI_RLC_FAMILY_LINK_FIELDS, PI_RLC_FAMILY_OVERLAY_COLUMNS,
    PI_RLC_FAMILY_OVERLAY_ROWS,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, build_scheduled_grouped_phase_low_norm_r1cs,
    lower_field_r1cs, ScheduledCursorBits, SparseR1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingProgram.lean";

fn lean_nat_list(values: &[usize]) -> String {
    let lines = values
        .chunks(20)
        .map(|chunk| {
            chunk
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect::<Vec<_>>();
    format!("[\n    {}\n  ]", lines.join("\n  , "))
}

fn lean_overlay_link_runs(values: &[NebulaFPrimeClaimCoordinateOverlayLinkRun]) -> String {
    let lines = values
        .iter()
        .map(|run| {
            format!(
                "{{ overlayKind := {}, phaseKind := {}, chunkIndex := {}, activeOffsetStart := {}, activeFieldCount := {} }}",
                run.overlay_kind(),
                run.phase_kind(),
                run.chunk_index(),
                run.active_offset_start(),
                run.active_field_count(),
            )
        })
        .collect::<Vec<_>>();
    format!("[\n    {}\n  ]", lines.join("\n  , "))
}

fn lean_pi_rlc_overlay_link_runs(values: &[NebulaFPrimePiRlcFamilyOverlayLinkRun]) -> String {
    let lines = values
        .iter()
        .map(|run| {
            format!(
                "{{ phaseFieldStart := {}, overlayFieldStart := {}, outerCount := {}, phaseStride := {}, overlayStride := {}, fieldCount := {} }}",
                run.phase_field_start(),
                run.overlay_field_start(),
                run.outer_count(),
                run.phase_stride(),
                run.overlay_stride(),
                run.field_count(),
            )
        })
        .collect::<Vec<_>>();
    format!("[\n    {}\n  ]", lines.join("\n  , "))
}

fn render_artifact(program: &NebulaFPrimeStreamingProgramAudit) -> String {
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgramSchema\n\n\
/-! Generated file: exact verifier-owned work schedule for the bounded-width\n\
Nebula F-prime relation.\n\n\
Owns: the Rust phase codes, compact phase runs, and production stream counts.\n\n\
Does not own: phase-local constraints, relation dimensions, recursive proof\n\
integration, or security reduction.\n\n\
Emits constraints: no.\n-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact\n\n\
def profileId : String := \"nebula-fprime-streaming-program\"\n\
def rawProgram : RawProgram where\n",
    );
    rendered.push_str("  schemaVersion := 9\n");
    writeln!(rendered, "  stateChunkFields := {}", program.state_chunk_fields()).unwrap();
    writeln!(
        rendered,
        "  priorStateFrameFields := {}",
        program.prior_state_frame_fields()
    )
    .unwrap();
    writeln!(rendered, "  priorStateChunks := {}", program.prior_state_chunks()).unwrap();
    writeln!(rendered, "  claimFrameFields := {}", program.claim_frame_fields()).unwrap();
    writeln!(rendered, "  claimChunkFields := {}", program.claim_chunk_fields()).unwrap();
    writeln!(rendered, "  claimChunks := {}", program.claim_chunks()).unwrap();
    writeln!(rendered, "  piCcsRounds := {}", program.pi_ccs_rounds()).unwrap();
    writeln!(rendered, "  piRlcFamilies := {}", program.pi_rlc_families()).unwrap();
    writeln!(
        rendered,
        "  firstPiRlcFamilyProgramCursor := {}",
        program.first_pi_rlc_family_program_cursor()
    )
    .unwrap();
    writeln!(
        rendered,
        "  successorPrefixFrameFields := {}",
        program.successor_prefix_frame_fields(),
    )
    .unwrap();
    writeln!(
        rendered,
        "  successorPrefixChunks := {}",
        program.successor_prefix_chunks()
    )
    .unwrap();
    writeln!(rendered, "  workItemCount := {}", program.work_items().len()).unwrap();
    writeln!(rendered, "  lifecycleGroupCount := {}", program.lifecycle_group_count()).unwrap();
    writeln!(rendered, "  circuitKindCount := {}", program.circuit_kind_count()).unwrap();
    writeln!(
        rendered,
        "  claimCoordinateOverlayKindCount := {}",
        production_claim_coordinate_overlay_kind_count()
    )
    .unwrap();
    writeln!(
        rendered,
        "  combinedOverlayKindCount := {}",
        production_claim_coordinate_overlay_kind_count() + PI_RLC_FAMILY_COUNT
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyFirstOverlayKind := {}",
        production_claim_coordinate_overlay_kind_count()
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyEvenPhaseKind := {}",
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyEven.code()
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyOddPhaseKind := {}",
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyOdd.code()
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyBodySourceRows := {PI_RLC_FAMILY_BODY_SOURCE_ROWS}"
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyBodyEvenSourceRows := {PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS}"
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyBodyOddSourceRows := {PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS}"
    )
    .unwrap();
    writeln!(rendered, "  piRlcFamilyBodyEvenRows := {PI_RLC_FAMILY_BODY_EVEN_ROWS}").unwrap();
    writeln!(rendered, "  piRlcFamilyBodyOddRows := {PI_RLC_FAMILY_BODY_ODD_ROWS}").unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyBodyEvenColumns := {PI_RLC_FAMILY_BODY_EVEN_COLUMNS}"
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyBodyOddColumns := {PI_RLC_FAMILY_BODY_ODD_COLUMNS}"
    )
    .unwrap();
    writeln!(rendered, "  piRlcFamilyOverlayRows := {PI_RLC_FAMILY_OVERLAY_ROWS}").unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyOverlayColumns := {PI_RLC_FAMILY_OVERLAY_COLUMNS}"
    )
    .unwrap();
    writeln!(rendered, "  piRlcFamilyLinkFieldCount := {PI_RLC_FAMILY_LINK_FIELDS}").unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyTotalLinkFieldCount := {}",
        PI_RLC_FAMILY_COUNT * PI_RLC_FAMILY_LINK_FIELDS
    )
    .unwrap();
    writeln!(rendered, "  phasePublicLogicalColumns := {}", public.logical_columns()).unwrap();
    writeln!(rendered, "  phasePublicColumns := {}", public.columns()).unwrap();
    writeln!(
        rendered,
        "  afterStateDigestStart := {}",
        public.after_state_digest_bits().start
    )
    .unwrap();
    writeln!(
        rendered,
        "  afterStateDigestEnd := {}",
        public.after_state_digest_bits().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  beforeStateDigestStart := {}",
        public.before_state_digest_bits().start
    )
    .unwrap();
    writeln!(
        rendered,
        "  beforeStateDigestEnd := {}",
        public.before_state_digest_bits().end
    )
    .unwrap();
    writeln!(rendered, "  beforeCursorStart := {}", public.before_cursor_bits().start).unwrap();
    writeln!(rendered, "  beforeCursorEnd := {}", public.before_cursor_bits().end).unwrap();
    writeln!(rendered, "  afterCursorStart := {}", public.after_cursor_bits().start).unwrap();
    writeln!(rendered, "  afterCursorEnd := {}", public.after_cursor_bits().end).unwrap();
    writeln!(
        rendered,
        "  phasePublicPaddingStart := {}",
        public.padding_columns().start
    )
    .unwrap();
    writeln!(rendered, "  phasePublicPaddingEnd := {}", public.padding_columns().end).unwrap();
    writeln!(
        rendered,
        "  lifecycleGroupMap := {}",
        lean_nat_list(&program.lifecycle_group_map())
    )
    .unwrap();
    writeln!(
        rendered,
        "  circuitKindMap := {}",
        lean_nat_list(&program.circuit_kind_map())
    )
    .unwrap();
    writeln!(
        rendered,
        "  claimCoordinateOverlayKindMap := {}",
        lean_nat_list(&production_claim_coordinate_overlay_kind_map())
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyOverlayKindMap := {}",
        lean_nat_list(&production_pi_rlc_family_overlay_kind_map(
            0,
            production_claim_coordinate_overlay_kind_count(),
        ))
    )
    .unwrap();
    writeln!(
        rendered,
        "  claimCoordinateOverlayLinkRuns := {}",
        lean_overlay_link_runs(&production_claim_coordinate_overlay_link_runs())
    )
    .unwrap();
    writeln!(
        rendered,
        "  piRlcFamilyOverlayLinkRuns := {}",
        lean_pi_rlc_overlay_link_runs(&production_pi_rlc_family_overlay_link_runs())
    )
    .unwrap();
    rendered.push_str("  runs := [\n");
    for (index, run) in program.runs().iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ phaseCode := {}, firstIndex := {}, count := {} }}",
            run.phase().code(),
            run.first_index(),
            run.count(),
        )
        .unwrap();
    }
    rendered.push_str(
        "  ]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram\n",
    );
    rendered
}

#[test]
fn production_streaming_program_matches_lean_artifact() {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    assert_eq!(program.state_chunk_fields(), 1_024);
    assert_eq!(program.prior_state_frame_fields(), 95_754);
    assert_eq!(program.prior_state_chunks(), 94);
    assert_eq!(program.claim_frame_fields(), 99_903);
    assert_eq!(program.claim_chunk_fields(), 1_024);
    assert_eq!(program.claim_chunks(), 98);
    assert_eq!(program.first_claim_program_cursor(), 95);
    assert_eq!(program.pi_ccs_rounds(), 26);
    assert_eq!(program.pi_rlc_families(), 110);
    assert_eq!(program.first_pi_rlc_family_program_cursor(), 223);
    assert_eq!(program.successor_prefix_frame_fields(), 95_636);
    assert_eq!(program.successor_prefix_chunks(), 94);
    assert_eq!(program.work_items().len(), 436);
    assert_eq!(program.runs().len(), 19);
    assert_eq!(program.lifecycle_group_count(), 2);
    assert_eq!(program.circuit_kind_count(), 23);
    assert_eq!(production_claim_coordinate_overlay_kind_count(), 99);
    let public = NebulaFPrimeStreamingPublicLayout::production();
    assert_eq!(public.after_state_digest_bits(), 1..257);
    assert_eq!(public.before_state_digest_bits(), 257..513);
    assert_eq!(public.before_cursor_bits(), 513..577);
    assert_eq!(public.after_cursor_bits(), 577..641);
    assert_eq!(public.padding_columns(), 641..648);

    let expanded = program
        .runs()
        .into_iter()
        .flat_map(|run| (0..run.count()).map(move |offset| (run.phase(), run.first_index() + offset)))
        .collect::<Vec<_>>();
    let direct = program
        .work_items()
        .iter()
        .map(|item| (item.phase(), item.index()))
        .collect::<Vec<_>>();
    assert_eq!(expanded, direct, "run compression must preserve every work item");

    assert_eq!(direct[0], (NebulaFPrimeStreamingPhase::Prelude, 0));
    assert_eq!(direct[1], (NebulaFPrimeStreamingPhase::PriorStateReplay, 0));
    assert_eq!(direct[94], (NebulaFPrimeStreamingPhase::PriorStateReplay, 93));
    assert_eq!(direct[95], (NebulaFPrimeStreamingPhase::ClaimReplay, 0));
    assert_eq!(direct[192], (NebulaFPrimeStreamingPhase::ClaimReplay, 97));
    assert_eq!(direct[193], (NebulaFPrimeStreamingPhase::PiCcsStart, 0));
    assert_eq!(direct[219], (NebulaFPrimeStreamingPhase::PiCcsRound, 25));
    assert_eq!(direct[223], (NebulaFPrimeStreamingPhase::PiRlcFamily, 0));
    assert_eq!(direct[332], (NebulaFPrimeStreamingPhase::PiRlcFamily, 109));
    assert_eq!(direct[338], (NebulaFPrimeStreamingPhase::SuccessorPrefixReplay, 0));
    assert_eq!(direct[431], (NebulaFPrimeStreamingPhase::SuccessorPrefixReplay, 93));
    assert_eq!(direct[435], (NebulaFPrimeStreamingPhase::SemanticLinks, 0));

    let lifecycle_groups = program.lifecycle_group_map();
    assert_eq!(lifecycle_groups.len(), 436);
    assert_eq!(&lifecycle_groups[..4], &[0, 1, 1, 1]);
    assert!(lifecycle_groups[1..].iter().all(|&group| group == 1));

    let circuit_kinds = program.circuit_kind_map();
    assert_eq!(circuit_kinds.len(), 436);
    assert_eq!(
        circuit_kinds[0],
        NebulaFPrimeStreamingCircuitKind::Prelude.code() as usize
    );
    assert_eq!(
        circuit_kinds[1],
        NebulaFPrimeStreamingCircuitKind::PriorStateReplayFull.code() as usize
    );
    assert_eq!(
        circuit_kinds[94],
        NebulaFPrimeStreamingCircuitKind::PriorStateReplayFinal.code() as usize
    );
    assert_eq!(
        circuit_kinds[95],
        NebulaFPrimeStreamingCircuitKind::ClaimReplayFull.code() as usize
    );
    assert_eq!(
        circuit_kinds[192],
        NebulaFPrimeStreamingCircuitKind::ClaimReplayFinal.code() as usize
    );
    assert_eq!(
        circuit_kinds[223],
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyEven.code() as usize
    );
    assert_eq!(
        circuit_kinds[224],
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyOdd.code() as usize
    );
    assert_eq!(
        circuit_kinds[332],
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyOdd.code() as usize
    );
    assert_eq!(
        circuit_kinds[338],
        NebulaFPrimeStreamingCircuitKind::SuccessorPrefixReplayFull.code() as usize
    );
    assert_eq!(
        circuit_kinds[431],
        NebulaFPrimeStreamingCircuitKind::SuccessorPrefixReplayFinal.code() as usize
    );
    assert_eq!(
        circuit_kinds[435],
        NebulaFPrimeStreamingCircuitKind::SemanticLinks.code() as usize
    );
    let mut seen = circuit_kinds.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen, (0..program.circuit_kind_count()).collect::<Vec<_>>());

    let overlay_kinds = production_claim_coordinate_overlay_kind_map();
    assert_eq!(overlay_kinds.len(), 436);
    assert!(overlay_kinds[..95].iter().all(|&kind| kind == 0));
    assert_eq!(overlay_kinds[95], 1);
    assert_eq!(overlay_kinds[96], 2);
    assert_eq!(overlay_kinds[155], 61);
    assert_eq!(overlay_kinds[176], 82);
    assert_eq!(overlay_kinds[177], 83);
    assert_eq!(overlay_kinds[192], 98);
    assert!(overlay_kinds[193..].iter().all(|&kind| kind == 0));

    let link_runs = production_claim_coordinate_overlay_link_runs();
    assert_eq!(link_runs.len(), 98);

    let pi_rlc_overlay_kinds =
        production_pi_rlc_family_overlay_kind_map(0, production_claim_coordinate_overlay_kind_count());
    assert_eq!(pi_rlc_overlay_kinds.len(), 436);
    assert!(pi_rlc_overlay_kinds[..223].iter().all(|&kind| kind == 0));
    assert_eq!(pi_rlc_overlay_kinds[223], 99);
    assert_eq!(pi_rlc_overlay_kinds[332], 208);
    assert!(pi_rlc_overlay_kinds[333..].iter().all(|&kind| kind == 0));

    let pi_rlc_link_runs = production_pi_rlc_family_overlay_link_runs();
    assert_eq!(
        pi_rlc_link_runs
            .iter()
            .map(|run| run.link_count())
            .sum::<usize>(),
        PI_RLC_FAMILY_LINK_FIELDS,
    );

    let combined_overlay_kinds = production_combined_overlay_kind_map();
    assert_eq!(production_combined_overlay_kind_count(), 209);
    assert_eq!(combined_overlay_kinds.len(), 436);
    assert_eq!(&combined_overlay_kinds[95..193], &overlay_kinds[95..193]);
    assert_eq!(&combined_overlay_kinds[223..333], &pi_rlc_overlay_kinds[223..333]);
    assert!(combined_overlay_kinds[193..223]
        .iter()
        .all(|&kind| kind == 0));
    assert!(combined_overlay_kinds[333..].iter().all(|&kind| kind == 0));

    let rendered = render_artifact(&program);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write streaming-program artifact candidate");
        panic!("streaming-program artifact drifted; wrote {expected}. Inspect and promote it explicitly");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_production_streaming_program_artifact() {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    std::fs::write(path, render_artifact(&program)).expect("write generated streaming-program artifact");
}

fn cursor_arm(arm: usize) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let mut public = Vec::with_capacity(640);
    for _ in 0..512 {
        let var = builder.alloc(F::ZERO);
        enforce_bit(&mut builder, var);
        public.push(var);
    }
    for value in [arm, arm + 1] {
        for bit in 0..64 {
            let var = builder.alloc(F::from_usize((value >> bit) & 1));
            enforce_bit(&mut builder, var);
            public.push(var);
        }
    }
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &public)
        .expect("lower production cursor fixture")
        .into_parts()
}

#[test]
fn production_schedule_composer_uses_two_lifecycle_and_twenty_three_phase_circuits() {
    let (shape, _) = cursor_arm(0);
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile two lifecycle circuits");
    let phase_arms = vec![shape; 23];
    let phase_kinds = build_multi_branch_selective_low_norm_r1cs_with_alignment(&phase_arms, 0, D, 0)
        .expect("compile 23 phase circuits");
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let relation = build_scheduled_grouped_phase_low_norm_r1cs(
        common,
        phase_kinds,
        program.lifecycle_group_map(),
        program.circuit_kind_map(),
        ScheduledCursorBits::new(public.before_cursor_bits(), public.after_cursor_bits()),
    )
    .expect("compose the exact production schedule maps");
    let layout = relation.layout();

    assert_eq!(layout.lifecycle_groups(), program.lifecycle_group_map());
    assert_eq!(layout.phase_kinds(), program.circuit_kind_map());
    assert_eq!(layout.common_selector_columns().len(), 2);
    assert_eq!(layout.phase_kind_selector_columns().len(), 23);
    assert_eq!(layout.schedule_selector_columns().len(), 436);
    assert_eq!(layout.cursor_binding_rows().len(), 872);
    assert_eq!(relation.public_input_len(), 648);
    assert_eq!(layout.cursor_bits().before(), 513..577);
    assert_eq!(layout.cursor_bits().after(), 577..641);

    for arm in [0, 1, 94, 95, 192, 193, 219, 223, 332, 338, 431, 435] {
        let (_, assignment) = cursor_arm(arm);
        let joint = relation
            .encode(arm, &assignment, &assignment)
            .expect("encode production schedule arm");
        assert!(relation.is_satisfied(&joint), "production arm {arm}");
    }
}

#[test]
#[ignore = "expensive production combined-overlay shape snapshot"]
fn production_combined_overlay_low_norm_shape_snapshot() {
    let relation = build_production_combined_overlay_low_norm_r1cs().expect("build 197 production overlay kinds");
    eprintln!(
        "combined overlay low-norm rows={}, columns={}, public={}, selectors={}",
        relation.structure().n,
        relation.structure().m,
        relation.public_input_len(),
        relation.selector_cols().len(),
    );
    assert_eq!(relation.selector_cols().len(), production_combined_overlay_kind_count());
    assert_eq!(relation.public_input_len(), 1);
    assert_eq!(relation.structure().n, 3_149_869);
    assert_eq!(relation.structure().m, 67_662);
    assert!(relation.structure().n < 1 << 24);
    assert!(relation.structure().m < 1 << 24);
}
