//! Drift check for the verifier-owned streaming F-prime work schedule.

use std::fmt::Write as _;

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit};

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingProgram.lean";

fn render_artifact(program: &NebulaFPrimeStreamingProgramAudit) -> String {
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
    rendered.push_str("  schemaVersion := 2\n");
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
    assert_eq!(program.prior_state_frame_fields(), 83_874);
    assert_eq!(program.prior_state_chunks(), 82);
    assert_eq!(program.claim_frame_fields(), 88_023);
    assert_eq!(program.claim_chunk_fields(), 1_024);
    assert_eq!(program.claim_chunks(), 86);
    assert_eq!(program.pi_ccs_rounds(), 26);
    assert_eq!(program.pi_rlc_families(), 110);
    assert_eq!(program.successor_prefix_frame_fields(), 83_756);
    assert_eq!(program.successor_prefix_chunks(), 82);
    assert_eq!(program.work_items().len(), 400);
    assert_eq!(program.runs().len(), 19);

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
    assert_eq!(direct[82], (NebulaFPrimeStreamingPhase::PriorStateReplay, 81));
    assert_eq!(direct[83], (NebulaFPrimeStreamingPhase::ClaimReplay, 0));
    assert_eq!(direct[168], (NebulaFPrimeStreamingPhase::ClaimReplay, 85));
    assert_eq!(direct[169], (NebulaFPrimeStreamingPhase::PiCcsStart, 0));
    assert_eq!(direct[195], (NebulaFPrimeStreamingPhase::PiCcsRound, 25));
    assert_eq!(direct[199], (NebulaFPrimeStreamingPhase::PiRlcFamily, 0));
    assert_eq!(direct[308], (NebulaFPrimeStreamingPhase::PiRlcFamily, 109));
    assert_eq!(direct[314], (NebulaFPrimeStreamingPhase::SuccessorPrefixReplay, 0));
    assert_eq!(direct[395], (NebulaFPrimeStreamingPhase::SuccessorPrefixReplay, 81));
    assert_eq!(direct[399], (NebulaFPrimeStreamingPhase::SemanticLinks, 0));

    let rendered = render_artifact(&program);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write streaming-program artifact candidate");
        panic!("streaming-program artifact drifted; wrote {expected}. Inspect and promote it explicitly");
    }
}
