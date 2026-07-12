use super::super::full_history_terminal_accumulator_artifact_support::{
    compare_accumulator_core_artifacts, AccumulatorCorePaths,
};
use super::super::*;

const CORE_ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursiveAccumulatorCoreArtifact.lean";
const CORE_SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveAccumulatorCoreSegment";
const CORE_HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.lean";
const CORE_CHECK_COVERAGE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.lean";
const CORE_SCHEDULE_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveAccumulatorCoreSchedule";
const CORE_SCHEDULES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursiveAccumulatorCoreSchedules.lean";
const RUNNING_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveAccumulatorRunningLinkArtifact.lean";
const OUTPUT_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveAccumulatorOutputLinkArtifact.lean";
const OWNER_ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursiveAccumulatorArtifact.lean";

fn owner<'a>(builder: &'a R1csBuilder, name: &str) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one {name} owner");
    matches[0]
}

fn compare(path: &Path, rendered: String, drifted: &mut Vec<PathBuf>) {
    if fs::read_to_string(path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        fs::write(&expected, rendered).expect("write recursive accumulator artifact");
        drifted.push(expected);
    }
}

fn render_owner_artifact(
    owner: &RowFamilyRange,
    core: &RowFamilyRange,
    state_output_accumulator: &[usize],
    owner_sha256: &str,
) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorRunningLinkArtifact\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorOutputLinkArtifact\n\n\
         /-! Exact aggregate for the recursive accumulator owner. Hashes are drift metadata only. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def rangeSha256 : String := \"{owner_sha256}\"\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         def runningDigestRows : List Row :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorRunningLink.rows\n\n\
         def coreRows : List Row :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorCore.rows\n\n\
         def outputDigestRows : List Row :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorOutputLink.rows\n\n\
         def rowPieces : List (List Row) :=\n\
         \x20 [runningDigestRows, coreRows, outputDigestRows]\n\n\
         def rows : List Row := rowPieces.flatten\n\n\
         theorem rows_length : rows.length = rowCount := by\n\
         \x20 simp [rows, rowPieces, runningDigestRows, coreRows, outputDigestRows,\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorRunningLink.rows_length,\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorCore.rows_length,\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorOutputLink.rows_length, rowCount]\n\
         \x20 native_decide\n\n\
         def runningAccumulatorDigestColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs.map Prod.fst\n\n\
         def stateInputAccumulatorDigestColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs.map Prod.snd\n\n\
         def claimedAccumulatorDigestColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs.map Prod.fst\n\n\
         def recomputedAccumulatorDigestColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs.map Prod.snd\n\n\
         def parentCeDigestColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryRecursiveAccumulatorCore.parentCeDigestColumns\n\n\
         def accumulatorDigestColumns : List Nat :=\n\
         \x20 recomputedAccumulatorDigestColumns\n\n\
         def stateOutputAccumulatorDigestColumns : List Nat :=\n\
         \x20 {}\n\n\
         theorem exact_owner_partition :\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowStart = rowStart ∧\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowEnd = {} ∧\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorCore.rowStart = {} ∧\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorCore.rowEnd = {} ∧\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowStart = {} ∧\n\
         \x20   FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowEnd = rowEnd := by native_decide\n\n\
         theorem recomputed_is_core_output :\n\
         \x20   recomputedAccumulatorDigestColumns =\n\
         \x20     FPrimeFullHistoryRecursiveAccumulatorCore.accumulatorDigestColumns := by native_decide\n\n\
         theorem recomputed_is_state_output :\n\
         \x20   recomputedAccumulatorDigestColumns =\n\
         \x20     stateOutputAccumulatorDigestColumns := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator\n",
        owner.row_start,
        owner.row_end,
        owner.row_end - owner.row_start,
        lean_nat_list(state_output_accumulator.iter().copied()),
        core.row_start,
        core.row_start,
        core.row_end,
        core.row_end,
    )
}

pub fn compare_recursive_accumulator_artifacts(
    builder: &R1csBuilder,
    recursive: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) {
    assert!(!recursive.is_base, "recursive accumulator needs recursive wire audit");
    assert_eq!(recursive.state_out_columns.len(), 31, "plain recursive state width");
    let accumulator = owner(builder, "fprime.recursive.accumulator");
    assert_eq!(
        accumulator.row_end - accumulator.row_start,
        37_303,
        "fixed-profile recursive accumulator row count"
    );
    let running_link = RowFamilyRange {
        name: "fprime.recursive.accumulator.running_link",
        row_start: accumulator.row_start,
        row_end: accumulator.row_start + 4,
    };
    let core = RowFamilyRange {
        name: "fprime.recursive.accumulator.core",
        row_start: accumulator.row_start + 4,
        row_end: accumulator.row_end - 4,
    };
    let output_link = RowFamilyRange {
        name: "fprime.recursive.accumulator.output_link",
        row_start: accumulator.row_end - 4,
        row_end: accumulator.row_end,
    };
    assert_eq!(core.row_end - core.row_start, 37_295, "shared accumulator core");

    compare_accumulator_core_artifacts(
        builder,
        &core,
        AccumulatorCorePaths {
            artifact: CORE_ARTIFACT_PATH,
            shard_prefix: CORE_SHARD_PREFIX,
            hashes: CORE_HASHES_PATH,
            check_coverage: CORE_CHECK_COVERAGE_PATH,
            schedule_prefix: CORE_SCHEDULE_PREFIX,
            schedules: CORE_SCHEDULES_PATH,
            recursive: true,
        },
    );

    let root = formal_repo_root();
    let mut drifted = Vec::new();
    compare(
        &root.join(RUNNING_LINK_PATH),
        render_equality_artifact(
            builder,
            &running_link,
            "FPrimeFullHistoryRecursiveAccumulatorRunningLink",
            "recursive running-accumulator input digest link",
            &range_hash(builder, &running_link),
        ),
        &mut drifted,
    );
    compare(
        &root.join(OUTPUT_LINK_PATH),
        render_equality_artifact(
            builder,
            &output_link,
            "FPrimeFullHistoryRecursiveAccumulatorOutputLink",
            "recursive claimed/recomputed output accumulator digest link",
            &range_hash(builder, &output_link),
        ),
        &mut drifted,
    );
    compare(
        &root.join(OWNER_ARTIFACT_PATH),
        render_owner_artifact(
            accumulator,
            &core,
            &recursive.state_out_columns[23..27],
            &range_hash(builder, accumulator),
        ),
        &mut drifted,
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "recursive accumulator aggregate artifacts drifted: {drifted:?}"
    );
}
