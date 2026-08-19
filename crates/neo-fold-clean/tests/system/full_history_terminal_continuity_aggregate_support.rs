use super::*;

const TERMINAL_CHILDREN: usize = 14;
const SHARD_PATH_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalContinuityShard";
const AGGREGATE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalContinuityArtifact.lean";

fn render_aggregate(range: &RowFamilyRange, range_digest: &str) -> String {
    let imports = (0..TERMINAL_CHILDREN)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let pairs = (0..TERMINAL_CHILDREN)
        .map(|index| format!("Generated{index}.pairs"))
        .collect::<Vec<_>>()
        .join(" ++\n  ");
    let lengths = (0..TERMINAL_CHILDREN)
        .map(|index| format!("Generated{index}.pairs_length"))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let mut partition = vec!["Generated0.rowStart = rowStart".to_owned()];
    partition.extend(
        (0..TERMINAL_CHILDREN - 1).map(|index| format!("Generated{index}.rowEnd = Generated{}.rowStart", index + 1)),
    );
    partition.push(format!("Generated{}.rowEnd = rowEnd", TERMINAL_CHILDREN - 1));
    let partition = partition.join(" ∧\n    ");
    let row_count = range.row_end - range.row_start;

    format!(
        r#"{imports}

/-! Generated aggregate for the exact 14-child terminal continuity owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "{range_digest}"
def rowStart : Nat := {}
def rowEnd : Nat := {}
def rowCount : Nat := {row_count}

def pairs : List (Nat × Nat) :=
  {pairs}

def rows : List Row := EqualityPins.rows pairs

theorem shard_ranges_partition :
    {partition} := by native_decide

theorem pairs_length : pairs.length = rowCount := by
  simp only [pairs, List.length_append,
    {lengths}]
  decide

theorem rows_length : rows.length = rowCount := by
  simpa [rows, EqualityPins.rows] using pairs_length

theorem sound {{assignment : Nat → Nat}}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  exact EqualityPins.rows_sound canonical one satisfies

theorem complete {{assignment : Nat → Nat}}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equalities : ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2) :
    Satisfies rows assignment := by
  exact EqualityPins.rows_complete canonical one equalities

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity
"#,
        range.row_start, range.row_end
    )
}

pub fn compare_terminal_continuity_artifacts(builder: &R1csBuilder) {
    let terminal_continuity = builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == "decider.terminal_continuity")
        .expect("terminal continuity owner");
    let row_count = terminal_continuity.row_end - terminal_continuity.row_start;
    assert_eq!(row_count % TERMINAL_CHILDREN, 0, "equal terminal child blocks");
    let rows_per_child = row_count / TERMINAL_CHILDREN;

    for index in 0..TERMINAL_CHILDREN {
        let range = RowFamilyRange {
            name: "decider.terminal_continuity.child",
            row_start: terminal_continuity.row_start + index * rows_per_child,
            row_end: terminal_continuity.row_start + (index + 1) * rows_per_child,
        };
        compare_full_history_artifact(
            &formal_repo_root().join(format!("{SHARD_PATH_PREFIX}{index}.lean")),
            &render_equality_artifact(
                builder,
                &range,
                &format!("FPrimeFullHistoryTerminalContinuity.Generated{index}"),
                &format!("terminal child/running continuity shard {index}"),
                &range_hash(builder, &range),
            ),
            "lean.expected",
        );
    }

    compare_full_history_artifact(
        &formal_repo_root().join(AGGREGATE_PATH),
        &render_aggregate(terminal_continuity, &range_hash(builder, terminal_continuity)),
        "lean.expected",
    );
}
