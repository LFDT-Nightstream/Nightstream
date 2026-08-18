import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLink

/-!
Contract: exact omission counterexample for the terminal Nebula-state-digest
family's final links.

Rust exports the baseline values on the violated equality-row support and
replays the complete assignment against every retained row. This leaf proves
that the same support values violate source row 23009.

Assurance tier: artifact-checked.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLinkNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

def changedColumn : Nat := rawArtifact.xOutStateColumns.getD 0 0

def digestColumn : Nat := rawArtifact.hashOutputColumns.getD 0 0

def baselineValue : Nat := rawArtifact.baselineDigestValue

def changedValue : Nat := baselineValue + 1

def projectedBaselineAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = changedColumn then baselineValue
  else if column = digestColumn then baselineValue
  else 0

def omissionAssignment (column : Nat) : Nat :=
  if column = changedColumn then changedValue
  else projectedBaselineAssignment column

theorem changed_value_canonical : changedValue < goldilocksP := by
  norm_num [changedValue, baselineValue, rawArtifact, goldilocksP]

theorem omissionAssignment_one : omissionAssignment 0 = 1 := by
  norm_num [omissionAssignment, projectedBaselineAssignment,
    changedColumn, rawArtifact]

theorem omissionAssignment_canonical (column : Nat) :
    omissionAssignment column < goldilocksP := by
  by_cases changed : column = changedColumn
  · simp [omissionAssignment, changed, changed_value_canonical]
  · by_cases zero : column = 0
    · subst column
      norm_num [omissionAssignment, projectedBaselineAssignment,
        changedColumn, rawArtifact, goldilocksP]
    · by_cases digest : column = digestColumn
      · subst column
        norm_num [omissionAssignment, projectedBaselineAssignment,
          changedColumn, digestColumn, baselineValue, rawArtifact,
          goldilocksP]
      · simp [omissionAssignment, changed, projectedBaselineAssignment,
          zero, digest, goldilocksP]

theorem replay_support_exact :
    omissionAssignment changedColumn = rawArtifact.baselineDigestValue + 1 ∧
      omissionAssignment digestColumn = rawArtifact.baselineDigestValue := by
  norm_num [omissionAssignment, projectedBaselineAssignment, changedValue,
    baselineValue, changedColumn, digestColumn, rawArtifact]

theorem agrees_outside_changed
    {column : Nat} (outside : column ≠ changedColumn) :
    omissionAssignment column = projectedBaselineAssignment column := by
  simp [omissionAssignment, outside]

private def violatedRow : Row :=
  builderLinearRow changedColumn [(digestColumn, 1)]

private theorem violatedRow_member :
    violatedRow ∈ rawArtifact.equalityRows := by
  rw [RawArtifact.equalityRows]
  apply List.mem_map.mpr
  refine ⟨0, ?_, ?_⟩
  · norm_num [digestFields]
  · norm_num [violatedRow, changedColumn, digestColumn, rawArtifact]

theorem selected_source_row : rawArtifact.selectedSourceRow = 23009 := by
  norm_num [rawArtifact]

theorem linkRows_fail :
    ¬ rawArtifact.LinkSatisfied omissionAssignment := by
  intro satisfied
  have holds := satisfied violatedRow violatedRow_member
  have exact := builderLinearRow_sound omissionAssignment_canonical
    omissionAssignment_one changedColumn [(digestColumn, 1)]
      (by norm_num [CanonicalTerms, goldilocksP]) holds
  norm_num [omissionAssignment, projectedBaselineAssignment, changedValue,
    baselineValue, changedColumn, digestColumn, rawArtifact,
    lcEval, rawLcEval, goldilocksP] at exact

/-- Lean-checked selected-family failure for the exact support values from the
Rust-replayed terminal audit assignment. -/
theorem exact_removal_counterexample :
    omissionAssignment 0 = 1 ∧
      (∀ column, omissionAssignment column < goldilocksP) ∧
      omissionAssignment changedColumn = rawArtifact.baselineDigestValue + 1 ∧
      omissionAssignment digestColumn = rawArtifact.baselineDigestValue ∧
      rawArtifact.selectedSourceRow = 23009 ∧
      ¬ rawArtifact.LinkSatisfied omissionAssignment :=
  ⟨omissionAssignment_one, omissionAssignment_canonical,
    replay_support_exact.1, replay_support_exact.2,
    selected_source_row, linkRows_fail⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLinkNecessity
