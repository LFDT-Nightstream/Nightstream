import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContext

/-!
Contract: exact omission counterexample for the terminal XOut context family.

The assignment projection changes only verifier-key XOut lane zero. Its two
values are the exact values in the full Rust-replayed cvc5 model. The exact
Rust row recipe rejects this projection at the corresponding public-source
equality. Exact Rust replay separately checks every retained row.

Assurance tier: artifact-checked.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContextNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext

def changedColumn : Nat := rawArtifact.xOutColumns.getD 1 0

def changedSource : Nat := rawArtifact.vkFsSourceColumns.getD 0 0

def baselineAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = changedColumn ∨ column = changedSource then
    rawArtifact.baselineChangedValue
  else 0

def omissionAssignment (column : Nat) : Nat :=
  if column = changedColumn then rawArtifact.mutatedChangedValue
  else baselineAssignment column

theorem omissionAssignment_one : omissionAssignment 0 = 1 := by
  norm_num [omissionAssignment, baselineAssignment, changedColumn, rawArtifact]

theorem omissionAssignment_canonical (column : Nat) :
    omissionAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    norm_num [omissionAssignment, baselineAssignment, changedColumn,
      rawArtifact, goldilocksP]
  · by_cases changed : column = changedColumn
    · simp only [omissionAssignment, if_pos changed]
      norm_num [rawArtifact, goldilocksP]
    · simp only [omissionAssignment, if_neg changed, baselineAssignment,
        if_neg zero]
      by_cases source : column = changedColumn ∨ column = changedSource
      · rw [if_pos source]
        norm_num [rawArtifact, goldilocksP]
      · rw [if_neg source]
        norm_num [goldilocksP]

theorem agrees_outside_changed
    {column : Nat} (outside : column ≠ changedColumn) :
    omissionAssignment column = baselineAssignment column := by
  simp [omissionAssignment, outside]

private def violatedRow : Row :=
  builderLinearRow changedColumn
    [(rawArtifact.vkFsSourceColumns.getD 0 0, 1)]

private theorem violatedRow_member : violatedRow ∈ rawArtifact.contextRows := by
  norm_num [violatedRow, changedColumn, RawArtifact.contextRows, copyRows,
    rawArtifact]

theorem candidate_projection_exact :
    omissionAssignment changedColumn = rawArtifact.mutatedChangedValue ∧
      omissionAssignment changedSource = rawArtifact.baselineChangedValue := by
  norm_num [omissionAssignment, baselineAssignment, changedColumn,
    changedSource, rawArtifact]

theorem contextRows_fail : ¬ rawArtifact.Satisfied omissionAssignment := by
  intro satisfied
  have holds := satisfied violatedRow violatedRow_member
  have exact := builderLinearRow_sound omissionAssignment_canonical
    omissionAssignment_one changedColumn
      [(rawArtifact.vkFsSourceColumns.getD 0 0, 1)]
      (by norm_num [CanonicalTerms, goldilocksP]) holds
  norm_num [omissionAssignment, baselineAssignment, changedColumn,
    changedSource, rawArtifact, lcEval, rawLcEval, goldilocksP] at exact

/-- Lean-checked counterexample for the exact Rust context-row recipe. Rust
replay proves that the same selected-family omission preserves every retained
row in the bounded terminal audit. -/
theorem exact_removal_counterexample :
    omissionAssignment 0 = 1 ∧
      (∀ column, omissionAssignment column < goldilocksP) ∧
      (∀ column, column ≠ changedColumn →
        omissionAssignment column = baselineAssignment column) ∧
      omissionAssignment changedColumn = rawArtifact.mutatedChangedValue ∧
      omissionAssignment changedSource = rawArtifact.baselineChangedValue ∧
      ¬ rawArtifact.Satisfied omissionAssignment :=
  ⟨omissionAssignment_one, omissionAssignment_canonical,
    fun _ outside => agrees_outside_changed outside,
    candidate_projection_exact.1, candidate_projection_exact.2,
    contextRows_fail⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContextNecessity
