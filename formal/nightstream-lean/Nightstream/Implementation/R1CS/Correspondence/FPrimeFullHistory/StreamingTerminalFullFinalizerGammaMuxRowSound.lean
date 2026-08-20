import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound

/-!
Contract: exact row soundness for the 16 terminal Nebula open muxes.

Owns the four gamma and twelve `d_pre` selector equations at the end of the
open phase. The selector is the decoded `input.open` bit proved by the open
algebra leaf. It does not own later leaf hashes, advance, close, or lifecycle
closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaMuxRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

namespace OpenSound

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound.Sound

end OpenSound

namespace TranscriptSound

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge.Sound

abbrev rowsSound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge.rows_sound

end TranscriptSound

private theorem orderedDifference_perm {positive negative : Nat}
    (different : positive ≠ negative) :
    (orderedDifferenceTerms positive negative).Perm
      [(positive, 1), (negative, goldilocksP - 1)] := by
  unfold orderedDifferenceTerms
  rw [if_neg different]
  by_cases before : positive < negative
  · simp [before]
  · rw [if_neg before]
    exact List.Perm.swap _ _ []

private theorem orderedDifference_eval
    (assignment : Nat → Nat) {positive negative : Nat}
    (different : positive ≠ negative) :
    lcEval assignment (orderedDifferenceTerms positive negative) =
      (assignment positive +
        (goldilocksP - 1) * assignment negative) % goldilocksP := by
  calc
    lcEval assignment (orderedDifferenceTerms positive negative) =
        lcEval assignment
          [(positive, 1), (negative, goldilocksP - 1)] :=
      lcEval_eq_of_perm assignment (orderedDifference_perm different)
    _ = _ := by simp [lcEval]

private theorem mux_columns_distinct (index : Nat) (bounded : index < 16) :
    rawArtifact.gammaMuxOpenedColumns.getD index 0 ≠
        rawArtifact.gammaMuxCarriedColumns.getD index 0 ∧
      rawArtifact.gammaMuxOutputColumns.getD index 0 ≠
        rawArtifact.gammaMuxCarriedColumns.getD index 0 := by
  interval_cases index <;> decide

private theorem mux_row_member (index : Nat) (bounded : index < 16) :
    rawArtifact.gammaMuxRow index ∈ rawArtifact.gammaMuxRows := by
  apply List.mem_map.mpr
  exact ⟨index, by simp [bounded], rfl⟩

/-- One exact Rust mux row selects the opened or carried source according to
the already-proved `input.open` bit. -/
theorem output_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (inputOpenExact : assignment rawArtifact.openColumn = 0 ∨
      assignment rawArtifact.openColumn = 1)
    (satisfied : rawArtifact.GammaMuxSatisfied assignment)
    (index : Nat) (bounded : index < 16) :
    assignment (rawArtifact.gammaMuxOutputColumns.getD index 0) =
      if assignment rawArtifact.openColumn = 1 then
        assignment (rawArtifact.gammaMuxOpenedColumns.getD index 0)
      else
        assignment (rawArtifact.gammaMuxCarriedColumns.getD index 0) := by
  let selector := rawArtifact.gammaMuxSelectorColumn
  let opened := rawArtifact.gammaMuxOpenedColumns.getD index 0
  let carried := rawArtifact.gammaMuxCarriedColumns.getD index 0
  let output := rawArtifact.gammaMuxOutputColumns.getD index 0
  have different := mux_columns_distinct index bounded
  have rowHolds := satisfied _ (mux_row_member index bounded)
  change
    lcEval assignment [(selector, 1)] *
        lcEval assignment (orderedDifferenceTerms opened carried) %
          goldilocksP =
      lcEval assignment (orderedDifferenceTerms output carried) at rowHolds
  rw [orderedDifference_eval assignment different.1,
    orderedDifference_eval assignment different.2] at rowHolds
  have selectorCanonical := canonical selector
  have selectorEval : lcEval assignment [(selector, 1)] =
      assignment selector := by
    simp [lcEval, Nat.mod_eq_of_lt selectorCanonical]
  rw [selectorEval] at rowHolds
  have selectorColumn : selector = rawArtifact.openColumn := by
    exact rawArtifact_valid.gammaMuxSelector
  rw [selectorColumn] at rowHolds
  change
    assignment rawArtifact.openColumn *
        ((assignment opened +
          (goldilocksP - 1) * assignment carried) % goldilocksP) %
          goldilocksP =
      (assignment output +
        (goldilocksP - 1) * assignment carried) % goldilocksP at rowHolds
  have openedCanonical := canonical opened
  have carriedCanonical := canonical carried
  have outputCanonical := canonical output
  change assignment output =
    if assignment rawArtifact.openColumn = 1 then assignment opened
    else assignment carried
  rcases inputOpenExact with inputZero | inputOne
  · have outputEq : assignment output = assignment carried := by
      rw [inputZero] at rowHolds
      simp only [zero_mul, Nat.zero_mod] at rowHolds
      simp only [goldilocksP] at rowHolds carriedCanonical outputCanonical
      omega
    rw [if_neg (by omega)]
    exact outputEq
  · have outputEq : assignment output = assignment opened := by
      rw [inputOne] at rowHolds
      simp only [one_mul, Nat.mod_mod] at rowHolds
      simp only [goldilocksP] at rowHolds openedCanonical carriedCanonical outputCanonical
      omega
    rw [if_pos inputOne]
    exact outputEq

structure Sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  transcript : TranscriptSound.Sound assignment canonical
  outputs : ∀ index < 16,
    assignment (rawArtifact.gammaMuxOutputColumns.getD index 0) =
      if assignment rawArtifact.openColumn = 1 then
        assignment (rawArtifact.gammaMuxOpenedColumns.getD index 0)
      else
        assignment (rawArtifact.gammaMuxCarriedColumns.getD index 0)

/-- The transcript rows and all 16 exact mux rows implement the complete
gamma-and-`d_pre` end of the open phase. -/
theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (openSound : OpenSound.Sound assignment)
    (transcriptSatisfied : rawArtifact.GammaTranscriptSatisfied assignment)
    (muxSatisfied : rawArtifact.GammaMuxSatisfied assignment) :
    Sound assignment canonical := by
  exact {
    transcript := TranscriptSound.rowsSound assignment canonical one
      transcriptSatisfied
    outputs := fun index bounded =>
      output_exact assignment canonical openSound.inputOpenExact muxSatisfied
        index bounded }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaMuxRowSound
