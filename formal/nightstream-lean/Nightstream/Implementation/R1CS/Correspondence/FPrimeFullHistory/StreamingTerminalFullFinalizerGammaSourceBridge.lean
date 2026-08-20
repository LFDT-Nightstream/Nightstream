import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition

/-!
Contract: bind the exact terminal gamma columns to the complete handwritten
transcript input schedule and its two computed challenges.

This module does not give transcript inputs semantic authority. A lifecycle
parent must bind all `inputValues`, not only the compressed challenges.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaSourceBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound

private abbrev artifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

/-- Every verifier-owned field absorbed by the gamma transcript, in exact
operation order. -/
def inputValues (assignment : Nat → Nat) : List Nat :=
  externalColumns.map assignment

/-- The two typed extension-field challenges computed by the complete
handwritten transcript replay. -/
def computedGamma
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (index : Fin 2) : K :=
  if index.val = 0 then
    ⟨computedGamma1 assignment canonical ⟨0, by decide⟩,
      computedGamma1 assignment canonical ⟨1, by decide⟩⟩
  else
    ⟨computedGamma2 assignment canonical ⟨0, by decide⟩,
      computedGamma2 assignment canonical ⟨1, by decide⟩⟩

private theorem opened_gamma1_column (coordinate : Fin 2) :
    artifact.gammaMuxOpenedColumns.getD coordinate.val 0 =
      artifact.gamma1Columns.getD coordinate.val 0 := by
  rw [rawArtifact_valid.gammaMuxOpenedOrder]
  fin_cases coordinate <;> rfl

private theorem opened_gamma2_column (coordinate : Fin 2) :
    artifact.gammaMuxOpenedColumns.getD (2 + coordinate.val) 0 =
      artifact.gamma2Columns.getD coordinate.val 0 := by
  rw [rawArtifact_valid.gammaMuxOpenedOrder]
  fin_cases coordinate <;> rfl

private theorem transition_field_eq_transcript_field
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    fieldValue assignment column =
      Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement.fieldAt
        assignment canonical column := by
  apply Fin.ext
  change assignment column % goldilocksP = assignment column
  exact Nat.mod_eq_of_lt (canonical column)

/-- Exact transcript rows fix both gamma values from all external transcript
inputs. The output equality is a computation theorem, not an authority
shortcut; lifecycle authority remains on `inputValues`. -/
theorem rows_bind_candidateGamma
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.GammaTranscriptSatisfied assignment) :
    candidateGamma assignment = computedGamma assignment canonical := by
  have sound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge.rows_sound
      assignment canonical one satisfied
  have gamma1Exact (coordinate : Fin 2) :
      fieldValue assignment
          (artifact.gamma1Columns.getD coordinate.val 0) =
        computedGamma1 assignment canonical coordinate := by
    rw [transition_field_eq_transcript_field assignment canonical]
    simpa [assignedGamma1] using
      congrFun sound.transcript.gamma1 coordinate
  have gamma2Exact (coordinate : Fin 2) :
      fieldValue assignment
          (artifact.gamma2Columns.getD coordinate.val 0) =
        computedGamma2 assignment canonical coordinate := by
    rw [transition_field_eq_transcript_field assignment canonical]
    simpa [assignedGamma2] using
      congrFun sound.transcript.gamma2 coordinate
  funext index
  fin_cases index
  · change K.mk
        (fieldValue assignment (artifact.gammaMuxOpenedColumns.getD 0 0))
        (fieldValue assignment (artifact.gammaMuxOpenedColumns.getD 1 0)) = _
    rw [opened_gamma1_column ⟨0, by decide⟩,
      opened_gamma1_column ⟨1, by decide⟩]
    exact congrArg₂ K.mk
      (gamma1Exact ⟨0, by decide⟩)
      (gamma1Exact ⟨1, by decide⟩)
  · change K.mk
        (fieldValue assignment (artifact.gammaMuxOpenedColumns.getD 2 0))
        (fieldValue assignment (artifact.gammaMuxOpenedColumns.getD 3 0)) = _
    have lowColumn :
        artifact.gammaMuxOpenedColumns.getD 2 0 =
          artifact.gamma2Columns.getD 0 0 := by
      simpa using opened_gamma2_column ⟨0, by decide⟩
    have highColumn :
        artifact.gammaMuxOpenedColumns.getD 3 0 =
          artifact.gamma2Columns.getD 1 0 := by
      simpa using opened_gamma2_column ⟨1, by decide⟩
    rw [lowColumn, highColumn]
    exact congrArg₂ K.mk
      (gamma2Exact ⟨0, by decide⟩)
      (gamma2Exact ⟨1, by decide⟩)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaSourceBridge
