import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptExecution

/-!
Contract: exact row soundness for the terminal Nebula gamma transcript.

Satisfying every exported constant pin and Poseidon2 call refines the
handwritten ten-append, two-challenge schedule and fixes both two-field
challenge outputs. The initial application-domain state remains a separate
named obligation.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptExecution

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

theorem pins_canonical :
    ConstantPins.ValuesCanonical trace.pins := by
  unfold ConstantPins.ValuesCanonical
  decide

theorem trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.GammaTranscriptSatisfied assignment) :
    trace.Accepted assignment := by
  constructor
  · exact ConstantPins.sound pins_canonical
      (rowsIncluded_self _) canonical one satisfied.1
  · intro call member
    exact Poseidon2Call.rows_sound call canonical one
      (satisfied.2 call member)

def semanticRun
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ColumnReplay.SemanticRun :=
  ColumnReplay.semanticExecute assignment canonical
    (ColumnReplay.decodeRun assignment canonical start) operations

def zeroDigest : Fin 4 → Field := fun _ => wordField 0

def computedGamma1
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 2 → Field := fun lane =>
  (semanticRun assignment canonical).digests.getD 0 zeroDigest
    ⟨lane.val, by omega⟩

def computedGamma2
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 2 → Field := fun lane =>
  (semanticRun assignment canonical).digests.getD 1 zeroDigest
    ⟨lane.val, by omega⟩

def assignedGamma1
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 2 → Field := fun lane =>
  fieldAt assignment canonical
    (rawArtifact.gamma1Columns.getD lane.val 0)

def assignedGamma2
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 2 → Field := fun lane =>
  fieldAt assignment canonical
    (rawArtifact.gamma2Columns.getD lane.val 0)

structure Sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  replay :
    semanticRun assignment canonical =
      ColumnReplay.decodeRun assignment canonical result
  gamma1 : assignedGamma1 assignment canonical =
    computedGamma1 assignment canonical
  gamma2 : assignedGamma2 assignment canonical =
    computedGamma2 assignment canonical

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.GammaTranscriptSatisfied assignment) :
    Sound assignment canonical := by
  have accepted := trace_accepted assignment canonical one satisfied
  have refined := ColumnReplay.execute_sound canonical pins_canonical one
    accepted execution
  refine {
    replay := refined
    gamma1 := ?_
    gamma2 := ?_ }
  · have digestsEqual :=
      congrArg ColumnReplay.SemanticRun.digests refined
    funext lane
    have selected := congrArg
      (fun digests => (digests.getD 0 zeroDigest)
        ⟨lane.val, by omega⟩) digestsEqual
    have normalized :
        computedGamma1 assignment canonical lane =
          fieldAt assignment canonical
            (gamma1DigestColumns ⟨lane.val, by omega⟩) := by
      simpa [computedGamma1, semanticRun, ColumnReplay.decodeRun,
        ColumnReplay.decodeDigest, gamma1DigestColumns] using selected
    rw [gamma1_columns_exact lane] at normalized
    exact normalized.symm
  · have digestsEqual :=
      congrArg ColumnReplay.SemanticRun.digests refined
    funext lane
    have selected := congrArg
      (fun digests => (digests.getD 1 zeroDigest)
        ⟨lane.val, by omega⟩) digestsEqual
    have normalized :
        computedGamma2 assignment canonical lane =
          fieldAt assignment canonical
            (gamma2DigestColumns ⟨lane.val, by omega⟩) := by
      simpa [computedGamma2, semanticRun, ColumnReplay.decodeRun,
        ColumnReplay.decodeDigest, gamma2DigestColumns] using selected
    rw [gamma2_columns_exact lane] at normalized
    exact normalized.symm

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound
