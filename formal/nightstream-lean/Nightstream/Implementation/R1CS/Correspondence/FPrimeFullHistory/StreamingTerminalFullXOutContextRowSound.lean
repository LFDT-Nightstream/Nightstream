import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutContextRowSound

/-!
Contract: the 24 exact full-layout Rust rows bind decoded XOut context lanes
to verifier-owned terminal columns and fixed lifecycle constants.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullXOutContextRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutContextRowSound
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext

private abbrev fullArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext.rawArtifact

private theorem canonical_row_implies_builder
    (assignment : Nat → Nat) (row : Row)
    (holds : RowHolds assignment
      (canonicalizeLinearRow row)) :
    RowHolds assignment row := by
  cases row with
  | mk a b c =>
      cases a with
      | nil => simpa [canonicalizeLinearRow] using holds
      | cons head tail =>
          have permutation : (tail ++ [head]).Perm (head :: tail) := by
            simpa using List.Perm.append_comm tail [head]
          have evalPermutation := lcEval_eq_of_perm assignment permutation
          simpa [canonicalizeLinearRow, RowHolds, evalPermutation] using holds

private theorem canonical_satisfied_implies_builder
    (assignment : Nat → Nat)
    (satisfied : Satisfied assignment) :
    fullArtifact.Satisfied assignment := by
  intro row member
  apply canonical_row_implies_builder assignment row
  apply satisfied
  exact List.mem_map.mpr ⟨row, member, rfl⟩

private theorem constants :
    ContextConstantsCanonical fullArtifact := by
  refine {
    domainPositive := by decide
    domainCanonical := by decide
    acceptedPositive := by decide
    acceptedCanonical := by decide
    markerPositive := by decide
    markerCanonical := by decide }

private theorem rows_present : ContextRowsPresent fullArtifact := by
  refine {
    domain := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    verifierKey := ?_
    piCcsHeader := ?_
    chunkCountLow := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    chunkCountHigh := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    stepCountLow := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    stepCountHigh := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    programCounterLow := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    programCounterHigh := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
    boundary := ?_
    accumulator := ?_
    nebulaMarker := by
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact] }
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, fullArtifact, rawArtifact]

abbrev Sound := SoundFor fullArtifact

/-- Full-layout Rust satisfaction implies the named terminal XOut context
relation after the exact CSC-to-builder operand permutation. -/
theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfied assignment) :
    Sound assignment := by
  apply rows_sound_for fullArtifact assignment canonical one
  · exact canonical_satisfied_implies_builder assignment satisfied
  · exact constants
  · exact rows_present

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullXOutContextRowSound
