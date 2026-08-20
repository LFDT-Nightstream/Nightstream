import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound

/-!
Contract: exact row soundness for the rank-one compression stage of the
terminal `ops` leaf.

The theorem binds the 108 primary outputs to 108 canonical openings and
proves that every one of the 54 assigned compression outputs equals the exact
compact seeded linear map. It records the exact verifier-owned schedule.

It does not own sampler no-rejection liveness, the Poseidon2 envelope,
Module-SIS security, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SeededPhi81
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFullFinalizerLeafScheduleCertificate

def opsCompressionOpeningRows : List Row :=
  ((List.range leafCompressionFields).map fun index =>
    (openingPiece rawArtifact.opsLeaf.compression index).rows).flatten

def OpsCompressionOpeningsSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsCompressionOpeningRows assignment

private theorem opening_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsCompressionOpeningsSatisfied assignment)
    (index : Nat) (bounded : index < leafCompressionFields) :
    Satisfies
      (openingPiece rawArtifact.opsLeaf.compression index).rows assignment := by
  have pieces := (satisfies_flatten_iff
    ((List.range leafCompressionFields).map fun fieldIndex =>
      (openingPiece rawArtifact.opsLeaf.compression fieldIndex).rows)
    assignment).mp (by
      simpa only [OpsCompressionOpeningsSatisfied,
        opsCompressionOpeningRows] using satisfied)
  exact pieces _ (List.mem_map.mpr
    ⟨index, List.mem_range.mpr bounded, rfl⟩)

def opsCompressionRows : List Row :=
  opsCompressionOpeningRows ++ rawArtifact.opsLeaf.compression.block.rows

def OpsCompressionSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsCompressionRows assignment

private theorem openings_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsCompressionSatisfied assignment) :
    OpsCompressionOpeningsSatisfied assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem block_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsCompressionSatisfied assignment) :
    Satisfies rawArtifact.opsLeaf.compression.block.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

structure Sound (assignment : Nat → Nat) : Prop where
  sourceOrder :
    rawArtifact.opsLeaf.compression.sourceColumns =
      rawArtifact.opsLeaf.primary.block.outputColumns
  schedule :
    rawArtifact.opsLeaf.compression.block.schedule =
      expectedCompressionSchedule
  openings : ∀ index, index < leafCompressionFields →
    CanonicalOpening
      (localAssignment assignment
        (rawArtifact.opsLeaf.compression.sourceColumns.getD index 0)
        (rawArtifact.opsLeaf.compression.wordStart index))
  inputDigits : ∀ fieldIndex, fieldIndex < leafCompressionFields →
    ∀ digitIndex, digitIndex < digitCount →
      assignment
          (rawArtifact.opsLeaf.compression.wordStart fieldIndex + digitIndex) =
        canonicalDigit
          (assignment
            (rawArtifact.opsLeaf.compression.sourceColumns.getD fieldIndex 0))
          digitIndex
  block : rawArtifact.opsLeaf.compression.block.Holds assignment
  outputs :
    ∀ (output : Fin rawArtifact.opsLeaf.compression.block.kappa)
      (coordinate : Fin dimension),
      assignment
          (rawArtifact.opsLeaf.compression.block.outputColumns.getD
            (output.val * dimension + coordinate.val) 0) =
        rawArtifact.opsLeaf.compression.block.linearValue assignment
          output.val coordinate.val

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsCompressionSatisfied assignment) :
    Sound assignment := by
  have openingRows := openings_satisfied assignment satisfied
  have blockRows := block_satisfied assignment satisfied
  have blockSound := SeededPhi81.sound canonical one blockRows
  exact {
    sourceOrder := rawArtifact_valid.opsLeafCompressionSources
    schedule := rawArtifact_valid.opsLeafCompressionSchedule
    openings := fun index bounded =>
      opening_rows_sound rawArtifact.opsLeaf.compression index assignment
        canonical one (opening_satisfied assignment openingRows index bounded)
    inputDigits := fun fieldIndex fieldBounded digitIndex digitBounded =>
      opening_digit_exact rawArtifact.opsLeaf.compression fieldIndex assignment
        canonical one
        (opening_satisfied assignment openingRows fieldIndex fieldBounded)
        digitIndex digitBounded
    block := blockSound
    outputs := fun output coordinate =>
      rawArtifact.opsLeaf.compression.block.output_eq_linearValue
        blockSound output coordinate }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound
