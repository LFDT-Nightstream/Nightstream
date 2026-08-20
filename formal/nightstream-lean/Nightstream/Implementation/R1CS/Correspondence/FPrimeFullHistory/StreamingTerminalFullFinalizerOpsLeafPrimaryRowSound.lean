import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound

/-!
Contract: exact row soundness for the primary rank-two seeded `Phi81` stage
of the terminal `ops` leaf.

The theorem binds all 981 source fields to canonical 41-coordinate words and
proves that every one of the 108 assigned outputs equals the exact compact
seeded linear map. It records the exact verifier-owned schedule identity.

It does not own sampler no-rejection liveness, the compression block,
Poseidon2 envelope, Module-SIS security, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound

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

def opsPrimaryRows : List Row :=
  opsPrimaryOpeningRows ++ rawArtifact.opsLeaf.primary.block.rows

def OpsPrimarySatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsPrimaryRows assignment

private theorem openings_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsPrimarySatisfied assignment) :
    OpsPrimaryOpeningsSatisfied assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem block_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsPrimarySatisfied assignment) :
    Satisfies rawArtifact.opsLeaf.primary.block.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

structure Sound (assignment : Nat → Nat) : Prop where
  sourceOrder :
    rawArtifact.opsLeaf.primary.sourceColumns =
      List.range' rawArtifact.opsLeaf.prefixConstantStartColumn
        leafPrefixFields ++ rawArtifact.opsColumns
  schedule :
    rawArtifact.opsLeaf.primary.block.schedule = expectedPrimarySchedule
  openings : ∀ index, index < leafPrimaryFields →
    CanonicalOpening
      (localAssignment assignment
        (rawArtifact.opsLeaf.primary.sourceColumns.getD index 0)
        (rawArtifact.opsLeaf.primary.wordStart index))
  inputDigits : ∀ fieldIndex, fieldIndex < leafPrimaryFields →
    ∀ digitIndex, digitIndex < digitCount →
      assignment
          (rawArtifact.opsLeaf.primary.wordStart fieldIndex + digitIndex) =
        canonicalDigit
          (assignment
            (rawArtifact.opsLeaf.primary.sourceColumns.getD fieldIndex 0))
          digitIndex
  block : rawArtifact.opsLeaf.primary.block.Holds assignment
  outputs :
    ∀ (output : Fin rawArtifact.opsLeaf.primary.block.kappa)
      (coordinate : Fin dimension),
      assignment
          (rawArtifact.opsLeaf.primary.block.outputColumns.getD
            (output.val * dimension + coordinate.val) 0) =
        rawArtifact.opsLeaf.primary.block.linearValue assignment
          output.val coordinate.val

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsPrimarySatisfied assignment) :
    Sound assignment := by
  have openingRows := openings_satisfied assignment satisfied
  have blockRows := block_satisfied assignment satisfied
  have blockSound := SeededPhi81.sound canonical one blockRows
  exact {
    sourceOrder := rawArtifact_valid.opsLeafPrimarySources
    schedule := rawArtifact_valid.opsLeafPrimarySchedule
    openings := fun index bounded =>
      opsPrimary_openings_sound assignment canonical one openingRows
        index bounded
    inputDigits := fun fieldIndex fieldBounded digitIndex digitBounded =>
      opsPrimary_digit_exact assignment canonical one openingRows
        fieldIndex fieldBounded digitIndex digitBounded
    block := blockSound
    outputs := fun output coordinate =>
      rawArtifact.opsLeaf.primary.block.output_eq_linearValue
        blockSound output coordinate }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound
