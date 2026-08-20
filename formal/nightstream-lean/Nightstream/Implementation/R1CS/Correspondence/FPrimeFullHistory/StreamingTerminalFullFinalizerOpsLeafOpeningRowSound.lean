import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CanonicalWord

/-!
Contract: the first authoritative `ops` field in the terminal leaf has the
exact Rust-owned 124-row canonical shifted-ternary opening.

This leaf theorem proves one concrete source placement before the same proof
is generalized over all `ops` leaf fields. It does not own the seeded Ajtai
block, compression block, Poseidon2 envelope, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

def openingPiece (binding : SeededBindingArtifact) (index : Nat) : Piece where
  rowStart := binding.openingRow index
  rowEnd := binding.openingRow index + canonicalOpeningRows
  payload := .shiftedTernary
    (binding.sourceColumns.getD index 0)
    (binding.wordStart index)

/-- The first field after the nine verifier-owned leaf-prefix constants. -/
def firstOpsFieldIndex : Nat := leafPrefixFields

def firstOpsFieldPiece : Piece :=
  openingPiece rawArtifact.opsLeaf.primary firstOpsFieldIndex

private theorem canonicalRows_count : canonicalRows.length = 124 := by
  decide

theorem openingPiece_valid (binding : SeededBindingArtifact) (index : Nat) :
    (openingPiece binding index).Valid := by
  constructor
  · simp [openingPiece]
  · simp only [openingPiece, Payload.rowCount, Nat.add_sub_cancel_left]
    rw [canonicalRows_count]
    decide

/-- Any one exact relocated opening reconstructs the typed canonical opening
of its named source field. -/
theorem opening_rows_sound
    (binding : SeededBindingArtifact) (index : Nat)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (openingPiece binding index).rows assignment) :
    CanonicalOpening
      (localAssignment assignment
        (binding.sourceColumns.getD index 0)
        (binding.wordStart index)) := by
  have mapped :
      Satisfies
        (canonicalRows.map
          (Relabel.row (shiftedTernaryColumnMap
            (binding.sourceColumns.getD index 0)
            (binding.wordStart index))))
        assignment := by
    simpa only [Piece.rows, openingPiece, Payload.rows] using satisfied
  have localSatisfied :
      Satisfies canonicalRows
        (localAssignment assignment
          (binding.sourceColumns.getD index 0)
          (binding.wordStart index)) := by
    exact (Relabel.satisfies_mapped_iff canonicalRows
      (shiftedTernaryColumnMap
        (binding.sourceColumns.getD index 0)
        (binding.wordStart index)) assignment).mp mapped
  have localCanonical :
      ∀ column,
        localAssignment assignment
            (binding.sourceColumns.getD index 0)
            (binding.wordStart index) column < goldilocksP := by
    intro column
    exact canonical _
  have localOne :
      localAssignment assignment
          (binding.sourceColumns.getD index 0)
          (binding.wordStart index) 0 = 1 := by
    simpa using one
  exact canonicalOpening_of_canonicalRows goldilocks_euclidPrime
    localCanonical localOne localSatisfied

theorem opening_digit_exact
    (binding : SeededBindingArtifact) (fieldIndex : Nat)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies
      (openingPiece binding fieldIndex).rows assignment)
    (digitIndex : Nat) (digitBounded : digitIndex < digitCount) :
    assignment (binding.wordStart fieldIndex + digitIndex) =
      canonicalDigit
        (assignment (binding.sourceColumns.getD fieldIndex 0)) digitIndex := by
  exact productionDigit_eq_canonicalDigit
    (opening_rows_sound binding fieldIndex assignment canonical one satisfied)
    digitIndex digitBounded

/-- Exact Rust source placement for the first authoritative `ops` field. -/
theorem firstOpsField_source :
    rawArtifact.opsLeaf.primary.sourceColumns.getD firstOpsFieldIndex 0 =
      28038963 := by
  rfl

theorem firstOpsField_digitStart :
    rawArtifact.opsLeaf.primary.wordStart firstOpsFieldIndex = 28425998 := by
  calc
    rawArtifact.opsLeaf.primary.wordStart firstOpsFieldIndex =
        rawArtifact.opsLeaf.primary.metadataStartColumn + 2 +
          leafPrimaryOutputs +
            canonicalOpeningColumns * firstOpsFieldIndex :=
      rawArtifact_valid.opsLeafPrimaryWordStart firstOpsFieldIndex (by decide)
    _ = 28425998 := by decide

theorem firstOpsField_rowStart : firstOpsFieldPiece.rowStart = 387013 := by
  rfl

theorem firstOpsField_rowEnd : firstOpsFieldPiece.rowEnd = 387137 := by
  rfl

theorem firstOpsField_payload :
    firstOpsFieldPiece.payload =
      .shiftedTernary 28038963 28425998 := by
  unfold firstOpsFieldPiece openingPiece
  rw [firstOpsField_source, firstOpsField_digitStart]

theorem firstOpsField_valid : firstOpsFieldPiece.Valid := by
  exact openingPiece_valid rawArtifact.opsLeaf.primary firstOpsFieldIndex

def FirstOpsFieldSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies firstOpsFieldPiece.rows assignment

/-- The exact relocated rows determine one canonical opening of the named
authoritative `ops` source field. -/
theorem firstOpsField_opening_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : FirstOpsFieldSatisfied assignment) :
    CanonicalOpening
      (localAssignment assignment 28038963 28425998) := by
  have sound := opening_rows_sound rawArtifact.opsLeaf.primary
    firstOpsFieldIndex assignment canonical one satisfied
  simpa only [firstOpsField_source, firstOpsField_digitStart] using sound

/-- Each of the 41 SIS input coordinates is the unique canonical digit of
the first authoritative `ops` field. -/
theorem firstOpsField_digit_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : FirstOpsFieldSatisfied assignment)
    (index : Nat) (bounded : index < digitCount) :
    assignment (28425998 + index) =
      canonicalDigit (assignment 28038963) index := by
  exact productionDigit_eq_canonicalDigit
    (firstOpsField_opening_sound assignment canonical one satisfied)
    index bounded

def opsPrimaryOpeningRows : List Row :=
  ((List.range leafPrimaryFields).map fun index =>
    (openingPiece rawArtifact.opsLeaf.primary index).rows).flatten

def OpsPrimaryOpeningsSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsPrimaryOpeningRows assignment

private theorem opsPrimaryOpening_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsPrimaryOpeningsSatisfied assignment)
    (index : Nat) (bounded : index < leafPrimaryFields) :
    Satisfies (openingPiece rawArtifact.opsLeaf.primary index).rows assignment := by
  have pieces := (satisfies_flatten_iff
    ((List.range leafPrimaryFields).map fun fieldIndex =>
      (openingPiece rawArtifact.opsLeaf.primary fieldIndex).rows)
    assignment).mp (by
      simpa only [OpsPrimaryOpeningsSatisfied, opsPrimaryOpeningRows]
        using satisfied)
  exact pieces _ (List.mem_map.mpr
    ⟨index, List.mem_range.mpr bounded, rfl⟩)

/-- All 981 fields in the primary leaf binding have exact canonical openings.
The theorem is indexed and does not evaluate the complete row family. -/
theorem opsPrimary_openings_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsPrimaryOpeningsSatisfied assignment)
    (index : Nat) (bounded : index < leafPrimaryFields) :
    CanonicalOpening
      (localAssignment assignment
        (rawArtifact.opsLeaf.primary.sourceColumns.getD index 0)
        (rawArtifact.opsLeaf.primary.wordStart index)) := by
  exact opening_rows_sound rawArtifact.opsLeaf.primary index assignment
    canonical one (opsPrimaryOpening_satisfied assignment satisfied index bounded)

theorem opsPrimary_digit_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsPrimaryOpeningsSatisfied assignment)
    (fieldIndex : Nat) (fieldBounded : fieldIndex < leafPrimaryFields)
    (digitIndex : Nat) (digitBounded : digitIndex < digitCount) :
    assignment
        (rawArtifact.opsLeaf.primary.wordStart fieldIndex + digitIndex) =
      canonicalDigit
        (assignment
          (rawArtifact.opsLeaf.primary.sourceColumns.getD fieldIndex 0))
        digitIndex := by
  exact opening_digit_exact rawArtifact.opsLeaf.primary fieldIndex assignment
    canonical one
    (opsPrimaryOpening_satisfied assignment satisfied fieldIndex fieldBounded)
    digitIndex digitBounded

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound
