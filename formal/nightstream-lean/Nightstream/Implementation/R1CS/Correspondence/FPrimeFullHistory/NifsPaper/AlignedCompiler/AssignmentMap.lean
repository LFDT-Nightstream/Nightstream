import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ColumnMap

/-!
Assignment-value refinement for the aligned F' compiler column map.

Owns: coordinate-by-coordinate preservation of every old assignment value and
the exact zero value of every newly inserted public-padding coordinate.

Does not own: matrix relocation, sparse storage, commitment setup, proof
serialization, generated R1CS rows, or production Rust conformance.

Emits constraints: no.

Authority boundary: an aligned assignment is constructed from the old
verifier-owned assignment. Padding values come from `List.replicate ... 0`;
they are not accepted from a prover-carried projection or digest.

| Protocol | Phase | Constraint family | Mathematical obligation | Result |
|---|---|---|---|---|
| F' / CCS | assignment lowering | old public values | old columns below 257 retain value and index | `getElem?_alignedIndex` |
| F' / CCS | assignment lowering | old private values | old columns from 257 retain value after the +13 shift | `getElem?_alignedIndex` |
| F' / CCS | assignment lowering | public padding | every new column in 257..269 is exactly zero | `getD_padding_zero` |
| SuperNeo | coefficient packing | scalar source | `packedCoeff` at the mapped block/lane reads the old scalar | `packedCoeff_alignedIndex` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap

/-- Exact source-value preservation at the old-to-aligned scalar index. -/
theorem getElem?_alignedIndex (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ assignment.length) (column : Nat) :
    (insertPublicPadding assignment)[alignedIndex column]? =
      assignment[column]? := by
  have takeLength :
      (assignment.take logicalPublicWidth).length = logicalPublicWidth := by
    simp [List.length_take, Nat.min_eq_left hasPublic]
  by_cases isPublic : column < logicalPublicWidth
  · rw [alignedIndex_public column isPublic]
    unfold insertPublicPadding
    rw [List.append_assoc]
    rw [List.getElem?_append_left]
    · exact List.getElem?_take_of_lt isPublic
    · simpa [takeLength] using isPublic
  · have isPrivate : logicalPublicWidth ≤ column := Nat.not_lt.mp isPublic
    rw [alignedIndex_private column isPrivate]
    unfold insertPublicPadding
    rw [List.append_assoc]
    rw [List.getElem?_append_right]
    · rw [takeLength]
      rw [List.getElem?_append_right]
      · rw [publicPadding_length]
        rw [List.getElem?_drop]
        have exactIndex :
            logicalPublicWidth +
                (column + paddingWidth - logicalPublicWidth - paddingWidth) =
              column := by
          simp [logicalPublicWidth, paddingWidth] at isPrivate ⊢
          omega
        rw [exactIndex]
      · rw [publicPadding_length]
        simp [logicalPublicWidth, paddingWidth] at isPrivate ⊢
        omega
    · rw [takeLength]
      simp [logicalPublicWidth, paddingWidth] at isPrivate ⊢
      omega

theorem getD_alignedIndex (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ assignment.length) (column : Nat) :
    (insertPublicPadding assignment).getD (alignedIndex column) 0 =
      assignment.getD column 0 := by
  simp only [List.getD_eq_getElem?_getD, getElem?_alignedIndex assignment
    hasPublic column]

/-- Every position in the aligned public-padding interval is definitionally
zero in the constructed assignment. -/
theorem getD_padding_zero (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ assignment.length) (column : Nat)
    (isPadding : logicalPublicWidth ≤ column ∧
      column < alignedPublicWidth) :
    (insertPublicPadding assignment).getD column 0 = 0 := by
  have takeLength :
      (assignment.take logicalPublicWidth).length = logicalPublicWidth := by
    simp [List.length_take, Nat.min_eq_left hasPublic]
  simp only [List.getD_eq_getElem?_getD]
  unfold insertPublicPadding
  rw [List.append_assoc]
  rw [List.getElem?_append_right]
  · rw [takeLength]
    rw [List.getElem?_append_left]
    · have paddingIndex :
          column - logicalPublicWidth < paddingWidth := by
        simp [logicalPublicWidth, alignedPublicWidth, paddingWidth]
          at isPadding ⊢
        omega
      simp [publicPadding, paddingIndex]
    · rw [publicPadding_length]
      simp [logicalPublicWidth, alignedPublicWidth, paddingWidth] at isPadding ⊢
      omega
  · rw [takeLength]
    exact isPadding.1

/-- The concrete assignment packer reads the preserved old scalar at the
mapped quotient/remainder block and lane. -/
theorem packedCoeff_alignedIndex (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ assignment.length) (column : Nat)
    (laneBound : packedLane column < ringDegree) :
    packedCoeff (insertPublicPadding assignment) (packedBlock column)
        ⟨packedLane column, laneBound⟩ =
      assignment.getD column 0 := by
  unfold packedCoeff
  rw [packedFlatIndex]
  exact getD_alignedIndex assignment hasPublic column

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap
