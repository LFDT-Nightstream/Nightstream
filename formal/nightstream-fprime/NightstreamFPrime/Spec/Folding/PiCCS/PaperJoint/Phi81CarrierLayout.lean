import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

/-!
Two-width Phi81 carrier layout for the SuperNeo coefficient embedding.

Protocol: SuperNeo coefficient embedding (v1.2 Section 5, Definitions 13 and 14).
Phase: original CCS width to exact `n_F = 54 * n_R` carrier completion.
Constraint family: logical column / complete ring block / carried tail lane.

Owns: the distinction between the original CCS column width and the complete
Phi81 coefficient-carrier width; the sole typed block/lane-to-carrier map;
zero extension of fresh assignments and matrices; exact preservation of every
original column; and recognition that all completed carrier columns are real
coordinates, even when they began as a fresh zero-extension suffix.

Does not own: the bar transform, matrix-vector multiplication, proof that Rust
uses this completion, mixed-accumulator construction, R1CS lowering, row
removal, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout

open NightstreamFPrime.Spec
open PaperLinearAlgebra
open MatrixCoefficientSource
open Phi81ColumnLayout

universe uValue

/-- Complete field-carrier width required by the paper equality
`n_F = d * n_R`. -/
def carrierWidth (logicalWidth : Nat) : Nat :=
  blockCount logicalWidth * ringDegree

theorem carrierWidth_eq (logicalWidth : Nat) :
    carrierWidth logicalWidth = blockCount logicalWidth * ringDegree := by
  rfl

theorem logicalWidth_le_carrierWidth (logicalWidth : Nat) :
    logicalWidth ≤ carrierWidth logicalWidth := by
  simp [carrierWidth, blockCount, ringDegree]
  omega

theorem blockCount_carrierWidth (logicalWidth : Nat) :
    blockCount (carrierWidth logicalWidth) = blockCount logicalWidth := by
  simp [carrierWidth, blockCount, ringDegree]
  omega

def layout (logicalWidth : Nat) :
    RingColumnLayout ringDegree
      (blockCount (carrierWidth logicalWidth))
      (carrierWidth logicalWidth) :=
  Phi81ColumnLayout.layout (carrierWidth logicalWidth)

def embedLogical
    {logicalWidth : Nat}
    (column : Fin logicalWidth) : Fin (carrierWidth logicalWidth) :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt
    (logicalWidth_le_carrierWidth logicalWidth)⟩

def logicalColumn?
    {logicalWidth : Nat}
    (column : Fin (carrierWidth logicalWidth)) : Option (Fin logicalWidth) :=
  if inLogicalRange : column.val < logicalWidth then
    some ⟨column.val, inLogicalRange⟩
  else
    none

theorem logicalColumn?_embedLogical
    {logicalWidth : Nat}
    (column : Fin logicalWidth) :
    logicalColumn? (embedLogical column) = some column := by
  simp [logicalColumn?, embedLogical, column.isLt]

def extendAssignment
    {Value : Type uValue}
    (zero : Value)
    {logicalWidth : Nat}
    (assignment : Assignment Value logicalWidth) :
    Assignment Value (carrierWidth logicalWidth) :=
  fun column =>
    match logicalColumn? column with
    | some logical => assignment logical
    | none => zero

theorem extendAssignment_embedLogical
    {Value : Type uValue}
    (zero : Value)
    {logicalWidth : Nat}
    (assignment : Assignment Value logicalWidth)
    (column : Fin logicalWidth) :
    extendAssignment zero assignment (embedLogical column) =
      assignment column := by
  simp [extendAssignment, logicalColumn?_embedLogical]

theorem extendAssignment_tail_zero
    {Value : Type uValue}
    (zero : Value)
    {logicalWidth : Nat}
    (assignment : Assignment Value logicalWidth)
    (column : Fin (carrierWidth logicalWidth))
    (tail : logicalWidth ≤ column.val) :
    extendAssignment zero assignment column = zero := by
  simp [extendAssignment, logicalColumn?, Nat.not_lt.mpr tail]

def extendMatrix
    {Value : Type uValue}
    (zero : Value)
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix Value variables logicalWidth) :
    BooleanMatrix Value variables (carrierWidth logicalWidth) :=
  fun vertex column =>
    match logicalColumn? column with
    | some logical => matrix vertex logical
    | none => zero

theorem extendMatrix_embedLogical
    {Value : Type uValue}
    (zero : Value)
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix Value variables logicalWidth)
    (vertex : BooleanVertex variables)
    (column : Fin logicalWidth) :
    extendMatrix zero matrix vertex (embedLogical column) =
      matrix vertex column := by
  simp [extendMatrix, logicalColumn?_embedLogical]

theorem extendMatrix_tail_zero
    {Value : Type uValue}
    (zero : Value)
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix Value variables logicalWidth)
    (vertex : BooleanVertex variables)
    (column : Fin (carrierWidth logicalWidth))
    (tail : logicalWidth ≤ column.val) :
    extendMatrix zero matrix vertex column = zero := by
  simp [extendMatrix, logicalColumn?, Nat.not_lt.mpr tail]

theorem flatIndex_lt_carrierWidth
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) :
    Phi81ColumnLayout.flatIndex block coefficient <
      carrierWidth logicalWidth := by
  have blockBound : block.val < blockCount logicalWidth := by
    calc
      block.val < blockCount (carrierWidth logicalWidth) := block.isLt
      _ = blockCount logicalWidth := blockCount_carrierWidth logicalWidth
  have coefficientBound : coefficient.val < 54 := by
    simp [ringDegree]
  change block.val * 54 + coefficient.val < blockCount logicalWidth * 54
  omega

def carrierColumn
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) : Fin (carrierWidth logicalWidth) :=
  ⟨Phi81ColumnLayout.flatIndex block coefficient,
    flatIndex_lt_carrierWidth block coefficient⟩

theorem layout_encode?_isSome
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) :
    (layout logicalWidth).encode? block coefficient =
      some ⟨Phi81ColumnLayout.flatIndex block coefficient,
        flatIndex_lt_carrierWidth block coefficient⟩ := by
  change Phi81ColumnLayout.encode? block coefficient = _
  unfold Phi81ColumnLayout.encode?
  rw [dif_pos (flatIndex_lt_carrierWidth block coefficient)]

theorem decode_carrierColumn
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) :
    Phi81ColumnLayout.decode (carrierColumn block coefficient) =
      (block, coefficient) := by
  apply Phi81ColumnLayout.decode_encode
  simpa [carrierColumn, layout,
    MatrixCoefficientSource.RingColumnLayout.encode?] using!
    (layout_encode?_isSome block coefficient)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
