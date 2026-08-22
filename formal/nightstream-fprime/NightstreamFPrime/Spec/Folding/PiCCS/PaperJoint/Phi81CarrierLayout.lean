import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/Phi81CarrierLayout.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Two-width Phi81 carrier layout for the SuperNeo coefficient embedding.

Protocol: SuperNeo coefficient embedding (Section 5, Definitions 7 and 8).
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

Authority boundary: the original matrix and fresh assignment determine their
completed values. No caller supplies the fresh suffix. Once a ring-linear
combination produces a carried CE assignment, however, every coordinate in
`carrierWidth` is authoritative; this module deliberately does not truncate
it back to `logicalWidth`.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | completion shape | `m` / `m_eff` | `m_eff = ceil(m / 54) * 54` |
| coefficient embedding | paper dimension | `n_F` / `n_R` | `carrierWidth = 54 * blockCount` |
| fresh CCS | assignment completion | logical prefix | original assignment entries are preserved exactly |
| fresh CCS | assignment completion | suffix | fresh tail entries are canonical zero |
| CCS structure | matrix completion | logical prefix / suffix | matrix is preserved then zero-extended |
| carried CE | carrier domain | all completed coordinates | the 54-lane layout has no absent coordinate |
| carried CE | carrier indexing | block / lane | `carrierColumn` is total and decodes to the exact source pair |
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

/-- The completed carrier is a whole number of Phi81 coefficient blocks. -/
theorem carrierWidth_eq (logicalWidth : Nat) :
    carrierWidth logicalWidth = blockCount logicalWidth * ringDegree := by
  rfl

/-- Ceiling completion never removes an original logical column. -/
theorem logicalWidth_le_carrierWidth (logicalWidth : Nat) :
    logicalWidth ≤ carrierWidth logicalWidth := by
  simp [carrierWidth, blockCount, ringDegree]
  omega

/-- Reapplying ceiling division to an already completed carrier preserves the
original block count. -/
theorem blockCount_carrierWidth (logicalWidth : Nat) :
    blockCount (carrierWidth logicalWidth) = blockCount logicalWidth := by
  simp [carrierWidth, blockCount, ringDegree]
  omega

/-- The total layout of the completed carrier. Unlike the partial layout at
the original width, every block/lane pair represents a semantic CE column. -/
def layout (logicalWidth : Nat) :
    RingColumnLayout ringDegree
      (blockCount (carrierWidth logicalWidth))
      (carrierWidth logicalWidth) :=
  Phi81ColumnLayout.layout (carrierWidth logicalWidth)

/-- Embed an original column into the completed carrier prefix. -/
def embedLogical
    {logicalWidth : Nat}
    (column : Fin logicalWidth) : Fin (carrierWidth logicalWidth) :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt
    (logicalWidth_le_carrierWidth logicalWidth)⟩

/-- Recover an original column exactly when a carrier coordinate lies in the
original prefix. -/
def logicalColumn?
    {logicalWidth : Nat}
    (column : Fin (carrierWidth logicalWidth)) : Option (Fin logicalWidth) :=
  if inLogicalRange : column.val < logicalWidth then
    some ⟨column.val, inLogicalRange⟩
  else
    none

/-- Original columns survive completion and prefix recognition exactly. -/
theorem logicalColumn?_embedLogical
    {logicalWidth : Nat}
    (column : Fin logicalWidth) :
    logicalColumn? (embedLogical column) = some column := by
  simp [logicalColumn?, embedLogical, column.isLt]

/-- Fresh assignment completion: preserve the original prefix and assign
canonical zero to the suffix. -/
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

/-- Every original assignment coordinate is preserved exactly. -/
theorem extendAssignment_embedLogical
    {Value : Type uValue}
    (zero : Value)
    {logicalWidth : Nat}
    (assignment : Assignment Value logicalWidth)
    (column : Fin logicalWidth) :
    extendAssignment zero assignment (embedLogical column) =
      assignment column := by
  simp [extendAssignment, logicalColumn?_embedLogical]

/-- Every completed coordinate outside the original width is zero in a fresh
assignment. -/
theorem extendAssignment_tail_zero
    {Value : Type uValue}
    (zero : Value)
    {logicalWidth : Nat}
    (assignment : Assignment Value logicalWidth)
    (column : Fin (carrierWidth logicalWidth))
    (tail : logicalWidth ≤ column.val) :
    extendAssignment zero assignment column = zero := by
  simp [extendAssignment, logicalColumn?, Nat.not_lt.mpr tail]

/-- Matrix completion uses the same prefix/suffix rule as a fresh assignment. -/
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

/-- Every original matrix column is preserved exactly. -/
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

/-- Every matrix entry beyond the original width is canonical zero. -/
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

/-- Every completed block/lane coordinate lies inside the carrier width. -/
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
    simpa [ringDegree] using coefficient.isLt
  change block.val * 54 + coefficient.val < blockCount logicalWidth * 54
  omega

/-- The sole typed block/lane-to-carrier map for a completed Phi81
assignment. Protocol layers should reuse this owner instead of rebuilding the
same bound from a relation-specific shape. -/
def carrierColumn
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) : Fin (carrierWidth logicalWidth) :=
  ⟨Phi81ColumnLayout.flatIndex block coefficient,
    flatIndex_lt_carrierWidth block coefficient⟩

/-- No block/lane coordinate is absent at the completed carrier width. -/
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

/-- Decoding the canonical completed-carrier coordinate recovers its exact
block/lane pair. -/
theorem decode_carrierColumn
    {logicalWidth : Nat}
    (block : Fin (blockCount (carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) :
    Phi81ColumnLayout.decode (carrierColumn block coefficient) =
      (block, coefficient) := by
  apply Phi81ColumnLayout.decode_encode
  simpa [carrierColumn, layout,
    MatrixCoefficientSource.RingColumnLayout.encode?] using
    (layout_encode?_isSome block coefficient)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
