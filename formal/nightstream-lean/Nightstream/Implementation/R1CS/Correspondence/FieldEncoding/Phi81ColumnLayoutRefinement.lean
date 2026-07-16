import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

/-!
Fresh-assignment packing refinement of the Phi81 logical-column layout.

Protocol: SuperNeo fresh CCS / CE assignment representation.
Phase: executable Lean carrier to paper-derived column-layout correspondence.
Constraint family: flat assignment / 54-lane ring block / final zero padding.

Owns: equality of the executable `packAssignment` block count with the
paper-derived Phi81 layout; a cell-by-cell theorem saying that every packed
cell is either the uniquely encoded logical assignment entry or canonical
zero padding; and logical-column and padding projections of that theorem.

Does not own: carried or mixed-accumulator assignments, whose completed suffix
may be nonzero after folding; Rust `Mat` construction; Rust
`validate_superneo_witness_mat`; matrix-cache construction; the runtime
Gram-matrix inverse; R1CS lowering; row removal; or constraint counts.

Emits constraints: no.

Assurance tier: model-level. The theorem connects two independently owned Lean
definitions. It is not Rust-conformant until an exported Rust witness/matrix
trace is proved to instantiate `packAssignment` and this ordering.

Authority boundary: the flat assignment and its verifier-owned logical width
determine every freshly packed cell. No fresh padding value is accepted from a
caller. This theorem must not be reused to conclude that the corresponding
suffix of a carried CE remains zero.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| fresh CCS / CE | shape | block count | executable packing allocates exactly `ceil(length / 54)` blocks |
| fresh CCS / CE | logical cell | quotient / remainder | decoded logical column reads the same flat assignment entry |
| fresh CCS / CE | arbitrary cell | partial encoding | packed cell equals the matching logical entry or zero |
| fresh CCS / CE | padding | final block suffix | every layout hole is canonical zero |
-/

namespace Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FreshAssignmentPacking
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Phi81ColumnLayout

/-- The executable packed assignment and the paper-derived layout allocate the
same number of 54-lane blocks. -/
theorem packAssignment_length_eq_blockCount (assignment : List F) :
    (packAssignment assignment).length = blockCount assignment.length := by
  simp [packAssignment, blockCount]

/-- Exact packed cell selected by the paper-derived block/lane layout. -/
def packedCell
    (assignment : List F)
    (block : Fin (blockCount assignment.length))
    (coefficient : Fin ringDegree) : F :=
  (packAssignment assignment).getD block.val ringFZero coefficient

/-- Cell-by-cell refinement: present layout positions read their unique flat
assignment entry, while absent final-block positions read zero. -/
theorem packedCell_eq_layout
    (assignment : List F)
    (block : Fin (blockCount assignment.length))
    (coefficient : Fin ringDegree) :
    packedCell assignment block coefficient =
      match encode? block coefficient with
      | some column => assignment.getD column.val 0
      | none => 0 := by
  have blockLt : block.val < (packAssignment assignment).length := by
    rw [packAssignment_length_eq_blockCount]
    exact block.isLt
  by_cases inLogicalRange : flatIndex block coefficient < assignment.length
  · have encoded :
        encode? block coefficient =
          some ⟨flatIndex block coefficient, inLogicalRange⟩ := by
      simp [encode?, inLogicalRange]
    rw [encoded]
    unfold packedCell
    rw [packAssignment_block assignment block.val blockLt]
    rfl
  · have encoded : encode? block coefficient = none := by
      simp [encode?, inLogicalRange]
    rw [encoded]
    unfold packedCell
    apply packAssignment_padding_zero assignment block.val coefficient blockLt
    simpa [flatIndex] using Nat.le_of_not_gt inLogicalRange

/-- Every logical column selects exactly the same scalar in the executable
packed assignment. -/
theorem logicalColumn_exact
    (assignment : List F)
    (column : Fin assignment.length) :
    packedCell assignment (decode column).1 (decode column).2 =
      assignment.getD column.val 0 := by
  rw [packedCell_eq_layout, encode_decode]

/-- Every absent final-block position is zero in the executable packing. -/
theorem paddingCell_zero
    (assignment : List F)
    (block : Fin (blockCount assignment.length))
    (coefficient : Fin ringDegree)
    (padding : encode? block coefficient = none) :
    packedCell assignment block coefficient = 0 := by
  rw [packedCell_eq_layout, padding]

end Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement
