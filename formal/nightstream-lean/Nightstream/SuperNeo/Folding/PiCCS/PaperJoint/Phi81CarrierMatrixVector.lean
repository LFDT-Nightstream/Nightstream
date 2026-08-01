import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout

/-!
Contract: preserve Goldilocks matrix images through Phi81 carrier completion.

Assurance tier: model-level.

Owns: exact equality between a logical-width matrix image and the image after
the matrix and assignment receive the same canonical zero suffix.

Does not own: public-column placement, a constraint polynomial, a protocol
program, commitments, Rust, or constraint counts.

Emits constraints: no.

| Owner | Equation or boundary |
|---|---|
| This module | `matrixVectorAt (extendMatrix 0 M) (extendAssignment 0 z) = matrixVectorAt M z` |
| Caller | public placement, CCS polynomial, and protocol acceptance |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierMatrixVector

open Nightstream.SuperNeo.Concrete
open MatrixCoefficientSource
open PaperLinearAlgebra
open ConcreteCarrier

private theorem sumRange_zero (term : Nat → F) :
    sumRange baseOps 0 term = 0 := by
  rfl

private theorem sumRange_succ (count : Nat) (term : Nat → F) :
    sumRange baseOps (count + 1) term =
      sumRange baseOps count term + term count := by
  rfl

private theorem sumRange_append
    (leftCount rightCount : Nat) (term : Nat → F) :
    sumRange baseOps (leftCount + rightCount) term =
      sumRange baseOps leftCount term +
        sumRange baseOps rightCount
          (fun index => term (leftCount + index)) := by
  induction rightCount with
  | zero =>
      rw [Nat.add_zero, sumRange_zero]
      exact (baseLaws.add_zero _).symm
  | succ rightCount inductionHypothesis =>
      rw [Nat.add_succ, sumRange_succ, sumRange_succ,
        inductionHypothesis]
      exact Lean.Grind.Fin.add_assoc _ _ _

private theorem sumRange_count_congr
    (leftCount rightCount : Nat) (term : Nat → F)
    (countsEqual : leftCount = rightCount) :
    sumRange baseOps leftCount term =
      sumRange baseOps rightCount term := by
  exact congrArg (fun count => sumRange baseOps count term) countsEqual

private def listSum {Index : Type}
    (indices : List Index) (term : Index → F) : F :=
  match indices with
  | [] => 0
  | index :: rest => term index + listSum rest term

private theorem foldl_eq_add_listSum
    {Index : Type} (indices : List Index) (term : Index → F)
    (initial : F) :
    indices.foldl (fun accumulated index => accumulated + term index) initial =
      initial + listSum indices term := by
  induction indices generalizing initial with
  | nil => exact (baseLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact baseLaws.add_assoc _ _ _

private theorem listSum_map
    {Left Right : Type} (indices : List Left) (map : Left → Right)
    (term : Right → F) :
    listSum (indices.map map) term =
      listSum indices (fun index => term (map index)) := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, listSum, inductionHypothesis]

private theorem listSum_append
    {Index : Type} (left right : List Index) (term : Index → F) :
    listSum (left ++ right) term =
      listSum left term + listSum right term := by
  induction left with
  | nil => exact (baseLaws.zero_add _).symm
  | cons index left inductionHypothesis =>
      simp only [List.cons_append, listSum, inductionHypothesis]
      exact (baseLaws.add_assoc _ _ _).symm

private theorem listSum_range_eq_sumRange
    (count : Nat) (term : Nat → F) :
    listSum (List.range count) term =
      sumRange baseOps count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, listSum_append, listSum, inductionHypothesis,
        sumRange_succ]
      rw [listSum]
      rw [Fin.add_zero]

private def matrixVectorTerm
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables) (index : Nat) : F :=
  if indexLt : index < columns then
    matrix vertex ⟨index, indexLt⟩ * assignment ⟨index, indexLt⟩
  else
    0

private theorem matrixVectorAt_eq_sumRange
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt baseOps matrix assignment vertex =
      sumRange baseOps columns
        (matrixVectorTerm matrix assignment vertex) := by
  unfold matrixVectorAt
  calc
    (canonicalFinIndices columns).foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * assignment column) 0 =
      0 + listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
          foldl_eq_add_listSum _ _ 0
    _ = listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
      baseLaws.zero_add _
    _ = listSum ((canonicalFinIndices columns).map Fin.val)
        (matrixVectorTerm matrix assignment vertex) := by
      rw [listSum_map]
      apply congrArg (listSum (canonicalFinIndices columns))
      funext column
      simp [matrixVectorTerm, column.isLt]
    _ = listSum (List.range columns)
        (matrixVectorTerm matrix assignment vertex) := by
      rw [canonicalFinIndices_values]
    _ = _ := listSum_range_eq_sumRange _ _

private theorem completedTerm_prefix
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment : Assignment F logicalWidth)
    (vertex : BooleanVertex variables)
    (index : Nat) (indexLt : index < logicalWidth) :
    matrixVectorTerm (Phi81CarrierLayout.extendMatrix 0 matrix)
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex index =
      matrixVectorTerm matrix assignment vertex index := by
  have carrierBound :
      index < Phi81CarrierLayout.carrierWidth logicalWidth :=
    Nat.lt_of_lt_of_le indexLt
      (Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth)
  let logicalColumn : Fin logicalWidth := ⟨index, indexLt⟩
  let carrierColumn : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) :=
    ⟨index, carrierBound⟩
  have mapped :
      carrierColumn = Phi81CarrierLayout.embedLogical logicalColumn := by
    apply Fin.ext
    rfl
  simp only [matrixVectorTerm, dif_pos indexLt, dif_pos carrierBound]
  change
    Phi81CarrierLayout.extendMatrix 0 matrix vertex carrierColumn *
        Phi81CarrierLayout.extendAssignment 0 assignment carrierColumn =
      matrix vertex logicalColumn * assignment logicalColumn
  rw [mapped, Phi81CarrierLayout.extendMatrix_embedLogical,
    Phi81CarrierLayout.extendAssignment_embedLogical]

private theorem completedTerm_tail
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment : Assignment F logicalWidth)
    (vertex : BooleanVertex variables)
    (offset : Nat)
    (offsetLt :
      offset <
        Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth) :
    matrixVectorTerm (Phi81CarrierLayout.extendMatrix 0 matrix)
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex
        (logicalWidth + offset) = 0 := by
  have logicalFits :
      logicalWidth ≤ Phi81CarrierLayout.carrierWidth logicalWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth
  have carrierBound :
      logicalWidth + offset <
        Phi81CarrierLayout.carrierWidth logicalWidth := by
    omega
  let column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) :=
    ⟨logicalWidth + offset, carrierBound⟩
  have isCompletion : logicalWidth ≤ column.val := by
    simp only [column]
    omega
  simp only [matrixVectorTerm, dif_pos carrierBound]
  rw [Phi81CarrierLayout.extendMatrix_tail_zero 0 matrix vertex column
      isCompletion,
    Phi81CarrierLayout.extendAssignment_tail_zero 0 assignment column
      isCompletion,
    Fin.zero_mul]

/-- Joint zero completion of a Goldilocks matrix and assignment preserves the
exact canonical finite matrix image. -/
theorem matrixVectorAt_extend_eq
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment : Assignment F logicalWidth)
    (vertex : BooleanVertex variables) :
    matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 matrix)
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex =
      matrixVectorAt baseOps matrix assignment vertex := by
  rw [matrixVectorAt_eq_sumRange]
  have logicalFits :
      logicalWidth ≤ Phi81CarrierLayout.carrierWidth logicalWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth
  have carrierWidthEq :
      Phi81CarrierLayout.carrierWidth logicalWidth =
        logicalWidth +
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth) := by
    omega
  have prefixSum :
      sumRange baseOps logicalWidth
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) =
        sumRange baseOps logicalWidth
          (matrixVectorTerm matrix assignment vertex) := by
    apply sumRange_congr
    intro index indexLt
    exact completedTerm_prefix matrix assignment vertex index indexLt
  have tailSum :
      sumRange baseOps
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth)
          (fun offset =>
            matrixVectorTerm
              (Phi81CarrierLayout.extendMatrix 0 matrix)
              (Phi81CarrierLayout.extendAssignment 0 assignment) vertex
              (logicalWidth + offset)) = 0 := by
    apply sumRange_eq_zero baseOps baseLaws
    intro offset offsetLt
    exact completedTerm_tail matrix assignment vertex offset offsetLt
  calc
    sumRange baseOps (Phi81CarrierLayout.carrierWidth logicalWidth)
        (matrixVectorTerm
          (Phi81CarrierLayout.extendMatrix 0 matrix)
          (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) =
      sumRange baseOps
          (logicalWidth +
            (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth))
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) := by
        exact sumRange_count_congr _ _ _ carrierWidthEq
    _ =
      sumRange baseOps logicalWidth
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) +
        sumRange baseOps
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth)
          (fun offset =>
            matrixVectorTerm
              (Phi81CarrierLayout.extendMatrix 0 matrix)
              (Phi81CarrierLayout.extendAssignment 0 assignment) vertex
              (logicalWidth + offset)) := by
        rw [sumRange_append]
    _ = sumRange baseOps logicalWidth
        (matrixVectorTerm matrix assignment vertex) := by
      rw [prefixSum, tailSum, Fin.add_zero]
    _ = matrixVectorAt baseOps matrix assignment vertex := by
      rw [matrixVectorAt_eq_sumRange]

private theorem arbitraryCompletedTerm_prefix
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment :
      Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (vertex : BooleanVertex variables)
    (index : Nat) (indexLt : index < logicalWidth) :
    matrixVectorTerm (Phi81CarrierLayout.extendMatrix 0 matrix)
        assignment vertex index =
      matrixVectorTerm matrix
        (fun logical =>
          assignment (Phi81CarrierLayout.embedLogical logical))
        vertex index := by
  have carrierBound :
      index < Phi81CarrierLayout.carrierWidth logicalWidth :=
    Nat.lt_of_lt_of_le indexLt
      (Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth)
  let logicalColumn : Fin logicalWidth := ⟨index, indexLt⟩
  let carrierColumn : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) :=
    ⟨index, carrierBound⟩
  have mapped :
      carrierColumn = Phi81CarrierLayout.embedLogical logicalColumn := by
    apply Fin.ext
    rfl
  simp only [matrixVectorTerm, dif_pos indexLt, dif_pos carrierBound]
  change
    Phi81CarrierLayout.extendMatrix 0 matrix vertex carrierColumn *
        assignment carrierColumn =
      matrix vertex logicalColumn *
        assignment (Phi81CarrierLayout.embedLogical logicalColumn)
  rw [mapped, Phi81CarrierLayout.extendMatrix_embedLogical]

private theorem arbitraryCompletedTerm_tail
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment :
      Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (vertex : BooleanVertex variables)
    (offset : Nat)
    (offsetLt :
      offset <
        Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth) :
    matrixVectorTerm (Phi81CarrierLayout.extendMatrix 0 matrix)
        assignment vertex (logicalWidth + offset) = 0 := by
  have logicalFits :
      logicalWidth ≤ Phi81CarrierLayout.carrierWidth logicalWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth
  have carrierBound :
      logicalWidth + offset <
        Phi81CarrierLayout.carrierWidth logicalWidth := by
    omega
  let column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) :=
    ⟨logicalWidth + offset, carrierBound⟩
  have isCompletion : logicalWidth ≤ column.val := by
    simp only [column]
    omega
  simp only [matrixVectorTerm, dif_pos carrierBound]
  rw [Phi81CarrierLayout.extendMatrix_tail_zero 0 matrix vertex column
      isCompletion,
    Fin.zero_mul]

/-- A completed matrix reads only the logical prefix of an arbitrary carrier
assignment. The assignment's completion suffix is not required to be zero. -/
theorem matrixVectorAt_extendMatrix_eq
    {variables logicalWidth : Nat}
    (matrix : BooleanMatrix F variables logicalWidth)
    (assignment :
      Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (vertex : BooleanVertex variables) :
    matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 matrix) assignment vertex =
      matrixVectorAt baseOps matrix
        (fun logical =>
          assignment (Phi81CarrierLayout.embedLogical logical))
        vertex := by
  rw [matrixVectorAt_eq_sumRange]
  have logicalFits :
      logicalWidth ≤ Phi81CarrierLayout.carrierWidth logicalWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth
  have carrierWidthEq :
      Phi81CarrierLayout.carrierWidth logicalWidth =
        logicalWidth +
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth) := by
    omega
  have prefixSum :
      sumRange baseOps logicalWidth
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            assignment vertex) =
        sumRange baseOps logicalWidth
          (matrixVectorTerm matrix
            (fun logical =>
              assignment (Phi81CarrierLayout.embedLogical logical))
            vertex) := by
    apply sumRange_congr
    intro index indexLt
    exact arbitraryCompletedTerm_prefix
      matrix assignment vertex index indexLt
  have tailSum :
      sumRange baseOps
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth)
          (fun offset =>
            matrixVectorTerm
              (Phi81CarrierLayout.extendMatrix 0 matrix)
              assignment vertex (logicalWidth + offset)) = 0 := by
    apply sumRange_eq_zero baseOps baseLaws
    intro offset offsetLt
    exact arbitraryCompletedTerm_tail
      matrix assignment vertex offset offsetLt
  calc
    sumRange baseOps (Phi81CarrierLayout.carrierWidth logicalWidth)
        (matrixVectorTerm
          (Phi81CarrierLayout.extendMatrix 0 matrix)
          assignment vertex) =
      sumRange baseOps
          (logicalWidth +
            (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth))
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            assignment vertex) := by
        exact sumRange_count_congr _ _ _ carrierWidthEq
    _ =
      sumRange baseOps logicalWidth
          (matrixVectorTerm
            (Phi81CarrierLayout.extendMatrix 0 matrix)
            assignment vertex) +
        sumRange baseOps
          (Phi81CarrierLayout.carrierWidth logicalWidth - logicalWidth)
          (fun offset =>
            matrixVectorTerm
              (Phi81CarrierLayout.extendMatrix 0 matrix)
              assignment vertex (logicalWidth + offset)) := by
        rw [sumRange_append]
    _ =
      sumRange baseOps logicalWidth
        (matrixVectorTerm matrix
          (fun logical =>
            assignment (Phi81CarrierLayout.embedLogical logical))
          vertex) := by
      rw [prefixSum, tailSum, Fin.add_zero]
    _ =
      matrixVectorAt baseOps matrix
        (fun logical =>
          assignment (Phi81CarrierLayout.embedLogical logical))
        vertex := by
      rw [matrixVectorAt_eq_sumRange]

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierMatrixVector
