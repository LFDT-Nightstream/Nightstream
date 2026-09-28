import Batteries.Data.Fin.Coding
import NightstreamFPrime.Layout.ProductionRelation

/-!
Owns ordered composition for production 14-matrix plans. `append` places one
plan after another. `indexed` places fixed-size blocks in block-major order.
Both constructors select only sparse forms and do not materialize matrices.
-/

namespace NightstreamFPrime.Layout.ProductionRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

namespace Plan

/-- Exact live-row acceptance predicate for any production plan. -/
def RowsZero {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin plan.rowCount,
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (plan.rowImage assignment (plan.rowLayout.toVertex row)) = 0

/-- On a live row, `rowImage` is exactly the selected sparse-form evaluation. -/
theorem rowImage_toVertex {logicalWidth : Nat}
    (plan : ProductionRelation.Plan logicalWidth)
    (assignment : Assignment F logicalWidth) (row : Fin plan.rowCount) :
    plan.rowImage assignment (plan.rowLayout.toVertex row) =
      fun port => (plan.portForm row port).eval assignment := by
  funext port
  unfold ProductionRelation.Plan.rowImage
  rw [plan.rowLayout.toColumn_toVertex]

def splitIndex (leftCount rightCount : Nat)
    (row : Fin (leftCount + rightCount)) :
    Fin leftCount ⊕ Fin rightCount :=
  if left : row.val < leftCount then
    Sum.inl ⟨row.val, left⟩
  else
    Sum.inr ⟨row.val - leftCount, by omega⟩

def leftIndex (leftCount rightCount : Nat) (row : Fin leftCount) :
    Fin (leftCount + rightCount) :=
  ⟨row.val, by omega⟩

def rightIndex (leftCount rightCount : Nat) (row : Fin rightCount) :
    Fin (leftCount + rightCount) :=
  ⟨leftCount + row.val, by omega⟩

@[simp] theorem leftIndex_val (leftCount rightCount : Nat)
    (row : Fin leftCount) :
    (leftIndex leftCount rightCount row).val = row.val := by
  rfl

@[simp] theorem rightIndex_val (leftCount rightCount : Nat)
    (row : Fin rightCount) :
    (rightIndex leftCount rightCount row).val = leftCount + row.val := by
  rfl

theorem leftIndex_of_splitIndex_eq (leftCount rightCount : Nat)
    (global : Fin (leftCount + rightCount)) (row : Fin leftCount)
    (decoded : splitIndex leftCount rightCount global = Sum.inl row) :
    leftIndex leftCount rightCount row = global := by
  unfold splitIndex at decoded
  split at decoded
  · cases decoded
    rfl
  · simp at decoded

theorem rightIndex_of_splitIndex_eq (leftCount rightCount : Nat)
    (global : Fin (leftCount + rightCount)) (row : Fin rightCount)
    (decoded : splitIndex leftCount rightCount global = Sum.inr row) :
    rightIndex leftCount rightCount row = global := by
  unfold splitIndex at decoded
  split at decoded
  · simp at decoded
  · cases decoded
    apply Fin.ext
    simp only [rightIndex_val]
    omega

/-- Canonical ordered concatenation of two plan row families. -/
def append {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth where
  rowCount := left.rowCount + right.rowCount
  rowCount_le := rowCount_le
  forms := fun row port =>
    match splitIndex left.rowCount right.rowCount row with
    | Sum.inl leftRow => left.forms leftRow port
    | Sum.inr rightRow => right.forms rightRow port

@[simp] theorem append_rowCount {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    (append left right rowCount_le).rowCount =
      left.rowCount + right.rowCount := by
  rfl

theorem append_forms_left {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (row : Fin left.rowCount)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (append left right rowCount_le).forms
        (leftIndex left.rowCount right.rowCount row) port =
      left.forms row port := by
  simp [append, splitIndex, leftIndex]

theorem append_forms_right {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (row : Fin right.rowCount)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (append left right rowCount_le).forms
        (rightIndex left.rowCount right.rowCount row) port =
      right.forms row port := by
  simp [append, splitIndex, rightIndex]

theorem append_portForm_left {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (row : Fin left.rowCount) (port : Fin Spec.ProductionRelation.matrixCount) :
    (append left right rowCount_le).portForm
        (leftIndex left.rowCount right.rowCount row) port =
      left.portForm row port := by
  unfold ProductionRelation.Plan.portForm
  split
  · exact append_forms_left left right rowCount_le row _
  · rfl

theorem append_portForm_right {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (row : Fin right.rowCount) (port : Fin Spec.ProductionRelation.matrixCount) :
    (append left right rowCount_le).portForm
        (rightIndex left.rowCount right.rowCount row) port =
      right.portForm row port := by
  unfold ProductionRelation.Plan.portForm
  split
  · exact append_forms_right left right rowCount_le row _
  · rfl

theorem append_residual_left {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) (row : Fin left.rowCount) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((append left right rowCount_le).rowImage assignment
          ((append left right rowCount_le).rowLayout.toVertex
            (leftIndex left.rowCount right.rowCount row))) =
      evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        (left.rowImage assignment (left.rowLayout.toVertex row)) := by
  rw [rowImage_toVertex, rowImage_toVertex]
  apply congrArg (evaluatePolynomial baseOps Spec.ProductionRelation.polynomial)
  funext port
  rw [append_portForm_left]

theorem append_residual_right {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) (row : Fin right.rowCount) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((append left right rowCount_le).rowImage assignment
          ((append left right rowCount_le).rowLayout.toVertex
            (rightIndex left.rowCount right.rowCount row))) =
      evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        (right.rowImage assignment (right.rowLayout.toVertex row)) := by
  rw [rowImage_toVertex, rowImage_toVertex]
  apply congrArg (evaluatePolynomial baseOps Spec.ProductionRelation.polynomial)
  funext port
  rw [append_portForm_right]

/-- Appended live rows vanish exactly when both child plans vanish. -/
theorem append_rowsZero_iff {logicalWidth : Nat}
    (left right : ProductionRelation.Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) :
    (append left right rowCount_le).RowsZero assignment ↔
      left.RowsZero assignment ∧ right.RowsZero assignment := by
  constructor
  · intro all
    constructor
    · intro row
      rw [← append_residual_left left right rowCount_le assignment row]
      exact all (leftIndex left.rowCount right.rowCount row)
    · intro row
      rw [← append_residual_right left right rowCount_le assignment row]
      exact all (rightIndex left.rowCount right.rowCount row)
  · rintro ⟨leftZero, rightZero⟩ row
    cases selected : splitIndex left.rowCount right.rowCount row with
    | inl leftRow =>
        have rowEqual : row = leftIndex left.rowCount right.rowCount leftRow := by
          unfold splitIndex at selected
          split at selected
          next inside =>
            simp only [Sum.inl.injEq] at selected
            subst leftRow
            apply Fin.ext
            rfl
          next outside => simp at selected
        subst row
        rw [append_residual_left]
        exact leftZero leftRow
    | inr rightRow =>
        have rowEqual : row =
            rightIndex left.rowCount right.rowCount rightRow := by
          unfold splitIndex at selected
          split at selected
          next inside => simp at selected
          next outside =>
            simp only [Sum.inr.injEq] at selected
            subst rightRow
            apply Fin.ext
            simp [rightIndex]
            omega
        subst row
        rw [append_residual_right]
        exact rightZero rightRow

/-- Invocation-major fixed-size plan. -/
def indexed {logicalWidth blockCount rowCount : Nat}
    (forms : Fin blockCount → Fin rowCount →
      Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth)
    (rowCount_le : blockCount * rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth where
  rowCount := blockCount * rowCount
  rowCount_le := rowCount_le
  forms := fun row port =>
    let decoded : Fin blockCount × Fin rowCount := Fin.decodeProd row
    forms decoded.1 decoded.2 port

def indexedRow (blockCount rowCount : Nat)
    (block : Fin blockCount) (row : Fin rowCount) :
    Fin (blockCount * rowCount) :=
  Fin.encodeProd (block, row)

@[simp] theorem indexed_rowCount {logicalWidth blockCount rowCount : Nat}
    (forms : Fin blockCount → Fin rowCount →
      Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth)
    (rowCount_le : blockCount * rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    (indexed forms rowCount_le).rowCount = blockCount * rowCount := by
  rfl

theorem indexed_forms {logicalWidth blockCount rowCount : Nat}
    (forms : Fin blockCount → Fin rowCount →
      Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth)
    (rowCount_le : blockCount * rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (block : Fin blockCount) (row : Fin rowCount)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (indexed forms rowCount_le).forms
        (indexedRow blockCount rowCount block row) port =
      forms block row port := by
  simp [indexed, indexedRow]

end Plan

end NightstreamFPrime.Layout.ProductionRelation
