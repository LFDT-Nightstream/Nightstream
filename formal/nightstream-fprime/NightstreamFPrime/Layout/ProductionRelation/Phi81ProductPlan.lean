import NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan
import NightstreamFPrime.Spec.Phi81Relation.QuotientProduct

/-!
Owns the direct quotient constraints for one Phi81 ring product. The 108
distinct evaluation points enforce the degree-at-most-107 polynomial identity
`left * right = output - prior + Phi81 * quotient`. Only the 54 quotient
coefficients are additional retained field values; evaluation is linear.

This module does not select Stage 1 source or assignment columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev State (logicalWidth : Nat) := Fin ringDegree → SparseForm logicalWidth

def evalState {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (state : State logicalWidth) : RingF :=
  fun lane => (state lane).eval assignment

/-- Sparse polynomial evaluation, without additional witness coordinates. -/
def evaluateForm {logicalWidth : Nat} (state : State logicalWidth) (point : F) :
    SparseForm logicalWidth :=
  ProductSumPlan.sumForms (List.ofFn fun lane : Fin ringDegree =>
    SparseForm.scale (point ^ lane.val) (state lane))

theorem evaluateForm_eval {logicalWidth : Nat}
    (state : State logicalWidth) (point : F)
    (assignment : Assignment F logicalWidth) :
    (evaluateForm state point).eval assignment =
      Phi81Relation.QuotientProduct.evaluate (evalState assignment state) point := by
  simp only [evaluateForm, ProductSumPlan.sumForms_eval, List.map_ofFn,
    Function.comp_def, SparseForm.scale_eval,
    Phi81Relation.QuotientProduct.evaluate, evalState, mul_comm]

/-- Forms for one complete ring product and its running sum. -/
structure Interface (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  left : State logicalWidth
  right : State logicalWidth
  quotient : State logicalWidth
  prior : State logicalWidth
  output : State logicalWidth

def outputForm {logicalWidth : Nat} (interface : Interface logicalWidth)
    (point : F) : SparseForm logicalWidth :=
  SparseForm.add
    (SparseForm.add (evaluateForm interface.output point)
      (SparseForm.scale (-1) (evaluateForm interface.prior point)))
    (SparseForm.scale (Phi81Relation.QuotientProduct.modulusValue point)
      (evaluateForm interface.quotient point))

def productRow {logicalWidth : Nat} (interface : Interface logicalWidth)
    (row : Fin 108) : ProductSumRow.Forms logicalWidth :=
  let point := Phi81Relation.QuotientProduct.node row
  { selector := SparseForm.singleton interface.oneColumn 1
    left := fun lane => if lane.val = 0 then evaluateForm interface.left point
      else .empty
    right := fun lane => if lane.val = 0 then evaluateForm interface.right point
      else .empty
    output := outputForm interface point }

def rowAt {logicalWidth : Nat} (interface : Interface logicalWidth)
    (row : Fin 108) : ProductSumPlan.Row logicalWidth :=
  .product (productRow interface row)

def rows {logicalWidth : Nat} (interface : Interface logicalWidth) :
    List (ProductSumPlan.Row logicalWidth) :=
  List.ofFn (rowAt interface)

@[simp] theorem rows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) : (rows interface).length = 108 := by
  simp only [rows, List.length_ofFn]

/-- The exact scalar polynomial identity at each fixed evaluation point. -/
def Equation {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (row : Fin 108) : Prop :=
  let point := Phi81Relation.QuotientProduct.node row
  Phi81Relation.QuotientProduct.evaluate
        (evalState assignment interface.left) point *
      Phi81Relation.QuotientProduct.evaluate
        (evalState assignment interface.right) point =
    Phi81Relation.QuotientProduct.evaluate
        (evalState assignment interface.output) point -
      Phi81Relation.QuotientProduct.evaluate
        (evalState assignment interface.prior) point +
      Phi81Relation.QuotientProduct.modulusValue point *
        Phi81Relation.QuotientProduct.evaluate
          (evalState assignment interface.quotient) point

def Equations {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin 108, Equation interface assignment row

def RowsZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin 108, (rowAt interface row).residual assignment = 0

theorem row_zero_iff {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) (row : Fin 108) :
    (rowAt interface row).residual assignment = 0 ↔
      Equation interface assignment row := by
  let point := Phi81Relation.QuotientProduct.node row
  have residual : (rowAt interface row).residual assignment =
      Phi81Relation.QuotientProduct.evaluate
          (evalState assignment interface.left) point *
        Phi81Relation.QuotientProduct.evaluate
          (evalState assignment interface.right) point -
        (Phi81Relation.QuotientProduct.evaluate
            (evalState assignment interface.output) point -
          Phi81Relation.QuotientProduct.evaluate
            (evalState assignment interface.prior) point +
          Phi81Relation.QuotientProduct.modulusValue point *
            Phi81Relation.QuotientProduct.evaluate
              (evalState assignment interface.quotient) point) := by
    change (productRow interface row).residual assignment = _
    rw [ProductSumRow.Forms.residual_eq]
    simp [productRow, Spec.ProductionRelation.RowSemantics.productTotal, outputForm,
      evaluateForm_eval, one, point, sub_eq_add_neg]
  rw [residual]
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff

theorem rowsZero_iff_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    RowsZero interface assignment ↔ Equations interface assignment := by
  unfold RowsZero Equations
  exact forall_congr' (fun row => row_zero_iff interface assignment one row)

/-- All fixed-point equations force the unchanged Phi81 multiplication. -/
theorem rowsZero_implies_ringProduct {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : RowsZero interface assignment) :
    evalState assignment interface.output =
      ringFAdd (evalState assignment interface.prior)
        (ringFMul (evalState assignment interface.left)
          (evalState assignment interface.right)) := by
  exact Phi81Relation.QuotientProduct.sound_add
    (evalState assignment interface.left) (evalState assignment interface.right)
    (evalState assignment interface.prior) (evalState assignment interface.output)
    (evalState assignment interface.quotient)
    ((rowsZero_iff_equations interface assignment one).mp rowsZero)

end NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
