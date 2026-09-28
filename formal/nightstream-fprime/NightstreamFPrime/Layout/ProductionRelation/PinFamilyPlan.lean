import NightstreamFPrime.Layout.ProductionRelation.PinRow
import NightstreamFPrime.Layout.ProductionRelation.PlanComposition

/-!
Owns an indexed family of zero-pin rows. Each row enforces one exact affine
form equal to zero through the fixed production polynomial.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure Interface (logicalWidth rowCount : Nat) where
  oneColumn : Fin logicalWidth
  value : Fin rowCount → SparseForm logicalWidth

def forms {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount) (row : Fin rowCount) :
    PinRow.Forms logicalWidth :=
  { selector := SparseForm.singleton interface.oneColumn 1
    value := interface.value row }

def plan {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount)
    (rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth where
  rowCount := rowCount
  rowCount_le := rowCount_le
  forms := fun row port => (forms interface row).meaningfulForm port

theorem plan_rowImage_at {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount)
    (rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) (row : Fin rowCount) :
    (plan interface rowCount_le).rowImage assignment
        ((plan interface rowCount_le).rowLayout.toVertex row) =
      (forms interface row).portImages assignment := by
  rw [ProductionRelation.Plan.rowImage_toVertex]
  funext port
  unfold PinRow.Forms.portImages
  cases found : ProductionRelation.meaningfulPort? port with
  | none =>
      simp [ProductionRelation.Plan.portForm, PinRow.Forms.portForm, found]
  | some meaningful =>
      simp only [ProductionRelation.Plan.portForm, PinRow.Forms.portForm, found]
      rfl

theorem plan_residual_at {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount)
    (rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) (row : Fin rowCount) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex row)) =
      (forms interface row).residual assignment := by
  rw [plan_rowImage_at]
  rfl

/-- The family rows vanish exactly when every indexed value is zero. -/
theorem planRowsZero_iff {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount)
    (rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    (plan interface rowCount_le).RowsZero assignment ↔
      ∀ row, (interface.value row).eval assignment = 0 := by
  constructor
  · intro rowsZero row
    have zero := rowsZero row
    rw [plan_residual_at, PinRow.Forms.residual_eq] at zero
    have negated := congrArg Neg.neg zero
    simpa [forms, one] using negated
  · intro equations row
    rw [plan_residual_at, PinRow.Forms.residual_eq]
    simp [forms, one, equations row]

end NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan
