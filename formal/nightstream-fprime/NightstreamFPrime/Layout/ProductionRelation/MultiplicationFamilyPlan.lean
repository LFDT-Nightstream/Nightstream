import NightstreamFPrime.Layout.ProductionRelation.OrdinaryRow
import NightstreamFPrime.Layout.ProductionRelation.PlanComposition

/-!
Owns an indexed family of ordinary multiplication rows. Each row enforces
one exact `left * right = output` equation through the fixed production
polynomial.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure Interface (logicalWidth rowCount : Nat) where
  oneColumn : Fin logicalWidth
  left : Fin rowCount → SparseForm logicalWidth
  right : Fin rowCount → SparseForm logicalWidth
  output : Fin rowCount → SparseForm logicalWidth

def forms {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount) (row : Fin rowCount) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton interface.oneColumn 1
    a := interface.left row
    b := interface.right row
    c := interface.output row }

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
  unfold OrdinaryRow.Forms.portImages
  cases found : ProductionRelation.meaningfulPort? port with
  | none =>
      simp [ProductionRelation.Plan.portForm,
        OrdinaryRow.Forms.portForm, found]
  | some meaningful =>
      simp only [ProductionRelation.Plan.portForm,
        OrdinaryRow.Forms.portForm, found]
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

/-- The family rows vanish exactly for their indexed multiplication
equations. -/
theorem planRowsZero_iff {logicalWidth rowCount : Nat}
    (interface : Interface logicalWidth rowCount)
    (rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    (plan interface rowCount_le).RowsZero assignment ↔
      ∀ row,
        (interface.left row).eval assignment *
          (interface.right row).eval assignment =
            (interface.output row).eval assignment := by
  constructor
  · intro rowsZero row
    have zero := rowsZero row
    rw [plan_residual_at, OrdinaryRow.Forms.residual_eq] at zero
    simp [forms, one] at zero
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp zero
  · intro equations row
    rw [plan_residual_at, OrdinaryRow.Forms.residual_eq]
    simp [forms, one, equations row]

end NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan
