import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
import NightstreamFPrime.Layout.ProductionRelation.PlanComposition

/-!
Owns an invocation-major family of 108-row Phi81 quotient product plans.
Each invocation owns one complete ring product, its prior and output rings,
and 54 retained quotient coefficients. Each row is one rank-one row of the
fixed gate polynomial.

This module does not select the concrete Stage 1 invocation schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Sparse forms for an ordered family of complete ring products. -/
structure Interface (logicalWidth invocationCount : Nat) where
  left : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  right : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  quotient : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  prior : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  output : Fin invocationCount → Phi81ProductPlan.State logicalWidth

def ringInterface {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) : Phi81ProductPlan.Interface logicalWidth :=
  { left := interface.left invocation
    right := interface.right invocation
    quotient := interface.quotient invocation
    prior := interface.prior invocation
    output := interface.output invocation }

def rowAt {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 108) :
    OrdinaryRow.Forms logicalWidth :=
  Phi81ProductPlan.rowAt (ringInterface interface invocation) row

def rowForms {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 108)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  (rowAt interface invocation row).meaningfulForm port

def plan {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.indexed (rowForms interface) rowCount_le

@[simp] theorem plan_rowCount {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    (plan interface rowCount_le).rowCount = invocationCount * 108 := by
  rfl

theorem plan_rowImage_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 108) :
    (plan interface rowCount_le).rowImage assignment
        ((plan interface rowCount_le).rowLayout.toVertex
          (ProductionRelation.Plan.indexedRow invocationCount 108
            invocation row)) =
      (rowAt interface invocation row).portImages assignment := by
  rw [ProductionRelation.Plan.rowImage_toVertex]
  funext port
  unfold OrdinaryRow.Forms.portImages
  cases found : ProductionRelation.meaningfulPort? port with
  | none =>
      simp [ProductionRelation.Plan.portForm, OrdinaryRow.Forms.portForm,
        found]
  | some meaningful =>
      simp only [ProductionRelation.Plan.portForm,
        OrdinaryRow.Forms.portForm, found]
      rw [show (plan interface rowCount_le).forms
            (ProductionRelation.Plan.indexedRow invocationCount 108
              invocation row) meaningful =
          rowForms interface invocation row meaningful by
        exact ProductionRelation.Plan.indexed_forms
          (rowForms interface) rowCount_le invocation row meaningful]
      rfl

theorem plan_residual_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 108) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex
            (ProductionRelation.Plan.indexedRow invocationCount 108
              invocation row))) =
      (rowAt interface invocation row).residual assignment := by
  rw [plan_rowImage_at]
  rfl

/-- The actual family rows encode exactly the quotient evaluation equations. -/
theorem planRowsZero_iff {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) :
    (plan interface rowCount_le).RowsZero assignment ↔
      ∀ invocation,
        Phi81ProductPlan.Equations (ringInterface interface invocation)
          assignment := by
  constructor
  · intro rowsZero invocation
    apply (Phi81ProductPlan.rowsZero_iff_equations
      (ringInterface interface invocation) assignment).mp
    intro row
    change (rowAt interface invocation row).residual assignment = 0
    rw [← plan_residual_at interface rowCount_le assignment invocation row]
    exact rowsZero
      (ProductionRelation.Plan.indexedRow invocationCount 108 invocation row)
  · intro equations globalRow
    let decoded : Fin invocationCount × Fin 108 := Fin.decodeProd globalRow
    have encodedEqual :
        ProductionRelation.Plan.indexedRow invocationCount 108
          decoded.1 decoded.2 = globalRow := by
      unfold ProductionRelation.Plan.indexedRow decoded
      exact Fin.encodeProd_decodeProd globalRow
    rw [← encodedEqual]
    rw [plan_residual_at]
    exact (Phi81ProductPlan.rowsZero_iff_equations
      (ringInterface interface decoded.1) assignment).mpr
        (equations decoded.1) decoded.2

/-- Satisfying family rows force every complete `prior + ringFMul` result. -/
theorem planRowsZero_implies_ringProduct {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 108 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (rowsZero : (plan interface rowCount_le).RowsZero assignment)
    (invocation : Fin invocationCount) :
    Phi81ProductPlan.evalState assignment (interface.output invocation) =
      ringFAdd
        (Phi81ProductPlan.evalState assignment (interface.prior invocation))
        (ringFMul
          (Phi81ProductPlan.evalState assignment (interface.left invocation))
          (Phi81ProductPlan.evalState assignment (interface.right invocation))) := by
  apply Phi81ProductPlan.rowsZero_implies_ringProduct
    (ringInterface interface invocation) assignment
  exact (Phi81ProductPlan.rowsZero_iff_equations
    (ringInterface interface invocation) assignment).mpr
      ((planRowsZero_iff interface rowCount_le assignment).mp rowsZero
        invocation)

end NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan
