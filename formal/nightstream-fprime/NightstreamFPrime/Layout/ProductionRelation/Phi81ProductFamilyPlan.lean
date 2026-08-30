import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
import NightstreamFPrime.Layout.ProductionRelation.PlanComposition

/-!
Owns an invocation-major family of fixed 34-row Phi81 product plans. Each
invocation selects its lane, input rings, prior value, output value, and 33
retained group-output forms. The family is one actual 14-matrix plan.

This module does not select a concrete Stage 1 invocation schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Sparse forms supplied by one ordered family of product invocations. -/
structure Interface (logicalWidth invocationCount : Nat) where
  oneColumn : Fin logicalWidth
  lane : Fin invocationCount → Fin ringDegree
  left : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  right : Fin invocationCount → Phi81ProductPlan.State logicalWidth
  groupOutput : Fin invocationCount → Fin 33 → SparseForm logicalWidth
  prior : Fin invocationCount → SparseForm logicalWidth
  output : Fin invocationCount → SparseForm logicalWidth

def groupOutputAt {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount)
    (group : Fin (ProductSumPlan.groups
      (Phi81ProductPlan.terms (interface.left invocation)
        (interface.right invocation) (interface.lane invocation))).length) :
    SparseForm logicalWidth :=
  interface.groupOutput invocation ⟨group.val, by
    simpa using group.isLt⟩

def laneInterface {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) :
    ProductSumPlan.Interface logicalWidth :=
  { oneColumn := interface.oneColumn
    terms := Phi81ProductPlan.terms (interface.left invocation)
      (interface.right invocation) (interface.lane invocation)
    groupOutput := groupOutputAt interface invocation
    prior := interface.prior invocation
    output := interface.output invocation }

@[simp] theorem laneRows_length {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) :
    (ProductSumPlan.rows (laneInterface interface invocation)).length = 34 := by
  simpa [laneInterface] using Phi81ProductPlan.rows_length
    interface.oneColumn (interface.left invocation) (interface.right invocation)
    (interface.lane invocation) (groupOutputAt interface invocation)
    (interface.prior invocation) (interface.output invocation)

def rowAt {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 34) :
    ProductSumPlan.Row logicalWidth :=
  (ProductSumPlan.rows (laneInterface interface invocation)).get
    ⟨row.val, by
      rw [laneRows_length]
      exact row.isLt⟩

def rowForms {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 34)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  (rowAt interface invocation row).meaningfulForm port

/-- Exact invocation-major family plan. -/
def plan {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.indexed (rowForms interface) rowCount_le

@[simp] theorem plan_rowCount {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    (plan interface rowCount_le).rowCount = invocationCount * 34 := by
  rfl

theorem plan_rowImage_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 34) :
    (plan interface rowCount_le).rowImage assignment
        ((plan interface rowCount_le).rowLayout.toVertex
          (ProductionRelation.Plan.indexedRow invocationCount 34
            invocation row)) =
      (rowAt interface invocation row).portImages assignment := by
  rw [ProductionRelation.Plan.rowImage_toVertex]
  funext port
  unfold ProductSumPlan.Row.portImages
  cases found : ProductionRelation.meaningfulPort? port with
  | none =>
      simp [ProductionRelation.Plan.portForm, ProductSumPlan.Row.portForm,
        found]
  | some meaningful =>
      simp only [ProductionRelation.Plan.portForm,
        ProductSumPlan.Row.portForm, found]
      rw [show (plan interface rowCount_le).forms
            (ProductionRelation.Plan.indexedRow invocationCount 34
              invocation row) meaningful =
          rowForms interface invocation row meaningful by
        exact ProductionRelation.Plan.indexed_forms
          (rowForms interface) rowCount_le invocation row meaningful]
      rfl

theorem plan_residual_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 34) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex
            (ProductionRelation.Plan.indexedRow invocationCount 34
              invocation row))) =
      (rowAt interface invocation row).residual assignment := by
  rw [plan_rowImage_at]
  exact ProductSumPlan.Row.polynomial_eq_residual _ _

theorem rowAt_mem {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 34) :
    rowAt interface invocation row ∈
      ProductSumPlan.rows (laneInterface interface invocation) := by
  unfold rowAt
  exact List.get_mem _ _

/-- The family plan vanishes exactly when every invocation satisfies its
grouped-product equations. -/
theorem planRowsZero_iff {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    (plan interface rowCount_le).RowsZero assignment ↔
      ∀ invocation,
        ProductSumPlan.Equations (laneInterface interface invocation)
          assignment := by
  constructor
  · intro rowsZero invocation
    apply (ProductSumPlan.rowsZero_iff_equations
      (laneInterface interface invocation) assignment one).mp
    intro localRow member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    let row : Fin 34 := ⟨index.val, by
      exact Nat.lt_of_lt_of_eq index.isLt
        (laneRows_length interface invocation)⟩
    have rowEqual :
        rowAt interface invocation row =
          (ProductSumPlan.rows (laneInterface interface invocation)).get
            index := by
      unfold rowAt row
      congr 1
    rw [← rowEqual]
    rw [← plan_residual_at interface rowCount_le assignment invocation row]
    exact rowsZero
      (ProductionRelation.Plan.indexedRow invocationCount 34 invocation row)
  · intro equations globalRow
    let decoded : Fin invocationCount × Fin 34 := Fin.decodeProd globalRow
    have encodedEqual :
        ProductionRelation.Plan.indexedRow invocationCount 34
          decoded.1 decoded.2 = globalRow := by
      unfold ProductionRelation.Plan.indexedRow decoded
      exact Fin.encodeProd_decodeProd globalRow
    rw [← encodedEqual]
    rw [plan_residual_at]
    have localRows :=
      (ProductSumPlan.rowsZero_iff_equations
        (laneInterface interface decoded.1) assignment one).mpr
          (equations decoded.1)
    exact localRows (rowAt interface decoded.1 decoded.2)
      (rowAt_mem interface decoded.1 decoded.2)

/-- Satisfying family rows force every exact `prior + ringFMul` output. -/
theorem planRowsZero_implies_ringProduct {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 34 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : (plan interface rowCount_le).RowsZero assignment)
    (invocation : Fin invocationCount) :
    (interface.output invocation).eval assignment =
      (interface.prior invocation).eval assignment +
        ringFMul
          (Phi81ProductPlan.evalState assignment (interface.left invocation))
          (Phi81ProductPlan.evalState assignment (interface.right invocation))
          (interface.lane invocation) := by
  have equations := (planRowsZero_iff interface rowCount_le assignment one).mp
    rowsZero invocation
  rw [← Phi81ProductPlan.terms_total]
  exact equations.final

end NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan
