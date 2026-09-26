import NightstreamFPrime.Layout.ProductionRelation.PlanComposition
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedRows

/-!
Owns an invocation-major family of fixed 86-row Poseidon2 plans. One shared
constant-one column and one indexed set of input, retained S-box, and output
forms produce one actual 14-matrix plan.

This module does not select a concrete invocation schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Sparse forms supplied by one ordered family of Poseidon2 invocations. -/
structure Interface (logicalWidth invocationCount : Nat) where
  oneColumn : Fin logicalWidth
  input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth
  sboxOutput : Fin invocationCount →
    Fin PoseidonRetainedSlots.rows.length → SparseForm logicalWidth

def Interface.output {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  SparseLayer.external fun lane =>
    interface.sboxOutput invocation (PoseidonRetainedSlots.finalRow lane)

def invocationInterface {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.Interface logicalWidth :=
  { oneColumn := interface.oneColumn
    input := interface.input invocation
    sboxOutput := interface.sboxOutput invocation
    output := interface.output invocation }

theorem invocation_output_eq_trace {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) :
    (invocationInterface interface invocation).output =
      (PoseidonSboxPlan.trace (invocationInterface interface invocation)).state := by
  rw [PoseidonSboxPlan.trace_state_eq_directOutput]
  unfold PoseidonSboxPlan.directOutput invocationInterface Interface.output
  apply congrArg SparseLayer.external
  funext lane
  unfold PoseidonSboxPlan.fullOutput PoseidonSboxPlan.sboxOutputAt
  have bounded : 78 + lane.val < PoseidonRetainedSlots.rows.length :=
    (PoseidonRetainedSlots.finalRow lane).isLt
  rw [dif_pos bounded]
  rfl

def rowAt {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 86) :
    PoseidonSboxPlan.Row logicalWidth :=
  (PoseidonRetainedRows.rows (invocationInterface interface invocation)).get
    ⟨row.val, by
      rw [PoseidonRetainedRows.rows_length]
      exact row.isLt⟩

def rowForms {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 86)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  (rowAt interface invocation row).meaningfulForm port

/-- Exact invocation-major family plan. -/
def plan {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.indexed (rowForms interface) rowCount_le

@[simp] theorem plan_rowCount {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    (plan interface rowCount_le).rowCount = invocationCount * 86 := by
  rfl

theorem plan_rowImage_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 86) :
    (plan interface rowCount_le).rowImage assignment
        ((plan interface rowCount_le).rowLayout.toVertex
          (ProductionRelation.Plan.indexedRow invocationCount 86
            invocation row)) =
      (rowAt interface invocation row).portImages assignment := by
  rw [ProductionRelation.Plan.rowImage_toVertex]
  funext port
  unfold PoseidonSboxPlan.Row.portImages
  cases found : ProductionRelation.meaningfulPort? port with
  | none =>
      simp [ProductionRelation.Plan.portForm,
        PoseidonSboxPlan.Row.portForm, found]
  | some meaningful =>
      simp only [ProductionRelation.Plan.portForm,
        PoseidonSboxPlan.Row.portForm, found]
      rw [show (plan interface rowCount_le).forms
            (ProductionRelation.Plan.indexedRow invocationCount 86
              invocation row) meaningful =
          rowForms interface invocation row meaningful by
        exact ProductionRelation.Plan.indexed_forms
          (rowForms interface) rowCount_le invocation row meaningful]
      rfl

theorem plan_residual_at {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) (row : Fin 86) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex
            (ProductionRelation.Plan.indexedRow invocationCount 86
              invocation row))) =
      (rowAt interface invocation row).residual assignment := by
  rw [plan_rowImage_at]
  exact PoseidonSboxPlan.Row.polynomial_eq_residual _ _

theorem rowAt_mem {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (invocation : Fin invocationCount) (row : Fin 86) :
    rowAt interface invocation row ∈
      PoseidonRetainedRows.rows (invocationInterface interface invocation) := by
  unfold rowAt
  exact List.get_mem _ _

/-- Family rows vanish exactly when every indexed Poseidon2 child row
vanishes. -/
theorem planRowsZero_iff {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) :
    (plan interface rowCount_le).RowsZero assignment ↔
      ∀ invocation, PoseidonSboxPlan.RowsZero
        (invocationInterface interface invocation) assignment := by
  constructor
  · intro rowsZero invocation
    apply (PoseidonRetainedRows.rowsZero_iff _ assignment
      (invocation_output_eq_trace interface invocation)).mpr
    intro localRow member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    let row : Fin 86 := ⟨index.val, by
      exact Nat.lt_of_lt_of_eq index.isLt
        (PoseidonRetainedRows.rows_length _)⟩
    have rowEqual :
        rowAt interface invocation row =
          (PoseidonRetainedRows.rows
            (invocationInterface interface invocation)).get index := by
      unfold rowAt row
      congr 1
    rw [← rowEqual]
    rw [← plan_residual_at interface rowCount_le assignment invocation row]
    exact rowsZero
      (ProductionRelation.Plan.indexedRow invocationCount 86 invocation row)
  · intro children globalRow
    let decoded : Fin invocationCount × Fin 86 := Fin.decodeProd globalRow
    have encodedEqual :
        ProductionRelation.Plan.indexedRow invocationCount 86
          decoded.1 decoded.2 = globalRow := by
      unfold ProductionRelation.Plan.indexedRow decoded
      exact Fin.encodeProd_decodeProd globalRow
    rw [← encodedEqual]
    rw [plan_residual_at]
    exact (PoseidonRetainedRows.rowsZero_iff _ assignment
      (invocation_output_eq_trace interface decoded.1)).mp (children decoded.1)
      (rowAt interface decoded.1 decoded.2) (rowAt_mem interface decoded.1 decoded.2)

/-- Family acceptance forces the exact Poseidon2 permutation for every
indexed invocation. -/
theorem planRowsZero_implies_permute {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : (plan interface rowCount_le).RowsZero assignment)
    (invocation : Fin invocationCount) :
    List.ofFn (SparseLayer.evalState assignment
        (interface.output invocation)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          (interface.input invocation))) := by
  exact PoseidonSboxPlan.rowsZero_implies_permute
    (invocationInterface interface invocation) assignment one
      ((planRowsZero_iff interface rowCount_le assignment).mp
        rowsZero invocation)

/-- Exact child equations make every row of the family plan vanish. -/
theorem equations_imply_planRowsZero {logicalWidth invocationCount : Nat}
    (interface : Interface logicalWidth invocationCount)
    (rowCount_le : invocationCount * 86 ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (equations : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
          (invocationInterface interface invocation) assignment ∧
        PoseidonSboxPlan.OutputEquations
          (invocationInterface interface invocation) assignment) :
    (plan interface rowCount_le).RowsZero assignment := by
  apply (planRowsZero_iff interface rowCount_le assignment).mpr
  intro invocation
  exact PoseidonSboxPlan.rowsZero_of_equations
    (invocationInterface interface invocation) assignment one
      (equations invocation).1 (equations invocation).2

end NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan
