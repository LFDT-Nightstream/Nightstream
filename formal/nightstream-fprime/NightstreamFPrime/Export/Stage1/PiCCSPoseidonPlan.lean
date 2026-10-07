import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.Retained

/-!
Owns the direct Poseidon2 plan for all four PiCCS transcript action families.
Each invocation uses retained S-box outputs, the previous invocation's
closed-form output, and the exact Lean action payload. No expanded invocation
list or recursive trace reconstruction is used by the plan.

This module does not close PiCCS status or bind the final package identity.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- The parent supplies the exact typed action-word forms. -/
abbrev Payload (logicalWidth : Nat) :=
  Fin PiCCSActionPayloadBlock.payloadCount → SparseForm logicalWidth

def payloadForm {logicalWidth : Nat} (payload : Payload logicalWidth)
    (invocation : Fin invocationCount) (lane : Fin Spec.Poseidon2.width) :
    SparseForm logicalWidth :=
  if rateLane : lane.val < Spec.Poseidon2.rate then
    payload (Fin.encodeProd (invocation, ⟨lane.val, rateLane⟩))
  else
    .empty

def inputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.State logicalWidth :=
  let previous := previousOutput geometry invocation
  match PiCCSActionPayloadBlock.kindAt invocation with
  | .absorb _ => fun lane =>
      SparseForm.add (previous lane) (payloadForm payload invocation lane)

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (schedule program)
    (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState payload geometry)

theorem familyRowCount_le : invocationCount * 150 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (interface payload geometry) familyRowCount_le

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    (plan payload geometry).rowCount = 142200 := by
  change invocationCount * 150 = 142200
  rw [invocationCount_eq]

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan payload geometry).RowsZero assignment ↔
      ∀ invocation, PoseidonSboxPlan.RowsZero
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface payload geometry) invocation) assignment := by
  rw [plan, PoseidonSboxFamilyPlan.planRowsZero_iff]

structure Semantics {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  invocation : ∀ current,
    List.ofFn (SparseLayer.evalState assignment
        ((interface payload geometry).output current)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          ((interface payload geometry).input current)))

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (rowsZero : (plan payload geometry).RowsZero assignment) :
    Semantics payload geometry assignment := by
  refine ⟨?_⟩
  intro invocation
  have sboxRows := (PoseidonSboxFamilyPlan.planRowsZero_iff
    (interface payload geometry) familyRowCount_le assignment).mpr
      ((rowsZero_iff payload geometry assignment).mp rowsZero)
  exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (interface payload geometry) familyRowCount_le assignment one sboxRows invocation

/-- Honest retained S-box equations are sufficient for all PiCCS Poseidon2
rows. Final-output equations are definitional custody checks. -/
theorem equations_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (sboxes : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface payload geometry) invocation) assignment) :
    (plan payload geometry).RowsZero assignment := by
  apply (rowsZero_iff payload geometry assignment).mpr
  intro invocation
  apply PoseidonSboxPlan.rowsZero_of_equations
    (PoseidonSboxFamilyPlan.invocationInterface
      (interface payload geometry) invocation) assignment
    (sboxes invocation)
  exact PoseidonRetainedFamily.outputEquations
    (schedule program) (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState payload geometry) assignment invocation

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
