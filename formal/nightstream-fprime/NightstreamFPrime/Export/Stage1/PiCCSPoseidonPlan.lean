import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.Retained
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

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
  | .squeezeFirst _ => previous
  | .squeezeSecond => previous

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (schedule program)
    (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState payload geometry)

def bindingActual {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) (component : Fin 2) :
    SparseForm logicalWidth :=
  if component.val = 0 then
    previousOutput geometry invocation 0
  else
    outputState geometry invocation 0

def bindingForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) (component : Fin 2) :
    SparseForm logicalWidth :=
  match PiCCSActionPayloadBlock.kindAt invocation with
  | .squeezeFirst _ =>
      SparseForm.add
        (payloadForm payload invocation
          ⟨component.val, Nat.lt_trans component.isLt (by
            norm_num [Spec.Poseidon2.width])⟩)
        (SparseForm.scale (-1) (bindingActual geometry invocation component))
  | .absorb _ | .squeezeSecond => .empty

theorem bindingForm_squeezeFirst_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    bindingForm payload geometry invocation (0 : Fin 2) =
      SparseForm.add (payloadForm payload invocation (0 : Fin 8))
        (SparseForm.scale (-1) (previousOutput geometry invocation 0)) := by
  unfold bindingForm bindingActual
  rw [found]
  rfl

theorem bindingForm_squeezeFirst_one
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    bindingForm payload geometry invocation (1 : Fin 2) =
      SparseForm.add (payloadForm payload invocation (1 : Fin 8))
        (SparseForm.scale (-1) (outputState geometry invocation 0)) := by
  unfold bindingForm bindingActual
  rw [found]
  rfl

def bindingRowCount : Nat := invocationCount * 2

def bindingInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth bindingRowCount where
  oneColumn := oneColumn geometry
  value := fun row =>
    let decoded : Fin invocationCount × Fin 2 := Fin.decodeProd row
    bindingForm payload geometry decoded.1 decoded.2

theorem bindingRowCount_le : bindingRowCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [bindingRowCount, invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

theorem familyRowCount_le : invocationCount * 94 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def sboxPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (interface payload geometry) familyRowCount_le

def bindingPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PinFamilyPlan.plan (bindingInterface payload geometry) bindingRowCount_le

theorem combinedRowCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    (sboxPlan payload geometry).rowCount + (bindingPlan payload geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  change invocationCount * 94 + bindingRowCount ≤ _
  rw [bindingRowCount, invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (sboxPlan payload geometry) (bindingPlan payload geometry)
    (combinedRowCount_le payload geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth) :
    (plan payload geometry).rowCount = 729984 := by
  change invocationCount * 94 + bindingRowCount = 729984
  rw [bindingRowCount, invocationCount_eq]

theorem bindingRowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (bindingPlan payload geometry).RowsZero assignment ↔
      ∀ invocation component,
        (bindingForm payload geometry invocation component).eval assignment = 0 := by
  rw [bindingPlan, PinFamilyPlan.planRowsZero_iff
    (bindingInterface payload geometry) bindingRowCount_le assignment one]
  constructor
  · intro rows invocation component
    simpa [bindingInterface] using
      rows (Fin.encodeProd (invocation, component))
  · intro rows row
    let decoded : Fin invocationCount × Fin 2 := Fin.decodeProd row
    change (bindingForm payload geometry decoded.1 decoded.2).eval assignment = 0
    exact rows decoded.1 decoded.2

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (plan payload geometry).RowsZero assignment ↔
      (∀ invocation, PoseidonSboxPlan.RowsZero
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface payload geometry) invocation) assignment) ∧
      (∀ invocation component,
        (bindingForm payload geometry invocation component).eval assignment = 0) := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [sboxPlan, PoseidonSboxFamilyPlan.planRowsZero_iff]
  rw [bindingRowsZero_iff payload geometry assignment one]

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
  squeezeBinding : ∀ current component,
    (bindingForm payload geometry current component).eval assignment = 0

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (rowsZero : (plan payload geometry).RowsZero assignment) :
    Semantics payload geometry assignment := by
  have children := (rowsZero_iff payload geometry assignment one).mp rowsZero
  refine ⟨?_, children.2⟩
  intro invocation
  have sboxRows := (PoseidonSboxFamilyPlan.planRowsZero_iff
    (interface payload geometry) familyRowCount_le assignment).mpr children.1
  exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (interface payload geometry) familyRowCount_le assignment one sboxRows invocation

/-- Honest retained S-box equations are sufficient for all PiCCS Poseidon2
rows. Final-output equations are definitional custody checks. -/
theorem equations_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payload : Payload logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (sboxes : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface payload geometry) invocation) assignment)
    (bindings : ∀ invocation component,
      (bindingForm payload geometry invocation component).eval assignment = 0) :
    (plan payload geometry).RowsZero assignment := by
  apply (rowsZero_iff payload geometry assignment one).mpr
  constructor
  · intro invocation
    apply PoseidonSboxPlan.rowsZero_of_equations
      (PoseidonSboxFamilyPlan.invocationInterface
        (interface payload geometry) invocation) assignment one
      (sboxes invocation)
    exact PoseidonRetainedFamily.outputEquations
      (schedule program) (retainedStart program) (retainedFits geometry)
      (oneColumn geometry) (inputState payload geometry) assignment invocation
  · exact bindings

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
