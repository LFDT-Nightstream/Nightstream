import NightstreamFPrime.Export.Stage1.LaterPoseidonRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily

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

def invocationCount : Nat := PiCCSActionPayloadBlock.invocationCount

@[simp] theorem invocationCount_eq : invocationCount = 7604 := by
  exact PiCCSActionPayloadBlock.invocationCount_eq

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSActionPayloadBlock.sourceWidth program

def prefixSourceFits (program : Lifecycle.Stage1.Application.Program) :
    PiRLCRetainedGeometry.sourceWidth program ≤ sourceWidth program := by
  unfold sourceWidth PiCCSActionPayloadBlock.sourceWidth
    PiCCSActionPayloadBlock.prefixSourceWidth FieldSuffixBlock.sourceWidth
  omega

def retainedBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (LaterPoseidonRetainedBlocks.piCcsBlock program).lift
    (prefixSourceFits program)

@[simp] theorem retainedBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (retainedBlock program).slotCount = 653944 := by
  rw [retainedBlock, LowNormBlock.Block.lift_slotCount,
    LaterPoseidonRetainedBlocks.piCcsBlock_slotCount]

@[simp] theorem retainedBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (retainedBlock program).coordinateCount = 26811704 := by
  rw [retainedBlock, LowNormBlock.Block.lift_coordinateCount,
    LaterPoseidonRetainedBlocks.piCcsBlock_coordinateCount]

def schedule (program : Lifecycle.Stage1.Application.Program) :
    PoseidonRetainedFamily.Schedule (sourceWidth program) invocationCount where
  block := retainedBlock program
  slotCount_eq := by
    rw [retainedBlock_slotCount, invocationCount_eq,
      PoseidonRetainedSlots.rows_length]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  payloadFits : PiCCSActionPayloadBlock.logicalWidth program ≤ logicalWidth

def pilotGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCPoseidonGeometry.Geometry program logicalWidth where
  pilotFits := by
    apply Nat.le_trans _ geometry.payloadFits
    unfold PiCCSActionPayloadBlock.logicalWidth
      PiCCSActionPayloadBlock.payloadStart
    omega

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  PiRLCPoseidonGeometry.prefixGeometry (pilotGeometry geometry)

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiRLCPoseidonGeometry.oneColumn (pilotGeometry geometry)

def retainedStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  LaterPoseidonRetainedBlocks.piCcsStart program

def retainedFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    retainedStart program + (retainedBlock program).coordinateCount ≤
      logicalWidth := by
  have whole := PiRLCRetainedGeometry.laterPoseidonFits
    (prefixGeometry geometry)
  rw [retainedBlock_coordinateCount]
  unfold retainedStart LaterPoseidonRetainedBlocks.piCcsStart
  rw [PiRLCRetainedGeometry.laterPoseidonBlock_coordinateCount] at whole
  omega

def payloadFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiCCSActionPayloadBlock.payloadStart program +
        (PiCCSActionPayloadBlock.block program).coordinateCount ≤ logicalWidth :=
  geometry.payloadFits

/-- The retained PiCCS block is exactly the zero-offset slice of the shared
later-Poseidon block, lifted through the payload suffix source domain. -/
theorem retainedBlock_encodesAt
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment : Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (parentFits : PiRLCRetainedGeometry.laterPoseidonStart program +
      (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount ≤
        logicalWidth)
    (parent : (PiRLCRetainedGeometry.laterPoseidonBlock program).EncodesAt
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits assignment
      prefixAssignment) :
    (retainedBlock program).EncodesAt (retainedStart program)
      (retainedFits geometry) assignment
      (PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment) := by
  have sliceFits :
      LaterPoseidonRetainedBlocks.piCcsStart program +
          (LaterPoseidonRetainedBlocks.piCcsBlock program).coordinateCount ≤
        logicalWidth := by
    simpa [retainedStart, retainedBlock] using retainedFits geometry
  have sliced :
      (LaterPoseidonRetainedBlocks.piCcsBlock program).EncodesAt
        (LaterPoseidonRetainedBlocks.piCcsStart program) sliceFits assignment
        prefixAssignment := by
    exact LaterPoseidonRetainedBlocks.piCcsBlock_encodesAt assignment
      prefixAssignment parentFits sliceFits parent
  apply LowNormBlock.Block.encodesAt_lift
    (LaterPoseidonRetainedBlocks.piCcsBlock program)
    (prefixSourceFits program) (retainedStart program) sliceFits
    (retainedFits geometry) assignment prefixAssignment
    (PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment) sliced
  intro slot
  rw [show
      ((LaterPoseidonRetainedBlocks.piCcsBlock program).lift
          (prefixSourceFits program)).source slot =
        PiCCSActionPayloadBlock.prefixColumn program
          ((LaterPoseidonRetainedBlocks.piCcsBlock program).source slot) by
    apply Fin.ext
    rfl]
  exact PiCCSActionPayloadBlock.sourceAssignment_prefix program
    prefixAssignment _

def previousOutput {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.State logicalWidth :=
  if first : invocation.val = 0 then
    fun _ => .empty
  else
    PoseidonRetainedFamily.outputState (schedule program)
      (retainedStart program) (retainedFits geometry)
      ⟨invocation.val - 1, by
        have invocationBound := invocation.isLt
        omega⟩

def payloadForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) (lane : Fin Spec.Poseidon2.width) :
    SparseForm logicalWidth :=
  if rateLane : lane.val < Spec.Poseidon2.rate then
    (PiCCSActionPayloadBlock.block program).form
      (PiCCSActionPayloadBlock.payloadStart program) (payloadFits geometry)
      (Fin.encodeProd (invocation, ⟨lane.val, rateLane⟩))
  else
    .empty

def inputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.State logicalWidth :=
  let previous := previousOutput geometry invocation
  match PiCCSActionPayloadBlock.kindAt invocation with
  | .absorb _ => fun lane =>
      SparseForm.add (previous lane) (payloadForm geometry invocation lane)
  | .squeezeFirst _ => previous
  | .squeezeSecond => previous

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (schedule program)
    (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState geometry)

def outputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  PoseidonRetainedFamily.outputState (schedule program)
    (retainedStart program) (retainedFits geometry) invocation

def bindingActual {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) (component : Fin 2) :
    SparseForm logicalWidth :=
  if component.val = 0 then
    previousOutput geometry invocation 0
  else
    outputState geometry invocation 0

def bindingForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) (component : Fin 2) :
    SparseForm logicalWidth :=
  match PiCCSActionPayloadBlock.kindAt invocation with
  | .squeezeFirst _ =>
      SparseForm.add
        (payloadForm geometry invocation
          ⟨component.val, Nat.lt_trans component.isLt (by
            norm_num [Spec.Poseidon2.width])⟩)
        (SparseForm.scale (-1) (bindingActual geometry invocation component))
  | .absorb _ | .squeezeSecond => .empty

theorem bindingForm_squeezeFirst_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    bindingForm geometry invocation (0 : Fin 2) =
      SparseForm.add (payloadForm geometry invocation (0 : Fin 8))
        (SparseForm.scale (-1) (previousOutput geometry invocation 0)) := by
  unfold bindingForm bindingActual
  rw [found]
  rfl

theorem bindingForm_squeezeFirst_one
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    bindingForm geometry invocation (1 : Fin 2) =
      SparseForm.add (payloadForm geometry invocation (1 : Fin 8))
        (SparseForm.scale (-1) (outputState geometry invocation 0)) := by
  unfold bindingForm bindingActual
  rw [found]
  rfl

def bindingRowCount : Nat := invocationCount * 2

def bindingInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth bindingRowCount where
  oneColumn := oneColumn geometry
  value := fun row =>
    let decoded : Fin invocationCount × Fin 2 := Fin.decodeProd row
    bindingForm geometry decoded.1 decoded.2

theorem bindingRowCount_le : bindingRowCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [bindingRowCount, invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

theorem familyRowCount_le : invocationCount * 94 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def sboxPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (interface geometry) familyRowCount_le

def bindingPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PinFamilyPlan.plan (bindingInterface geometry) bindingRowCount_le

theorem combinedRowCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    (sboxPlan geometry).rowCount + (bindingPlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  change invocationCount * 94 + bindingRowCount ≤ _
  rw [bindingRowCount, invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (sboxPlan geometry) (bindingPlan geometry)
    (combinedRowCount_le geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (plan geometry).rowCount = 729984 := by
  change invocationCount * 94 + bindingRowCount = 729984
  rw [bindingRowCount, invocationCount_eq]

theorem bindingRowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (bindingPlan geometry).RowsZero assignment ↔
      ∀ invocation component,
        (bindingForm geometry invocation component).eval assignment = 0 := by
  rw [bindingPlan, PinFamilyPlan.planRowsZero_iff
    (bindingInterface geometry) bindingRowCount_le assignment one]
  constructor
  · intro rows invocation component
    simpa [bindingInterface] using
      rows (Fin.encodeProd (invocation, component))
  · intro rows row
    let decoded : Fin invocationCount × Fin 2 := Fin.decodeProd row
    change (bindingForm geometry decoded.1 decoded.2).eval assignment = 0
    exact rows decoded.1 decoded.2

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (plan geometry).RowsZero assignment ↔
      (∀ invocation, PoseidonSboxPlan.RowsZero
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface geometry) invocation) assignment) ∧
      (∀ invocation component,
        (bindingForm geometry invocation component).eval assignment = 0) := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [sboxPlan, PoseidonSboxFamilyPlan.planRowsZero_iff]
  rw [bindingRowsZero_iff geometry assignment one]

structure Semantics {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  invocation : ∀ current,
    List.ofFn (SparseLayer.evalState assignment
        ((interface geometry).output current)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          ((interface geometry).input current)))
  squeezeBinding : ∀ current component,
    (bindingForm geometry current component).eval assignment = 0

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (rowsZero : (plan geometry).RowsZero assignment) :
    Semantics geometry assignment := by
  have children := (rowsZero_iff geometry assignment one).mp rowsZero
  refine ⟨?_, children.2⟩
  intro invocation
  have sboxRows := (PoseidonSboxFamilyPlan.planRowsZero_iff
    (interface geometry) familyRowCount_le assignment).mpr children.1
  exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (interface geometry) familyRowCount_le assignment one sboxRows invocation

/-- Honest retained S-box equations are sufficient for all PiCCS Poseidon2
rows. Final-output equations are definitional custody checks. -/
theorem equations_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (sboxes : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface geometry) invocation) assignment)
    (bindings : ∀ invocation component,
      (bindingForm geometry invocation component).eval assignment = 0) :
    (plan geometry).RowsZero assignment := by
  apply (rowsZero_iff geometry assignment one).mpr
  constructor
  · intro invocation
    apply PoseidonSboxPlan.rowsZero_of_equations
      (PoseidonSboxFamilyPlan.invocationInterface
        (interface geometry) invocation) assignment one
      (sboxes invocation)
    exact PoseidonRetainedFamily.outputEquations
      (schedule program) (retainedStart program) (retainedFits geometry)
      (oneColumn geometry) (inputState geometry) assignment invocation
  · exact bindings

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
