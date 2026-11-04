import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceCount : Nat := 17
def invocationsPerSource : Nat := 2
def invocationCount : Nat := sourceCount * invocationsPerSource

@[simp] theorem sourceCount_eq : sourceCount = 17 := by rfl
@[simp] theorem invocationsPerSource_eq : invocationsPerSource = 2 := by rfl
@[simp] theorem invocationCount_eq : invocationCount = 34 := by rfl

def descriptor (invocation : Fin invocationCount) :
    Fin sourceCount × Fin invocationsPerSource :=
  Fin.decodeProd invocation

def invocation (source : Fin sourceCount) (step : Fin invocationsPerSource) :
    Fin invocationCount :=
  Fin.encodeProd (source, step)

@[simp] theorem descriptor_invocation
    (source : Fin sourceCount) (step : Fin invocationsPerSource) :
    descriptor (invocation source step) = (source, step) := by
  exact Fin.decodeProd_encodeProd (source, step)

@[implicit_reducible]
def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSActionPayloadBlock.sourceWidth program

def retainedBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (LaterPoseidonRetainedBlocks.samplerBlock program).lift
    (PiCCSPoseidonPlan.prefixSourceFits program)

@[simp] theorem retainedBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (retainedBlock program).slotCount = 5100 := by
  rw [retainedBlock, LowNormBlock.Block.lift_slotCount,
    LaterPoseidonRetainedBlocks.samplerBlock_slotCount]

@[simp] theorem retainedBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (retainedBlock program).coordinateCount = 209100 := by
  rw [retainedBlock, LowNormBlock.Block.lift_coordinateCount,
    LaterPoseidonRetainedBlocks.samplerBlock_coordinateCount]

def schedule (program : Lifecycle.Stage1.Application.Program) :
    PoseidonRetainedFamily.Schedule (sourceWidth program) invocationCount where
  block := retainedBlock program
  slotCount_eq := by
    rw [retainedBlock_slotCount, invocationCount_eq,
      PoseidonRetainedSlots.rows_length]

def retainedStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  LaterPoseidonRetainedBlocks.samplerStart program

theorem retainedFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    retainedStart program + (retainedBlock program).coordinateCount ≤
      logicalWidth := by
  have whole := PiRLCRetainedGeometry.laterPoseidonFits
    (PiCCSPoseidonPlan.prefixGeometry geometry)
  have startEq : retainedStart program = LaterPoseidonRetainedBlocks.samplerStart program := rfl
  have samplerEq := LaterPoseidonRetainedBlocks.samplerStart_eq program
  have count := retainedBlock_coordinateCount program
  have later := PiRLCRetainedGeometry.laterPoseidonBlock_coordinateCount program
  omega

theorem retainedBlock_encodesAt
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
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
      LaterPoseidonRetainedBlocks.samplerStart program +
          (LaterPoseidonRetainedBlocks.samplerBlock program).coordinateCount ≤
        logicalWidth := by
    have fits := retainedFits geometry
    rw [retainedBlock_coordinateCount] at fits
    rw [LaterPoseidonRetainedBlocks.samplerBlock_coordinateCount]
    exact fits
  have sliced :
      (LaterPoseidonRetainedBlocks.samplerBlock program).EncodesAt
        (LaterPoseidonRetainedBlocks.samplerStart program) sliceFits assignment
        prefixAssignment := by
    exact LaterPoseidonRetainedBlocks.samplerBlock_encodesAt assignment
      prefixAssignment parentFits sliceFits parent
  apply LowNormBlock.Block.encodesAt_lift
    (LaterPoseidonRetainedBlocks.samplerBlock program)
    (PiCCSPoseidonPlan.prefixSourceFits program) (retainedStart program)
    sliceFits (retainedFits geometry) assignment prefixAssignment
    (PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment) sliced
  intro slot
  rw [show
      ((LaterPoseidonRetainedBlocks.samplerBlock program).lift
          (PiCCSPoseidonPlan.prefixSourceFits program)).source slot =
        PiCCSActionPayloadBlock.prefixColumn program
          ((LaterPoseidonRetainedBlocks.samplerBlock program).source slot) by
    apply Fin.ext
    rfl]
  exact PiCCSActionPayloadBlock.sourceAssignment_prefix program
    prefixAssignment _

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiCCSPoseidonPlan.oneColumn geometry

def piCcsFinalOutput {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    PoseidonSboxPlan.State logicalWidth :=
  PiCCSPoseidonPlan.outputState geometry
    ⟨PiCCSPoseidonPlan.invocationCount - 1, by
      rw [PiCCSPoseidonPlan.invocationCount_eq]
      omega⟩

def previousOutput {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  if first : current.val = 0 then
    piCcsFinalOutput geometry
  else
    PoseidonRetainedFamily.outputState (schedule program)
      (retainedStart program) (retainedFits geometry)
      ⟨current.val - 1, by
        have currentBound := current.isLt
        omega⟩

def entryWord (source : Fin sourceCount) (lane : Fin Spec.Poseidon2.width) : F :=
  if lane.val = 0 then
    NightstreamFPrime.Lifecycle.natWord 4
  else if lane.val = 1 then
    NightstreamFPrime.Lifecycle.natWord source.val
  else
    0

def inputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  let decoded := descriptor current
  let previous := previousOutput geometry current
  if decoded.2.val = 0 then
    fun lane => SparseForm.add (previous lane)
      (SparseForm.singleton (oneColumn geometry) (entryWord decoded.1 lane))
  else
    previous

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (schedule program)
    (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState geometry)

theorem familyRowCount_le : invocationCount * 150 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (interface geometry) familyRowCount_le

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (plan geometry).rowCount = 5100 := by
  rw [plan, PoseidonSboxFamilyPlan.plan_rowCount, invocationCount_eq]

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan geometry).RowsZero assignment ↔
      ∀ current, PoseidonSboxPlan.RowsZero
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface geometry) current) assignment := by
  exact PoseidonSboxFamilyPlan.planRowsZero_iff
    (interface geometry) familyRowCount_le assignment

structure Semantics {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  invocation : ∀ current,
    List.ofFn (SparseLayer.evalState assignment
        ((interface geometry).output current)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          ((interface geometry).input current)))

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (rowsZero : (plan geometry).RowsZero assignment) :
    Semantics geometry assignment := by
  refine ⟨?_⟩
  intro current
  exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
    (interface geometry) familyRowCount_le assignment one rowsZero current

theorem equations_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (sboxes : ∀ current,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (interface geometry) current) assignment) :
    (plan geometry).RowsZero assignment := by
  apply PoseidonSboxFamilyPlan.equations_imply_planRowsZero
    (interface geometry) familyRowCount_le assignment
  intro current
  refine ⟨sboxes current, ?_⟩
  exact PoseidonRetainedFamily.outputEquations
    (schedule program) (retainedStart program) (retainedFits geometry)
    (oneColumn geometry) (inputState geometry) assignment current

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPlan
