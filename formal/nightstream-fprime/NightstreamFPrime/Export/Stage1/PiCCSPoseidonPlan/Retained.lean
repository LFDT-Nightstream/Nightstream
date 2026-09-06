import NightstreamFPrime.Export.Stage1.LaterPoseidonRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily

/-!
Owns the shared PiCCS retained coordinates and output forms. The phase plan
and its parent source map use these contracts without importing each other.
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

def outputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  PoseidonRetainedFamily.outputState (schedule program)
    (retainedStart program) (retainedFits geometry) invocation

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
