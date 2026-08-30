import NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry

/-!
Owns zero-copy PiCCS and PiRLC sampler views of the retained later-Poseidon2
block. PiCCS owns the first 7,550 invocations and the sampler owns the next
153. Both views keep the original invocation-major coordinate order.

This module does not construct matrix rows or duplicate retained values.
-/

namespace NightstreamFPrime.Export.Stage1.LaterPoseidonRetainedBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

def piCcsInvocationCount : Nat := 7550
def samplerInvocationCount : Nat := 153

def piCcsSlotCount : Nat :=
  piCcsInvocationCount * PoseidonRetainedSlots.rows.length

def samplerSlotCount : Nat :=
  samplerInvocationCount * PoseidonRetainedSlots.rows.length

theorem totalSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    piCcsSlotCount + samplerSlotCount =
      (PiRLCRetainedGeometry.laterPoseidonBlock program).slotCount := by
  rw [PiRLCRetainedGeometry.laterPoseidonBlock_slotCount]
  rfl

def piCcsFits (program : Lifecycle.Stage1.Application.Program) :
    0 + piCcsSlotCount ≤
      (PiRLCRetainedGeometry.laterPoseidonBlock program).slotCount := by
  rw [PiRLCRetainedGeometry.laterPoseidonBlock_slotCount]
  norm_num [piCcsSlotCount, piCcsInvocationCount]

def samplerFits (program : Lifecycle.Stage1.Application.Program) :
    piCcsSlotCount + samplerSlotCount ≤
      (PiRLCRetainedGeometry.laterPoseidonBlock program).slotCount := by
  rw [totalSlotCount_eq program]

def piCcsBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (PiRLCRetainedGeometry.laterPoseidonBlock program).slice
    0 piCcsSlotCount (piCcsFits program)

def samplerBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (PiRLCRetainedGeometry.laterPoseidonBlock program).slice
    piCcsSlotCount samplerSlotCount (samplerFits program)

def piCcsStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.laterPoseidonStart program

def samplerStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piCcsStart program + (piCcsBlock program).coordinateCount

@[simp] theorem piCcsBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (piCcsBlock program).slotCount = 649300 := by
  calc
    (piCcsBlock program).slotCount = piCcsSlotCount :=
      LowNormBlock.Block.slice_slotCount
        (PiRLCRetainedGeometry.laterPoseidonBlock program)
        0 piCcsSlotCount (piCcsFits program)
    _ = 649300 := by rfl

@[simp] theorem samplerBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (samplerBlock program).slotCount = 13158 := by
  calc
    (samplerBlock program).slotCount = samplerSlotCount :=
      LowNormBlock.Block.slice_slotCount
        (PiRLCRetainedGeometry.laterPoseidonBlock program)
        piCcsSlotCount samplerSlotCount (samplerFits program)
    _ = 13158 := by rfl

@[simp] theorem piCcsBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (piCcsBlock program).coordinateCount = 26621300 := by
  calc
    (piCcsBlock program).coordinateCount =
        piCcsSlotCount *
          (PiRLCRetainedGeometry.laterPoseidonBlock program).kind.width :=
      LowNormBlock.Block.slice_coordinateCount
        (PiRLCRetainedGeometry.laterPoseidonBlock program)
        0 piCcsSlotCount (piCcsFits program)
    _ = 26621300 := by
      rw [PiRLCRetainedGeometry.laterPoseidonBlock_kind]
      rfl

@[simp] theorem samplerBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (samplerBlock program).coordinateCount = 539478 := by
  calc
    (samplerBlock program).coordinateCount =
        samplerSlotCount *
          (PiRLCRetainedGeometry.laterPoseidonBlock program).kind.width :=
      LowNormBlock.Block.slice_coordinateCount
        (PiRLCRetainedGeometry.laterPoseidonBlock program)
        piCcsSlotCount samplerSlotCount (samplerFits program)
    _ = 539478 := by
      rw [PiRLCRetainedGeometry.laterPoseidonBlock_kind]
      rfl

@[simp] theorem samplerStart_eq
    (program : Lifecycle.Stage1.Application.Program) :
    samplerStart program =
      PiRLCRetainedGeometry.laterPoseidonStart program + 26621300 := by
  unfold samplerStart piCcsStart
  rw [piCcsBlock_coordinateCount]

theorem sampler_end_eq_later_end
    (program : Lifecycle.Stage1.Application.Program) :
    samplerStart program + (samplerBlock program).coordinateCount =
      PiRLCRetainedGeometry.laterPoseidonStart program +
        (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount := by
  rw [samplerStart_eq, samplerBlock_coordinateCount]
  rw [PiRLCRetainedGeometry.laterPoseidonBlock_coordinateCount]

/-- A parent later-Poseidon encoding restricts to the exact PiCCS prefix. -/
theorem piCcsBlock_encodesAt
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F)
    (parentFits : PiRLCRetainedGeometry.laterPoseidonStart program +
      (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount ≤
        logicalWidth)
    (childFits : piCcsStart program +
      (piCcsBlock program).coordinateCount ≤ logicalWidth)
    (parent : (PiRLCRetainedGeometry.laterPoseidonBlock program).EncodesAt
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits assignment
      source) :
    (piCcsBlock program).EncodesAt (piCcsStart program) childFits assignment
      source := by
  let indexedStart := PiRLCRetainedGeometry.laterPoseidonStart program +
    0 * (PiRLCRetainedGeometry.laterPoseidonBlock program).kind.width
  have startEq : indexedStart = piCcsStart program := by
    simp [indexedStart, piCcsStart]
  have indexedFits : indexedStart +
      (piCcsBlock program).coordinateCount ≤ logicalWidth := by
    rw [startEq]
    exact childFits
  have indexed : (piCcsBlock program).EncodesAt indexedStart indexedFits
      assignment source := by
    change ((PiRLCRetainedGeometry.laterPoseidonBlock program).slice
      0 piCcsSlotCount (piCcsFits program)).EncodesAt
        indexedStart indexedFits assignment source
    exact LowNormBlock.Block.encodesAt_slice
      (PiRLCRetainedGeometry.laterPoseidonBlock program) 0 piCcsSlotCount
      (piCcsFits program)
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits
      indexedFits assignment source parent
  exact LowNormBlock.Block.encodesAt_start_eq (piCcsBlock program)
    indexedStart (piCcsStart program) startEq indexedFits childFits
    assignment source indexed

theorem samplerStart_eq_indexed
    (program : Lifecycle.Stage1.Application.Program) :
    samplerStart program =
      PiRLCRetainedGeometry.laterPoseidonStart program +
        piCcsSlotCount *
          (PiRLCRetainedGeometry.laterPoseidonBlock program).kind.width := by
  rw [samplerStart_eq, PiRLCRetainedGeometry.laterPoseidonBlock_kind]
  rfl

/-- A parent later-Poseidon encoding restricts to the exact sampler suffix. -/
theorem samplerBlock_encodesAt
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F)
    (parentFits : PiRLCRetainedGeometry.laterPoseidonStart program +
      (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount ≤
        logicalWidth)
    (childFits : samplerStart program +
      (samplerBlock program).coordinateCount ≤ logicalWidth)
    (parent : (PiRLCRetainedGeometry.laterPoseidonBlock program).EncodesAt
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits assignment
      source) :
    (samplerBlock program).EncodesAt (samplerStart program) childFits assignment
      source := by
  let indexedStart := PiRLCRetainedGeometry.laterPoseidonStart program +
    piCcsSlotCount *
      (PiRLCRetainedGeometry.laterPoseidonBlock program).kind.width
  have startEq : indexedStart = samplerStart program := by
    exact (samplerStart_eq_indexed program).symm
  have indexedFits : indexedStart +
      (samplerBlock program).coordinateCount ≤ logicalWidth := by
    rw [startEq]
    exact childFits
  have indexed : (samplerBlock program).EncodesAt indexedStart indexedFits
      assignment source := by
    change ((PiRLCRetainedGeometry.laterPoseidonBlock program).slice
      piCcsSlotCount samplerSlotCount (samplerFits program)).EncodesAt
        indexedStart indexedFits assignment source
    exact LowNormBlock.Block.encodesAt_slice
      (PiRLCRetainedGeometry.laterPoseidonBlock program) piCcsSlotCount
      samplerSlotCount (samplerFits program)
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits
      indexedFits assignment source parent
  exact LowNormBlock.Block.encodesAt_start_eq (samplerBlock program)
    indexedStart (samplerStart program) startEq indexedFits childFits
    assignment source indexed

end NightstreamFPrime.Export.Stage1.LaterPoseidonRetainedBlocks
