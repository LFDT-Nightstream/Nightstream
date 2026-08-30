import NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectPlan
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns the five retained low-norm blocks required by the direct First54 matrix
plan. It retains only reject bits, decoded symbols, position outputs, value
outputs, and the shared accepted-symbol products.

This module does not construct the complete Stage 1 retained assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54RetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCFirst54DirectPlan.sourceWidth program

def rejectBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .bit
  slotCount := PiRLCFirst54DirectSchedule.candidateCount
  source := fun candidate =>
    PiRLCFirst54DirectPlan.retainedRejectColumn program
      (PiRLCFirst54DirectSchedule.candidate candidate)

def symbolBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCFirst54DirectSchedule.candidateCount
  source := fun candidate =>
    PiRLCFirst54DirectPlan.retainedSymbolColumn program
      (PiRLCFirst54DirectSchedule.candidate candidate)

def positionBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .bit
  slotCount := PiRLCFirst54DirectSchedule.positionCount
  source := fun position =>
    PiRLCFirst54DirectPlan.retainedPositionColumn program
      (PiRLCFirst54DirectSchedule.position position)

def valueBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCFirst54DirectSchedule.valueCount
  source := fun value =>
    PiRLCFirst54DirectPlan.retainedValueColumn program
      (PiRLCFirst54DirectSchedule.value value)

def productBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  FieldSuffixBlock.block (PiRLCFirst54DirectPlan.prefixSourceWidth program)
    PiRLCFirst54DirectSchedule.candidateCount

@[simp] theorem rejectBlock_kind (program : Lifecycle.Stage1.Application.Program) :
    (rejectBlock program).kind = .bit := by
  rfl

@[simp] theorem symbolBlock_kind (program : Lifecycle.Stage1.Application.Program) :
    (symbolBlock program).kind = .field := by
  rfl

@[simp] theorem positionBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (positionBlock program).kind = .bit := by
  rfl

@[simp] theorem valueBlock_kind (program : Lifecycle.Stage1.Application.Program) :
    (valueBlock program).kind = .field := by
  rfl

@[simp] theorem productBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (productBlock program).kind = .field := by
  rfl

@[simp] theorem rejectBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (rejectBlock program).slotCount = 1088 := by
  rfl

@[simp] theorem symbolBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (symbolBlock program).slotCount = 1088 := by
  rfl

@[simp] theorem positionBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (positionBlock program).slotCount = 59840 := by
  rfl

@[simp] theorem valueBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (valueBlock program).slotCount = 58752 := by
  rfl

@[simp] theorem productBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (productBlock program).slotCount = 1088 := by
  rfl

theorem rejectBlock_source (program : Lifecycle.Stage1.Application.Program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    (rejectBlock program).source candidate =
      PiRLCFirst54DirectPlan.retainedRejectColumn program
        (PiRLCFirst54DirectSchedule.candidate candidate) := by
  rfl

theorem symbolBlock_source (program : Lifecycle.Stage1.Application.Program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    (symbolBlock program).source candidate =
      PiRLCFirst54DirectPlan.retainedSymbolColumn program
        (PiRLCFirst54DirectSchedule.candidate candidate) := by
  rfl

theorem positionBlock_source (program : Lifecycle.Stage1.Application.Program)
    (position : Fin PiRLCFirst54DirectSchedule.positionCount) :
    (positionBlock program).source position =
      PiRLCFirst54DirectPlan.retainedPositionColumn program
        (PiRLCFirst54DirectSchedule.position position) := by
  rfl

theorem valueBlock_source (program : Lifecycle.Stage1.Application.Program)
    (value : Fin PiRLCFirst54DirectSchedule.valueCount) :
    (valueBlock program).source value =
      PiRLCFirst54DirectPlan.retainedValueColumn program
        (PiRLCFirst54DirectSchedule.value value) := by
  rfl

theorem productBlock_source (program : Lifecycle.Stage1.Application.Program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    (productBlock program).source candidate =
      PiRLCFirst54DirectPlan.productColumn program candidate := by
  rfl

@[simp] theorem rejectBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (rejectBlock program).coordinateCount = 1088 := by
  rfl

@[simp] theorem symbolBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (symbolBlock program).coordinateCount = 44608 := by
  norm_num [symbolBlock, LowNormBlock.Block.coordinateCount,
    LowNormSlot.Kind.width, BalancedTernary.width,
    PiRLCFirst54DirectSchedule.candidateCount]

@[simp] theorem positionBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (positionBlock program).coordinateCount = 59840 := by
  rfl

@[simp] theorem valueBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (valueBlock program).coordinateCount = 2408832 := by
  change PiRLCFirst54DirectSchedule.valueCount * 41 = 2408832
  rw [PiRLCFirst54DirectSchedule.valueCount_eq]

@[simp] theorem productBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (productBlock program).coordinateCount = 44608 := by
  norm_num [productBlock, FieldSuffixBlock.block,
    LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width,
    BalancedTernary.width, PiRLCFirst54DirectSchedule.candidateCount]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (rejectBlock program).coordinateCount +
    (symbolBlock program).coordinateCount +
    (positionBlock program).coordinateCount +
    (valueBlock program).coordinateCount +
    (productBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 2558976 := by
  simp [retainedCoordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCFirst54RetainedBlocks
