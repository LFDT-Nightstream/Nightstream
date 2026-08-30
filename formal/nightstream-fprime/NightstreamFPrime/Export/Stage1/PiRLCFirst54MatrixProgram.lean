import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

/-!
Owns the compact matrix program for the canonical PiRLC First54 plan. All
source, round, slot, region, stride, coefficient, retained-block, and family
order data is Lean-authored package data.

This module does not prove semantic row equality; the adjacent semantics
module owns that bridge.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Export.MatrixProgram.AffineGrid
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

def sourceCount : Nat := 17
def roundCount : Nat := 64
def positionSlotCount : Nat := 55
def valueSlotCount : Nat := 54

def positionShape : MultiplicationGrid.Shape :=
  ⟨sourceCount, roundCount, positionSlotCount⟩

def acceptedShape : MultiplicationGrid.Shape :=
  ⟨sourceCount, roundCount, 1⟩

def valueShape : MultiplicationGrid.Shape :=
  ⟨sourceCount, roundCount, valueSlotCount⟩

def region (middleStart middleCount minorStart minorCount : Nat) : Region :=
  { majorStart := 0
    majorCount := sourceCount
    middleStart
    middleCount
    minorStart
    minorCount }

def allPosition : Region := region 0 roundCount 0 positionSlotCount
def allAccepted : Region := region 0 roundCount 0 1
def allValue : Region := region 0 roundCount 0 valueSlotCount

def firstSlotZero : Region := region 0 1 0 1
def firstSlotOne : Region := region 0 1 1 1
def laterPositionZero : Region := region 1 63 0 1
def laterPositionMiddle : Region := region 1 63 1 53
def laterPositionFull : Region := region 1 63 54 1
def laterPositionAll : Region := region 1 63 0 positionSlotCount
def laterValueAll : Region := region 1 63 0 valueSlotCount

def constantRule (selected : Region) (coefficient : F) : Rule :=
  ⟨selected, .constant coefficient.val⟩

def retainedRule (selected : Region) (block : RetainedBlock)
    (slotBase majorStride middleStride minorStride : Nat)
    (coefficient : F) : Rule :=
  ⟨selected, .retained block slotBase majorStride middleStride minorStride
    coefficient.val⟩

def rejectWire (program : Lifecycle.Stage1.Application.Program) : RetainedBlock :=
  RetainedBlock.ofSemantic (PiRLCFirst54RetainedBlocks.rejectBlock program)
    (PiRLCRetainedGeometry.rejectStart program)

def symbolWire (program : Lifecycle.Stage1.Application.Program) : RetainedBlock :=
  RetainedBlock.ofSemantic (PiRLCFirst54RetainedBlocks.symbolBlock program)
    (PiRLCRetainedGeometry.symbolStart program)

def positionWire (program : Lifecycle.Stage1.Application.Program) : RetainedBlock :=
  RetainedBlock.ofSemantic (PiRLCFirst54RetainedBlocks.positionBlock program)
    (PiRLCRetainedGeometry.positionStart program)

def valueWire (program : Lifecycle.Stage1.Application.Program) : RetainedBlock :=
  RetainedBlock.ofSemantic (PiRLCFirst54RetainedBlocks.valueBlock program)
    (PiRLCRetainedGeometry.valueStart program)

def productWire (program : Lifecycle.Stage1.Application.Program) : RetainedBlock :=
  RetainedBlock.ofSemantic (PiRLCFirst54RetainedBlocks.productBlock program)
    (PiRLCRetainedGeometry.first54ProductStart program)

def acceptedProgram (program : Lifecycle.Stage1.Application.Program)
    (selected : Region) : AffineGrid.Program where
  rules := [
    constantRule selected 1,
    retainedRule selected (rejectWire program) 0 64 1 0 (-1)]

def positionLeftProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program := acceptedProgram program allPosition

def positionRightProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [
    constantRule firstSlotZero (-1),
    constantRule firstSlotOne 1,
    retainedRule laterPositionZero (positionWire program) 0 3520 55 0 (-1),
    retainedRule laterPositionMiddle (positionWire program) 0 3520 55 1 1,
    retainedRule laterPositionMiddle (positionWire program) 1 3520 55 1 (-1),
    retainedRule laterPositionFull (positionWire program) 53 3520 55 0 1]

def positionOutputProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [
    retainedRule allPosition (positionWire program) 0 3520 55 1 1,
    constantRule firstSlotZero (-1),
    retainedRule laterPositionAll (positionWire program) 0 3520 55 1 (-1)]

def acceptedLeftProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program := acceptedProgram program allAccepted

def acceptedRightProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [retainedRule allAccepted (symbolWire program) 0 64 1 0 1]

def acceptedOutputProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [retainedRule allAccepted (productWire program) 0 64 1 0 1]

def valueLeftProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [
    constantRule firstSlotZero 1,
    retainedRule laterValueAll (positionWire program) 0 3520 55 1 1]

def valueRightProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [retainedRule allValue (productWire program) 0 64 1 0 1]

def valueOutputProgram (program : Lifecycle.Stage1.Application.Program) :
    AffineGrid.Program where
  rules := [
    retainedRule allValue (valueWire program) 0 3456 54 1 1,
    retainedRule laterValueAll (valueWire program) 0 3456 54 1 (-1)]

def positionGrid {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MultiplicationGrid.Block where
  shape := positionShape
  oneColumn := (PiRLCRetainedGeometry.oneColumn geometry).val
  left := positionLeftProgram program
  right := positionRightProgram program
  output := positionOutputProgram program

def acceptedGrid {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MultiplicationGrid.Block where
  shape := acceptedShape
  oneColumn := (PiRLCRetainedGeometry.oneColumn geometry).val
  left := acceptedLeftProgram program
  right := acceptedRightProgram program
  output := acceptedOutputProgram program

def valueGrid {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MultiplicationGrid.Block where
  shape := valueShape
  oneColumn := (PiRLCRetainedGeometry.oneColumn geometry).val
  left := valueLeftProgram program
  right := valueRightProgram program
  output := valueOutputProgram program

def finalPin {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Pin.Block :=
  MatrixProgram.Pin.Block.ofSemantic
    (PiRLCFirst54DirectPlan.finalInterface
      (PiRLCRetainedInputs.first54Inputs geometry))

@[simp] theorem positionGrid_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (positionGrid geometry).rowCount = 59840 := by
  change positionShape.rowCount = 59840
  norm_num [positionShape, MultiplicationGrid.Shape.rowCount,
    sourceCount, roundCount, positionSlotCount]

@[simp] theorem acceptedGrid_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (acceptedGrid geometry).rowCount = 1088 := by
  change acceptedShape.rowCount = 1088
  norm_num [acceptedShape, MultiplicationGrid.Shape.rowCount,
    sourceCount, roundCount]

@[simp] theorem valueGrid_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (valueGrid geometry).rowCount = 58752 := by
  change valueShape.rowCount = 58752
  norm_num [valueShape, MultiplicationGrid.Shape.rowCount,
    sourceCount, roundCount, valueSlotCount]

@[simp] theorem finalPin_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (finalPin geometry).rowCount = 17 := by
  exact MatrixProgram.Pin.Block.ofSemantic_rowCount _

/-- Exact First54 family order: position, accepted-symbol product, value,
then final full-position pin. -/
def matrixProgram {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [
    .multiplicationGrid (positionGrid geometry),
    .multiplicationGrid (acceptedGrid geometry),
    .multiplicationGrid (valueGrid geometry),
    .pin (finalPin geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 119697 := by
  norm_num [matrixProgram, MatrixProgram.Program.rowCount,
    MatrixProgram.Block.rowCount]

end NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram
