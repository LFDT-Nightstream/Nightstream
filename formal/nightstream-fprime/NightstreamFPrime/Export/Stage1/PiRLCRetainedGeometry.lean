import NightstreamFPrime.Export.Stage1.DirectLowNormFootprint
import NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Layout.ProductionRelation.ProductRetainedBlock

/-!
Owns the canonical retained-block order and column starts for the current
direct Stage 1 prefix through PiRLC. All blocks are lifted into the one final
PiRLC source domain. Later phases may extend the logical width without moving
these prefix columns.

This module does not construct matrix rows or claim the complete Stage 1 fit.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCFirst54DirectPlan.sourceWidth program

theorem poseidonSourceFits (program : Lifecycle.Stage1.Application.Program) :
    PoseidonRetainedBlock.basePackage.layout.constantColumn ≤
      sourceWidth program := by
  have baseFits :
      PoseidonRetainedBlock.basePackage.layout.constantColumn ≤
        PiRLCProductPlan.baseSourceWidth program := by
    simpa [PoseidonRetainedBlock.basePackage, PiRLCProductPlan.basePackage] using
      PiRLCProductPlan.basePackage_fits program
  unfold sourceWidth PiRLCFirst54DirectPlan.sourceWidth
    PiRLCFirst54DirectPlan.prefixSourceWidth
    PiRLCProductPlan.sourceWidth ProductRetainedBlock.sourceWidth
    FieldSuffixBlock.sourceWidth
  omega

theorem productSourceFits (program : Lifecycle.Stage1.Application.Program) :
    PiRLCProductPlan.sourceWidth program ≤ sourceWidth program := by
  unfold sourceWidth PiRLCFirst54DirectPlan.sourceWidth
    PiRLCFirst54DirectPlan.prefixSourceWidth FieldSuffixBlock.sourceWidth
  omega

def priorPoseidonBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PoseidonRetainedBlock.priorBlock.lift (poseidonSourceFits program)

def outputPoseidonBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PoseidonRetainedBlock.outputBlock.lift (poseidonSourceFits program)

def laterPoseidonBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PoseidonRetainedBlock.laterBlock.lift (poseidonSourceFits program)

@[simp] theorem laterPoseidonBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (laterPoseidonBlock program).kind = .field := by
  rw [laterPoseidonBlock, LowNormBlock.Block.lift_kind,
    PoseidonRetainedBlock.laterBlock_kind]

@[simp] theorem laterPoseidonBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (laterPoseidonBlock program).slotCount = 667102 := by
  rw [laterPoseidonBlock, LowNormBlock.Block.lift_slotCount,
    PoseidonRetainedBlock.laterBlock_slotCount]

@[simp] theorem laterPoseidonBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (laterPoseidonBlock program).coordinateCount = 27351182 := by
  rw [laterPoseidonBlock, LowNormBlock.Block.lift_coordinateCount,
    PoseidonRetainedBlock.laterBlock_coordinateCount]

def productGroupBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (ProductRetainedBlock.block (PiRLCProductPlan.baseSourceWidth program)
    PiRLCProductSchedule.invocationCount).lift (productSourceFits program)

def productOutputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  (PiRLCProductSourceBlocks.outputBlock program).lift (productSourceFits program)

def priorPoseidonStart (_program : Lifecycle.Stage1.Application.Program) : Nat :=
  ProductionAssignment.publicWidth

def outputPoseidonStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  priorPoseidonStart program + (priorPoseidonBlock program).coordinateCount

def laterPoseidonStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputPoseidonStart program + (outputPoseidonBlock program).coordinateCount

def productGroupStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  laterPoseidonStart program + (laterPoseidonBlock program).coordinateCount

def rejectStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  productGroupStart program + (productGroupBlock program).coordinateCount

def symbolStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  rejectStart program +
    (PiRLCFirst54RetainedBlocks.rejectBlock program).coordinateCount

def positionStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  symbolStart program +
    (PiRLCFirst54RetainedBlocks.symbolBlock program).coordinateCount

def valueStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  positionStart program +
    (PiRLCFirst54RetainedBlocks.positionBlock program).coordinateCount

def first54ProductStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  valueStart program +
    (PiRLCFirst54RetainedBlocks.valueBlock program).coordinateCount

def productOutputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  first54ProductStart program +
    (PiRLCFirst54RetainedBlocks.productBlock program).coordinateCount

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  productOutputStart program + (productOutputBlock program).coordinateCount

@[simp] theorem prefixLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    prefixLogicalWidth program = 189945072 := by
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart rejectStart
    productGroupStart laterPoseidonStart outputPoseidonStart priorPoseidonStart
    priorPoseidonBlock outputPoseidonBlock laterPoseidonBlock productGroupBlock
    productOutputBlock PoseidonRetainedBlock.priorBlock
    PoseidonRetainedBlock.outputBlock PoseidonRetainedBlock.laterBlock
  simp only [LowNormBlock.Block.lift_coordinateCount]
  rw [Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount]
  rw [Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount]
  rw [Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount]
  rw [ProductRetainedBlock.block_coordinateCount]
  simp

/-- The prefix owns a fixed number of coordinates. Reading its width does
not need the selected application's source-domain size or circuit. -/
def directPrefixLogicalWidth (_program : Lifecycle.Stage1.Application.Program) : Nat :=
  189945072

@[csimp] theorem prefixLogicalWidth_eq_directPrefixLogicalWidth :
    @prefixLogicalWidth = @directPrefixLogicalWidth := by
  funext program
  exact prefixLogicalWidth_eq program

theorem prefixLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    prefixLogicalWidth program ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [prefixLogicalWidth_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  prefixFits : prefixLogicalWidth program ≤ logicalWidth

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  ⟨NightstreamFPrime.Lifecycle.encHashMarkerIndex.val, by
    have prefixPositive : 0 < prefixLogicalWidth program := by
      rw [prefixLogicalWidth_eq]
      omega
    have := geometry.prefixFits
    simpa [NightstreamFPrime.Lifecycle.encHashMarkerIndex] using
      Nat.lt_of_lt_of_le prefixPositive this⟩

def priorPoseidonFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    priorPoseidonStart program + (priorPoseidonBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart rejectStart
    productGroupStart laterPoseidonStart outputPoseidonStart
  omega

def outputPoseidonFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputPoseidonStart program +
        (outputPoseidonBlock program).coordinateCount ≤ logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart rejectStart
    productGroupStart laterPoseidonStart
  omega

def laterPoseidonFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    laterPoseidonStart program + (laterPoseidonBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart rejectStart
    productGroupStart
  omega

def productGroupFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    productGroupStart program + (productGroupBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart rejectStart
  omega

def rejectFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    rejectStart program +
        (PiRLCFirst54RetainedBlocks.rejectBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart symbolStart
  omega

def symbolFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    symbolStart program +
        (PiRLCFirst54RetainedBlocks.symbolBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart positionStart
  omega

def positionFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    positionStart program +
        (PiRLCFirst54RetainedBlocks.positionBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart valueStart
  omega

def valueFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    valueStart program +
        (PiRLCFirst54RetainedBlocks.valueBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
    first54ProductStart
  omega

def first54ProductFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    first54ProductStart program +
        (PiRLCFirst54RetainedBlocks.productBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.prefixFits
  unfold prefixLogicalWidth productOutputStart
  omega

def productOutputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    productOutputStart program +
        (productOutputBlock program).coordinateCount ≤ logicalWidth := by
  exact geometry.prefixFits

end NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry
