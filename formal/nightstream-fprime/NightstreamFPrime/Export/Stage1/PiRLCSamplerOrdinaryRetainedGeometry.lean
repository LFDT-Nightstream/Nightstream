import NightstreamFPrime.Export.Stage1.PiDECRetainedGeometry
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks

/-!
Owns the no-gap placement of the two PiRLC sampler ordinary retained blocks
after the complete PiDEC retained prefix. Earlier retained coordinates do not
move.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiDECRetainedGeometry.completeLogicalWidth program

def logicalStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  prefixLogicalWidth program

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  logicalStart program +
    (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock program).coordinateCount

def completeLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program +
    (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 264311405 := by
  unfold completeLogicalWidth freshStart logicalStart prefixLogicalWidth
  rw [PiDECRetainedGeometry.completeLogicalWidth_eq,
    PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock_coordinateCount,
    PiRLCSamplerOrdinaryRetainedBlocks.freshBlock_coordinateCount]

theorem completeLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [completeLogicalWidth_eq]
  norm_num [Lifecycle.cubeVariables]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  completeFits : completeLogicalWidth program ≤ logicalWidth

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiDECRetainedGeometry.Geometry program logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth freshStart logicalStart prefixLogicalWidth
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiDECRetainedGeometry.oneColumn (prefixGeometry geometry)

def logicalFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    logicalStart program +
        (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart
  omega

def freshFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    freshStart program +
        (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock program).coordinateCount ≤
      logicalWidth :=
  geometry.completeFits

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth program) →
      NightstreamFPrime.Spec.F) : Prop where
  logical : (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock program).EncodesAt
    (logicalStart program) (logicalFits geometry) assignment source
  fresh : (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock program).EncodesAt
    (freshStart program) (freshFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry
