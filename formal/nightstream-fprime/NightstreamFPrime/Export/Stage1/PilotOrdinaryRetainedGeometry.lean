import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedGeometry
import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedBlocks

/-!
Owns the no-gap placement of the three pilot ordinary retained blocks after
the existing PiCCS ordinary retained prefix.
-/

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSOrdinaryRetainedGeometry.completeLogicalWidth program

def canonicalLocalStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  prefixLogicalWidth program

def canonicalFreshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  canonicalLocalStart program +
    (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program).coordinateCount

def outputDigestStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  canonicalFreshStart program +
    (PilotOrdinaryRetainedBlocks.canonicalFreshBlock program).coordinateCount

def completeLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputDigestStart program +
    (PilotOrdinaryRetainedBlocks.outputDigestBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 255046225 := by
  unfold completeLogicalWidth outputDigestStart canonicalFreshStart
    canonicalLocalStart prefixLogicalWidth
  rw [PiCCSOrdinaryRetainedGeometry.completeLogicalWidth_eq]
  have count := PilotOrdinaryRetainedBlocks.retainedCoordinateCount_eq program
  unfold PilotOrdinaryRetainedBlocks.retainedCoordinateCount at count
  omega

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
    PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth outputDigestStart canonicalFreshStart
      canonicalLocalStart prefixLogicalWidth
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiCCSOrdinaryRetainedGeometry.oneColumn (prefixGeometry geometry)

def canonicalLocalFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    canonicalLocalStart program +
        (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth outputDigestStart canonicalFreshStart
  omega

def canonicalFreshFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    canonicalFreshStart program +
        (PilotOrdinaryRetainedBlocks.canonicalFreshBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth outputDigestStart
  omega

def outputDigestFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputDigestStart program +
        (PilotOrdinaryRetainedBlocks.outputDigestBlock program).coordinateCount ≤
      logicalWidth :=
  geometry.completeFits

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (PilotOrdinaryRetainedBlocks.sourceWidth program) →
      NightstreamFPrime.Spec.F) : Prop where
  canonicalLocal :
    (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program).EncodesAt
      (canonicalLocalStart program) (canonicalLocalFits geometry) assignment source
  canonicalFresh :
    (PilotOrdinaryRetainedBlocks.canonicalFreshBlock program).EncodesAt
      (canonicalFreshStart program) (canonicalFreshFits geometry) assignment source
  outputDigest :
    (PilotOrdinaryRetainedBlocks.outputDigestBlock program).EncodesAt
      (outputDigestStart program) (outputDigestFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry
