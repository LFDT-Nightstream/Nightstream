import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedBlocks
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry

/-!
Owns the no-gap placement of the conservative PiCCS ordinary retained blocks.

The candidate begins after the direct running-prefix geometry and stays below
the fixed `2^28` domain. It does not prove that the PiCCS ordinary rows use
only these sources; that source-support edge remains separate.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  RunningTransitionRetainedGeometry.completeLogicalWidth program

def priorInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  prefixLogicalWidth program

def outputInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  priorInputStart program +
    (PiCCSOrdinaryRetainedBlocks.priorInputBlock program).coordinateCount

def freshPublicInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputInputStart program +
    (PiCCSOrdinaryRetainedBlocks.outputInputBlock program).coordinateCount

def priorLastStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshPublicInputStart program +
    (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program).coordinateCount

def outputLastStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  priorLastStart program +
    (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).coordinateCount

def expectedContextStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputLastStart program +
    (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).coordinateCount

def proofLogicalStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  expectedContextStart program +
    (PiCCSOrdinaryRetainedBlocks.expectedContextBlock program).coordinateCount

def outputEndpointStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  proofLogicalStart program +
    (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program).coordinateCount

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputEndpointStart program +
    (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).coordinateCount

def completeLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program +
    (PiCCSOrdinaryRetainedBlocks.freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 252392541 := by
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart, priorLastStart, freshPublicInputStart,
    outputInputStart, priorInputStart,
    prefixLogicalWidth]
  rw [RunningTransitionRetainedGeometry.completeLogicalWidth_eq]
  have count :=
    PiCCSOrdinaryRetainedBlocks.retainedCoordinateCount_eq program
  unfold PiCCSOrdinaryRetainedBlocks.retainedCoordinateCount at count
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
    RunningTransitionRetainedGeometry.Geometry program logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth freshStart outputEndpointStart proofLogicalStart
      expectedContextStart outputLastStart priorLastStart freshPublicInputStart
      outputInputStart priorInputStart prefixLogicalWidth
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  RunningTransitionRetainedGeometry.oneColumn (prefixGeometry geometry)

def priorInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    priorInputStart program +
        (PiCCSOrdinaryRetainedBlocks.priorInputBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart, priorLastStart, freshPublicInputStart,
    outputInputStart, priorInputStart]
  omega

def outputInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputInputStart program +
        (PiCCSOrdinaryRetainedBlocks.outputInputBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart, priorLastStart, freshPublicInputStart,
    outputInputStart]
  omega

def priorLastFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    priorLastStart program +
        (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart, priorLastStart,
    freshPublicInputStart]
  omega

def outputLastFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputLastStart program +
        (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart]
  omega

def freshPublicInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    freshPublicInputStart program +
        (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart, outputLastStart, priorLastStart,
    freshPublicInputStart]
  omega

def expectedContextFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    expectedContextStart program +
        (PiCCSOrdinaryRetainedBlocks.expectedContextBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart,
    expectedContextStart]
  omega

def proofLogicalFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    proofLogicalStart program +
        (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart,
    proofLogicalStart]
  omega

def outputEndpointFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputEndpointStart program +
        (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  simp only [completeLogicalWidth, freshStart, outputEndpointStart]
  omega

def freshFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    freshStart program +
        (PiCCSOrdinaryRetainedBlocks.freshBlock program).coordinateCount ≤
      logicalWidth := by
  exact geometry.completeFits

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment Spec.F logicalWidth)
    (source : Fin (PiCCSOrdinaryRetainedBlocks.sourceWidth program) → Spec.F) :
    Prop where
  priorInput : (PiCCSOrdinaryRetainedBlocks.priorInputBlock program).EncodesAt
    (priorInputStart program) (priorInputFits geometry) assignment source
  outputInput : (PiCCSOrdinaryRetainedBlocks.outputInputBlock program).EncodesAt
    (outputInputStart program) (outputInputFits geometry) assignment source
  freshPublicInput :
    (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program).EncodesAt
      (freshPublicInputStart program) (freshPublicInputFits geometry) assignment
      source
  priorLast : (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).EncodesAt
    (priorLastStart program) (priorLastFits geometry) assignment source
  outputLast : (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).EncodesAt
    (outputLastStart program) (outputLastFits geometry) assignment source
  expectedContext :
    (PiCCSOrdinaryRetainedBlocks.expectedContextBlock program).EncodesAt
      (expectedContextStart program) (expectedContextFits geometry) assignment
      source
  proofLogical :
    (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program).EncodesAt
      (proofLogicalStart program) (proofLogicalFits geometry) assignment source
  outputEndpoint :
    (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock program).EncodesAt
      (outputEndpointStart program) (outputEndpointFits geometry) assignment
      source
  fresh : (PiCCSOrdinaryRetainedBlocks.freshBlock program).EncodesAt
    (freshStart program) (freshFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedGeometry
