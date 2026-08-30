import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

/-!
Owns the canonical low-norm placement for the direct running-transition
source support. The six blocks extend the proved PiRLC prefix without gaps.

This module does not compile transition rows or construct an assignment.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionRetainedBlocks

def stateStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSActionPayloadBlock.logicalWidth program

/-- The running-transition support starts exactly after the PiCCS payload. -/
theorem payloadEnd_eq_stateStart
    (program : Lifecycle.Stage1.Application.Program) :
    PiCCSActionPayloadBlock.payloadStart program +
        (PiCCSActionPayloadBlock.block program).coordinateCount =
      stateStart program := by
  rfl

theorem retainedPrefix_le_stateStart
    (program : Lifecycle.Stage1.Application.Program) :
    PiRLCRetainedGeometry.prefixLogicalWidth program ≤ stateStart program := by
  rw [PiRLCRetainedGeometry.prefixLogicalWidth_eq]
  unfold stateStart
  rw [PiCCSActionPayloadBlock.logicalWidth_eq]
  omega

def outputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  stateStart program + (stateBlock program).coordinateCount

def roundC0Start (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputStart program + (outputBlock program).coordinateCount

def roundC1Start (program : Lifecycle.Stage1.Application.Program) : Nat :=
  roundC0Start program + (roundC0Block program).coordinateCount

def piDecStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  roundC1Start program + (roundC1Block program).coordinateCount

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piDecStart program + (piDecBlock program).coordinateCount

def completeLogicalWidth
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program + (freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 200597938 := by
  unfold completeLogicalWidth freshStart piDecStart roundC1Start roundC0Start
    outputStart stateStart
  rw [PiCCSActionPayloadBlock.logicalWidth_eq]
  change 185542820 + 11 * 41 + 45937 * 41 + 28 * 41 + 28 * 41 +
    45792 * 41 + 275402 * 41 = 200597938
  norm_num

theorem completeLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [completeLogicalWidth_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  completeFits : completeLogicalWidth program ≤ logicalWidth

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth where
  prefixFits := by
    apply Nat.le_trans (retainedPrefix_le_stateStart program)
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth freshStart piDecStart roundC1Start roundC0Start
      outputStart stateStart
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiRLCRetainedGeometry.oneColumn (prefixGeometry geometry)

def stateFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    stateStart program + (stateBlock program).coordinateCount ≤ logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart piDecStart roundC1Start roundC0Start
    outputStart
  omega

def outputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputStart program + (outputBlock program).coordinateCount ≤ logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart piDecStart roundC1Start roundC0Start
  omega

def roundC0Fits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    roundC0Start program + (roundC0Block program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart piDecStart roundC1Start
  omega

def roundC1Fits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    roundC1Start program + (roundC1Block program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart piDecStart
  omega

def piDecFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    piDecStart program + (piDecBlock program).coordinateCount ≤ logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth freshStart
  omega

def freshFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    freshStart program + (freshBlock program).coordinateCount ≤ logicalWidth :=
  geometry.completeFits

/-- Exact encoding obligation for the six transition support blocks. -/
structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (sourceWidth program) → NightstreamFPrime.Spec.F) : Prop where
  state : (stateBlock program).EncodesAt
    (stateStart program) (stateFits geometry) assignment source
  output : (outputBlock program).EncodesAt
    (outputStart program) (outputFits geometry) assignment source
  roundC0 : (roundC0Block program).EncodesAt
    (roundC0Start program) (roundC0Fits geometry) assignment source
  roundC1 : (roundC1Block program).EncodesAt
    (roundC1Start program) (roundC1Fits geometry) assignment source
  piDec : (piDecBlock program).EncodesAt
    (piDecStart program) (piDecFits geometry) assignment source
  fresh : (freshBlock program).EncodesAt
    (freshStart program) (freshFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry
