import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.Retained

/-!
Owns the canonical low-norm placement for the direct running-transition
source support. Four blocks extend the PiCCS payload without gaps. State and
output use the coordinates of the actual pilot preimages.

This module does not compile transition rows or construct an assignment.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionRetainedBlocks

def stateStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCPoseidonGeometry.priorInputStart program + 28 * 41

theorem retainedPrefix_le_stateStart
    (program : Lifecycle.Stage1.Application.Program) :
    PiRLCRetainedGeometry.prefixLogicalWidth program ≤ stateStart program := by
  unfold stateStart PiRLCPoseidonGeometry.priorInputStart
  omega

def outputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCPoseidonGeometry.outputInputStart program

def roundC0Start (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSActionPayloadBlock.logicalWidth program

/-- The first allocated transition block starts exactly after the PiCCS
payload. The state and output views allocate no suffix coordinates. -/
theorem payloadEnd_eq_roundC0Start
    (program : Lifecycle.Stage1.Application.Program) :
    PiCCSActionPayloadBlock.payloadStart program +
        (PiCCSActionPayloadBlock.block program).coordinateCount =
      roundC0Start program := by
  rfl

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
    completeLogicalWidth program = 209405476 := by
  unfold completeLogicalWidth freshStart piDecStart roundC1Start roundC0Start
  rw [PiCCSActionPayloadBlock.logicalWidth_eq]
  change 195242354 + 28 * 41 + 28 * 41 +
    49248 * 41 + 296138 * 41 = 209405476
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

def poseidonGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiCCSPoseidonPlan.Geometry program logicalWidth where
  payloadFits := by
    apply Nat.le_trans _ geometry.completeFits
    rw [PiCCSActionPayloadBlock.logicalWidth_eq, completeLogicalWidth_eq]
    omega

/-- The actual pilot preimages already belong to the preceding prefix. -/
def pilotGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCPoseidonGeometry.Geometry program logicalWidth where
  pilotFits := by
    have complete := geometry.completeFits
    rw [completeLogicalWidth_eq] at complete
    rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq]
    omega

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  PiRLCPoseidonGeometry.prefixGeometry (pilotGeometry geometry)

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiRLCRetainedGeometry.oneColumn (prefixGeometry geometry)

def stateFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    stateStart program + (stateBlock program).coordinateCount ≤ logicalWidth := by
  have parent := PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry)
  have width : (PiRLCPoseidonGeometry.priorInputBlock program).coordinateCount =
      2025113 := by simp [PiRLCPoseidonGeometry.priorInputBlock]
  rw [width] at parent
  change PiRLCPoseidonGeometry.priorInputStart program + 28 * 41 + 11 * 41 ≤
    logicalWidth
  omega

def outputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputStart program + (outputBlock program).coordinateCount ≤ logicalWidth := by
  have width : (outputBlock program).coordinateCount =
      (PiRLCPoseidonGeometry.outputInputBlock program).coordinateCount := by
    rw [outputBlock, fieldBlock_coordinateCount,
      Layout.Stage1.RunningTransitionSourceSupport.outputCount_eq]
    simp [PiRLCPoseidonGeometry.outputInputBlock]
  rw [width]
  exact PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry)

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

/-- Exact encoding obligation for the four allocated blocks and the two
shared pilot-preimage views. -/
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
  sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits (poseidonGeometry geometry)) assignment
    (PiCCSActionPayloadBlock.sourceAssignment program source)

end NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry
