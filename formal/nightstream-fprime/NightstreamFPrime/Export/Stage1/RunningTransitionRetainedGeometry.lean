import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.Retained

/-!
Owns the canonical low-norm placement for the direct running-transition
source support. Two blocks extend the pilot prefix without gaps. State and
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

def piDecStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCPoseidonGeometry.pilotLogicalWidth program

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piDecStart program + (piDecBlock program).coordinateCount

def completeLogicalWidth
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program + (freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 208156124 := by
  have retained := retainedCoordinateCount_eq program
  simp only [retainedCoordinateCount] at retained
  unfold completeLogicalWidth freshStart piDecStart
  rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq]
  omega

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
  pilotFits := by
    apply Nat.le_trans _ geometry.completeFits
    rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq, completeLogicalWidth_eq]
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

/-- Exact encoding obligation for the two allocated blocks and the two
shared pilot-preimage views. -/
structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (sourceWidth program) → NightstreamFPrime.Spec.F) : Prop where
  state : (stateBlock program).EncodesAt
    (stateStart program) (stateFits geometry) assignment source
  output : (outputBlock program).EncodesAt
    (outputStart program) (outputFits geometry) assignment source
  piDec : (piDecBlock program).EncodesAt
    (piDecStart program) (piDecFits geometry) assignment source
  fresh : (freshBlock program).EncodesAt
    (freshStart program) (freshFits geometry) assignment source
  sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits (poseidonGeometry geometry)) assignment
    (PiCCSActionPayloadBlock.sourceAssignment program source)

end NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry
