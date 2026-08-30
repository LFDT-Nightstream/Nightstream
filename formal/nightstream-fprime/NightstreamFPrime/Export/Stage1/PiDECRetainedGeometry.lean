import NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks
import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry

/-!
Owns the no-gap placement of the seven PiDEC retained blocks after the PiCCS
ordinary retained prefix.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PilotOrdinaryRetainedGeometry.completeLogicalWidth program

def parentCommitmentStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  prefixLogicalWidth program

def parentPublicInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  parentCommitmentStart program +
    (PiDECRetainedBlocks.parentCommitmentBlock program).coordinateCount

def parentEvalKStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  parentPublicInputStart program +
    (PiDECRetainedBlocks.parentPublicInputBlock program).coordinateCount

def parentEvalAStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  parentEvalKStart program +
    (PiDECRetainedBlocks.parentEvalKBlock program).coordinateCount

def proofStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  parentEvalAStart program +
    (PiDECRetainedBlocks.parentEvalABlock program).coordinateCount

def logicalStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  proofStart program + (PiDECRetainedBlocks.proofBlock program).coordinateCount

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  logicalStart program +
    (PiDECRetainedBlocks.logicalBlock program).coordinateCount

def completeLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program + (PiDECRetainedBlocks.freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = 257782729 := by
  simp only [completeLogicalWidth, freshStart, logicalStart, proofStart,
    parentEvalAStart, parentEvalKStart, parentPublicInputStart,
    parentCommitmentStart, prefixLogicalWidth]
  rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
  have count := PiDECRetainedBlocks.retainedCoordinateCount_eq program
  unfold PiDECRetainedBlocks.retainedCoordinateCount at count
  omega

theorem completeLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [completeLogicalWidth_eq]
  norm_num [Lifecycle.cubeVariables]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  completeFits : completeLogicalWidth program ≤ logicalWidth

def pilotOrdinaryGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PilotOrdinaryRetainedGeometry.Geometry program logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth freshStart logicalStart proofStart
      parentEvalAStart parentEvalKStart parentPublicInputStart
      parentCommitmentStart prefixLogicalWidth
    omega

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth :=
  PilotOrdinaryRetainedGeometry.prefixGeometry
    (pilotOrdinaryGeometry geometry)

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiCCSOrdinaryRetainedGeometry.oneColumn (prefixGeometry geometry)

def parentCommitmentFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentCommitmentStart program +
        (PiDECRetainedBlocks.parentCommitmentBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart proofStart parentEvalAStart
    parentEvalKStart parentPublicInputStart parentCommitmentStart
  omega

def parentPublicInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentPublicInputStart program +
        (PiDECRetainedBlocks.parentPublicInputBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart proofStart parentEvalAStart
    parentEvalKStart parentPublicInputStart
  omega

def parentEvalKFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentEvalKStart program +
        (PiDECRetainedBlocks.parentEvalKBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart proofStart parentEvalAStart
    parentEvalKStart
  omega

def parentEvalAFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentEvalAStart program +
        (PiDECRetainedBlocks.parentEvalABlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart proofStart parentEvalAStart
  omega

def proofFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    proofStart program + (PiDECRetainedBlocks.proofBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart proofStart
  omega

def logicalFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    logicalStart program +
        (PiDECRetainedBlocks.logicalBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart
  omega

def freshFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    freshStart program + (PiDECRetainedBlocks.freshBlock program).coordinateCount ≤
      logicalWidth :=
  geometry.completeFits

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (PiDECRetainedBlocks.sourceWidth program) →
      NightstreamFPrime.Spec.F) : Prop where
  parentCommitment :
    (PiDECRetainedBlocks.parentCommitmentBlock program).EncodesAt
      (parentCommitmentStart program) (parentCommitmentFits geometry)
      assignment source
  parentPublicInput :
    (PiDECRetainedBlocks.parentPublicInputBlock program).EncodesAt
      (parentPublicInputStart program) (parentPublicInputFits geometry)
      assignment source
  parentEvalK : (PiDECRetainedBlocks.parentEvalKBlock program).EncodesAt
    (parentEvalKStart program) (parentEvalKFits geometry) assignment source
  parentEvalA : (PiDECRetainedBlocks.parentEvalABlock program).EncodesAt
    (parentEvalAStart program) (parentEvalAFits geometry) assignment source
  proof : (PiDECRetainedBlocks.proofBlock program).EncodesAt
    (proofStart program) (proofFits geometry) assignment source
  logical : (PiDECRetainedBlocks.logicalBlock program).EncodesAt
    (logicalStart program) (logicalFits geometry) assignment source
  fresh : (PiDECRetainedBlocks.freshBlock program).EncodesAt
    (freshStart program) (freshFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.PiDECRetainedGeometry
