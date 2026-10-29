import NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks
import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry

namespace NightstreamFPrime.Export.Stage1.PiDECRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PilotOrdinaryRetainedGeometry.completeLogicalWidth program

def parentCommitmentSlot : Nat := 16 * 22 * 54

def parentPublicInputSlot : Nat := 17 * 22 * 54 + 16 * 5 * 54

def parentEvalKSlot : Nat := 17 * (22 + 5) * 54 + 16 * 54 * 2

def parentEvalASlot : Nat := 17 * (22 + 5 + 2) * 54 + 16 * 4 * 54 * 2

def parentCommitmentStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.productOutputStart program + parentCommitmentSlot * 41

def parentPublicInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.productOutputStart program + parentPublicInputSlot * 41

def parentEvalKStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.productOutputStart program + parentEvalKSlot * 41

def parentEvalAStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.productOutputStart program + parentEvalASlot * 41

def proofStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  RunningTransitionRetainedGeometry.piDecStart program

def logicalStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  prefixLogicalWidth program

def freshStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  logicalStart program +
    (PiDECRetainedBlocks.logicalBlock program).coordinateCount

def completeLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  freshStart program + (PiDECRetainedBlocks.freshBlock program).coordinateCount

@[simp] theorem completeLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program = prefixLogicalWidth program +
      PiDECRetainedBlocks.retainedCoordinateCount program := by
  simp only [completeLogicalWidth, freshStart, logicalStart,
    PiDECRetainedBlocks.retainedCoordinateCount, Nat.add_assoc]

theorem completeLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth program ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [completeLogicalWidth_eq, prefixLogicalWidth,
    PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq,
    PiDECRetainedBlocks.retainedCoordinateCount_eq]
  decide

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  completeFits : completeLogicalWidth program ≤ logicalWidth

theorem pilotOrdinaryGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PilotOrdinaryRetainedGeometry.Geometry program logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth freshStart logicalStart prefixLogicalWidth
    omega

theorem prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth :=
  PilotOrdinaryRetainedGeometry.prefixGeometry
    (pilotOrdinaryGeometry geometry)

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiCCSOrdinaryRetainedGeometry.oneColumn (prefixGeometry geometry)

theorem piRlcGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  PiCCSPoseidonPlan.prefixGeometry <|
    PiCCSOrdinaryRetainedGeometry.poseidonGeometry (prefixGeometry geometry)

theorem parentCommitmentFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentCommitmentStart program +
        (PiDECRetainedBlocks.parentCommitmentBlock program).coordinateCount ≤
      logicalWidth := by
  have fits := PiRLCRetainedGeometry.productOutputFits (piRlcGeometry geometry)
  change PiRLCRetainedGeometry.productOutputStart program + 33966 * 41 ≤
    logicalWidth at fits
  change PiRLCRetainedGeometry.productOutputStart program + 19008 * 41 +
    1188 * 41 ≤ logicalWidth
  omega

theorem parentPublicInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentPublicInputStart program +
        (PiDECRetainedBlocks.parentPublicInputBlock program).coordinateCount ≤
      logicalWidth := by
  have fits := PiRLCRetainedGeometry.productOutputFits (piRlcGeometry geometry)
  change PiRLCRetainedGeometry.productOutputStart program + 33966 * 41 ≤
    logicalWidth at fits
  change PiRLCRetainedGeometry.productOutputStart program + 24516 * 41 +
    270 * 41 ≤ logicalWidth
  omega

theorem parentEvalKFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentEvalKStart program +
        (PiDECRetainedBlocks.parentEvalKBlock program).coordinateCount ≤
      logicalWidth := by
  have fits := PiRLCRetainedGeometry.productOutputFits (piRlcGeometry geometry)
  change PiRLCRetainedGeometry.productOutputStart program + 33966 * 41 ≤
    logicalWidth at fits
  change PiRLCRetainedGeometry.productOutputStart program + 26514 * 41 +
    108 * 41 ≤ logicalWidth
  omega

theorem parentEvalAFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    parentEvalAStart program +
        (PiDECRetainedBlocks.parentEvalABlock program).coordinateCount ≤
      logicalWidth := by
  have fits := PiRLCRetainedGeometry.productOutputFits (piRlcGeometry geometry)
  have outputCount :
      (PiRLCRetainedGeometry.productOutputBlock program).coordinateCount = 1392606 := by
    simp [PiRLCRetainedGeometry.productOutputBlock]
  have evalACount : (PiDECRetainedBlocks.parentEvalABlock program).coordinateCount =
      Lifecycle.productionShape.matrixCount * (Spec.ringDegree * 2) * 41 := rfl
  have matrices : Lifecycle.productionShape.matrixCount = Spec.productionProfile.ccsMatrices := rfl
  have ccs : Spec.productionProfile.ccsMatrices = 4 := by decide
  have degree : Spec.ringDegree = 54 := rfl
  rw [matrices, ccs, degree] at evalACount
  have startEq : parentEvalAStart program =
      PiRLCRetainedGeometry.productOutputStart program + parentEvalASlot * 41 := rfl
  have slotEq : parentEvalASlot = 33534 := rfl
  omega

theorem proofFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    proofStart program + (PiDECRetainedBlocks.proofBlock program).coordinateCount ≤
      logicalWidth := by
  exact RunningTransitionRetainedGeometry.piDecFits
    (PiCCSOrdinaryRetainedGeometry.prefixGeometry (prefixGeometry geometry))

theorem logicalFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    logicalStart program +
        (PiDECRetainedBlocks.logicalBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans (m := completeLogicalWidth program) _ geometry.completeFits
  unfold completeLogicalWidth freshStart logicalStart
  omega

theorem freshFits {program : Lifecycle.Stage1.Application.Program}
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
