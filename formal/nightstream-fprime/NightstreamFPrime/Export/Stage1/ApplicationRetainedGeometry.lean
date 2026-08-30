import NightstreamFPrime.Export.Stage1.ApplicationRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry

/-!
Owns the canonical low-norm placement of one verifier-selected application.
The four application blocks extend the running-prefix geometry without gaps.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ApplicationRetainedBlocks

def inputStart (application : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth application

def witnessStart (application : Lifecycle.Stage1.Application.Program) : Nat :=
  inputStart application + (inputBlock application).coordinateCount

def outputStart (application : Lifecycle.Stage1.Application.Program) : Nat :=
  witnessStart application + (witnessBlock application).coordinateCount

def localStart (application : Lifecycle.Stage1.Application.Program) : Nat :=
  outputStart application + (outputBlock application).coordinateCount

def completeLogicalWidth
    (application : Lifecycle.Stage1.Application.Program) : Nat :=
  localStart application + (localBlock application).coordinateCount

theorem completeLogicalWidth_eq
    (application : Lifecycle.Stage1.Application.Program) :
    completeLogicalWidth application =
      266771241 + retainedCoordinateCount application := by
  unfold completeLogicalWidth localStart outputStart witnessStart inputStart
  rw [PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq]
  unfold retainedCoordinateCount
  omega

/-- One concrete application must separately prove its low-norm retained
width fits the owner-selected row cube. -/
structure FitsTwoPow28
    (application : Lifecycle.Stage1.Application.Program) : Prop where
  complete : completeLogicalWidth application ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables

theorem completeLogicalWidth_le_cube
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    completeLogicalWidth application ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables :=
  fits.complete

structure Geometry (application : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  completeFits : completeLogicalWidth application ≤ logicalWidth

def prefixGeometry {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    PiRLCSamplerOrdinaryRetainedGeometry.Geometry application logicalWidth where
  completeFits := by
    apply Nat.le_trans _ geometry.completeFits
    unfold completeLogicalWidth localStart outputStart witnessStart inputStart
    omega

def oneColumn {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    Fin logicalWidth :=
  PiRLCSamplerOrdinaryRetainedGeometry.oneColumn (prefixGeometry geometry)

def inputFits {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    inputStart application + (inputBlock application).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth localStart outputStart witnessStart
  omega

def witnessFits {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    witnessStart application + (witnessBlock application).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth localStart outputStart
  omega

def outputFits {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    outputStart application + (outputBlock application).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth localStart
  omega

def localFits {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    localStart application + (localBlock application).coordinateCount ≤
      logicalWidth :=
  geometry.completeFits

structure Encodes {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (sourceWidth application) → NightstreamFPrime.Spec.F) : Prop where
  input : (inputBlock application).EncodesAt
    (inputStart application) (inputFits geometry) assignment source
  witness : (witnessBlock application).EncodesAt
    (witnessStart application) (witnessFits geometry) assignment source
  output : (outputBlock application).EncodesAt
    (outputStart application) (outputFits geometry) assignment source
  localValues : (localBlock application).EncodesAt
    (localStart application) (localFits geometry) assignment source

end NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry
