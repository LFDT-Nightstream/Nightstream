import NightstreamFPrime.Export.MatrixProgram.PlanBridge
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

/-!
Owns the compact matrix program for the running-instance transition. Lean
fixes the one contiguous row schedule and the six retained source families.
The two round-challenge families use affine grids; all other families use
contiguous ranges.

This module defines executable package data. Exact row equality to the
canonical direct plan is proved separately.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionRetainedBlocks
open RunningTransitionRetainedGeometry

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def stateRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (stateBlock program) (stateStart program)
    (Spartan.sourceToSpartan RunningTransitionSourceSupport.stateStart)
    RunningTransitionSourceSupport.stateCount 0

def outputRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (outputBlock program) (outputStart program)
    (Spartan.sourceToSpartan RunningTransitionSourceSupport.outputStart)
    RunningTransitionSourceSupport.outputCount 0

def roundC0SourceStart : Nat :=
  PiCCSStarts.roundTranscriptWitnessStart +
    RunningTransitionInputs.roundSampleC0Offset

def roundC1SourceStart : Nat :=
  PiCCSStarts.roundTranscriptWitnessStart +
    RunningTransitionInputs.roundSampleC1Offset

def roundC0Grid (program : ApplicationProgram) : SourceGrid :=
  SourceGrid.ofSemantic (roundC0Block program) (roundC0Start program)
    (Spartan.sourceToSpartan roundC0SourceStart)
    productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 1 1 0 1 0

def roundC1Grid (program : ApplicationProgram) : SourceGrid :=
  SourceGrid.ofSemantic (roundC1Block program) (roundC1Start program)
    (Spartan.sourceToSpartan roundC1SourceStart)
    productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 1 1 0 1 0

def piDecRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (piDecBlock program) (piDecStart program)
    (Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart)
    RunningTransitionSourceSupport.piDecCount 0

def freshRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (freshBlock program) (freshStart program)
    (Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset) freshCount 0

/-- Complete fail-closed running-transition source substitution. -/
def substitution (program : ApplicationProgram) : SourceSubstitution where
  ranges := [stateRange program, outputRange program, piDecRange program,
    freshRange program]
  grids := [roundC0Grid program, roundC1Grid program]

def rowSchedule : IndexSchedule :=
  .rangeList [⟨RunningTransitionArithmetic.rowStart, 321303⟩]

@[simp] theorem rowSchedule_count : rowSchedule.count = 321303 := by
  rfl

def ordinaryBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block where
  rows := rowSchedule
  oneColumn := (oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.base program

/-- The transition is one ordered ordinary-row block. -/
def matrixProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (ordinaryBlock geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 321303 := by
  rfl

end NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram
