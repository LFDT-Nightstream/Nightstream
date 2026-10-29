import NightstreamFPrime.Layout.MatrixProgram.PlanBridge
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

namespace NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
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

def piDecRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (piDecBlock program) (piDecStart program)
    (Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart)
    RunningTransitionSourceSupport.piDecCount 0

def freshRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (freshBlock program) (freshStart program)
    (Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset) freshCount 0

def substitution (program : ApplicationProgram) : SourceSubstitution where
  ranges := [stateRange program, outputRange program, piDecRange program,
    freshRange program]
  grids := [PiCCSTranscriptOutputForms.pointGrid program]

def rowSchedule : IndexSchedule :=
  .rangeList [⟨RunningTransitionArithmetic.rowStart, 27800⟩]

@[simp] theorem rowSchedule_count : rowSchedule.count = 27800 := by
  rfl

def ordinaryBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block where
  rows := rowSchedule
  oneColumn := (oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.base program

def matrixProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (ordinaryBlock geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 27800 := by
  rfl

end NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram
