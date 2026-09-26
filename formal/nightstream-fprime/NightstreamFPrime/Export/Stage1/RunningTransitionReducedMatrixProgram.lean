import NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedBlocks
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputForms
import NightstreamFPrime.Layout.MatrixProgram.Program

/-!
Compact candidate matrix data for the reduced running transition. Every
operand uses the existing pilot, PiDEC, or Poseidon retained coordinates.
Only the inverse field and shared Boolean flag use local coordinates.
No block reads the physical R1CS row accessor. The selected package does not
use this program yet.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixProgram

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.MatrixProgram.AffineGrid

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def region (majorCount middleCount minorStart minorCount : Nat) : Region :=
  ⟨0, majorCount, 0, middleCount, minorStart, minorCount⟩

def constant (selected : Region) (coefficient : F) : Rule :=
  ⟨selected, .constant coefficient.val⟩

def retained (selected : Region) (wire : RetainedBlock)
    (slot majorStride middleStride minorStride : Nat) (coefficient : F) : Rule :=
  ⟨selected, .retained wire slot majorStride middleStride minorStride coefficient.val⟩

def stateWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (RunningTransitionRetainedBlocks.stateBlock program)
    (RunningTransitionRetainedGeometry.stateStart program)

def outputWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (RunningTransitionRetainedBlocks.outputBlock program)
    (RunningTransitionRetainedGeometry.outputStart program)

def piDecWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (RunningTransitionRetainedBlocks.piDecBlock program)
    (RunningTransitionRetainedGeometry.piDecStart program)

def poseidonWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)

def inverseWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (RunningTransitionReducedRetainedBlocks.inverseBlock program)
    (RunningTransitionReducedRetainedBlocks.inverseStart program)

def flagWire (program : ApplicationProgram) : RetainedBlock :=
  .ofSemantic (RunningTransitionReducedRetainedBlocks.flagBlock program)
    (RunningTransitionReducedRetainedBlocks.flagStart program)

def flagProgram (program : ApplicationProgram) (selected : Region) :
    AffineGrid.Program :=
  ⟨[retained selected (flagWire program) 0 0 0 0 1]⟩

def baseProgram (program : ApplicationProgram) (selected : Region) :
    AffineGrid.Program :=
  ⟨[constant selected 1, retained selected (flagWire program) 0 0 0 0 (-1)]⟩

def flagGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  let selected := region 1 1 0 1
  { shape := ⟨1, 1, 1⟩
    oneColumn
    left := ⟨[retained selected (stateWire program) 0 0 0 0 1]⟩
    right := ⟨[retained selected (inverseWire program) 0 0 0 0 1]⟩
    output := flagProgram program selected }

def bindingGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  let selected := region 1 1 0 1
  { shape := ⟨1, 1, 1⟩
    oneColumn
    left := ⟨[retained selected (stateWire program) 0 0 0 0 1]⟩
    right := baseProgram program selected
    output := ⟨[]⟩ }

def pointHeaderGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  { shape := ⟨1, 1, 1⟩
    oneColumn
    left := flagProgram program (region 1 1 0 1)
    right := ⟨[]⟩
    output := ⟨[]⟩ }

/-- Keep the exact ordered external-layer summands used by `SparseLayer.external`.
Repeated first-block terms express the two copies in output lane zero. -/
def pointTerms : List (Nat × Nat) :=
  [(0, 2), (1, 3), (2, 1), (3, 1),
   (0, 2), (1, 3), (2, 1), (3, 1),
   (4, 2), (5, 3), (6, 1), (7, 1)]

def pointGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  let selected := region Lifecycle.productionShape.cubeVariables 2 0 1
  let c0 := PiCCSTranscriptOutputForms.pointGrid program 0
  let c1 := PiCCSTranscriptOutputForms.pointGrid program 1
  { shape := ⟨Lifecycle.productionShape.cubeVariables, 2, 1⟩
    oneColumn
    left := flagProgram program selected
    right := ⟨pointTerms.map fun term =>
      retained selected (poseidonWire program) (c0.slotStart + term.1)
        c0.majorSlotStride (c1.slotStart - c0.slotStart) 0
        (Spec.Poseidon2.ofNat term.2)⟩
    output := ⟨[retained selected (outputWire program)
      PiCCSInputs.runningPointStart 2 1 0 1]⟩ }

def groupsGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  let count := Lifecycle.productionShape.runningCount
  let commitment := region count 1
    (PiCCSInputs.runningCommitmentStart 0 - PiCCSInputs.runningGroupStart 0)
    PiCCSInputs.runningCommitmentWords
  let publicInput := region count 1
    (PiCCSInputs.runningPublicStart 0 - PiCCSInputs.runningGroupStart 0)
    PiCCSInputs.runningPublicWords
  let evalStart := PiCCSInputs.runningEvaluationStart 0 - PiCCSInputs.runningGroupStart 0
  let evalK := region count 1 evalStart PiDECInputs.evalKWordsPerChild
  let evalA := region count 1 (evalStart + PiDECInputs.evalKWordsPerChild)
    PiDECInputs.evalAWordsPerChild
  { shape := ⟨count, 1, PiCCSInputs.runningGroupWords⟩
    oneColumn
    left := flagProgram program (region count 1 0 PiCCSInputs.runningGroupWords)
    right := ⟨[
      retained commitment (piDecWire program) 0
        PiDECInputs.commitmentWordsPerChild 0 1 1,
      retained publicInput (piDecWire program)
        (PiDECInputs.publicInputStart - PiDECInputs.proofInputStart)
        PiDECInputs.publicInputWordsPerChild 0 1 1,
      retained evalK (piDecWire program)
        (PiDECInputs.evalKInputStart - PiDECInputs.proofInputStart)
        PiDECInputs.evalKWordsPerChild 0 1 1,
      retained evalA (piDecWire program)
        (PiDECInputs.evalAInputStart - PiDECInputs.proofInputStart)
        PiDECInputs.evalAWordsPerChild 0 1 1]⟩
    output := ⟨[
      retained commitment (outputWire program) (PiCCSInputs.runningCommitmentStart 0)
        PiCCSInputs.runningGroupWords 0 1 1,
      retained publicInput (outputWire program) (PiCCSInputs.runningPublicStart 0)
        PiCCSInputs.runningGroupWords 0 1 1,
      retained evalK (outputWire program) (PiCCSInputs.runningEvaluationStart 0)
        PiCCSInputs.runningGroupWords 0 1 1,
      retained evalA (outputWire program)
        (PiCCSInputs.runningEvaluationStart 0 + PiDECInputs.evalKWordsPerChild)
        PiCCSInputs.runningGroupWords 0 1 1]⟩ }

def stateGrid (program : ApplicationProgram) (oneColumn : Nat) :
    MultiplicationGrid.Block :=
  let selected := region 1 1 0 Lifecycle.Stage1.RunningTransition.stateWordCount
  { shape := ⟨1, 1, Lifecycle.Stage1.RunningTransition.stateWordCount⟩
    oneColumn
    left := baseProgram program selected
    right := ⟨[
      retained selected (stateWire program)
        (RunningTransitionInputs.initialStateWordStart - RunningTransitionInputs.iterationWordIndex)
        0 0 1 1,
      retained selected (stateWire program)
        (RunningTransitionInputs.currentStateWordStart - RunningTransitionInputs.iterationWordIndex)
        0 0 1 (-1)]⟩
    output := ⟨[]⟩ }

def matrixProgram (program : ApplicationProgram) (oneColumn : Nat) :
    MatrixProgram.Program :=
  ⟨[.multiplicationGrid (flagGrid program oneColumn),
    .multiplicationGrid (bindingGrid program oneColumn),
    .multiplicationGrid (pointHeaderGrid program oneColumn),
    .multiplicationGrid (pointGrid program oneColumn),
    .multiplicationGrid (groupsGrid program oneColumn),
    .multiplicationGrid (stateGrid program oneColumn)]⟩

@[simp] theorem row_count (program : ApplicationProgram) (oneColumn : Nat) :
    (matrixProgram program oneColumn).rowCount = 49359 := by
  simp only [matrixProgram, MatrixProgram.Program.rowCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, MatrixProgram.Block.rowCount,
    MultiplicationGrid.Block.rowCount, flagGrid, bindingGrid, pointHeaderGrid,
    pointGrid, groupsGrid, stateGrid, MultiplicationGrid.Shape.rowCount]
  rfl

theorem local_coordinate_count (program : ApplicationProgram) :
    (inverseWire program).coordinateCount + (flagWire program).coordinateCount = 42 := by
  rfl

/-- This candidate cannot read an old R1CS source row. -/
theorem sourceRow_independent (program : ApplicationProgram)
    (oneColumn logicalWidth ordinal : Nat)
    (before after : Nat → Option R1CS.Row) :
    (matrixProgram program oneColumn).row? logicalWidth before ordinal =
      (matrixProgram program oneColumn).row? logicalWidth after ordinal := by
  rfl

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixProgram
