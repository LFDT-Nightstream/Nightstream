import NightstreamFPrime.Layout.MatrixProgram.PlanBridge
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PiDECDirectPlan

/-!
Owns the compact matrix program for the four nonempty PiDEC row families.
Lean fixes all source ranges, physical row ranges, and the exact parent order:
public split, commitment, Eval_K, then Eval_A.

This module defines executable package data. Row equality to the canonical
PiDEC plan is proved separately.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle
open PiDECRetainedBlocks
open PiDECRetainedGeometry

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def parentCommitmentRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (parentCommitmentBlock program)
    (parentCommitmentStart program)
    (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart)
    PiDECInputs.commitmentWordsPerChild 0

def parentPublicInputRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (parentPublicInputBlock program)
    (parentPublicInputStart program)
    (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart)
    PiDECInputs.publicInputWordsPerChild 0

def parentEvalKRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (parentEvalKBlock program)
    (parentEvalKStart program)
    (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart)
    PiDECInputs.evalKWordsPerChild 0

def parentEvalARange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (parentEvalABlock program)
    (parentEvalAStart program)
    (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart)
    PiDECInputs.evalAWordsPerChild 0

/-- The child proof source resolves to the same logical forms used by the
recursive running transition. This range adds no allocation or copy row. -/
def proofRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (proofBlock program) (proofStart program)
    (Spartan.sourceToSpartan PiDECInputs.proofInputStart)
    PiDECInputs.proofInputColumnCount 0

def logicalRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (logicalBlock program) (logicalStart program)
    (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart) logicalCount 0

def freshRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (freshBlock program) (freshStart program)
    (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart) freshCount 0

/-- Complete fail-closed PiDEC source substitution in increasing Spartan
column order. -/
def substitution (program : ApplicationProgram) : SourceSubstitution where
  ranges := [parentCommitmentRange program, parentPublicInputRange program,
    parentEvalKRange program, parentEvalARange program, proofRange program,
    logicalRange program, freshRange program]

def publicSchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.publicInputRowStart, Layout.PiDEC.v1_1.PublicInputSplit.physicalRowCount⟩]

def commitmentSchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.commitmentRowStart, Layout.PiDEC.v1_1.CommitmentRecomposition.physicalRowCount⟩]

def evalKSchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.evalKRowStart, Layout.PiDEC.v1_1.EvalKRecomposition.physicalRowCount⟩]

def evalASchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.evalARowStart, Layout.PiDEC.v1_1.EvalARecomposition.physicalRowCount⟩]

@[simp] theorem publicSchedule_count : publicSchedule.count =
    Layout.PiDEC.v1_1.PublicInputSplit.physicalRowCount := by
  rfl

@[simp] theorem commitmentSchedule_count : commitmentSchedule.count =
    Layout.PiDEC.v1_1.CommitmentRecomposition.physicalRowCount := by
  rfl

@[simp] theorem evalKSchedule_count : evalKSchedule.count =
    Layout.PiDEC.v1_1.EvalKRecomposition.physicalRowCount := by
  rfl

@[simp] theorem evalASchedule_count : evalASchedule.count =
    Layout.PiDEC.v1_1.EvalARecomposition.physicalRowCount := by
  rfl

def ordinaryBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (rows : IndexSchedule) :
    Ordinary.Block where
  rows := rows
  oneColumn := (oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.base program

def publicBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block :=
  ordinaryBlock geometry publicSchedule

def commitmentBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block :=
  ordinaryBlock geometry commitmentSchedule

def evalKBlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block :=
  ordinaryBlock geometry evalKSchedule

def evalABlock {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block :=
  ordinaryBlock geometry evalASchedule

def singletonProgram (block : Ordinary.Block) : MatrixProgram.Program where
  blocks := [.ordinary block]

def publicProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  singletonProgram (publicBlock geometry)

def commitmentProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  singletonProgram (commitmentBlock geometry)

def evalKProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  singletonProgram (evalKBlock geometry)

def evalAProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  singletonProgram (evalABlock geometry)

def evaluationProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  (evalKProgram geometry).append (evalAProgram geometry)

def recompositionProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  (commitmentProgram geometry).append (evaluationProgram geometry)

/-- Exact canonical parent order, matching `PiDECDirectPlan.plan`. -/
def matrixProgram {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program :=
  (publicProgram geometry).append (recompositionProgram geometry)

@[simp] theorem publicProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (publicProgram geometry).rowCount =
      Layout.PiDEC.v1_1.PublicInputSplit.physicalRowCount := by
  rfl

@[simp] theorem commitmentProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (commitmentProgram geometry).rowCount =
      Layout.PiDEC.v1_1.CommitmentRecomposition.physicalRowCount := by
  rfl

@[simp] theorem evalKProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evalKProgram geometry).rowCount =
      Layout.PiDEC.v1_1.EvalKRecomposition.physicalRowCount := by
  rfl

@[simp] theorem evalAProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evalAProgram geometry).rowCount =
      Layout.PiDEC.v1_1.EvalARecomposition.physicalRowCount := by
  rfl

@[simp] theorem evaluationProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evaluationProgram geometry).rowCount =
      Layout.PiDEC.v1_1.EvalKRecomposition.physicalRowCount +
        Layout.PiDEC.v1_1.EvalARecomposition.physicalRowCount := by
  simp [evaluationProgram]

@[simp] theorem recompositionProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (recompositionProgram geometry).rowCount =
      Layout.PiDEC.v1_1.CommitmentRecomposition.physicalRowCount +
        (Layout.PiDEC.v1_1.EvalKRecomposition.physicalRowCount +
          Layout.PiDEC.v1_1.EvalARecomposition.physicalRowCount) := by
  simp [recompositionProgram]

@[simp] theorem matrixProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount =
      Layout.PiDEC.v1_1.exactRowCount := by
  simp [matrixProgram, recompositionProgram, evaluationProgram,
    publicProgram, commitmentProgram, evalKProgram, evalAProgram,
    singletonProgram, publicBlock, commitmentBlock, evalKBlock, evalABlock,
    ordinaryBlock, MatrixProgram.Block.rowCount, Ordinary.Block.rowCount,
    publicSchedule, commitmentSchedule, evalKSchedule, evalASchedule,
    IndexSchedule.count, Layout.PiDEC.v1_1.exactRowCount,
    Layout.PiDEC.v1_1.exactRowDeltas, Nat.add_assoc]

end NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
