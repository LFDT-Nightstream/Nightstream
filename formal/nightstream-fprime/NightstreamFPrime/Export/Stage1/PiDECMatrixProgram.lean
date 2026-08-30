import NightstreamFPrime.Export.MatrixProgram.PlanBridge
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

open NightstreamFPrime.Export.MatrixProgram
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

def proofRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (proofBlock program) (proofStart program)
    (Spartan.sourceToSpartan PiDECInputs.proofInputStart)
    PiDECInputs.proofInputColumnCount 0

def logicalRange (program : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (logicalBlock program) (logicalStart program)
    (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart) 270 0

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
  .rangeList [⟨PiDECStarts.publicInputRowStart, 22680⟩]

def commitmentSchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.commitmentRowStart, 972⟩]

def evalKSchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.evalKRowStart, 108⟩]

def evalASchedule : IndexSchedule :=
  .rangeList [⟨PiDECStarts.evalARowStart, 1512⟩]

@[simp] theorem publicSchedule_count : publicSchedule.count = 22680 := by
  rfl

@[simp] theorem commitmentSchedule_count : commitmentSchedule.count = 972 := by
  rfl

@[simp] theorem evalKSchedule_count : evalKSchedule.count = 108 := by
  rfl

@[simp] theorem evalASchedule_count : evalASchedule.count = 1512 := by
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
    (publicProgram geometry).rowCount = 22680 := by
  rfl

@[simp] theorem commitmentProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (commitmentProgram geometry).rowCount = 972 := by
  rfl

@[simp] theorem evalKProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evalKProgram geometry).rowCount = 108 := by
  rfl

@[simp] theorem evalAProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evalAProgram geometry).rowCount = 1512 := by
  rfl

@[simp] theorem evaluationProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (evaluationProgram geometry).rowCount = 1620 := by
  simp [evaluationProgram]

@[simp] theorem recompositionProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (recompositionProgram geometry).rowCount = 2592 := by
  simp [recompositionProgram]

@[simp] theorem matrixProgram_rowCount
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 25272 := by
  simp [matrixProgram, recompositionProgram, evaluationProgram,
    publicProgram, commitmentProgram, evalKProgram, evalAProgram,
    singletonProgram, publicBlock, commitmentBlock, evalKBlock, evalABlock,
    ordinaryBlock, MatrixProgram.Block.rowCount, Ordinary.Block.rowCount,
    publicSchedule, commitmentSchedule, evalKSchedule, evalASchedule,
    IndexSchedule.count]

end NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
