import NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

/-!
Owns the seven exact source intervals retained by the nonempty PiDEC rows:
four parent outputs, the proof input, the logical split cells, and the R1CS
fresh interval.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

def sourceFieldBlock (program : Lifecycle.Stage1.Application.Program)
    (count start : Nat) (bounded : start + count ≤ Spartan.SourceColumnCount) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := count
  source := fun index =>
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (start + index.val) (by
        have indexBound := index.isLt
        omega)

def parentCommitmentBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.commitmentWordsPerChild
    parentCommitmentStart (by
      rw [parentCommitmentStart_eq, Spartan.sourceColumnCount_eq]
      norm_num [PiDECInputs.commitmentWordsPerChild])

def parentPublicInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.publicInputWordsPerChild
    parentPublicInputStart (by
      rw [parentPublicInputStart_eq, Spartan.sourceColumnCount_eq]
      norm_num [PiDECInputs.publicInputWordsPerChild])

def parentEvalKBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.evalKWordsPerChild parentEvalKStart (by
    rw [parentEvalKStart_eq, Spartan.sourceColumnCount_eq]
    norm_num [PiDECInputs.evalKWordsPerChild])

def parentEvalABlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.evalAWordsPerChild parentEvalAStart (by
    rw [parentEvalAStart_eq, Spartan.sourceColumnCount_eq]
    norm_num [PiDECInputs.evalAWordsPerChild])

def proofBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.proofInputColumnCount
    PiDECInputs.proofInputStart (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
        PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
        PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild])

def logicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program 270 PiDECStarts.phaseLogicalStart (by
    rw [Spartan.sourceColumnCount_eq]
    norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild])

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program freshCount PiDECStarts.phaseFreshStart (by
    rw [Spartan.sourceColumnCount_eq]
    norm_num [freshCount, PiDECStarts.phaseFreshStart,
      PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild, PiDEC.v1_1.Formal.logicalPrivateCount])

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    (parentCommitmentBlock program).slotCount +
      (parentPublicInputBlock program).slotCount +
      (parentEvalKBlock program).slotCount +
      (parentEvalABlock program).slotCount +
      (proofBlock program).slotCount +
      (logicalBlock program).slotCount +
      (freshBlock program).slotCount = 66744 := by
  norm_num [parentCommitmentBlock, parentPublicInputBlock, parentEvalKBlock,
    parentEvalABlock, proofBlock, logicalBlock, freshBlock, sourceFieldBlock,
    freshCount, PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) :
    Nat :=
  (parentCommitmentBlock program).coordinateCount +
    (parentPublicInputBlock program).coordinateCount +
    (parentEvalKBlock program).coordinateCount +
    (parentEvalABlock program).coordinateCount +
    (proofBlock program).coordinateCount +
    (logicalBlock program).coordinateCount +
    (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 2736504 := by
  simp only [retainedCoordinateCount, LowNormBlock.Block.coordinateCount,
    parentCommitmentBlock, parentPublicInputBlock, parentEvalKBlock,
    parentEvalABlock, proofBlock, logicalBlock, freshBlock, sourceFieldBlock]
  norm_num [freshCount, PiDECInputs.proofInputColumnCount,
    PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
    PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
    PiDECInputs.publicInputWordsPerChild, LowNormSlot.Kind.width,
    BalancedTernary.width]

end NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks
