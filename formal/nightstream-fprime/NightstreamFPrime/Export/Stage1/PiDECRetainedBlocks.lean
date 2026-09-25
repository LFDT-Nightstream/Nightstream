import NightstreamFPrime.Layout.Stage1.PiDECSourceSupportData
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

/-!
Owns the seven exact source intervals retained by the nonempty PiDEC rows:
four parent outputs, the proof input, the logical split cells, and the R1CS
fresh interval. The proof view reuses the running-transition allocation;
the four parent views reuse PiRLC outputs. Only logical and fresh views
allocate coordinates.
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
    parentCommitmentStart (by decide)

def parentPublicInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.publicInputWordsPerChild
    parentPublicInputStart (by decide)

def parentEvalKBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.evalKWordsPerChild parentEvalKStart (by decide)

def parentEvalABlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiDECInputs.evalAWordsPerChild parentEvalAStart (by decide)

def proofBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  RunningTransitionRetainedBlocks.piDecBlock program

def logicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program logicalCount PiDECStarts.phaseLogicalStart
    logical_end_le_sourceColumnCount

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program freshCount PiDECStarts.phaseFreshStart
    fresh_end_le_sourceColumnCount

/-- Retained slots follow the logical and fresh allocation owners. -/
@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).slotCount + (freshBlock program).slotCount =
      logicalCount + freshCount := by
  rfl

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) :
    Nat :=
          (logicalBlock program).coordinateCount +
    (freshBlock program).coordinateCount

/-- Each retained field slot uses the existing balanced-ternary encoding. -/
@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program =
      (logicalCount + freshCount) * LowNormSlot.Kind.field.width := by
  simp only [retainedCoordinateCount, LowNormBlock.Block.coordinateCount,
    logicalBlock, freshBlock, sourceFieldBlock, Nat.add_mul]

end NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks
