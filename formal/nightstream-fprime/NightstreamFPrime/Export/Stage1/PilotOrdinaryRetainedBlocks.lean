import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedBlocks
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource

/-!
Owns the three field blocks missing from the existing PiCCS retained set for
the non-Poseidon pilot rows: canonical-word locals, R1CS fresh values, and
the four public output-digest words.
-/

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSOrdinaryRetainedBlocks.sourceWidth program

def canonicalLocalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PiCCSOrdinaryRetainedBlocks.sourceFieldBlock program 264
    (PriorStateHash.hashEnd PilotProduction.priorInterface
      PilotProduction.witnessOffset) (by
        rw [Spartan.sourceColumnCount_eq]
        unfold PriorStateHash.hashEnd
        rw [PilotProduction.priorHashLogicalLength_eq,
          PilotProduction.witnessOffset_eq]
        norm_num)

def canonicalFreshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PiCCSOrdinaryRetainedBlocks.sourceFieldBlock program 788
    PilotValues.logicalColumnCount (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PilotValues.logicalColumnCount,
        PilotValues.externalColumnCount, PilotValues.outputDigestStart,
        PilotValues.outputPreimageStart, PilotValues.priorPublicInputStart,
        PilotValues.priorPreimageStart, PilotValues.stateHashWords,
        PilotValues.stateHashBaseWords, PilotValues.hashWitnessCount,
        PilotValues.absorbCount, PilotValues.permutationRecipeCount,
        Spec.Poseidon2.rate, PilotValues.priorCanonicalPrivateCount])

def outputDigestBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PiCCSOrdinaryRetainedBlocks.sourceFieldBlock program 4
    PilotProduction.outputDigestStart (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PilotProduction.outputDigestStart,
        PilotProduction.outputPreimageStart,
        PilotProduction.priorPublicInputStart,
        PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq, PriorStateHash.publicWidth_eq])

@[simp] theorem canonicalLocalBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (canonicalLocalBlock program).slotCount = 264 := by rfl

@[simp] theorem canonicalFreshBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (canonicalFreshBlock program).slotCount = 788 := by rfl

@[simp] theorem outputDigestBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputDigestBlock program).slotCount = 4 := by rfl

def retainedSlotCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (canonicalLocalBlock program).slotCount +
    (canonicalFreshBlock program).slotCount +
    (outputDigestBlock program).slotCount

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedSlotCount program = 1056 := by
  simp [retainedSlotCount]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (canonicalLocalBlock program).coordinateCount +
    (canonicalFreshBlock program).coordinateCount +
    (outputDigestBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 43296 := by
  change 264 * 41 + 788 * 41 + 4 * 41 = 43296
  norm_num

end NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedBlocks
