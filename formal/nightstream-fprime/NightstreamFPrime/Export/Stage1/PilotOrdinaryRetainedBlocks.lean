import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedBlocks

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

@[implicit_reducible]
def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiCCSOrdinaryRetainedBlocks.sourceWidth program

def canonicalLocalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PiCCSOrdinaryRetainedBlocks.sourceFieldBlock program 268
    (PriorStateHash.hashEnd PilotProduction.priorInterface
      PilotProduction.witnessOffset) (by
        rw [Spartan.sourceColumnCount_eq]
        unfold PriorStateHash.hashEnd
        rw [PilotProduction.priorHashLogicalLength_eq,
          PilotProduction.witnessOffset_eq]
        norm_num)

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
    (canonicalLocalBlock program).slotCount = 268 := by rfl

@[simp] theorem outputDigestBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputDigestBlock program).slotCount = 4 := by rfl

def retainedSlotCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (canonicalLocalBlock program).slotCount +
    (outputDigestBlock program).slotCount

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedSlotCount program = 272 := by
  simp [retainedSlotCount]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (canonicalLocalBlock program).coordinateCount +
    (outputDigestBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 11152 := by
  change 268 * 41 + 4 * 41 = 11152
  norm_num

end NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedBlocks
