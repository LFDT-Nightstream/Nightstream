import NightstreamFPrime.Export.Stage1.PiRLCFirst54RetainedBlocks
import NightstreamFPrime.Export.Stage1.PiRLCProductPlan
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns the retained input and accumulator-output blocks used by the direct
PiRLC product plan. Challenge coefficients reuse the final First54 value
slots; group outputs remain owned by `ProductRetainedBlock`.

This module does not construct the complete Stage 1 retained assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCProductPlan.sourceWidth program

def inputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCProductSchedule.invocationCount
  source := fun invocation =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    PiRLCProductPlan.valueColumn program descriptor descriptor.lane

def outputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCProductSchedule.invocationCount
  source := fun invocation =>
    PiRLCProductPlan.outputColumn program
      (PiRLCProductSchedule.descriptor invocation)

def challengeValueDescriptor
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) : PiRLCFirst54DirectSchedule.Value :=
  { candidate :=
      { source := ⟨source.val, by simpa [PiRLCFirst54DirectSchedule.sourceCount,
          PiRLCFirst54Invocations.sourceCount,
          PiRLCCombinationInvocations.sourceCount] using source.isLt⟩
        round := ⟨First54.candidateCount - 1, by
          norm_num [PiRLCFirst54DirectSchedule.roundCount,
            PiRLCFirst54Invocations.roundCount, First54.candidateCount]⟩ }
    slot := ⟨lane.val, by
      simpa [First54ValueStep.outputCount, ringDegree] using lane.isLt⟩ }

/-- Every product challenge coefficient is the already-retained final
First54 output coefficient for the same source and lane. -/
theorem challengeColumn_eq_first54Value
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    descriptor.challengeColumn lane =
      (challengeValueDescriptor descriptor.source lane).valueColumn := by
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  have sourceBound := source.isLt
  have laneBound := lane.isLt
  norm_num [PiRLCProductSchedule.Descriptor.challengeColumn,
    PiRLCCombinationInvocations.challengeSourceStart,
    challengeValueDescriptor,
    PiRLCFirst54DirectSchedule.Value.valueColumn,
    PiRLCFirst54Invocations.valueSourceStart,
    PiRLCStarts.challengeWordStart, First54.valueOffset,
    First54.positionOffset, First54.roundPrivateCount,
    First54Step.slotCount, First54ValueStep.outputCount,
    First54.candidateCount, PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.roundCount, ringDegree] at sourceBound laneBound ⊢

@[simp] theorem inputBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (inputBlock program).kind = .field := by
  rfl

@[simp] theorem outputBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).kind = .field := by
  rfl

@[simp] theorem inputBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (inputBlock program).slotCount = 52326 := by
  exact PiRLCProductSchedule.invocationCount_eq

@[simp] theorem outputBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).slotCount = 52326 := by
  exact PiRLCProductSchedule.invocationCount_eq

theorem inputBlock_source (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (inputBlock program).source invocation =
      let descriptor := PiRLCProductSchedule.descriptor invocation
      PiRLCProductPlan.valueColumn program descriptor descriptor.lane := by
  rfl

theorem outputBlock_source (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (outputBlock program).source invocation =
      PiRLCProductPlan.outputColumn program
        (PiRLCProductSchedule.descriptor invocation) := by
  rfl

@[simp] theorem inputBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (inputBlock program).coordinateCount = 2145366 := by
  change PiRLCProductSchedule.invocationCount * 41 = 2145366
  rw [PiRLCProductSchedule.invocationCount_eq]

@[simp] theorem outputBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).coordinateCount = 2145366 := by
  change PiRLCProductSchedule.invocationCount * 41 = 2145366
  rw [PiRLCProductSchedule.invocationCount_eq]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (inputBlock program).coordinateCount +
    (outputBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 4290732 := by
  simp [retainedCoordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks
