import NightstreamFPrime.Export.Stage1.PiRLCProductPlan
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns the checked challenge words and accumulator outputs used by the direct
PiRLC product plan. Each source has 54 challenge words. Product-group outputs
remain owned by `ProductRetainedBlock`.

This module does not construct the complete Stage 1 retained assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCProductPlan.sourceWidth program

def outputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCProductSchedule.invocationCount
  source := fun invocation =>
    PiRLCProductPlan.outputColumn program
      (PiRLCProductSchedule.descriptor invocation)

abbrev Challenge := Fin PiRLCCombinationInvocations.sourceCount × Fin ringDegree

def challengeIndex (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) : Fin (PiRLCCombinationInvocations.sourceCount * ringDegree) :=
  Fin.encodeProd (source, lane)

private theorem challengeSource_lt (position : Challenge) :
    PiRLCStarts.challengeWordStart position.1.val + position.2.val <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
  have sourceBound := position.1.isLt
  have laneBound := position.2.isLt
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn = 14750353 :=
    Package.circuitPackage_layout_values.2.2.1
  rw [constant, PiRLCStarts.challengeWordStart_eq, PiRLCStarts.phaseLogicalStart_eq]
  norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree] at sourceBound laneBound
  omega

def challengeBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := PiRLCCombinationInvocations.sourceCount * ringDegree
  source := fun index =>
    let position : Challenge := Fin.decodeProd index
    PiRLCProductPlan.baseColumn program
      (PiRLCStarts.challengeWordStart position.1.val + position.2.val)
      (challengeSource_lt position)

@[simp] theorem challengeBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (challengeBlock program).coordinateCount = 37638 := by
  change PiRLCCombinationInvocations.sourceCount * ringDegree * 41 = 37638
  norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree]

/-- The retained word is the exact coefficient source read by the product. -/
theorem challengeBlock_source (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    (challengeBlock program).source (challengeIndex descriptor.source lane) =
      PiRLCProductPlan.challengeColumn program descriptor lane := by
  simp only [challengeBlock, challengeIndex, Fin.decodeProd_encodeProd]
  rfl

@[simp] theorem outputBlock_kind
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).kind = .field := by
  rfl

@[simp] theorem outputBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).slotCount = 39474 := by
  exact PiRLCProductSchedule.invocationCount_eq

theorem outputBlock_source (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (outputBlock program).source invocation =
      PiRLCProductPlan.outputColumn program
        (PiRLCProductSchedule.descriptor invocation) := by
  rfl

@[simp] theorem outputBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).coordinateCount = 1618434 := by
  change PiRLCProductSchedule.invocationCount * 41 = 1618434
  rw [PiRLCProductSchedule.invocationCount_eq]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (challengeBlock program).coordinateCount + (outputBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 1656072 := by
  simp [retainedCoordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks
