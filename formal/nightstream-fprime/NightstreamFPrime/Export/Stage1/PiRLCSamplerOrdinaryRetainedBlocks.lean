import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks

/-! Field blocks for the checked sampler core and its R1CS lowering values.
Poseidon2 owns the four input lanes; the product-source block owns the checked
coefficient words. Temporary witness helpers are excluded by read support. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks

open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiDECRetainedBlocks.sourceWidth program

def sourceCount : Nat := PiRLCSamplerInvocations.sourceCount
def logicalCountPerSource : Nat := 617
def freshCountPerSource : Nat := 1548
def logicalSlotCount : Nat := sourceCount * logicalCountPerSource
def freshSlotCount : Nat := sourceCount * freshCountPerSource

@[simp] theorem sourceCount_eq : sourceCount = 17 := rfl
@[simp] theorem logicalSlotCount_eq : logicalSlotCount = 10489 := rfl
@[simp] theorem freshSlotCount_eq : freshSlotCount = 26316 := rfl

def logicalSlot (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    Fin logicalSlotCount := Fin.encodeProd (source, position)

def logicalDescriptor (slot : Fin logicalSlotCount) :
    Fin sourceCount × Fin logicalCountPerSource := Fin.decodeProd slot

@[simp] theorem logicalDescriptor_logicalSlot (source : Fin sourceCount)
    (position : Fin logicalCountPerSource) :
    logicalDescriptor (logicalSlot source position) = (source, position) := by
  simp [logicalDescriptor, logicalSlot]

def freshSlot (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    Fin freshSlotCount := Fin.encodeProd (source, position)

def freshDescriptor (slot : Fin freshSlotCount) :
    Fin sourceCount × Fin freshCountPerSource := Fin.decodeProd slot

@[simp] theorem freshDescriptor_freshSlot (source : Fin sourceCount)
    (position : Fin freshCountPerSource) :
    freshDescriptor (freshSlot source position) = (source, position) := by
  simp [freshDescriptor, freshSlot]

def logicalSource (source : Fin sourceCount) (position : Fin logicalCountPerSource) : Nat :=
  PiRLCSamplerOrdinaryDirectSource.coreStart source.val + position.val

def freshSource (source : Fin sourceCount) (position : Fin freshCountPerSource) : Nat :=
  PiRLCStarts.rangeFreshStart source.val + position.val

theorem logicalSource_lt (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    logicalSource source position < Spartan.SourceColumnCount :=
  (PiRLCSamplerOrdinaryDirectSource.Source.logical source.val position.val source.isLt position.isLt).bounded

theorem freshSource_lt (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    freshSource source position < Spartan.SourceColumnCount :=
  (PiRLCSamplerOrdinaryDirectSource.Source.fresh source.val position.val source.isLt position.isLt).bounded

def logicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := logicalSlotCount
  source := fun slot =>
    let descriptor := logicalDescriptor slot
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (logicalSource descriptor.1 descriptor.2) (logicalSource_lt descriptor.1 descriptor.2)

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := freshSlotCount
  source := fun slot =>
    let descriptor := freshDescriptor slot
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (freshSource descriptor.1 descriptor.2) (freshSource_lt descriptor.1 descriptor.2)

@[simp] theorem logicalBlock_slotCount (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).slotCount = 10489 := rfl

@[simp] theorem freshBlock_slotCount (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).slotCount = 26316 := rfl

theorem logicalBlock_source (program : Lifecycle.Stage1.Application.Program)
    (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    (logicalBlock program).source (logicalSlot source position) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (logicalSource source position) (logicalSource_lt source position) := by
  unfold logicalBlock
  simp only [logicalDescriptor_logicalSlot]

theorem freshBlock_source (program : Lifecycle.Stage1.Application.Program)
    (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    (freshBlock program).source (freshSlot source position) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (freshSource source position) (freshSource_lt source position) := by
  unfold freshBlock
  simp only [freshDescriptor_freshSlot]

@[simp] theorem logicalBlock_coordinateCount (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).coordinateCount = 430049 := by
  change logicalSlotCount * 41 = 430049
  rw [logicalSlotCount_eq]

@[simp] theorem freshBlock_coordinateCount (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).coordinateCount = 1078956 := by
  change freshSlotCount * 41 = 1078956
  rw [freshSlotCount_eq]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (logicalBlock program).coordinateCount + (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 1509005 := by
  unfold retainedCoordinateCount
  rw [logicalBlock_coordinateCount, freshBlock_coordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks
