import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks

open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Lifecycle

@[implicit_reducible]
def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiDECRetainedBlocks.sourceWidth program

def sourceCount : Nat := PiRLCSamplerInvocations.sourceCount
def logicalCountPerSource : Nat := 769
def logicalSlotCount : Nat := sourceCount * logicalCountPerSource

@[simp] theorem sourceCount_eq : sourceCount = 17 := rfl
@[simp] theorem logicalSlotCount_eq : logicalSlotCount = 13073 := rfl

def logicalSlot (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    Fin logicalSlotCount := Fin.encodeProd (source, position)

def logicalDescriptor (slot : Fin logicalSlotCount) :
    Fin sourceCount × Fin logicalCountPerSource := Fin.decodeProd slot

@[simp] theorem logicalDescriptor_logicalSlot (source : Fin sourceCount)
    (position : Fin logicalCountPerSource) :
    logicalDescriptor (logicalSlot source position) = (source, position) := by
  simp [logicalDescriptor, logicalSlot]

def logicalSource (source : Fin sourceCount) (position : Fin logicalCountPerSource) : Nat :=
  PiRLCSamplerOrdinaryDirectSource.coreStart source.val + position.val

theorem logicalSource_lt (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    logicalSource source position < PackageOrder.SourceColumnCount :=
  (PiRLCSamplerOrdinaryDirectSource.Source.logical source.val position.val source.isLt position.isLt).bounded

def logicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := logicalSlotCount
  source := fun slot =>
    let descriptor := logicalDescriptor slot
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (logicalSource descriptor.1 descriptor.2) (logicalSource_lt descriptor.1 descriptor.2)

@[simp] theorem logicalBlock_slotCount (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).slotCount = 13073 := rfl

theorem logicalBlock_source (program : Lifecycle.Stage1.Application.Program)
    (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    (logicalBlock program).source (logicalSlot source position) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (logicalSource source position) (logicalSource_lt source position) := by
  unfold logicalBlock
  simp only [logicalDescriptor_logicalSlot]

@[simp] theorem logicalBlock_coordinateCount (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).coordinateCount = 535993 := by
  change logicalSlotCount * 41 = 535993
  rw [logicalSlotCount_eq]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (logicalBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 535993 := by
  unfold retainedCoordinateCount
  rw [logicalBlock_coordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks
