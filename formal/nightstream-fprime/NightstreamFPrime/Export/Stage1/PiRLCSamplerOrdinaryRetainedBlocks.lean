import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.PiDECRetainedBlocks

/-!
Owns the two selective field blocks needed by the direct PiRLC sampler
ordinary rows. The blocks retain only digest-lane logical values and their
R1CS-fresh values. Existing sampler Poseidon2 and First54 blocks own the two
external endpoint families.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiDECRetainedBlocks.sourceWidth program

def sourceCount : Nat := PiRLCSamplerOrdinaryRows.sourceCount
def roundCount : Nat := PiRLCSamplerOrdinaryRows.digestRoundCount
def laneCount : Nat := 4
def laneInvocationCount : Nat := sourceCount * roundCount * laneCount
def logicalCountPerLane : Nat := 100
def freshCountPerLane : Nat := 303
def logicalSlotCount : Nat := laneInvocationCount * logicalCountPerLane
def freshSlotCount : Nat := laneInvocationCount * freshCountPerLane

@[simp] theorem sourceCount_eq : sourceCount = 17 := by rfl
@[simp] theorem roundCount_eq : roundCount = 8 := by rfl
@[simp] theorem laneCount_eq : laneCount = 4 := by rfl
@[simp] theorem laneInvocationCount_eq : laneInvocationCount = 544 := by rfl
@[simp] theorem logicalSlotCount_eq : logicalSlotCount = 54400 := by rfl
@[simp] theorem freshSlotCount_eq : freshSlotCount = 164832 := by rfl

structure Lane where
  source : Fin sourceCount
  round : Fin roundCount
  lane : Fin laneCount

def laneIndex (descriptor : Lane) : Fin laneInvocationCount :=
  Fin.encodeProd (Fin.encodeProd (descriptor.source, descriptor.round),
    descriptor.lane)

def laneDescriptor (index : Fin laneInvocationCount) : Lane :=
  let outer : Fin (sourceCount * roundCount) × Fin laneCount :=
    Fin.decodeProd index
  let inner : Fin sourceCount × Fin roundCount := Fin.decodeProd outer.1
  ⟨inner.1, inner.2, outer.2⟩

@[simp] theorem laneDescriptor_laneIndex (descriptor : Lane) :
    laneDescriptor (laneIndex descriptor) = descriptor := by
  cases descriptor
  simp [laneDescriptor, laneIndex]

def logicalSlot (descriptor : Lane) (position : Fin logicalCountPerLane) :
    Fin logicalSlotCount :=
  Fin.encodeProd (laneIndex descriptor, position)

def logicalDescriptor (slot : Fin logicalSlotCount) :
    Lane × Fin logicalCountPerLane :=
  let decoded : Fin laneInvocationCount × Fin logicalCountPerLane :=
    Fin.decodeProd slot
  (laneDescriptor decoded.1, decoded.2)

@[simp] theorem logicalDescriptor_logicalSlot (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalDescriptor (logicalSlot descriptor position) =
      (descriptor, position) := by
  simp [logicalDescriptor, logicalSlot]

def freshSlot (descriptor : Lane) (position : Fin freshCountPerLane) :
    Fin freshSlotCount :=
  Fin.encodeProd (laneIndex descriptor, position)

def freshDescriptor (slot : Fin freshSlotCount) :
    Lane × Fin freshCountPerLane :=
  let decoded : Fin laneInvocationCount × Fin freshCountPerLane :=
    Fin.decodeProd slot
  (laneDescriptor decoded.1, decoded.2)

@[simp] theorem freshDescriptor_freshSlot (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    freshDescriptor (freshSlot descriptor position) =
      (descriptor, position) := by
  simp [freshDescriptor, freshSlot]

def logicalSource (descriptor : Lane)
    (position : Fin logicalCountPerLane) : Nat :=
  PiRLCStarts.digestLaneLogicalStart descriptor.source.val
    descriptor.round.val descriptor.lane.val + position.val

def freshSource (descriptor : Lane)
    (position : Fin freshCountPerLane) : Nat :=
  PiRLCStarts.digestLaneFreshStart descriptor.source.val
    descriptor.round.val descriptor.lane.val + position.val

theorem logicalSource_lt (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalSource descriptor position < Spartan.SourceColumnCount := by
  rcases descriptor with ⟨source, round, lane⟩
  have sourceLt := source.isLt
  have roundLt := round.isLt
  have laneLt := lane.isLt
  have positionLt := position.isLt
  change source.val < 17 at sourceLt
  change round.val < 8 at roundLt
  change lane.val < 4 at laneLt
  change position.val < 100 at positionLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [logicalSource, logicalCountPerLane,
    PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset] at sourceLt roundLt laneLt positionLt ⊢
  omega

theorem freshSource_lt (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    freshSource descriptor position < Spartan.SourceColumnCount := by
  rcases descriptor with ⟨source, round, lane⟩
  have sourceLt := source.isLt
  have roundLt := round.isLt
  have laneLt := lane.isLt
  have positionLt := position.isLt
  change source.val < 17 at sourceLt
  change round.val < 8 at roundLt
  change lane.val < 4 at laneLt
  change position.val < 303 at positionLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [freshSource, freshCountPerLane,
    PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
    PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
    PiRLCStarts.phaseFreshStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.logicalPrivateCount] at sourceLt roundLt laneLt positionLt ⊢
  omega

def logicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := logicalSlotCount
  source := fun slot =>
    let descriptor := logicalDescriptor slot
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (logicalSource descriptor.1 descriptor.2)
      (logicalSource_lt descriptor.1 descriptor.2)

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := freshSlotCount
  source := fun slot =>
    let descriptor := freshDescriptor slot
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (freshSource descriptor.1 descriptor.2)
      (freshSource_lt descriptor.1 descriptor.2)

@[simp] theorem logicalBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).slotCount = 54400 := by rfl

@[simp] theorem freshBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).slotCount = 164832 := by rfl

theorem logicalBlock_source (program : Lifecycle.Stage1.Application.Program)
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (logicalBlock program).source (logicalSlot descriptor position) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (logicalSource descriptor position)
        (logicalSource_lt descriptor position) := by
  unfold logicalBlock
  simp only [logicalDescriptor_logicalSlot]

theorem freshBlock_source (program : Lifecycle.Stage1.Application.Program)
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (freshBlock program).source (freshSlot descriptor position) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (freshSource descriptor position)
        (freshSource_lt descriptor position) := by
  unfold freshBlock
  simp only [freshDescriptor_freshSlot]

@[simp] theorem logicalBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (logicalBlock program).coordinateCount = 2230400 := by
  norm_num [logicalBlock, LowNormBlock.Block.coordinateCount,
    logicalSlotCount, laneInvocationCount, sourceCount, roundCount, laneCount,
    logicalCountPerLane, PiRLCSamplerOrdinaryRows.sourceCount,
    PiRLCSamplerOrdinaryRows.digestRoundCount,
    LowNormSlot.Kind.width, BalancedTernary.width]

@[simp] theorem freshBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).coordinateCount = 6758112 := by
  norm_num [freshBlock, LowNormBlock.Block.coordinateCount,
    freshSlotCount, laneInvocationCount, sourceCount, roundCount, laneCount,
    freshCountPerLane, PiRLCSamplerOrdinaryRows.sourceCount,
    PiRLCSamplerOrdinaryRows.digestRoundCount,
    LowNormSlot.Kind.width, BalancedTernary.width]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) :
    Nat :=
  (logicalBlock program).coordinateCount +
    (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 8988512 := by
  simp [retainedCoordinateCount]

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedBlocks
