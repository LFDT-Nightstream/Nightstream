import NightstreamFPrime.Export.Stage1.PiRLCProductSchedule

/-!
Owns the complete-ring view of the existing PiRLC product schedule. Ring
order is family, source, block, cell. `withLane` returns the exact existing
source-lane descriptor, including the interleaved extension-field cells.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductRingSchedule

open NightstreamFPrime.Spec
open PiRLCCombinationInvocations

abbrev Family := PiRLCProductSchedule.Family

def zeroLane : Fin ringDegree := ⟨0, by decide⟩

def familyCount (family : Family) : Nat :=
  sourceCount * (family.blockCount * family.cellCount)

structure Descriptor where
  family : Family
  source : Fin sourceCount
  block : Fin family.blockCount
  cell : Fin family.cellCount

def Descriptor.withLane (value : Descriptor) (lane : Fin ringDegree) :
    PiRLCProductSchedule.Descriptor :=
  ⟨value.family, value.source, value.block, lane, value.cell⟩

def ofLane (value : PiRLCProductSchedule.Descriptor) : Descriptor :=
  ⟨value.family, value.source, value.block, value.cell⟩

@[simp] theorem ofLane_withLane (value : Descriptor) (lane : Fin ringDegree) :
    ofLane (value.withLane lane) = value := by
  cases value
  rfl

@[simp] theorem withLane_ofLane (value : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    (ofLane value).withLane lane = value.withLane lane := by
  cases value
  rfl

@[simp] theorem withLane_own (value : PiRLCProductSchedule.Descriptor) :
    value.withLane value.lane = value := by
  cases value
  rfl

def familyDescriptor (family : Family) (index : Fin (familyCount family)) :
    Descriptor :=
  let outer : Fin sourceCount × Fin (family.blockCount * family.cellCount) :=
    Fin.decodeProd index
  let inner : Fin family.blockCount × Fin family.cellCount :=
    Fin.decodeProd outer.2
  ⟨family, outer.1, inner.1, inner.2⟩

def Descriptor.familyIndex (value : Descriptor) : Fin (familyCount value.family) :=
  Fin.encodeProd (value.source, Fin.encodeProd (value.block, value.cell))

@[simp] theorem familyDescriptor_familyIndex (value : Descriptor) :
    familyDescriptor value.family value.familyIndex = value := by
  cases value
  simp [familyDescriptor, Descriptor.familyIndex]

@[simp] theorem familyIndex_familyDescriptor (family : Family)
    (index : Fin (familyCount family)) :
    (familyDescriptor family index).familyIndex = index := by
  simp [familyDescriptor, Descriptor.familyIndex]

def invocationCount : Nat :=
  familyCount .commitment +
    (familyCount .publicInput + (familyCount .evalK + familyCount .evalA))

@[simp] theorem invocationCount_eq : invocationCount = 969 := by
  rfl

def descriptor : Fin invocationCount → Descriptor :=
  Fin.append (familyDescriptor .commitment) <|
    Fin.append (familyDescriptor .publicInput) <|
      Fin.append (familyDescriptor .evalK) (familyDescriptor .evalA)

def Descriptor.invocation (value : Descriptor) : Fin invocationCount :=
  match value with
  | ⟨.commitment, source, block, cell⟩ =>
      Fin.castAdd (familyCount .publicInput +
        (familyCount .evalK + familyCount .evalA))
        ({ family := .commitment, source, block, cell } : Descriptor).familyIndex
  | ⟨.publicInput, source, block, cell⟩ =>
      Fin.natAdd (familyCount .commitment) <|
        Fin.castAdd (familyCount .evalK + familyCount .evalA)
          ({ family := .publicInput, source, block, cell } : Descriptor).familyIndex
  | ⟨.evalK, source, block, cell⟩ =>
      Fin.natAdd (familyCount .commitment) <|
        Fin.natAdd (familyCount .publicInput) <|
          Fin.castAdd (familyCount .evalA)
            ({ family := .evalK, source, block, cell } : Descriptor).familyIndex
  | ⟨.evalA, source, block, cell⟩ =>
      Fin.natAdd (familyCount .commitment) <|
        Fin.natAdd (familyCount .publicInput) <|
          Fin.natAdd (familyCount .evalK)
            ({ family := .evalA, source, block, cell } : Descriptor).familyIndex

@[simp] theorem descriptor_invocation (value : Descriptor) :
    descriptor value.invocation = value := by
  rcases value with ⟨family, source, block, cell⟩
  cases family
  · simp [Descriptor.invocation, descriptor]
    exact familyDescriptor_familyIndex ⟨.commitment, source, block, cell⟩
  · simp [Descriptor.invocation, descriptor]
    exact familyDescriptor_familyIndex ⟨.publicInput, source, block, cell⟩
  · simp [Descriptor.invocation, descriptor]
    exact familyDescriptor_familyIndex ⟨.evalK, source, block, cell⟩
  · simp [Descriptor.invocation, descriptor]
    exact familyDescriptor_familyIndex ⟨.evalA, source, block, cell⟩

@[simp] theorem invocation_descriptor (index : Fin invocationCount) :
    (descriptor index).invocation = index := by
  unfold descriptor
  refine Fin.addCases (fun first => ?_) (fun remaining => ?_) index
  · simp only [Fin.append_left]
    change Fin.castAdd _ (familyDescriptor .commitment first).familyIndex =
      Fin.castAdd _ first
    rw [familyIndex_familyDescriptor]
    rfl
  · simp only [Fin.append_right]
    refine Fin.addCases (fun second => ?_) (fun remaining => ?_) remaining
    · simp only [Fin.append_left]
      change Fin.natAdd (familyCount .commitment)
          (Fin.castAdd _ (familyDescriptor .publicInput second).familyIndex) =
        Fin.natAdd (familyCount .commitment) (Fin.castAdd _ second)
      rw [familyIndex_familyDescriptor]
      rfl
    · simp only [Fin.append_right]
      refine Fin.addCases (fun third => ?_) (fun fourth => ?_) remaining
      · simp only [Fin.append_left]
        change Fin.natAdd (familyCount .commitment)
            (Fin.natAdd (familyCount .publicInput)
              (Fin.castAdd _ (familyDescriptor .evalK third).familyIndex)) =
          Fin.natAdd (familyCount .commitment)
            (Fin.natAdd (familyCount .publicInput) (Fin.castAdd _ third))
        rw [familyIndex_familyDescriptor]
        rfl
      · simp only [Fin.append_right]
        change Fin.natAdd (familyCount .commitment)
            (Fin.natAdd (familyCount .publicInput)
              (Fin.natAdd (familyCount .evalK)
                (familyDescriptor .evalA fourth).familyIndex)) =
          Fin.natAdd (familyCount .commitment)
            (Fin.natAdd (familyCount .publicInput)
              (Fin.natAdd (familyCount .evalK) fourth))
        rw [familyIndex_familyDescriptor]
        rfl

/-- Exact old source-lane index of one coefficient in a complete ring. -/
def laneInvocation (ring : Fin invocationCount) (lane : Fin ringDegree) :
    Fin PiRLCProductSchedule.invocationCount :=
  ((descriptor ring).withLane lane).invocation

@[simp] theorem descriptor_laneInvocation
    (ring : Fin invocationCount) (lane : Fin ringDegree) :
    PiRLCProductSchedule.descriptor (laneInvocation ring lane) =
      (descriptor ring).withLane lane := by
  exact PiRLCProductSchedule.descriptor_invocation _

def ringInvocation (lane : Fin PiRLCProductSchedule.invocationCount) :
    Fin invocationCount :=
  (ofLane (PiRLCProductSchedule.descriptor lane)).invocation

@[simp] theorem laneInvocation_ringInvocation
    (lane : Fin PiRLCProductSchedule.invocationCount) :
    laneInvocation (ringInvocation lane)
      (PiRLCProductSchedule.descriptor lane).lane = lane := by
  simp [laneInvocation, ringInvocation]

@[simp] theorem ringInvocation_laneInvocation
    (ring : Fin invocationCount) (lane : Fin ringDegree) :
    ringInvocation (laneInvocation ring lane) = ring := by
  simp [ringInvocation]

end NightstreamFPrime.Export.Stage1.PiRLCProductRingSchedule
