import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations

/-!
Owns constant-time descriptors for the direct PiRLC First54 relation. Orders
are source-major, candidate-major, and then slot-major, matching the canonical
First54 parent traversal.

This module does not construct matrix rows or assignment forms.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectSchedule

open NightstreamFPrime.Gadgets.Sampling

def sourceCount : Nat := PiRLCFirst54Invocations.sourceCount
def roundCount : Nat := PiRLCFirst54Invocations.roundCount
def candidateCount : Nat := sourceCount * roundCount
def positionCount : Nat := candidateCount * First54Step.slotCount
def valueCount : Nat := candidateCount * First54ValueStep.outputCount
def finalCount : Nat := sourceCount

@[simp] theorem sourceCount_eq : sourceCount = 17 := by
  rfl

@[simp] theorem roundCount_eq : roundCount = 64 := by
  rfl

@[simp] theorem candidateCount_eq : candidateCount = 1088 := by
  rfl

@[simp] theorem positionCount_eq : positionCount = 59840 := by
  rfl

@[simp] theorem valueCount_eq : valueCount = 58752 := by
  rfl

@[simp] theorem finalCount_eq : finalCount = 17 := by
  rfl

structure Candidate where
  source : Fin sourceCount
  round : Fin roundCount
deriving Repr

def candidate (index : Fin candidateCount) : Candidate :=
  let decoded : Fin sourceCount × Fin roundCount := Fin.decodeProd index
  ⟨decoded.1, decoded.2⟩

def candidateIndex (candidate : Candidate) : Fin candidateCount :=
  Fin.encodeProd (candidate.source, candidate.round)

@[simp] theorem candidate_candidateIndex (descriptor : Candidate) :
    candidate (candidateIndex descriptor) = descriptor := by
  rcases descriptor with ⟨source, round⟩
  simp [candidate, candidateIndex]

@[simp] theorem candidateIndex_candidate (index : Fin candidateCount) :
    candidateIndex (candidate index) = index := by
  simp [candidate, candidateIndex]

structure Position where
  candidate : Candidate
  slot : Fin First54Step.slotCount
deriving Repr

def position (index : Fin positionCount) : Position :=
  let decoded : Fin candidateCount × Fin First54Step.slotCount :=
    Fin.decodeProd index
  ⟨candidate decoded.1, decoded.2⟩

def positionIndex (descriptor : Position) : Fin positionCount :=
  Fin.encodeProd (candidateIndex descriptor.candidate, descriptor.slot)

@[simp] theorem position_positionIndex (descriptor : Position) :
    position (positionIndex descriptor) = descriptor := by
  rcases descriptor with ⟨candidate, slot⟩
  simp [position, positionIndex]

@[simp] theorem positionIndex_position (index : Fin positionCount) :
    positionIndex (position index) = index := by
  simp [position, positionIndex]

structure Value where
  candidate : Candidate
  slot : Fin First54ValueStep.outputCount
deriving Repr

def value (index : Fin valueCount) : Value :=
  let decoded : Fin candidateCount × Fin First54ValueStep.outputCount :=
    Fin.decodeProd index
  ⟨candidate decoded.1, decoded.2⟩

def valueIndex (descriptor : Value) : Fin valueCount :=
  Fin.encodeProd (candidateIndex descriptor.candidate, descriptor.slot)

@[simp] theorem value_valueIndex (descriptor : Value) :
    value (valueIndex descriptor) = descriptor := by
  rcases descriptor with ⟨candidate, slot⟩
  simp [value, valueIndex]

@[simp] theorem valueIndex_value (index : Fin valueCount) :
    valueIndex (value index) = index := by
  simp [value, valueIndex]

namespace Candidate

def rejectColumn (descriptor : Candidate) : Nat :=
  PiRLCFirst54Invocations.rejectSourceColumn descriptor.source.val
    descriptor.round.val

def symbolColumn (descriptor : Candidate) : Nat :=
  PiRLCFirst54Invocations.remainderSourceColumn descriptor.source.val
    descriptor.round.val

end Candidate

namespace Position

def positionColumn (descriptor : Position) : Nat :=
  PiRLCFirst54Invocations.positionSourceStart
      descriptor.candidate.source.val descriptor.candidate.round.val +
    descriptor.slot.val

def priorPositionColumn (descriptor : Position) : Nat :=
  PiRLCFirst54Invocations.previousPositionSourceStart
      descriptor.candidate.source.val descriptor.candidate.round.val +
    descriptor.slot.val

end Position

namespace Value

def valueColumn (descriptor : Value) : Nat :=
  PiRLCFirst54Invocations.valueSourceStart
      descriptor.candidate.source.val descriptor.candidate.round.val +
    descriptor.slot.val

def priorValueColumn (descriptor : Value) : Nat :=
  PiRLCFirst54Invocations.previousValueSourceStart
      descriptor.candidate.source.val descriptor.candidate.round.val +
    descriptor.slot.val

end Value

def finalColumn (source : Fin finalCount) : Nat :=
  PiRLCFirst54Invocations.positionSourceStart source.val (roundCount - 1) +
    First54Step.fullSlot

end NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectSchedule
