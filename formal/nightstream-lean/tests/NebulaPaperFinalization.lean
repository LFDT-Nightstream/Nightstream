import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaPaperFinalization

open Nightstream.SuperNeo.Concrete
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.PaperFinalization

def commitments : Commitments Nat where
  programAdvice := 10
  operations := 10
  scan := 11
  reads := 12
  writes := 13
  initialMemory := 20
  finalMemory := 21

def challengeValues : Challenges := ⟨K.zero, K.one⟩

def segment : Segment Nat Nat PUnit where
  iterations := 2
  stateIn := 7
  stateOut := 9
  timestampIn := 3
  timestampOut := 5
  commitments := commitments
  challenges := challengeValues
  initialProducts := fun _ => K.one
  finalProducts := fun _ => K.one
  proofs := ⟨(), (), ()⟩

def semantics : Semantics Nat Nat PUnit Nat where
  deriveChallenges := fun _ => challengeValues
  verifyProgram := fun _ => true
  verifyOperations := fun _ => true
  verifyScan := fun _ => true
  foldProofs := fun accumulator _ => some (accumulator + 1)

def prior : Running Nat Nat Nat where
  step := 4
  initialState := 1
  currentState := 7
  timestamp := 3
  finalMemory := 20
  accumulator := 30

def next : Running Nat Nat Nat where
  step := 6
  initialState := 1
  currentState := 9
  timestamp := 5
  finalMemory := 21
  accumulator := 31

theorem accepted_segment : Advances semantics prior segment next := by
  exact ⟨
    ⟨fun _ => rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem accepted_segment_continuity :
    segment.commitments.initialMemory = prior.finalMemory ∧
      next.finalMemory = segment.commitments.finalMemory :=
  advances_memory_continuity accepted_segment

end tests.NebulaPaperFinalization
