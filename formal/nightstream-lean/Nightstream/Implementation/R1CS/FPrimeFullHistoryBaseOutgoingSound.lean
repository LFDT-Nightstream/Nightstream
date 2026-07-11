import Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound

/-!
Contract: close the base step's one-invocation-delayed public link using only
the exact generated base owner, adjacent-state rows, and recursive prior-link
owner.

This is the first composed full-history edge theorem. The recursive consumer
recomputes the prior state `x_out`; the proof below shows that its pure
Poseidon schedule and ordered values are exactly those of the base producer,
then connects the consumer's canonical encoding to its fresh public input.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

def priorTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace

def baseTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryBaseFacts.xOutTrace

theorem priorTrace_inputColumns :
    priorTrace.inputColumns =
      [(FPrimeFullHistoryPriorLink.constantPins.getD 0 (0, 0)).1,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 0 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 1 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 2 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 3 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 4 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 5 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 6 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 7 0,
       FPrimeFullHistoryPriorLinkSound.priorChunkLowHalfCol,
       FPrimeFullHistoryPriorLinkSound.priorChunkHighHalfCol,
       FPrimeFullHistoryPriorLinkSound.priorStepLowHalfCol,
       FPrimeFullHistoryPriorLinkSound.priorStepHighHalfCol,
       FPrimeFullHistoryPriorLinkSound.priorProgramCounterLowHalfCol,
       FPrimeFullHistoryPriorLinkSound.priorProgramCounterHighHalfCol,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 14 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 15 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 16 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 17 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 23 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 24 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 25 0,
       FPrimeFullHistoryPriorLink.stateInColumns.getD 26 0] := by
  native_decide

theorem priorTrace_outputColumns :
    priorTrace.outputColumns = FPrimeFullHistoryPriorLink.digestColumns :=
  FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace_output

theorem valueSchedules_equal :
    Poseidon2Sponge.valueSchedules priorTrace.rounds =
      Poseidon2Sponge.valueSchedules baseTrace.rounds := by
  native_decide

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by
  decide

theorem priorInputs_eq_baseInputs
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (priorFacts : FPrimeFullHistoryPriorLinkSound.Facts assignment)
    (links : ∀ pair ∈ FPrimeFullHistoryStateLink.pairs,
      assignment pair.1 = assignment pair.2) :
    priorTrace.inputColumns.map assignment =
      baseTrace.inputColumns.map assignment := by
  have recursiveChunkOne :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 8 0) = 1 :=
    (links (FPrimeFullHistoryBase.stateOutColumns.getD 8 0,
      FPrimeFullHistoryPriorLink.stateInColumns.getD 8 0)
      (by native_decide)).symm.trans
      (baseFacts.constants
        (FPrimeFullHistoryBase.stateOutColumns.getD 8 0, 1)
        (by native_decide))
  have recursiveStepOne :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 9 0) = 1 :=
    (links (FPrimeFullHistoryBase.stateOutColumns.getD 9 0,
      FPrimeFullHistoryPriorLink.stateInColumns.getD 9 0)
      (by native_decide)).symm.trans
      (baseFacts.constants
        (FPrimeFullHistoryBase.stateOutColumns.getD 9 0, 1)
        (by native_decide))
  have recursivePcOne :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0) = 1 :=
    (links (FPrimeFullHistoryBase.stateOutColumns.getD 18 0,
      FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0)
      (by native_decide)).symm.trans
      (baseFacts.stateIn
        (FPrimeFullHistoryBase.stateOutColumns.getD 18 0, 1)
        (by native_decide))
  have priorHalves := FPrimeFullHistoryPriorLinkSound.counterHalves_sound
    priorFacts recursiveChunkOne recursiveStepOne recursivePcOne
  have baseHalves := FPrimeFullHistoryBaseFacts.counterHalves_sound baseFacts
  have tag :=
    (priorFacts.constants
      (FPrimeFullHistoryPriorLink.constantPins.getD 0 (0, 0))
      (by native_decide)).trans
      (baseFacts.constants
        (baseTrace.inputColumns.getD 0 0, 1313210370)
        (by native_decide)).symm
  have vk0 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 0 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 0 0) (by native_decide)).symm
  have vk1 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 1 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 1 0) (by native_decide)).symm
  have vk2 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 2 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 2 0) (by native_decide)).symm
  have vk3 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 3 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 3 0) (by native_decide)).symm
  have header0 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 4 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 4 0) (by native_decide)).symm
  have header1 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 5 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 5 0) (by native_decide)).symm
  have header2 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 6 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 6 0) (by native_decide)).symm
  have header3 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 7 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 7 0) (by native_decide)).symm
  have boundary0 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 14 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 14 0) (by native_decide)).symm
  have boundary1 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 15 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 15 0) (by native_decide)).symm
  have boundary2 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 16 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 16 0) (by native_decide)).symm
  have boundary3 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 17 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 17 0) (by native_decide)).symm
  have accumulator0 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 23 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 23 0) (by native_decide)).symm
  have accumulator1 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 24 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 24 0) (by native_decide)).symm
  have accumulator2 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 25 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 25 0) (by native_decide)).symm
  have accumulator3 := (links (FPrimeFullHistoryBase.stateOutColumns.getD 26 0,
    FPrimeFullHistoryPriorLink.stateInColumns.getD 26 0) (by native_decide)).symm
  have chunkLow := priorHalves.chunk.1.trans baseHalves.chunk.1.symm
  have chunkHigh := priorHalves.chunk.2.trans baseHalves.chunk.2.symm
  have stepLow := priorHalves.step.1.trans baseHalves.step.1.symm
  have stepHigh := priorHalves.step.2.trans baseHalves.step.2.symm
  have pcLow := priorHalves.programCounter.1.trans
    baseHalves.programCounter.1.symm
  have pcHigh := priorHalves.programCounter.2.trans
    baseHalves.programCounter.2.symm
  simp only [Prod.fst, Prod.snd] at tag vk0 vk1 vk2 vk3 header0 header1 header2 header3 boundary0 boundary1 boundary2 boundary3 accumulator0 accumulator1 accumulator2 accumulator3
  unfold baseTrace at tag
  rw [FPrimeFullHistoryBaseFacts.xOutTrace_inputColumns] at tag
  rw [priorTrace_inputColumns]
  unfold baseTrace
  rw [FPrimeFullHistoryBaseFacts.xOutTrace_inputColumns]
  simp only [List.map_cons, List.map_nil]
  rw [tag, vk0, vk1, vk2, vk3, header0, header1, header2, header3,
    chunkLow, chunkHigh, stepLow, stepHigh, pcLow, pcHigh,
    boundary0, boundary1, boundary2, boundary3,
    accumulator0, accumulator1, accumulator2, accumulator3]
  rfl

theorem priorDigest_eq_baseXOut
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (priorFacts : FPrimeFullHistoryPriorLinkSound.Facts assignment)
    (links : ∀ pair ∈ FPrimeFullHistoryStateLink.pairs,
      assignment pair.1 = assignment pair.2) :
    FPrimeFullHistoryPriorLink.digestColumns.map assignment =
      xOutDigestValue := by
  have inputValues : priorTrace.inputColumns.map assignment =
      FPrimeFullHistoryBaseFacts.xOutInputValues :=
    (priorInputs_eq_baseInputs baseFacts priorFacts links).trans
      (FPrimeFullHistoryBaseFacts.xOutInputValues_sound baseFacts)
  have scheduleEq := Poseidon2Sponge.runValueRounds_eq_of_schedules
    valueSchedules_equal FPrimeFullHistoryBaseFacts.xOutInputValues (fun _ => 0)
  have laneSound : ∀ lane, lane < 4 →
      assignment (priorTrace.outputColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds baseTrace.rounds
          FPrimeFullHistoryBaseFacts.xOutInputValues (fun _ => 0) lane := by
    intro lane laneLt
    have produced := priorFacts.sponge lane laneLt
    change assignment (priorTrace.outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds priorTrace.rounds
        (priorTrace.inputColumns.map assignment) (fun _ => 0) lane at produced
    rw [inputValues] at produced
    exact produced.trans (congrFun scheduleEq lane)
  have lane0 := laneSound 0 (by decide)
  have lane1 := laneSound 1 (by decide)
  have lane2 := laneSound 2 (by decide)
  have lane3 := laneSound 3 (by decide)
  rw [priorTrace_outputColumns] at lane0
  rw [priorTrace_outputColumns] at lane1
  rw [priorTrace_outputColumns] at lane2
  rw [priorTrace_outputColumns] at lane3
  simpa [FPrimeFullHistoryPriorLink.digestColumns, xOutDigestValue,
    FPrimeFullHistoryBaseFacts.traceOutputValues, baseTrace, rangeFour] using
      And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

/-- Exact non-vacuous closure of the base producer's delayed public edge. -/
theorem outgoing_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment) :
    Step.OutgoingLinked stepSemantics
      (input (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment))
      (decodedProof assignment) := by
  have baseFacts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical
    one baseSatisfies
  have priorFacts := FPrimeFullHistoryPriorLinkSound.sound goldilocksPrime
    canonical one priorLinkSatisfies
  have links := FPrimeFullHistoryStateLinkSound.sound canonical one
    stateLinkSatisfies
  have freshDigest := FPrimeFullHistoryPriorLinkSound.decodedFresh_digest
    priorFacts
  have priorDigest := priorDigest_eq_baseXOut baseFacts priorFacts links
  have proofEq := decodedProof_eq baseFacts
  have proofDigest : (decodedProof assignment).xOut = xOutDigestValue := by
    rw [proofEq]
    rfl
  have linkedDigest :
      (decodedProof assignment).xOut =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut :=
    proofDigest.trans (priorDigest.symm.trans freshDigest.symm)
  simp [Step.OutgoingLinked, Step.FreshLinked, stepSemantics, input,
    linkedDigest]

/-- `CIR-SOUND` for the first closed generated full-history edge. -/
theorem base_step_holds
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment) :
    Step.Holds hashSemantics stepSemantics .stateless context
      (decodedPrior assignment)
      (decodedNext assignment
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment))
      (input (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment))
      (decodedProof assignment) := by
  rw [Step.holds_iff_local_and_outgoing]
  exact ⟨
    fPrimeFullHistoryBase_step_local_sound goldilocksPrime canonical one
      baseSatisfies _,
    outgoing_sound goldilocksPrime canonical one baseSatisfies
      stateLinkSatisfies priorLinkSatisfies⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound
