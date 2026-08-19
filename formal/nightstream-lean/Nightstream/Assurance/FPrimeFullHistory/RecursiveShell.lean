import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryBaseGenericSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProjectionSound
import Nightstream.Assurance.FPrimeConcreteNifs

/-!
Contract: structural correspondence for the recursive step in the exact
plain/stateless `[1,1]` full-history profile.

This module decodes the real adjacent input and output state columns.  State
link, counter, and recursive-output conclusions are derived from their exact
generated rows. NIFS is the executable fixed-row verifier from
`FPrimeConcreteNifs`; no caller may supply a native-verifier success or an
exact-projection refinement premise. `CoreLaws` owns only the two surrounding
digest computations not implemented by that NIFS verifier.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.R1CS

abbrev Digest := FPrimeFullHistoryBaseStepSound.Digest
abbrev ConcreteFresh := Nightstream.Assurance.FPrimeConcreteNifs.Fresh
abbrev ConcreteAccumulator :=
  Nightstream.Assurance.FPrimeConcreteNifs.Accumulator
abbrev ConcreteProof := Nightstream.Assurance.FPrimeConcreteNifs.Proof

universe uRunning uFresh uNifsProof

section

variable
  {Running : Type uRunning}
  {Fresh : Type uFresh}
  {NifsProof : Type uNifsProof}

def priorValues (assignment : Nat → Nat) : List Nat :=
  FPrimeFullHistoryStateLinkSound.recursiveStateInColumns.map assignment

def nextValues (assignment : Nat → Nat) : List Nat :=
  FPrimeFullHistoryRecursiveOutput.stateOutColumns.map assignment

def prior
    (assignment : Nat → Nat)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh : Fresh) : State Digest Running Fresh Unit :=
  FPrimeFullHistoryBaseStepSound.stateOfValues
    (priorValues assignment)
    FPrimeFullHistoryBaseStepSound.initialSemantic
    (.active semantics.emptyRunning [priorFresh])

def next
    (assignment : Nat → Nat)
    (nextRunning : Running)
    (nextFresh : Fresh) : State Digest Running Fresh Unit :=
  FPrimeFullHistoryBaseStepSound.stateOfValues
    (nextValues assignment)
    FPrimeFullHistoryBaseStepSound.initialSemantic
    (.active nextRunning [nextFresh])

def input (nextFresh : Fresh) : Step.Input Fresh Unit Unit where
  nextLatest := [nextFresh]
  nebulaOpen := none
  nebulaNext := none

def proof
    (assignment : Nat → Nat)
    (nifsProof : NifsProof) : Step.Proof Digest NifsProof Unit where
  fold := .recursive nifsProof
  nebulaOpen := none
  semanticStateDigest :=
    FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 19
  xOut := FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment

theorem priorValues_sound
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment) :
    priorValues assignment = FPrimeFullHistoryBaseStepSound.nextValues := by
  calc
    priorValues assignment =
        FPrimeFullHistoryBase.stateOutColumns.map assignment :=
      (FPrimeFullHistoryStateLinkSound.stateVectors_sound canonical one
        stateLinkSatisfies).symm
    _ = FPrimeFullHistoryBaseStepSound.nextValues :=
      FPrimeFullHistoryBaseStepSound.stateOutValues_sound baseFacts

theorem prior_eq_baseNext
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh : Fresh) :
    prior assignment semantics priorFresh =
      FPrimeFullHistoryBaseGenericSound.next semantics priorFresh := by
  rw [prior, priorValues_sound baseFacts canonical one stateLinkSatisfies]
  have template := FPrimeFullHistoryBaseStepSound.nextValues_state
    (FPrimeFullHistoryBaseStepSound.Fresh.mk [])
  apply FPrimeFullHistoryBaseGenericSound.state_ext
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.chunkCount template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.stepCount template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.z0 template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.zi template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.initialSemanticState template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.semanticState template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.pc template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.accumulatorDigest template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.publicTrace template
  · rfl
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseGenericSound.next] using
      congrArg State.nebula template

/-- The chunk digest is the only surrounding digest result not owned by the
concrete NIFS verifier.  The running-accumulator digest is recomputed by the
exact recursive accumulator owner and is therefore not a caller law. -/
structure CoreLaws
    (assignment : Nat → Nat)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (nextFresh : Fresh) : Prop where
  nextChunkDigest :
    semantics.chunkDigest 1 [nextFresh] =
      FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 14

theorem prior_active
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh : Fresh)
    (baseLaws :
      FPrimeFullHistoryBaseGenericSound.BaseLaws semantics priorFresh) :
    Step.ActiveState FPrimeFullHistoryBaseStepSound.hashSemantics semantics
      .stateless FPrimeFullHistoryBaseStepSound.context
      (prior assignment semantics priorFresh)
      semantics.emptyRunning [priorFresh] := by
  rw [prior_eq_baseNext baseFacts canonical one stateLinkSatisfies semantics
    priorFresh]
  refine ⟨rfl, ?_, ?_, rfl, ?_, ?_⟩
  · change (1 : Nat) ≠ 0
    decide
  · change (1 : Nat) ≠ 0
    decide
  · exact baseLaws.emptyRunningDigest.symm
  · exact {
      initialBoundaryPinned := rfl
      initialSemanticStatePinned := rfl
      publicTraceMirrorsBoundary := rfl
      statelessSemanticEqualsAccumulator := fun _ => rfl
    }

theorem counterInputs_one
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment) :
    assignment FPrimeFullHistoryCounterSound.chunkInputCol = 1 ∧
      assignment FPrimeFullHistoryCounterSound.stepInputCol = 1 := by
  have links := FPrimeFullHistoryStateLinkSound.sound canonical one
    stateLinkSatisfies
  rw [FPrimeFullHistoryCounterSound.concreteColumns.1,
    FPrimeFullHistoryCounterSound.concreteColumns.2.1]
  exact ⟨
    (links (5947, 10842) (by native_decide)).symm.trans
      (baseFacts.constants (5947, 1) (by native_decide)),
    (links (5948, 10843) (by native_decide)).symm.trans
      (baseFacts.constants (5948, 1) (by native_decide))⟩

theorem outputCounters_two
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment) :
    assignment FPrimeFullHistoryCounterSound.chunkOutputCol = 2 ∧
      assignment FPrimeFullHistoryCounterSound.stepOutputCol = 2 := by
  have inputs := counterInputs_one baseFacts canonical one stateLinkSatisfies
  have transition := FPrimeFullHistoryCounterSound.sound goldilocksPrime
    canonical one counterSatisfies
  have chunkAdvance := transition.1
  have stepAdvance := transition.2.1
  exact ⟨chunkAdvance.trans (by omega), stepAdvance.trans (by omega)⟩

theorem semanticDigestValues (assignment : Nat → Nat) :
    FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 19 =
      FPrimeFullHistoryRecursiveOutputSound.semanticStateColumns.map
        assignment := by
  simp [nextValues, FPrimeFullHistoryBaseStepSound.digestAt,
    FPrimeFullHistoryRecursiveOutputSound.semanticStateColumns,
    FPrimeFullHistoryRecursiveOutput.stateOutColumns,
    show List.range 4 = [0, 1, 2, 3] by decide]

theorem accumulatorDigestValues (assignment : Nat → Nat) :
    FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23 =
      FPrimeFullHistoryRecursiveOutputSound.accumulatorStateColumns.map
        assignment := by
  simp [nextValues, FPrimeFullHistoryBaseStepSound.digestAt,
    FPrimeFullHistoryRecursiveOutputSound.accumulatorStateColumns,
    FPrimeFullHistoryRecursiveOutput.stateOutColumns,
    show List.range 4 = [0, 1, 2, 3] by decide]

/-- The exact accumulator owner, rather than a caller-supplied digest law,
binds the verifier-returned accumulator to the recursive state output. -/
theorem recursiveAccumulator_handle_eq_output
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : FPrimeFullHistoryRecursiveAccumulatorSound.Facts assignment) :
    (Nightstream.Assurance.FPrimeConcreteNifs.recursiveAccumulator
      (Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
        assignment canonical)).handle =
      FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23 := by
  change FPrimeFullHistoryRecursiveAccumulatorSound.handle assignment = _
  rw [facts.handle_eq_stateOutput]
  simp [FPrimeFullHistoryRecursiveAccumulatorSound.stateOutputHandle,
    FPrimeFullHistoryRecursiveAccumulator.stateOutputAccumulatorDigestColumns,
    nextValues, FPrimeFullHistoryBaseStepSound.digestAt,
    FPrimeFullHistoryRecursiveOutput.stateOutColumns,
    show List.range 4 = [0, 1, 2, 3] by decide]

/-- The shell context and carried singleton are decoded from the same exact
columns consumed by the recursive transcript verifier. -/
theorem concrete_context_binding
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (chunkDigest : Nat → List ConcreteFresh → Digest)
    (freshLink : Digest → ConcreteFresh → Bool)
    (applicationStep : Digest → List ConcreteFresh → Digest → Bool)
    (priorFresh nextFresh : ConcreteFresh)
    (priorFreshPublic :
      priorFresh.publicXOut =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut)
    (laws : CoreLaws assignment
      (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
        chunkDigest freshLink applicationStep) nextFresh) :
    FPrimeFullHistoryTranscriptSound.ContextBinding assignment
      (Step.nifsContext
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep)
        (prior assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
            chunkDigest freshLink applicationStep) priorFresh)
        (input nextFresh)) [priorFresh] := by
  have initialSemantic :
      FPrimeFullHistoryBaseStepSound.initialSemantic =
        [assignment 20, assignment 21, assignment 22, assignment 23] := by
    have pinned := congrArg
      (fun stateValues =>
        FPrimeFullHistoryBaseStepSound.digestAt stateValues 19)
      (FPrimeFullHistoryBaseStepSound.stateInValues_sound baseFacts)
    simpa [FPrimeFullHistoryBaseStepSound.initialSemantic,
      FPrimeFullHistoryBaseStepSound.digestAt,
      FPrimeFullHistoryBase.stateInColumns,
      FPrimeFullHistoryBase.stateInValues,
      show List.range 4 = [0, 1, 2, 3] by decide] using pinned.symm
  have stepOne :=
    (counterInputs_one baseFacts canonical one stateLinkSatisfies).2
  rw [FPrimeFullHistoryCounterSound.concreteColumns.2.1] at stepOne
  change assignment 10843 = 1 at stepOne
  refine ⟨?_, ?_⟩
  · simp [Step.nifsContext, prior, priorValues, input,
      FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseStepSound.digestAt,
      FPrimeFullHistoryStateLinkSound.recursiveStateInColumns,
      FPrimeFullHistoryStateLink.pairs,
      FPrimeFullHistoryTranscriptSound.decodedContext,
      FPrimeFullHistoryRecursiveTranscriptArtifact.contextColumns,
      initialSemantic, stepOne, laws.nextChunkDigest,
      nextValues, FPrimeFullHistoryRecursiveOutput.stateOutColumns,
      show List.range 4 = [0, 1, 2, 3] by decide]
  · simp only [FPrimeFullHistoryTranscriptSound.decodedLatest]
    congr 1
    have samePublic :
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut =
          (FPrimeFullHistoryTranscriptSound.decodedFresh assignment).publicXOut :=
      rfl
    cases priorFresh with
    | mk priorPublic =>
        cases decoded : FPrimeFullHistoryTranscriptSound.decodedFresh assignment with
        | mk decodedPublic =>
            have equal : priorPublic = decodedPublic := by
              calc
                priorPublic =
                    (FPrimeFullHistoryPriorLinkSound.decodedFresh
                      assignment).publicXOut := priorFreshPublic
                _ = (FPrimeFullHistoryTranscriptSound.decodedFresh
                      assignment).publicXOut := samePublic
                _ = decodedPublic := by simp [decoded]
            cases equal
            rfl

theorem semantic_advance
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh nextFresh : Fresh)
    (nextRunning : Running)
    (nifsProof : NifsProof)
    (nextRunningDigest :
      semantics.runningDigest nextRunning =
        FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23) :
    Step.SemanticAdvance semantics .stateless
      (prior assignment semantics priorFresh) nextRunning (input nextFresh)
      (proof assignment nifsProof) := by
  have equal := FPrimeFullHistoryRecursiveOutputSound.semanticAccumulator_sound
    canonical one outputSatisfies
  change FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 19 =
    semantics.runningDigest nextRunning
  rw [nextRunningDigest]
  rw [semanticDigestValues, accumulatorDigestValues]
  exact equal

theorem nebula_advance
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (assignment : Nat → Nat)
    (priorFresh nextFresh : Fresh)
    (nifsProof : NifsProof)
    (baseLaws :
      FPrimeFullHistoryBaseGenericSound.BaseLaws semantics priorFresh) :
    Step.NebulaAdvance semantics (prior assignment semantics priorFresh)
      (input nextFresh) (proof assignment nifsProof) := by
  simp [Step.NebulaAdvance, Step.installedNebula, prior, input, proof,
    FPrimeFullHistoryBaseStepSound.stateOfValues, baseLaws.nebulaNone]

theorem state_advance
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh nextFresh : Fresh)
    (nextRunning : Running)
    (nifsProof : NifsProof)
    (nextRunningDigest :
      semantics.runningDigest nextRunning =
        FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23)
    (laws : CoreLaws assignment semantics nextFresh) :
    next assignment nextRunning nextFresh =
      Step.advancedState semantics (prior assignment semantics priorFresh)
        nextRunning (input nextFresh) (proof assignment nifsProof) := by
  have priorEq := prior_eq_baseNext baseFacts canonical one stateLinkSatisfies
    semantics priorFresh
  have priorChunk : (prior assignment semantics priorFresh).chunkCount = 1 := by
    rw [priorEq]
    rfl
  have priorStep : (prior assignment semantics priorFresh).stepCount = 1 := by
    rw [priorEq]
    rfl
  have counters := outputCounters_two goldilocksPrime baseFacts canonical one
    stateLinkSatisfies counterSatisfies
  apply FPrimeFullHistoryBaseGenericSound.state_ext
  · change assignment
        (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 8 0) =
      (prior assignment semantics priorFresh).chunkCount + 1
    rw [← FPrimeFullHistoryCounterSound.concreteColumns.2.2.1,
      priorChunk, counters.1]
  · change assignment
        (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 9 0) =
      (prior assignment semantics priorFresh).stepCount + [nextFresh].length
    rw [← FPrimeFullHistoryCounterSound.concreteColumns.2.2.2,
      priorStep, counters.2]
    rfl
  · simp [next, prior, nextValues, priorValues,
      Step.advancedState, FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryBaseStepSound.digestAt,
      FPrimeFullHistoryRecursiveOutput.stateOutColumns,
      FPrimeFullHistoryStateLinkSound.recursiveStateInColumns,
      FPrimeFullHistoryStateLink.pairs,
      show List.range 4 = [0, 1, 2, 3] by decide]
  · change FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 14 =
      semantics.chunkDigest (prior assignment semantics priorFresh).stepCount
        [nextFresh]
    rw [priorStep, laws.nextChunkDigest]
  · rfl
  · rfl
  · simp [next, prior, nextValues, priorValues,
      Step.advancedState, FPrimeFullHistoryBaseStepSound.stateOfValues,
      FPrimeFullHistoryRecursiveOutput.stateOutColumns,
      FPrimeFullHistoryStateLinkSound.recursiveStateInColumns,
      FPrimeFullHistoryStateLink.pairs]
  · change FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23 =
      semantics.runningDigest nextRunning
    exact nextRunningDigest.symm
  · change FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 27 =
      semantics.chunkDigest (prior assignment semantics priorFresh).stepCount
        [nextFresh]
    rw [priorStep, laws.nextChunkDigest]
    simp [nextValues, FPrimeFullHistoryBaseStepSound.digestAt,
      FPrimeFullHistoryRecursiveOutput.stateOutColumns,
      show List.range 4 = [0, 1, 2, 3] by decide]
  · rfl
  · simp [next, prior, input, Step.advancedState, Step.installedNebula,
      FPrimeFullHistoryBaseStepSound.stateOfValues]

/-- Exact base producer, adjacent link, and recursive prior-link rows establish
the recursive branch's delayed link to its carried prior fresh claim. -/
theorem prior_linked
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (priorFresh : Fresh)
    (publicXOut : Fresh → Digest)
    (linkLaws :
      FPrimeFullHistoryBaseGenericSound.FreshLinkLaws semantics publicXOut)
    (freshPublic :
      publicXOut priorFresh =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut) :
    Step.FreshLinked semantics.freshLink
      (XOut.compute FPrimeFullHistoryBaseStepSound.hashSemantics .stateless
        FPrimeFullHistoryBaseStepSound.context
        (prior assignment semantics priorFresh)) [priorFresh] := by
  have baseFacts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical
    one baseSatisfies
  have outgoing := FPrimeFullHistoryBaseGenericSound.outgoing_sound
    goldilocksPrime canonical one baseSatisfies stateLinkSatisfies
    priorLinkSatisfies semantics priorFresh publicXOut linkLaws freshPublic
  have proofEq := FPrimeFullHistoryBaseGenericSound.decodedProof_eq
    (NifsProof := NifsProof) baseFacts
  have priorEq := prior_eq_baseNext baseFacts canonical one stateLinkSatisfies
    semantics priorFresh
  have outputBinding := FPrimeFullHistoryBaseGenericSound.output_binding
    (NifsProof := NifsProof) semantics priorFresh
  change Step.FreshLinked semantics.freshLink
    (FPrimeFullHistoryBaseGenericSound.decodedProof assignment).xOut
    [priorFresh] at outgoing
  rw [proofEq] at outgoing
  rw [priorEq, ← outputBinding]
  exact outgoing

def verifierHeaderColumns : List Nat :=
  FPrimeFullHistoryPriorLink.stateInColumns.take 8

def counterHalfColumns : List Nat :=
  [ FPrimeFullHistoryRecursiveOutputSound.chunkLowHalfCol
  , FPrimeFullHistoryRecursiveOutputSound.chunkHighHalfCol
  , FPrimeFullHistoryRecursiveOutputSound.stepLowHalfCol
  , FPrimeFullHistoryRecursiveOutputSound.stepHighHalfCol
  , FPrimeFullHistoryRecursiveOutputSound.programCounterLowHalfCol
  , FPrimeFullHistoryRecursiveOutputSound.programCounterHighHalfCol ]

def boundaryColumns : List Nat :=
  (FPrimeFullHistoryRecursiveOutput.stateOutColumns.drop 14).take 4

def accumulatorColumns : List Nat :=
  FPrimeFullHistoryRecursiveOutputSound.accumulatorStateColumns

def outputTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryRecursiveOutputPoseidonHashes.xOutTrace

theorem outputTrace_inputColumns :
    outputTrace.inputColumns =
      [FPrimeFullHistoryRecursiveOutputSound.tagColumn] ++
        verifierHeaderColumns ++ counterHalfColumns ++
        boundaryColumns ++ accumulatorColumns := by
  native_decide

theorem outputTrace_outputColumns :
    outputTrace.outputColumns = FPrimeFullHistoryRecursiveOutput.xOutColumns := by
  native_decide

theorem outputTrace_valueSchedule :
    Poseidon2Sponge.valueSchedules outputTrace.rounds =
      Poseidon2Sponge.valueSchedules
        FPrimeFullHistoryBaseFacts.xOutTrace.rounds := by
  native_decide

theorem verifierHeaderValues
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment) :
    verifierHeaderColumns.map assignment =
      FPrimeFullHistoryBaseStepSound.vkDigest ++
        FPrimeFullHistoryBaseStepSound.headerDigest := by
  have links := FPrimeFullHistoryStateLinkSound.sound canonical one
    stateLinkSatisfies
  have value0 : assignment 10834 =
      FPrimeFullHistoryBase.stateInValues.getD 0 0 :=
    (links (2, 10834) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (2, FPrimeFullHistoryBase.stateInValues.getD 0 0) (by native_decide))
  have value1 : assignment 10835 =
      FPrimeFullHistoryBase.stateInValues.getD 1 0 :=
    (links (3, 10835) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (3, FPrimeFullHistoryBase.stateInValues.getD 1 0) (by native_decide))
  have value2 : assignment 10836 =
      FPrimeFullHistoryBase.stateInValues.getD 2 0 :=
    (links (4, 10836) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (4, FPrimeFullHistoryBase.stateInValues.getD 2 0) (by native_decide))
  have value3 : assignment 10837 =
      FPrimeFullHistoryBase.stateInValues.getD 3 0 :=
    (links (5, 10837) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (5, FPrimeFullHistoryBase.stateInValues.getD 3 0) (by native_decide))
  have value4 : assignment 10838 =
      FPrimeFullHistoryBase.stateInValues.getD 4 0 :=
    (links (6, 10838) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (6, FPrimeFullHistoryBase.stateInValues.getD 4 0) (by native_decide))
  have value5 : assignment 10839 =
      FPrimeFullHistoryBase.stateInValues.getD 5 0 :=
    (links (7, 10839) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (7, FPrimeFullHistoryBase.stateInValues.getD 5 0) (by native_decide))
  have value6 : assignment 10840 =
      FPrimeFullHistoryBase.stateInValues.getD 6 0 :=
    (links (8, 10840) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (8, FPrimeFullHistoryBase.stateInValues.getD 6 0) (by native_decide))
  have value7 : assignment 10841 =
      FPrimeFullHistoryBase.stateInValues.getD 7 0 :=
    (links (9, 10841) (by native_decide)).symm.trans
      (baseFacts.stateIn
        (9, FPrimeFullHistoryBase.stateInValues.getD 7 0) (by native_decide))
  simp [verifierHeaderColumns, FPrimeFullHistoryBaseStepSound.vkDigest,
    FPrimeFullHistoryBaseStepSound.headerDigest,
    FPrimeFullHistoryBaseStepSound.digestAt,
    FPrimeFullHistoryBaseStepSound.priorValues,
    FPrimeFullHistoryPriorLink.stateInColumns,
    show List.range 4 = [0, 1, 2, 3] by decide,
    value0, value1, value2, value3, value4, value5, value6, value7]

theorem programCounter_one
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment) :
    assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0) = 1 := by
  have links := FPrimeFullHistoryStateLinkSound.sound canonical one
    stateLinkSatisfies
  exact (links (1, 10833) (by native_decide)).symm.trans
    (baseFacts.stateIn (1, 1) (by native_decide))

def expectedXOutInputs (assignment : Nat → Nat) : List Nat :=
  [1313210370] ++
  FPrimeFullHistoryBaseStepSound.vkDigest ++
  FPrimeFullHistoryBaseStepSound.headerDigest ++
  [2, 0, 2, 0, 1, 0] ++
  boundaryColumns.map assignment ++ accumulatorColumns.map assignment

theorem preimageValues
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (nextRunning : Running)
    (nextFresh : Fresh) :
    FPrimeFullHistoryBaseStepSound.stateOutputValues
        (XOut.preimage FPrimeFullHistoryBaseStepSound.hashSemantics .stateless
          FPrimeFullHistoryBaseStepSound.context
          (next assignment nextRunning nextFresh)) =
      expectedXOutInputs assignment := by
  have counters := outputCounters_two goldilocksPrime baseFacts canonical one
    stateLinkSatisfies counterSatisfies
  have pcOne := programCounter_one baseFacts canonical one stateLinkSatisfies
  have chunkTwo :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 8 0) =
        2 := by
    rw [← FPrimeFullHistoryCounterSound.concreteColumns.2.2.1]
    exact counters.1
  have stepTwo :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 9 0) =
        2 := by
    rw [← FPrimeFullHistoryCounterSound.concreteColumns.2.2.2]
    exact counters.2
  have chunkMod :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 8 0) %
          4294967296 = 2 := by
    rw [chunkTwo]
  have chunkLt :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 8 0) <
          4294967296 := by
    rw [chunkTwo]
    decide
  have stepMod :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 9 0) %
          4294967296 = 2 := by
    rw [stepTwo]
  have stepLt :
      assignment (FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 9 0) <
          4294967296 := by
    rw [stepTwo]
    decide
  have pcMod :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0) %
          4294967296 = 1 := by
    rw [pcOne]
  have pcLt :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0) <
          4294967296 := by
    rw [pcOne]
    decide
  have chunkMod' := chunkMod
  have chunkLt' := chunkLt
  have stepMod' := stepMod
  have stepLt' := stepLt
  have pcMod' := pcMod
  have pcLt' := pcLt
  simp only [FPrimeFullHistoryRecursiveOutput.stateOutColumns] at chunkMod' chunkLt' stepMod' stepLt'
  simp only [FPrimeFullHistoryPriorLink.stateInColumns] at pcMod' pcLt'
  simp [FPrimeFullHistoryBaseStepSound.stateOutputValues, XOut.preimage,
    FPrimeFullHistoryBaseStepSound.hashSemantics,
    FPrimeFullHistoryBaseStepSound.profileHash,
    XOut.verifierDigest, FPrimeFullHistoryBaseStepSound.context,
    next, nextValues, FPrimeFullHistoryBaseStepSound.stateOfValues,
    FPrimeFullHistoryBaseStepSound.digestAt,
    FPrimeFullHistoryRecursiveOutput.stateOutColumns,
    FPrimeFullHistoryBaseStepSound.low32,
    FPrimeFullHistoryBaseStepSound.high32,
    expectedXOutInputs, boundaryColumns, accumulatorColumns,
    FPrimeFullHistoryRecursiveOutputSound.accumulatorStateColumns,
    FPrimeFullHistoryPriorLink.stateInColumns,
    show List.range 4 = [0, 1, 2, 3] by decide]
  exact ⟨chunkMod', chunkLt', stepMod', stepLt', pcMod', pcLt'⟩

theorem outputTraceInputs
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment) :
    outputTrace.inputColumns.map assignment = expectedXOutInputs assignment := by
  have inputsOne := counterInputs_one baseFacts canonical one stateLinkSatisfies
  have pcOne := programCounter_one baseFacts canonical one stateLinkSatisfies
  have outputFacts := FPrimeFullHistoryRecursiveOutputSound.sound
    goldilocksPrime canonical one outputSatisfies
  have priorFacts := FPrimeFullHistoryPriorLinkSound.sound goldilocksPrime
    canonical one priorLinkSatisfies
  have counterHalves :=
    FPrimeFullHistoryRecursiveOutputSound.counterHalves_sound
      goldilocksPrime canonical one counterSatisfies outputSatisfies
      inputsOne.1 inputsOne.2
  have pcHalves :=
    FPrimeFullHistoryRecursiveOutputSound.programCounterHalves_sound
      outputFacts priorFacts pcOne
  have tag := FPrimeFullHistoryRecursiveOutputSound.tag_sound outputFacts one
  have header := verifierHeaderValues baseFacts canonical one
    stateLinkSatisfies
  rw [outputTrace_inputColumns]
  simp only [List.map_append, List.map_cons, List.map_nil]
  rw [tag, header]
  simp [counterHalfColumns, boundaryColumns, accumulatorColumns,
    expectedXOutInputs, counterHalves.chunk.1, counterHalves.chunk.2,
    counterHalves.step.1, counterHalves.step.2, pcHalves.1, pcHalves.2]

/-- Exact recursive output rows compute the modeled `XOut.compute` value for
the decoded post-state. -/
theorem output_binding
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (baseFacts : FPrimeFullHistoryBaseFacts.Facts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (nextRunning : Running)
    (nextFresh : Fresh)
    (nifsProof : NifsProof) :
    (proof assignment nifsProof).xOut =
      XOut.compute FPrimeFullHistoryBaseStepSound.hashSemantics .stateless
        FPrimeFullHistoryBaseStepSound.context
        (next assignment nextRunning nextFresh) := by
  have outputFacts := FPrimeFullHistoryRecursiveOutputSound.sound
    goldilocksPrime canonical one outputSatisfies
  have inputValues := outputTraceInputs goldilocksPrime baseFacts canonical one
    stateLinkSatisfies priorLinkSatisfies counterSatisfies outputSatisfies
  have preimage := preimageValues goldilocksPrime baseFacts canonical one
    stateLinkSatisfies counterSatisfies nextRunning nextFresh
  have scheduleEq := Poseidon2Sponge.runValueRounds_eq_of_schedules
    outputTrace_valueSchedule (expectedXOutInputs assignment) (fun _ => 0)
  have laneSound : ∀ lane, lane < 4 →
      assignment (outputTrace.outputColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds
          FPrimeFullHistoryBaseFacts.xOutTrace.rounds
          (FPrimeFullHistoryBaseStepSound.stateOutputValues
            (XOut.preimage FPrimeFullHistoryBaseStepSound.hashSemantics
              .stateless FPrimeFullHistoryBaseStepSound.context
              (next assignment nextRunning nextFresh)))
          (fun _ => 0) lane := by
    intro lane laneLt
    have produced := outputFacts.sponge lane laneLt
    change assignment (outputTrace.outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds outputTrace.rounds
        (outputTrace.inputColumns.map assignment) (fun _ => 0) lane at produced
    rw [inputValues] at produced
    rw [preimage]
    exact produced.trans (congrFun scheduleEq lane)
  have lane0 := laneSound 0 (by decide)
  have lane1 := laneSound 1 (by decide)
  have lane2 := laneSound 2 (by decide)
  have lane3 := laneSound 3 (by decide)
  rw [outputTrace_outputColumns] at lane0 lane1 lane2 lane3
  simpa [proof, XOut.compute,
    FPrimeFullHistoryBaseStepSound.hashSemantics,
    FPrimeFullHistoryBaseStepSound.profileHash,
    FPrimeFullHistoryBaseStepSound.hashValues,
    FPrimeFullHistoryBaseFacts.traceOutputValues,
    FPrimeFullHistoryRecursiveOutput.xOutColumns,
    show List.range 4 = [0, 1, 2, 3] by decide] using
      And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

/-- Supported-profile recursive correspondence.  Exact NIFS rows first enter
the independent sampled verifier.  Its 31 projection identities are then
coefficient-exact, in which case the concrete native NIFS callback returns the
decoded PiDEC accumulator, or they expose the named `BadRoot` event. -/
theorem local_sound_or_badRoot
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (nifsRows :
      Nightstream.Assurance.FPrimeConcreteNifs.RecursiveRows assignment)
    (chunkDigest : Nat → List ConcreteFresh → Digest)
    (freshLink : Digest → ConcreteFresh → Bool)
    (applicationStep : Digest → List ConcreteFresh → Digest → Bool)
    (priorFresh nextFresh : ConcreteFresh)
    (baseLaws :
      FPrimeFullHistoryBaseGenericSound.BaseLaws
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep) priorFresh)
    (linkLaws :
      FPrimeFullHistoryBaseGenericSound.FreshLinkLaws
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep)
        (fun fresh : ConcreteFresh => fresh.publicXOut))
    (priorFreshPublic :
      priorFresh.publicXOut =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut)
    (coreLaws : CoreLaws assignment
      (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
        chunkDigest freshLink applicationStep) nextFresh) :
    Step.LocalHolds FPrimeFullHistoryBaseStepSound.hashSemantics
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep)
        .stateless FPrimeFullHistoryBaseStepSound.context
        (prior assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
            chunkDigest freshLink applicationStep) priorFresh)
        (next assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.recursiveAccumulator
            (Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
              assignment canonical)) nextFresh)
        (input nextFresh)
        (proof assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
            assignment canonical)) ∨
      Nightstream.SuperNeo.ProjectionCheck.BatchBadRoot
        ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          FPrimeFullHistoryProjection.recursiveTraces assignment) := by
  let semantics := Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
    chunkDigest freshLink applicationStep
  let nifsProof := Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
    assignment canonical
  let nextRunning :=
    Nightstream.Assurance.FPrimeConcreteNifs.recursiveAccumulator nifsProof
  have semanticAccepted :=
    Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_sound
      goldilocksPrime canonical one nifsRows
  have baseFacts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical
    one baseSatisfies
  have binding := concrete_context_binding baseFacts canonical one
    stateLinkSatisfies chunkDigest freshLink applicationStep priorFresh
    nextFresh priorFreshPublic coreLaws
  have nextRunningDigest :
      semantics.runningDigest nextRunning =
        FPrimeFullHistoryBaseStepSound.digestAt (nextValues assignment) 23 := by
    change nextRunning.handle = _
    simpa [nextRunning] using
      (recursiveAccumulator_handle_eq_output canonical
        semanticAccepted.accumulator)
  rcases Nightstream.Assurance.FPrimeConcreteNifs.recursive_semantic_sound_or_badRoot
      semanticAccepted with artifact | bad
  · left
    have semanticAccepted' :
        Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted
          nifsProof.assignment := by
      simpa [nifsProof] using semanticAccepted
    have exact' : Nightstream.SuperNeo.ProjectionCheck.BatchExact
        (ProjectionProgram.BatchIdentity
          FPrimeFullHistoryProjection.recursiveTraces nifsProof.assignment) := by
      simpa [nifsProof] using artifact.projection.exact
    have nifsResult :=
      Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics_nifsVerify
        chunkDigest freshLink applicationStep
        (Step.nifsContext semantics
          (prior assignment semantics priorFresh) (input nextFresh))
        [priorFresh] nifsProof (by simpa [semantics, nifsProof] using binding)
        semanticAccepted' exact'
    have nifsResult' :
        semantics.nifsVerify
            (Step.nifsContext semantics
              (prior assignment semantics priorFresh) (input nextFresh))
            semantics.emptyRunning [priorFresh] nifsProof = some nextRunning := by
      simpa [semantics, nextRunning] using nifsResult
    have recursive : Step.RecursiveLocalHolds
        FPrimeFullHistoryBaseStepSound.hashSemantics semantics .stateless
        FPrimeFullHistoryBaseStepSound.context
        (prior assignment semantics priorFresh)
        (next assignment nextRunning nextFresh)
        (input nextFresh) (proof assignment nifsProof)
        semantics.emptyRunning [priorFresh] nifsProof := by
      refine ⟨
        prior_active baseFacts canonical one stateLinkSatisfies semantics
          priorFresh baseLaws,
        rfl,
        by simp,
        prior_linked goldilocksPrime canonical one baseSatisfies
          stateLinkSatisfies priorLinkSatisfies semantics priorFresh
          (fun fresh : ConcreteFresh => fresh.publicXOut)
          linkLaws priorFreshPublic,
        ?_⟩
      rw [nifsResult']
      exact ⟨
        by simp [input],
        semantic_advance canonical one outputSatisfies semantics priorFresh
          nextFresh nextRunning nifsProof nextRunningDigest,
        nebula_advance semantics assignment priorFresh nextFresh nifsProof
          baseLaws,
        state_advance goldilocksPrime baseFacts canonical one stateLinkSatisfies
          counterSatisfies semantics priorFresh nextFresh nextRunning nifsProof
          nextRunningDigest coreLaws,
        output_binding goldilocksPrime baseFacts canonical one stateLinkSatisfies
          priorLinkSatisfies counterSatisfies outputSatisfies nextRunning
          nextFresh nifsProof⟩
    simpa [semantics, nifsProof, nextRunning, Step.LocalHolds, prior, proof,
      FPrimeFullHistoryBaseStepSound.stateOfValues] using recursive
  · exact Or.inr bad

/-- Exact recursive-output and terminal-link rows close the trailing delayed
fresh link for an arbitrary full claim carrier with the specified public
projection. -/
theorem terminal_outgoing_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (terminalLinkSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (nextFresh : Fresh)
    (nifsProof : NifsProof)
    (publicXOut : Fresh → Digest)
    (linkLaws :
      FPrimeFullHistoryBaseGenericSound.FreshLinkLaws semantics publicXOut)
    (nextFreshPublic :
      publicXOut nextFresh =
        (FPrimeFullHistoryOutputEncodingSound.decodedTerminalFresh
          assignment).publicXOut) :
    Step.OutgoingLinked semantics (input nextFresh)
      (proof assignment nifsProof) := by
  have terminalDigest :=
    FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut
      goldilocksPrime canonical one outputSatisfies terminalLinkSatisfies
  have linked : (proof assignment nifsProof).xOut = publicXOut nextFresh := by
    change FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment =
      publicXOut nextFresh
    exact terminalDigest.symm.trans nextFreshPublic.symm
  simp [Step.OutgoingLinked, Step.FreshLinked, input,
    linkLaws.freshLink_eq, linked]

/-- Closed supported-profile edge, retaining the projection-root alternative. -/
theorem step_holds_or_badRoot
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (nifsRows :
      Nightstream.Assurance.FPrimeConcreteNifs.RecursiveRows assignment)
    (terminalLinkSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment)
    (chunkDigest : Nat → List ConcreteFresh → Digest)
    (freshLink : Digest → ConcreteFresh → Bool)
    (applicationStep : Digest → List ConcreteFresh → Digest → Bool)
    (priorFresh nextFresh : ConcreteFresh)
    (baseLaws :
      FPrimeFullHistoryBaseGenericSound.BaseLaws
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep) priorFresh)
    (linkLaws :
      FPrimeFullHistoryBaseGenericSound.FreshLinkLaws
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep)
        (fun fresh : ConcreteFresh => fresh.publicXOut))
    (priorFreshPublic :
      priorFresh.publicXOut =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut)
    (nextFreshPublic :
      nextFresh.publicXOut =
        (FPrimeFullHistoryOutputEncodingSound.decodedTerminalFresh
          assignment).publicXOut)
    (coreLaws : CoreLaws assignment
      (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
        chunkDigest freshLink applicationStep) nextFresh) :
    Step.Holds FPrimeFullHistoryBaseStepSound.hashSemantics
        (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
          chunkDigest freshLink applicationStep)
        .stateless FPrimeFullHistoryBaseStepSound.context
        (prior assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
            chunkDigest freshLink applicationStep) priorFresh)
        (next assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.recursiveAccumulator
            (Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
              assignment canonical)) nextFresh)
        (input nextFresh)
        (proof assignment
          (Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
            assignment canonical)) ∨
      Nightstream.SuperNeo.ProjectionCheck.BatchBadRoot
        ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          FPrimeFullHistoryProjection.recursiveTraces assignment) := by
  let semantics := Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics
    chunkDigest freshLink applicationStep
  let nifsProof := Nightstream.Assurance.FPrimeConcreteNifs.proofOfAssignment
    assignment canonical
  let nextRunning :=
    Nightstream.Assurance.FPrimeConcreteNifs.recursiveAccumulator nifsProof
  rcases local_sound_or_badRoot goldilocksPrime canonical one baseSatisfies
      stateLinkSatisfies priorLinkSatisfies counterSatisfies outputSatisfies
      nifsRows chunkDigest freshLink applicationStep priorFresh nextFresh
      baseLaws linkLaws priorFreshPublic coreLaws with
    localProof | bad
  · left
    exact Step.closeLocal
      FPrimeFullHistoryBaseStepSound.hashSemantics semantics .stateless
      FPrimeFullHistoryBaseStepSound.context
      (prior assignment semantics priorFresh)
      (next assignment nextRunning nextFresh)
      (input nextFresh) (proof assignment nifsProof) (by
        simpa [semantics, nifsProof, nextRunning] using localProof)
      (terminal_outgoing_sound goldilocksPrime canonical one outputSatisfies
        terminalLinkSatisfies semantics nextFresh nifsProof
        (fun fresh : ConcreteFresh => fresh.publicXOut) linkLaws nextFreshPublic)
  · exact Or.inr bad

end

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound
