import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryBaseOutgoingSound

/-!
Contract: composable base-step correspondence for the exact plain/stateless
full-history base owner.

Unlike the earlier unit-carrier theorem, this module is polymorphic in the
real running accumulator, fresh claim, and NIFS proof types.  Its laws expose
only primitive executable-semantics equalities needed by the base branch;
they do not carry `LocalHolds`, `Holds`, or any verifier conclusion.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBase
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseFacts

abbrev Digest := FPrimeFullHistoryBaseStepSound.Digest

universe uRunning uFresh uNifsProof

section

variable
  {Running : Type uRunning}
  {Fresh : Type uFresh}
  {NifsProof : Type uNifsProof}

/-- Primitive base-branch equations owed by a concrete executable semantics.
No circuit or protocol acceptance is stored in this interface. -/
structure BaseLaws
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) : Prop where
  emptyRunningDigest :
    semantics.runningDigest semantics.emptyRunning =
      FPrimeFullHistoryBaseStepSound.emptyAccumulator
  initialNebula : semantics.initialNebula = none
  chunkDigest :
    semantics.chunkDigest 0 [fresh] =
      FPrimeFullHistoryBaseStepSound.chunkDigestValue
  nebulaNone : semantics.nebulaVerify none none none = true

def prior : State Digest Running Fresh Unit :=
  FPrimeFullHistoryBaseStepSound.stateOfValues
    FPrimeFullHistoryBaseStepSound.priorValues
    FPrimeFullHistoryBaseStepSound.initialSemantic .initial

def next
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) : State Digest Running Fresh Unit :=
  { chunkCount := 1
    stepCount := 1
    z0 := FPrimeFullHistoryBaseStepSound.initialBoundary
    zi := FPrimeFullHistoryBaseStepSound.chunkDigestValue
    initialSemanticState := FPrimeFullHistoryBaseStepSound.initialSemantic
    semanticState := FPrimeFullHistoryBaseStepSound.emptyAccumulator
    pc := 1
    accumulatorDigest := FPrimeFullHistoryBaseStepSound.emptyAccumulator
    publicTrace := FPrimeFullHistoryBaseStepSound.chunkDigestValue
    proof := .active semantics.emptyRunning [fresh]
    nebula := none }

def input (fresh : Fresh) : Step.Input Fresh Unit Unit where
  nextLatest := [fresh]
  nebulaOpen := none
  nebulaNext := none

def proof : Step.Proof Digest NifsProof Unit where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := FPrimeFullHistoryBaseStepSound.emptyAccumulator
  xOut := FPrimeFullHistoryBaseStepSound.xOutDigestValue

def decodedPrior (assignment : Nat → Nat) : State Digest Running Fresh Unit :=
  FPrimeFullHistoryBaseStepSound.stateOfValues
    (stateInColumns.map assignment)
    FPrimeFullHistoryBaseStepSound.initialSemantic .initial

def decodedNext
    (assignment : Nat → Nat)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) : State Digest Running Fresh Unit :=
  FPrimeFullHistoryBaseStepSound.stateOfValues
    (stateOutColumns.map assignment)
    FPrimeFullHistoryBaseStepSound.initialSemantic
    (.active semantics.emptyRunning [fresh])

def decodedProof (assignment : Nat → Nat) : Step.Proof Digest NifsProof Unit where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest :=
    FPrimeFullHistoryBaseStepSound.digestAt
      (stateOutColumns.map assignment) 19
  xOut := xOutColumns.map assignment

private theorem prior_z0_eq_zi :
    FPrimeFullHistoryBaseStepSound.digestAt
        FPrimeFullHistoryBaseStepSound.priorValues 10 =
      FPrimeFullHistoryBaseStepSound.digestAt
        FPrimeFullHistoryBaseStepSound.priorValues 14 := by
  native_decide

private theorem prior_semantic_eq_accumulator :
    FPrimeFullHistoryBaseStepSound.digestAt
        FPrimeFullHistoryBaseStepSound.priorValues 19 =
      FPrimeFullHistoryBaseStepSound.digestAt
        FPrimeFullHistoryBaseStepSound.priorValues 23 := by
  native_decide

theorem prior_initial
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    Step.InitialState FPrimeFullHistoryBaseStepSound.hashSemantics semantics
      .stateless FPrimeFullHistoryBaseStepSound.context prior := by
  unfold Step.InitialState
  refine ⟨rfl, rfl, rfl, prior_z0_eq_zi, rfl, rfl, rfl, ?_, ?_, rfl, ?_⟩
  · exact laws.emptyRunningDigest.symm
  · exact laws.initialNebula.symm
  · exact prior_semantic_eq_accumulator

theorem semantic_advance
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    Step.SemanticAdvance semantics .stateless prior semantics.emptyRunning
      (input fresh) proof := by
  simp [Step.SemanticAdvance, proof, laws.emptyRunningDigest]

theorem nebula_advance
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    Step.NebulaAdvance semantics prior (input fresh) proof := by
  simp [Step.NebulaAdvance, Step.installedNebula, prior, input, proof,
    FPrimeFullHistoryBaseStepSound.stateOfValues, laws.nebulaNone]

theorem next_advanced
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    next semantics fresh =
      Step.advancedState semantics prior semantics.emptyRunning
        (input fresh) proof := by
  simp [Step.advancedState, next, prior, input, proof,
    Step.installedNebula, FPrimeFullHistoryBaseStepSound.stateOfValues,
    FPrimeFullHistoryBaseStepSound.priorValues,
    FPrimeFullHistoryBaseStepSound.initialSemantic,
    FPrimeFullHistoryBaseStepSound.initialBoundary,
    FPrimeFullHistoryBaseStepSound.emptyAccumulator,
    FPrimeFullHistoryBaseStepSound.chunkDigestValue,
    FPrimeFullHistoryBaseStepSound.digestAt, stateInValues,
    laws.chunkDigest, laws.emptyRunningDigest]

theorem output_binding
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) :
    (proof (NifsProof := NifsProof)).xOut =
      XOut.compute FPrimeFullHistoryBaseStepSound.hashSemantics
      .stateless FPrimeFullHistoryBaseStepSound.context
      (next semantics fresh) := by
  change FPrimeFullHistoryBaseStepSound.xOutDigestValue =
    FPrimeFullHistoryBaseStepSound.hashValues
      (FPrimeFullHistoryBaseStepSound.stateOutputValues
        (XOut.preimage FPrimeFullHistoryBaseStepSound.hashSemantics .stateless
          FPrimeFullHistoryBaseStepSound.context (next semantics fresh)))
  have preimage :
      FPrimeFullHistoryBaseStepSound.stateOutputValues
        (XOut.preimage FPrimeFullHistoryBaseStepSound.hashSemantics .stateless
          FPrimeFullHistoryBaseStepSound.context (next semantics fresh)) =
        FPrimeFullHistoryBaseFacts.xOutInputValues := by
    simpa [next] using
      FPrimeFullHistoryBaseStepSound.next_preimage_values
        (FPrimeFullHistoryBaseStepSound.Fresh.mk [])
  rw [preimage]
  rfl

theorem profile_baseLocal
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    Step.BaseLocalHolds FPrimeFullHistoryBaseStepSound.hashSemantics semantics
      .stateless FPrimeFullHistoryBaseStepSound.context prior
      (next semantics fresh) (input fresh) proof := by
  refine ⟨prior_initial semantics fresh laws, rfl, ?_,
    semantic_advance semantics fresh laws, nebula_advance semantics fresh laws,
    next_advanced semantics fresh laws, output_binding semantics fresh⟩
  simp [input]

theorem decodedPrior_eq {assignment : Nat → Nat}
    (facts : Facts assignment) :
    (decodedPrior assignment : State Digest Running Fresh Unit) = prior := by
  rw [decodedPrior, FPrimeFullHistoryBaseStepSound.stateInValues_sound facts]
  rfl

/-- Fieldwise equality principle for the intentionally non-derived generic
Construction-2 state carrier. -/
theorem state_ext
    {left right : State Digest Running Fresh Unit}
    (chunkCount : left.chunkCount = right.chunkCount)
    (stepCount : left.stepCount = right.stepCount)
    (z0 : left.z0 = right.z0)
    (zi : left.zi = right.zi)
    (initialSemanticState :
      left.initialSemanticState = right.initialSemanticState)
    (semanticState : left.semanticState = right.semanticState)
    (pc : left.pc = right.pc)
    (accumulatorDigest : left.accumulatorDigest = right.accumulatorDigest)
    (publicTrace : left.publicTrace = right.publicTrace)
    (proofState : left.proof = right.proof)
    (nebula : left.nebula = right.nebula) :
    left = right := by
  cases left
  cases right
  simp_all

theorem decodedNext_eq {assignment : Nat → Nat}
    (facts : Facts assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) :
    decodedNext assignment semantics fresh = next semantics fresh := by
  rw [decodedNext, FPrimeFullHistoryBaseStepSound.stateOutValues_sound facts]
  have template := FPrimeFullHistoryBaseStepSound.nextValues_state
    (FPrimeFullHistoryBaseStepSound.Fresh.mk [])
  apply state_ext
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.chunkCount template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.stepCount template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.z0 template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.zi template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.initialSemanticState template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.semanticState template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.pc template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.accumulatorDigest template
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.publicTrace template
  · rfl
  · simpa [FPrimeFullHistoryBaseStepSound.stateOfValues, next] using
      congrArg State.nebula template

theorem decodedProof_eq {assignment : Nat → Nat}
    (facts : Facts assignment) :
    (decodedProof assignment : Step.Proof Digest NifsProof Unit) = proof := by
  rw [decodedProof,
    FPrimeFullHistoryBaseStepSound.stateOutValues_sound facts,
    FPrimeFullHistoryBaseStepSound.xOutValues_sound facts]
  have nextState := FPrimeFullHistoryBaseStepSound.nextValues_state
    (FPrimeFullHistoryBaseStepSound.Fresh.mk [])
  have semanticEq := congrArg
    (fun state : State Digest Unit FPrimeFullHistoryBaseStepSound.Fresh Unit =>
      state.semanticState) nextState
  simpa [proof, FPrimeFullHistoryBaseStepSound.stateOfValues] using
    congrArg
      (fun digest =>
        ({ fold := Step.FoldProof.noFold
           nebulaOpen := (none : Option Unit)
           semanticStateDigest := digest
           xOut := FPrimeFullHistoryBaseStepSound.xOutDigestValue } :
          Step.Proof Digest NifsProof Unit)) semanticEq

/-- Exact base-owner rows imply the composable M3 local relation for any
concrete accumulator/fresh semantics satisfying the primitive base laws. -/
theorem local_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (laws : BaseLaws semantics fresh) :
    Step.LocalHolds FPrimeFullHistoryBaseStepSound.hashSemantics semantics
      .stateless FPrimeFullHistoryBaseStepSound.context
      (decodedPrior assignment) (decodedNext assignment semantics fresh)
      (input fresh) (decodedProof assignment) := by
  have facts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical one
    satisfies
  rw [decodedPrior_eq facts, decodedNext_eq facts semantics fresh,
    decodedProof_eq facts]
  have base := profile_baseLocal semantics fresh laws
  simpa [Step.LocalHolds, prior, proof,
    FPrimeFullHistoryBaseStepSound.stateOfValues] using base

/-- The only fresh-claim view needed by the delayed public link.  A concrete
claim type may carry arbitrarily more authority; this projection does not
replace that payload. -/
structure FreshLinkLaws
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (publicXOut : Fresh → Digest) : Prop where
  freshLink_eq : ∀ digest fresh,
    semantics.freshLink digest fresh = decide (digest = publicXOut fresh)

/-- Exact consumer rows close the base producer's delayed link for the same
full fresh-claim type that the recursive NIFS verifier consumes. -/
theorem outgoing_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (publicXOut : Fresh → Digest)
    (linkLaws : FreshLinkLaws semantics publicXOut)
    (freshPublic :
      publicXOut fresh =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut) :
    Step.OutgoingLinked semantics (input fresh) (decodedProof assignment) := by
  have baseFacts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical
    one baseSatisfies
  have priorFacts := FPrimeFullHistoryPriorLinkSound.sound goldilocksPrime
    canonical one priorLinkSatisfies
  have links := FPrimeFullHistoryStateLinkSound.sound canonical one
    stateLinkSatisfies
  have priorDigest := FPrimeFullHistoryBaseOutgoingSound.priorDigest_eq_baseXOut
    baseFacts priorFacts links
  have decodedFreshDigest :=
    FPrimeFullHistoryPriorLinkSound.decodedFresh_digest priorFacts
  have proofEq := decodedProof_eq (NifsProof := NifsProof) baseFacts
  have proofDigest :
      (decodedProof assignment : Step.Proof Digest NifsProof Unit).xOut =
        FPrimeFullHistoryBaseStepSound.xOutDigestValue := by
    rw [proofEq]
    rfl
  have linkedDigest :
      (decodedProof assignment : Step.Proof Digest NifsProof Unit).xOut =
        publicXOut fresh :=
    proofDigest.trans
      (priorDigest.symm.trans (decodedFreshDigest.symm.trans freshPublic.symm))
  simp [Step.OutgoingLinked, Step.FreshLinked, input,
    linkLaws.freshLink_eq, linkedDigest]

/-- First exact closed edge, now parameterized by the real accumulator and
fresh-claim carriers instead of the non-composable unit surrogate. -/
theorem step_holds
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (priorLinkSatisfies :
      Satisfies FPrimeFullHistoryPriorLink.rows assignment)
    (semantics : Step.Semantics Digest Running Fresh NifsProof Unit Unit)
    (fresh : Fresh) (publicXOut : Fresh → Digest)
    (baseLaws : BaseLaws semantics fresh)
    (linkLaws : FreshLinkLaws semantics publicXOut)
    (freshPublic :
      publicXOut fresh =
        (FPrimeFullHistoryPriorLinkSound.decodedFresh assignment).publicXOut) :
    Step.Holds FPrimeFullHistoryBaseStepSound.hashSemantics semantics
      .stateless FPrimeFullHistoryBaseStepSound.context
      (decodedPrior assignment) (decodedNext assignment semantics fresh)
      (input fresh) (decodedProof assignment) := by
  rw [Step.holds_iff_local_and_outgoing]
  exact ⟨
    local_sound goldilocksPrime canonical one baseSatisfies semantics fresh
      baseLaws,
    outgoing_sound goldilocksPrime canonical one baseSatisfies
      stateLinkSatisfies priorLinkSatisfies semantics fresh publicXOut
      linkLaws freshPublic⟩

end

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound
