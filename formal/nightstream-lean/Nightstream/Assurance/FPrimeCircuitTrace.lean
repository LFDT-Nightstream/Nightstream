import Nightstream.Assurance.FPrimeTrace

/-!
Contract: composition of split circuit edges into the closed F' trace relation.

Each generated producer contributes `Step.LocalHolds` or a named bad event.
The unique next consumer, or the terminal owner, contributes
`Step.OutgoingLinked`.  This module proves that exact branches form the same
`AcceptedTrace` consumed by TRACE-VALID, while a bad branch remains explicit.

No artifact acceptance is represented by a proof field here.  Row-family
theorems must derive each constructor premise from R1CS satisfaction.
-/

namespace Nightstream.Assurance.FPrimeCircuitTrace

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Assurance.FPrimeTrace

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

section

variable
  {Params : Type uParams}
  {StructureDigest : Type uStructure}
  {Header : Type uHeader}
  {Digest : Type uDigest}
  {Running : Type uRunning}
  {Fresh : Type uFresh}
  {NifsProof : Type uNifsProof}
  {Nebula : Type}
  {NebulaDigest : Type uNebulaDigest}
  {NebulaOpen : Type uNebulaOpen}

local notation "Environment" =>
  FPrimeTrace.Environment Params StructureDigest Header Digest Running Fresh
    NifsProof Nebula NebulaDigest NebulaOpen

local notation "StepState" => State Digest Running Fresh Nebula

local notation "Invocation" =>
  FPrimeTrace.Invocation Digest Fresh NifsProof Nebula NebulaOpen

/-- A decoded circuit history.  Exact edges carry both halves of the split
contract; recursive projection failures carry only their named bad event. -/
inductive CandidateTrace
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (initial : StepState)
    (Bad : StepState → StepState → Invocation → Prop) :
    List Nat → StepState → Prop where
  | nil : CandidateTrace environment initial Bad [] initial
  | snoc
      {schedule : List Nat}
      {prior next : StepState}
      (tracePrefix : CandidateTrace environment initial Bad schedule prior)
      (invocation : Invocation)
      (edge :
        (Step.LocalHolds environment.hashSemantics environment.stepSemantics
            environment.mode environment.context prior next
            invocation.input invocation.proof ∧
          Step.OutgoingLinked environment.stepSemantics invocation.input
            invocation.proof) ∨
        Bad prior next invocation) :
      CandidateTrace environment initial Bad
        (schedule ++ [invocation.input.nextLatest.length]) next

def HasBad
    (Bad : StepState → StepState → Invocation → Prop) : Prop :=
  ∃ prior next invocation, Bad prior next invocation

/-- `CIR-SOUND` composition rule: exact producer/consumer branches form a
closed accepted trace; any inexact recursive branch is returned, not erased. -/
theorem candidate_sound_or_bad
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (initial final : StepState)
    (Bad : StepState → StepState → Invocation → Prop)
    (schedule : List Nat)
    (trace : CandidateTrace environment initial Bad schedule final) :
    FPrimeTrace.AcceptedTrace environment initial schedule final ∨ HasBad Bad := by
  induction trace with
  | nil => exact Or.inl .nil
  | @snoc schedule prior next tracePrefix invocation edge inductionHypothesis =>
      rcases inductionHypothesis with acceptedPrefix | bad
      · rcases edge with exactEdge | badEdge
        · left
          have holds := Step.closeLocal environment.hashSemantics
            environment.stepSemantics environment.mode environment.context
            prior next invocation.input invocation.proof exactEdge.1 exactEdge.2
          have accepted := (Step.check_eq_true_iff_holds
            environment.hashSemantics environment.stepSemantics
            environment.mode environment.context prior next invocation.input
            invocation.proof).2 holds
          exact .snoc acceptedPrefix invocation accepted
        · exact Or.inr ⟨prior, next, invocation, badEdge⟩
      · exact Or.inr bad

/-- `CIR-COMPLETE` composition rule: an accepted closed trace splits into the
standalone producer relation and its unique consumer/terminal link, with no
bad event needed. -/
theorem accepted_to_candidate
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment)
    (initial final : StepState)
    (Bad : StepState → StepState → Invocation → Prop)
    (schedule : List Nat)
    (trace : FPrimeTrace.AcceptedTrace environment initial schedule final) :
    CandidateTrace environment initial Bad schedule final := by
  induction trace with
  | nil => exact .nil
  | @snoc schedule prior next tracePrefix invocation accepted inductionHypothesis =>
      have holds := Step.check_sound environment.hashSemantics
        environment.stepSemantics environment.mode environment.context
        prior next invocation.input invocation.proof accepted
      have split := (Step.holds_iff_local_and_outgoing
        environment.hashSemantics environment.stepSemantics environment.mode
        environment.context prior next invocation.input invocation.proof).1 holds
      exact .snoc inductionHypothesis invocation (Or.inl split)

end

end Nightstream.Assurance.FPrimeCircuitTrace
