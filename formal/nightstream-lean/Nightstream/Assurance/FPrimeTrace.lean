import Nightstream.Protocol.FPrime.Step
import Nightstream.Assurance.ValidExecution

/-!
Contract: exact trace validity for the executable direct-F' model.

An `AcceptedTrace` retains every verifier input and proof, not merely a chain of
digests. `accepted_trace_sound` converts those Boolean acceptances into exact
semantic reachability, proves the split counters equal the invocation schedule,
and preserves all verifier-pinned state authority at the final nonempty state.

This is model-level assurance. Circuit soundness and Rust refinement remain the
separate M4 and M5 obligations.
-/

namespace Nightstream.Assurance.FPrimeTrace

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

/-- Fixed verifier-owned semantics for one F' trace. -/
structure Environment
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest)
    (NebulaOpen : Type uNebulaOpen) where
  hashSemantics : XOut.Semantics
    Params StructureDigest Header Digest Nebula NebulaDigest
  stepSemantics : Step.Semantics
    Digest Running Fresh NifsProof Nebula NebulaOpen
  mode : XOut.Mode
  context : XOut.Context Params StructureDigest Header Digest

/-- Data consumed by one F' invocation. -/
structure Invocation
    (Digest : Type uDigest)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaOpen : Type uNebulaOpen) where
  input : Step.Input Fresh Nebula NebulaOpen
  proof : Step.Proof Digest NifsProof NebulaOpen
deriving Repr, DecidableEq

/-- One rich semantic edge, with the accepting witness retained. -/
def Edge
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
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (prior next : State Digest Running Fresh Nebula) : Prop :=
  ∃ invocation : Invocation Digest Fresh NifsProof Nebula NebulaOpen,
    Step.Holds environment.hashSemantics environment.stepSemantics
      environment.mode environment.context prior next invocation.input invocation.proof

/-- Trace index: each entry is the cardinality of the installed fresh batch. -/
inductive AcceptedTrace
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial : State Digest Running Fresh Nebula) :
    List Nat → State Digest Running Fresh Nebula → Prop where
  | nil : AcceptedTrace environment initial [] initial
  | snoc
      {schedule : List Nat}
      {prior next : State Digest Running Fresh Nebula}
      (tracePrefix : AcceptedTrace environment initial schedule prior)
      (invocation : Invocation Digest Fresh NifsProof Nebula NebulaOpen)
      (accepted :
        Step.check environment.hashSemantics environment.stepSemantics
          environment.mode environment.context prior next
          invocation.input invocation.proof = true) :
      AcceptedTrace environment initial
        (schedule ++ [invocation.input.nextLatest.length]) next

/-- Exact interpretation of the two Rust counters over a trace schedule. -/
def CounterRefines
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    (schedule : List Nat)
    (state : State Digest Running Fresh Nebula) : Prop :=
  0 ∉ schedule ∧
  state.chunkCount = schedule.length ∧
  state.stepCount = schedule.sum

/-- The result established by `TRACE-VALID`. -/
structure SoundResult
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
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat) : Prop where
  exactReachability :
    Nightstream.Assurance.Reachable (Edge environment) initial schedule.length final
  counterRefinement : CounterRefines schedule final
  finalPinned : schedule ≠ [] →
    XOut.StatePinned environment.hashSemantics environment.mode
      environment.context final

theorem accepted_edge_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (prior next : State Digest Running Fresh Nebula)
    (invocation : Invocation Digest Fresh NifsProof Nebula NebulaOpen)
    (accepted :
      Step.check environment.hashSemantics environment.stepSemantics
        environment.mode environment.context prior next
        invocation.input invocation.proof = true) :
    Edge environment prior next := by
  refine ⟨invocation, ?_⟩
  exact Step.check_sound environment.hashSemantics environment.stepSemantics
    environment.mode environment.context prior next invocation.input
    invocation.proof accepted

/-- Every accepted trace is reachable by exactly its number of invocations. -/
theorem accepted_trace_reachable
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat)
    (trace : AcceptedTrace environment initial schedule final) :
    Nightstream.Assurance.Reachable
      (Edge environment) initial schedule.length final := by
  induction trace with
  | nil => exact .zero
  | snoc tracePrefix invocation accepted inductionHypothesis =>
      simpa using Nightstream.Assurance.Reachable.succ inductionHypothesis
        (accepted_edge_sound environment _ _ invocation accepted)

/-- The schedule cardinalities are nonzero and exactly refine both counters. -/
theorem accepted_trace_counter_refines
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat)
    (initialValid :
      Step.InitialState environment.hashSemantics environment.stepSemantics
        environment.mode environment.context initial)
    (trace : AcceptedTrace environment initial schedule final) :
    CounterRefines schedule final := by
  induction trace with
  | nil =>
      exact ⟨by simp, initialValid.2.1, initialValid.2.2.1⟩
  | snoc tracePrefix invocation accepted inductionHypothesis =>
      have holds := Step.check_sound environment.hashSemantics
        environment.stepSemantics environment.mode environment.context _ _
        invocation.input invocation.proof accepted
      have facts := Step.holds_advance_facts environment.hashSemantics
        environment.stepSemantics environment.mode environment.context _ _
        invocation.input invocation.proof holds
      rcases inductionHypothesis with
        ⟨priorScheduleValid, priorChunks, priorSteps⟩
      have batchNonzero : invocation.input.nextLatest.length ≠ 0 := by
        intro lengthZero
        apply facts.freshNonempty
        exact List.eq_nil_of_length_eq_zero lengthZero
      refine ⟨?_, ?_, ?_⟩
      · intro zeroMember
        simp only [List.mem_append, List.mem_singleton] at zeroMember
        rcases zeroMember with oldMember | newIsZero
        · exact priorScheduleValid oldMember
        · exact batchNonzero newIsZero.symm
      · calc
          _ = _ + 1 := facts.chunkCount
          _ = _ := by rw [priorChunks]; simp
      · calc
          _ = _ + invocation.input.nextLatest.length := facts.stepCount
          _ = _ := by rw [priorSteps]; simp

/-- Every nonempty accepted trace ends in a fully pinned state. -/
theorem accepted_trace_final_pinned
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat)
    (trace : AcceptedTrace environment initial schedule final)
    (nonempty : schedule ≠ []) :
    XOut.StatePinned environment.hashSemantics environment.mode
      environment.context final := by
  cases trace with
  | nil => exact False.elim (nonempty rfl)
  | snoc tracePrefix invocation accepted =>
      have holds := Step.check_sound environment.hashSemantics
        environment.stepSemantics environment.mode environment.context _ _
        invocation.input invocation.proof accepted
      exact Step.next_state_pinned environment.hashSemantics
        environment.stepSemantics environment.mode environment.context _ _
        invocation.input invocation.proof holds

/-- `TRACE-VALID`: accepted F' processing establishes exact model validity. -/
theorem accepted_trace_sound
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat)
    (initialValid :
      Step.InitialState environment.hashSemantics environment.stepSemantics
        environment.mode environment.context initial)
    (trace : AcceptedTrace environment initial schedule final) :
    SoundResult environment initial final schedule where
  exactReachability := accepted_trace_reachable environment initial final schedule trace
  counterRefinement := accepted_trace_counter_refines environment initial final
    schedule initialValid trace
  finalPinned := accepted_trace_final_pinned environment initial final schedule trace

/-- Bridge from `TRACE-VALID` to the top-level `ValidExecution` target. -/
theorem accepted_trace_valid_execution
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
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (environment : Environment Params StructureDigest Header Digest Running Fresh
      NifsProof Nebula NebulaDigest NebulaOpen)
    (initial final : State Digest Running Fresh Nebula)
    (schedule : List Nat)
    (initialValid :
      Step.InitialState environment.hashSemantics environment.stepSemantics
        environment.mode environment.context initial)
    (trace : AcceptedTrace environment initial schedule final)
    (TerminalValid : State Digest Running Fresh Nebula → Prop)
    (terminal : TerminalValid final) :
    Nightstream.Assurance.ValidExecution (Edge environment) TerminalValid
      initial final schedule.length :=
  ⟨(accepted_trace_sound environment initial final schedule initialValid trace).exactReachability,
    terminal⟩

end Nightstream.Assurance.FPrimeTrace
