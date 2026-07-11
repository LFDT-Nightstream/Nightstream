import Nightstream.Implementation.FPrime.Envelope

/-!
Contract: Construction-2 counter refinement.

Assumes: the initial state satisfies the existing envelope base condition and
every trace edge satisfies `AdvanceCoherent`.

Guarantees: `chunkCount` is the number of F' invocations, `stepCount` is the
sum of their nonempty fresh-batch cardinalities, and both facts are preserved
for an arbitrary finite trace.

Non-goals: Rust representation refinement, counter serialization, NIFS
correctness, application semantics, and surrounding F' R1CS soundness. Rust's
successful `u64` transition agrees with this `Nat` model because
`advance_state` uses checked addition and rejects overflow.
-/

namespace Nightstream.Implementation.FPrime.CounterRefinement

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.FPrime.Envelope

universe uDigest uRunning uFresh

/-- Every Construction-2 invocation installs at least one fresh instance. -/
def ValidSchedule (schedule : List Nat) : Prop :=
  0 ∉ schedule

/-- Rust's two counters refine one schedule of fresh-batch cardinalities. -/
def CounterRefines
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    (schedule : List Nat)
    (state : Carrier Digest Running Fresh) : Prop :=
  ValidSchedule schedule ∧
  state.chunkCount = schedule.length ∧
  state.stepCount = schedule.sum

/-- A finite trace assembled only from the authoritative envelope transition. -/
inductive TraceCoherent
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    (initial : Carrier Digest Running Fresh) :
    List Nat → Carrier Digest Running Fresh → Prop where
  | nil : TraceCoherent initial [] initial
  | snoc
      {schedule : List Nat}
      {prior next : Carrier Digest Running Fresh}
      {freshCount : Nat}
      (tracePrefix : TraceCoherent initial schedule prior)
      (last : AdvanceCoherent freshCount prior next) :
      TraceCoherent initial (schedule ++ [freshCount]) next

/-- The true Construction-2 base state refines the empty schedule. -/
theorem initial_refines
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {state : Carrier Digest Running Fresh}
    (inputCoherent : InputCoherent state)
    (isInitial : state.proof = .initial) :
    CounterRefines [] state := by
  unfold InputCoherent at inputCoherent
  rw [isInitial] at inputCoherent
  rcases inputCoherent with ⟨_, chunkZero, stepZero, _⟩
  exact ⟨by simp [ValidSchedule], by simpa using chunkZero, by simpa using stepZero⟩

/-- `AdvanceCoherent` excludes zero-cardinality trace edges. -/
theorem freshCount_ne_zero_of_advance
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {freshCount : Nat}
    {prior next : Carrier Digest Running Fresh}
    (advance : AdvanceCoherent freshCount prior next) :
    freshCount ≠ 0 := by
  rcases advance with ⟨_, _, _, _, _, _, activeBatch⟩
  cases proofShape : next.proof with
  | initial =>
      simp [proofShape] at activeBatch
  | active running latest =>
      simp [proofShape] at activeBatch
      exact activeBatch.1

/-- One coherent F' edge preserves the counter/schedule refinement. -/
theorem advance_preserves
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {schedule : List Nat}
    {freshCount : Nat}
    {prior next : Carrier Digest Running Fresh}
    (priorRefines : CounterRefines schedule prior)
    (advance : AdvanceCoherent freshCount prior next) :
    CounterRefines (schedule ++ [freshCount]) next := by
  rcases priorRefines with ⟨validPrior, priorChunks, priorSteps⟩
  have freshNonzero := freshCount_ne_zero_of_advance advance
  rcases advance with ⟨nextChunks, nextSteps, _, _, _, _, _⟩
  refine ⟨?_, ?_, ?_⟩
  · intro zeroMembership
    simp only [List.mem_append, List.mem_singleton] at zeroMembership
    rcases zeroMembership with oldMembership | zeroIsFresh
    · exact validPrior oldMembership
    · exact freshNonzero zeroIsFresh.symm
  · calc
      next.chunkCount = prior.chunkCount + 1 := nextChunks
      _ = schedule.length + 1 := by rw [priorChunks]
      _ = (schedule ++ [freshCount]).length := by simp
  · calc
      next.stepCount = prior.stepCount + freshCount := nextSteps
      _ = schedule.sum + freshCount := by rw [priorSteps]
      _ = (schedule ++ [freshCount]).sum := by simp

/-- Repeated coherent F' edges give the exact split-counter interpretation. -/
theorem trace_preserves
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {schedule : List Nat}
    {initial final : Carrier Digest Running Fresh}
    (initialRefines : CounterRefines [] initial)
    (trace : TraceCoherent initial schedule final) :
    CounterRefines schedule final := by
  induction trace with
  | nil => exact initialRefines
  | snoc tracePrefix last inductionHypothesis =>
      exact advance_preserves inductionHypothesis last

/-- Main `FPR-COUNTER-REFINE` theorem. -/
theorem counter_refinement
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {schedule : List Nat}
    {initial final : Carrier Digest Running Fresh}
    (inputCoherent : InputCoherent initial)
    (isInitial : initial.proof = .initial)
    (trace : TraceCoherent initial schedule final) :
    CounterRefines schedule final :=
  trace_preserves (initial_refines inputCoherent isInitial) trace

end Nightstream.Implementation.FPrime.CounterRefinement
