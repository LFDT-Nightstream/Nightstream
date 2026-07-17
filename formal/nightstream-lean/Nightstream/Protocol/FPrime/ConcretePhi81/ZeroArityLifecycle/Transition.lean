import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Context

/-!
Transitions and reachability for the concrete zero-running F-prime lifecycle.

Protocol: candidate F-prime scheduling over the concrete Phi81 NIFS.
Phase: base deposit, contextual zero-running fold, and steady recursive folds.
Constraint family: model-level transition semantics only; this file emits no
rows.

Owns: the assumed outer obligation on each deposited claim; three indexed
transition arms; provenance of each produced NIFS result; validity evidence
for raw lifecycle states; reachability; and closure of the running phase.

Does not own: the concrete application/control predicate instantiating the
outgoing obligation, proof that `Setup` is the production transcript, paper
or HyperNova refinement, an executable verifier, Rust/R1CS refinement, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: a raw `State` is not authority. `StateValid` pairs a
stored latest claim with its assumed outer obligation and a running payload
with the exact model-level NIFS transition that produced it. The two claim
arguments occupy separate delayed roles but are not assumed unequal.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.zero_arity.branch.base` | deposit the first claim without invoking NIFS | model-level premise | `Transition.base` |
| `fprime.zero_arity.branch.bootstrap` | fold the prior claim through exact `1 + 0` NIFS semantics | model-level semantic premise | `Transition.bootstrap` |
| `fprime.zero_arity.branch.recursive` | fold the prior claim plus the complete active payload | model-level semantic premise | `Transition.recursive` |
| `fprime.zero_arity.outgoing` | retain the explicit outer obligation for the newly deposited claim | assumed semantic premise | `NextClaimObligation` |
| `fprime.zero_arity.result.provenance` | bind each running payload to one bootstrap or active NIFS result transition | derived | `NifsOutputRealized`, `Transition.output_realized` |
| `fprime.zero_arity.state.valid` | pair raw phase data with claim and NIFS provenance evidence | derived | `StateValid`, `Transition.produces_valid`, `Reachable.valid_from_initial` |
| `fprime.zero_arity.running.closed` | every state reachable from any running state remains running | derived | `Reachable.from_running_is_running` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

universe uTranscriptState

/-- Assumed outer semantic obligation for the claim deposited by the current
step and folded only by the following step.

This predicate is not itself authentication and may not be instantiated with
`fun _ => True` by a later production refinement. -/
abbrev NextClaimObligation
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Fresh shape publicRingColumns publicFits verifierRows -> Prop

/-- Exact three-phase model-level lifecycle.

`nextLatest` is a separate delayed role from the `currentLatest` stored in the
input state. The values may be equal; no inequality is claimed. -/
inductive Transition
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows) :
    State shape publicRingColumns publicFits verifierRows ->
      Fresh shape publicRingColumns publicFits verifierRows ->
        State shape publicRingColumns publicFits verifierRows -> Prop where
  | base
      {nextLatest : Fresh shape publicRingColumns publicFits verifierRows}
      (nextValid : nextClaimObligation nextLatest) :
      Transition setup nextClaimObligation .initial nextLatest
        (.primed nextLatest)
  | bootstrap
      {currentLatest nextLatest :
        Fresh shape publicRingColumns publicFits verifierRows}
      {result :
        Accumulator shape publicRingColumns publicFits verifierRows}
      (nextValid : nextClaimObligation nextLatest)
      (fold :
        FixedBootstrap.ResultTransition
          (bootstrapContext setup currentLatest nextLatest) result) :
      Transition setup nextClaimObligation (.primed currentLatest) nextLatest
        (.running result nextLatest)
  | recursive
      {accumulator result :
        Accumulator shape publicRingColumns publicFits verifierRows}
      {currentLatest nextLatest :
        Fresh shape publicRingColumns publicFits verifierRows}
      (nextValid : nextClaimObligation nextLatest)
      (fold :
        FixedActive.ResultTransition
          (activeContext setup accumulator currentLatest nextLatest) result) :
      Transition setup nextClaimObligation
        (.running accumulator currentLatest) nextLatest
        (.running result nextLatest)

/-- Model-level NIFS provenance for a produced result payload. -/
def NifsOutputRealized
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (result : Accumulator shape publicRingColumns publicFits verifierRows) :
    Prop :=
  (exists currentLatest nextLatest,
    FixedBootstrap.ResultTransition
      (bootstrapContext setup currentLatest nextLatest) result) ∨
  (exists accumulator currentLatest nextLatest,
    FixedActive.ResultTransition
      (activeContext setup accumulator currentLatest nextLatest) result)

/-- Evidence required before a raw lifecycle state may be treated as the
output of this model-level transition system. -/
def StateValid
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows) :
    State shape publicRingColumns publicFits verifierRows -> Prop
  | .initial => True
  | .primed latest => nextClaimObligation latest
  | .running result latest =>
      NifsOutputRealized setup result ∧ nextClaimObligation latest

namespace Transition

/-- Every transition that enters or remains in `running` exposes the exact
model-level NIFS transition that produced its result payload. -/
theorem output_realized
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows}
    {before : State shape publicRingColumns publicFits verifierRows}
    {nextLatest : Fresh shape publicRingColumns publicFits verifierRows}
    {result : Accumulator shape publicRingColumns publicFits verifierRows}
    (transition :
      Transition setup nextClaimObligation before nextLatest
        (.running result nextLatest)) :
    NifsOutputRealized setup result := by
  cases transition with
  | bootstrap _ fold => exact Or.inl ⟨_, _, fold⟩
  | recursive _ fold => exact Or.inr ⟨_, _, _, fold⟩

/-- Every accepted arm produces a state with the explicit evidence required
by `StateValid`. -/
theorem produces_valid
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows}
    {before after : State shape publicRingColumns publicFits verifierRows}
    {nextLatest : Fresh shape publicRingColumns publicFits verifierRows}
    (transition :
      Transition setup nextClaimObligation before nextLatest after) :
    StateValid setup nextClaimObligation after := by
  cases transition with
  | base nextValid => exact nextValid
  | bootstrap nextValid fold =>
      exact ⟨Or.inl ⟨_, _, fold⟩, nextValid⟩
  | recursive nextValid fold =>
      exact ⟨Or.inr ⟨_, _, _, fold⟩, nextValid⟩

end Transition

/-- Reflexive-transitive closure of the exact lifecycle transition. -/
inductive Reachable
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows) :
    State shape publicRingColumns publicFits verifierRows ->
      State shape publicRingColumns publicFits verifierRows -> Prop where
  | refl (state) : Reachable setup nextClaimObligation state state
  | step
      {start before after :
        State shape publicRingColumns publicFits verifierRows}
      {nextLatest : Fresh shape publicRingColumns publicFits verifierRows}
      (prior : Reachable setup nextClaimObligation start before)
      (transition :
        Transition setup nextClaimObligation before nextLatest after) :
      Reachable setup nextClaimObligation start after

namespace Reachable

/-- Any running state reachable from `initial` exposes model-level NIFS
provenance for its current result payload. -/
theorem running_realized
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows}
    {result : Accumulator shape publicRingColumns publicFits verifierRows}
    {currentLatest : Fresh shape publicRingColumns publicFits verifierRows}
    (reachable :
      Reachable setup nextClaimObligation .initial
        (.running result currentLatest)) :
    NifsOutputRealized setup result := by
  cases reachable with
  | step _ transition =>
      cases transition with
      | bootstrap _ fold => exact Or.inl ⟨_, _, fold⟩
      | recursive _ fold => exact Or.inr ⟨_, _, _, fold⟩

/-- Every state reachable from `initial` carries the evidence described by
`StateValid`. -/
theorem valid_from_initial
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows}
    {state : State shape publicRingColumns publicFits verifierRows}
    (reachable :
      Reachable setup nextClaimObligation .initial state) :
    StateValid setup nextClaimObligation state := by
  cases reachable with
  | refl => exact True.intro
  | step _ transition => exact transition.produces_valid

/-- Every state reachable from any raw running state remains running. This
structural fact prevents a later transition from selecting bootstrap again;
it does not assert provenance for the starting payload. -/
theorem from_running_is_running
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {nextClaimObligation :
      NextClaimObligation shape publicRingColumns publicFits verifierRows}
    {initialAccumulator :
      Accumulator shape publicRingColumns publicFits verifierRows}
    {initialLatest : Fresh shape publicRingColumns publicFits verifierRows}
    {state : State shape publicRingColumns publicFits verifierRows}
    (reachable :
      Reachable setup nextClaimObligation
        (.running initialAccumulator initialLatest) state) :
    exists accumulator latest, state = .running accumulator latest := by
  induction reachable with
  | refl => exact ⟨_, _, rfl⟩
  | step _ transition ih =>
      rcases ih with ⟨accumulator, latest, rfl⟩
      cases transition with
      | recursive _ _ => exact ⟨_, _, rfl⟩

end Reachable

end Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle
