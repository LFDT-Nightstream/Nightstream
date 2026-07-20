import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Context

/-!
Production-only carrier for one pending delayed packed-`yZcol` value.

Protocol: fixed-one F-prime over the concrete Phi81 NIFS.
Phase: public carrier construction before the combined NC check.
Constraint family: typed carrier only; this file emits no rows.

Assurance tier: model-level.

Owns: a narrow wrapper around the existing payload-minimal fixed-one input;
installation of one complete optional `ProductionDelayedBlockLane` into the
canonical NIFS context; exact projection to `FixedActive.Context`; and
definitional agreement with the existing fixed-one context when no delayed
value is pending.

Does not own: physical acceptance, transcript sampling, raw assignment
decoding, child authority, one-fold continuity, base or terminal closure,
commitment binding, Rust/R1CS refinement, generated rows, costs, or row
removal. In particular, merely carrying `pending` proves no projection fact.

Emits constraints: none.

No raw `Sources.Data` or certificate wrapper is defined here. Raw data
presence is not authority; a later production checker must derive its
semantic meaning from concrete decoder and acceptance evidence.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_one.delayed.carrier.input` | retain the exact existing fixed-one payload and one optional complete delayed value | direct dataflow | `Input` |
| `nifs.fixed_one.delayed.carrier.canonical` | change only the canonical context's `pending` field | computed | `canonical` |
| `nifs.fixed_one.delayed.carrier.full` | materialize the exact fixed-active context with the supplied pending value | computed | `full` |
| `nifs.fixed_one.delayed.carrier.none` | recover the existing generic fixed-one context when no value is pending | exact model theorem | `canonical_eq_nifsContext_of_pending_eq_none`, `full_eq_nifsContext_of_pending_eq_none` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionContext

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator

universe uOuterKey uAppState uWitness uTranscriptState

/-- Production input for the delayed path. The fixed-one payload is unchanged;
the only additional public datum is the complete optional pending value. -/
structure Input
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  fixedOne :
    ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
      publicRingColumns publicFits verifierRows
  pending : Option ProductionDelayedBlockLane

namespace Input

/-- Explicit constructor used by the production wrapper. This constructor
does not claim that `pending` is true or authoritative. -/
def withPending
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fixedOne :
      ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (pending : Option ProductionDelayedBlockLane) :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows :=
  { fixedOne := fixedOne, pending := pending }

@[simp] theorem withPending_fixedOne
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fixedOne :
      ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (pending : Option ProductionDelayedBlockLane) :
    (withPending fixedOne pending).fixedOne = fixedOne := by
  rfl

@[simp] theorem withPending_pending
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fixedOne :
      ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (pending : Option ProductionDelayedBlockLane) :
    (withPending fixedOne pending).pending = pending := by
  rfl

end Input

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Canonical production context. Definitionally, this is the existing
fixed-one context with only its `pending` field replaced. -/
def canonical
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    FixedActive.Canonical.Context shape TranscriptState
      publicRingColumns publicFits verifierRows :=
  { FixedOneCanonical.nifsContext setup input.fixedOne with
    pending := input.pending }

/-- Exact independent fixed-active context reached by the production carrier. -/
def full
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    FixedActive.Context shape TranscriptState
      publicRingColumns publicFits verifierRows :=
  (canonical setup input).materialize

/-! ## Canonical-context projections -/

@[simp] theorem canonical_covers
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).covers =
      (FixedOneCanonical.nifsContext setup input.fixedOne).covers := by
  rfl

@[simp] theorem canonical_key
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).key =
      (FixedOneCanonical.nifsContext setup input.fixedOne).key := by
  rfl

@[simp] theorem canonical_alignment
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).alignment =
      (FixedOneCanonical.nifsContext setup input.fixedOne).alignment := by
  rfl

@[simp] theorem canonical_input
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).input =
      (FixedOneCanonical.nifsContext setup input.fixedOne).input := by
  rfl

@[simp] theorem canonical_pending
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).pending = input.pending := by
  rfl

@[simp] theorem canonical_piCcsInput
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).piCcsInput =
      (FixedOneCanonical.nifsContext setup input.fixedOne).piCcsInput := by
  rfl

@[simp] theorem canonical_priorState
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).priorState =
      (FixedOneCanonical.nifsContext setup input.fixedOne).priorState := by
  rfl

@[simp] theorem canonical_piCcsSchedule
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).piCcsSchedule =
      (FixedOneCanonical.nifsContext setup input.fixedOne).piCcsSchedule := by
  rfl

@[simp] theorem canonical_piRlcMachine
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).piRlcMachine =
      (FixedOneCanonical.nifsContext setup input.fixedOne).piRlcMachine := by
  rfl

@[simp] theorem canonical_profile
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).profile =
      (FixedOneCanonical.nifsContext setup input.fixedOne).profile := by
  rfl

@[simp] theorem canonical_challengeSetSize
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (canonical setup input).challengeSetSize =
      (FixedOneCanonical.nifsContext setup input.fixedOne).challengeSetSize := by
  rfl

/-! ## Materialized fixed-active projections -/

@[simp] theorem full_covers
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).covers =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.covers := by
  rfl

@[simp] theorem full_key
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).key =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.key := by
  rfl

@[simp] theorem full_alignment
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).alignment =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.alignment := by
  rfl

@[simp] theorem full_input
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).input =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.input := by
  rfl

@[simp] theorem full_runningParent
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).runningParent =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.runningParent := by
  rfl

@[simp] theorem full_pending
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).pending = input.pending := by
  rfl

@[simp] theorem full_piCcsInput
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).piCcsInput =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.piCcsInput := by
  rfl

@[simp] theorem full_priorState
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).priorState =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.priorState := by
  rfl

@[simp] theorem full_piCcsSchedule
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).piCcsSchedule =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.piCcsSchedule := by
  rfl

@[simp] theorem full_piRlcMachine
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).piRlcMachine =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.piRlcMachine := by
  rfl

@[simp] theorem full_profile
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).profile =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.profile := by
  rfl

@[simp] theorem full_challengeSetSize
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (full setup input).challengeSetSize =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize.challengeSetSize := by
  rfl

/-! ## Exact recovery of the generic fixed-one carrier -/

/-- If no delayed value is pending, the canonical production context is
definitionally the existing generic fixed-one context. -/
theorem canonical_eq_nifsContext_of_pending_eq_none
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (pendingNone : input.pending = none) :
    canonical setup input =
      FixedOneCanonical.nifsContext setup input.fixedOne := by
  cases input with
  | mk fixedOne pending =>
      simp only at pendingNone
      subst pending
      rfl

/-- If no delayed value is pending, materialization recovers the exact
existing independent fixed-active context. -/
theorem full_eq_nifsContext_of_pending_eq_none
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (pendingNone : input.pending = none) :
    full setup input =
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize := by
  rw [full, canonical_eq_nifsContext_of_pending_eq_none setup input
    pendingNone]

@[simp] theorem canonical_withPending_none
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows) :
    canonical setup (Input.withPending input none) =
      FixedOneCanonical.nifsContext setup input := by
  rfl

@[simp] theorem full_withPending_none
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ActiveEvaluator.FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows) :
    full setup (Input.withPending input none) =
      (FixedOneCanonical.nifsContext setup input).materialize := by
  rfl

end

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionContext
