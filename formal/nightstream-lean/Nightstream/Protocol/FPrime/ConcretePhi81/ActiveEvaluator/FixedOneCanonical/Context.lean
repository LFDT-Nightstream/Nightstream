import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context

/-!
Payload-minimal fixed-one F-prime carrier and canonical NIFS context.

Protocol: HyperNova Construction 2 over the concrete SuperNeo NIFS.
Phase: fixed-one active F-prime input to the sole selected NIFS invocation.
Constraint family: carrier construction only; this file emits no rows.

Owns: a public input with no selected slot, raw prior counter, relation
structure, or norm stages; canonical reconstruction of the incoming parent,
fourteen running children, fresh statement, active input, and exact NIFS
context; and equality with the independent active-semantics context.

Does not own: prior-link hashing, application dispatch, raw NIFS messages,
physical checking, semantic openings, output truth, Rust/R1CS decoding, rows,
costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Input` carries only finite statement payloads. The sole
slot, one-based counter, expected relation structure, source/parent stages,
and parent presence are installed from verifier setup. Production may omit
their old comparisons only after its decoder is proved to construct this
carrier exactly.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.carrier.slot` | the sole selected slot is zero | computed | `FixedOneCanonical.selected` |
| `fprime.fixed_one.carrier.prior_pc` | the active prior counter is exactly one | computed | `Input.toActive` |
| `fprime.fixed_one.carrier.system` | setup installs the sole selected relation structure everywhere | computed | `Input.system`, `nifsContext` |
| `fprime.fixed_one.carrier.stages` | fresh/running sources are fresh and incoming parent is combined | computed | payload materialization |
| `fprime.fixed_one.carrier.parent` | the complete incoming parent is always present | computed | `nifsContext` |
| `fprime.fixed_one.context.exact` | canonical NIFS context equals the independent active-semantics context | exact model theorem | `nifsContext_materialize` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uTranscriptState

/-- Fixed-one F-prime input after every structure/stage/selection field is
made verifier owned. -/
structure Input
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  verifierKey : OuterKey
  iteration : Nat
  z0 : AppState
  zi : AppState
  parent :
    FixedActive.Canonical.ParentPayload
      shape publicRingColumns publicFits verifierRows
  running : Fin productionGlobalParams.k ->
    FixedActive.Canonical.RunningPayload
      shape publicRingColumns publicFits verifierRows
  fresh :
    FixedActive.Canonical.FreshPayload
      shape publicRingColumns publicFits verifierRows
  witness : Witness

namespace Input

/-- Sole verifier-owned relation structure for this fixed-one invocation. -/
def system
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :=
  setup.expectedStructure input.verifierKey
    ActiveSemantics.FixedOneCanonical.selected

/-- Materialize the complete incoming parent-and-children slot. -/
def slot
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    Slot shape publicRingColumns publicFits verifierRows where
  parent := input.parent.materialize (input.system setup)
  children := fun child =>
    (input.running child).materialize (input.system setup)

/-- Project to the existing payload-reduced semantic carrier. Its remaining
stage field is filled canonically rather than accepted from the caller. -/
def toSemantic
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    ActiveSemantics.FixedOneCanonical.Input OuterKey AppState Witness shape
      publicRingColumns publicFits verifierRows where
  verifierKey := input.verifierKey
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := fun _ => input.slot setup
  fresh := {
    commitment := input.fresh.commitment
    publicInput := input.fresh.publicInput
    stage := .fresh
  }
  witness := input.witness

/-- Complete active-semantics input with all omitted authority fields
reconstructed. -/
def toActive
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
      publicFits verifierRows 1 :=
  (input.toSemantic setup).toActive setup

@[simp] theorem toActive_fresh_stage
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (input.toActive setup).fresh.stage = .fresh := by
  rfl

@[simp] theorem toActive_priorPc
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (input.toActive setup).priorPc = 1 := by
  rfl

end Input

/-- Build the canonical NIFS context directly from verifier setup and the
payload-minimal F-prime input. -/
def nifsContext
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    FixedActive.Canonical.Context shape TranscriptState
      publicRingColumns publicFits verifierRows :=
  let template :=
    setup.template input.verifierKey ActiveSemantics.FixedOneCanonical.selected
  {
    covers := template.covers
    key := template.key
    alignment := template.alignment
    input := {
      system := input.system setup
      fresh := input.fresh
      running := input.running
      parent := input.parent
    }
    piCcsInput :=
      setup.piCcsInput (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected
    priorState :=
      setup.priorTranscriptState
        (input.toActive setup) ActiveSemantics.FixedOneCanonical.selected
    piCcsSchedule := template.piCcsSchedule
    piRlcMachine := template.piRlcMachine
    profile := template.profile
    challengeSetSize := template.challengeSetSize
  }

/-- Canonical payload materialization is exactly the NIFS context used by the
independent fixed-one active semantics. -/
theorem nifsContext_materialize
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    (nifsContext setup input).materialize =
      contextAt setup (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected := by
  rfl

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
