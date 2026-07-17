import Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext
import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.State

/-!
Context construction for the concrete zero-running F-prime lifecycle.

Protocol: candidate F-prime scheduling over the concrete Phi81 NIFS.
Phase: bootstrap and steady recursive context construction.
Constraint family: model-level context wiring only; this file emits no rows.

Owns: one fixed assumed verifier configuration; the delayed distinction
between the claim folded now and the claim deposited for the next step; and
construction of the exact bootstrap and active NIFS contexts.

Does not own: proof that the configuration is the production transcript,
application/control semantics, transition acceptance, state validity,
HyperNova refinement, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: `Setup` is fixed verifier configuration assumed by this
model-level layer. Its provenance from one canonical production transcript
and public-input derivation remains an explicit later refinement obligation.
Invocation witnesses cannot inject a completed context or optional parent.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.zero_arity.branch.bootstrap.nifs.context` | construct the exact one-fresh, zero-running context with absent parent | assumed setup plus computed wiring | `bootstrapContext` |
| `fprime.zero_arity.branch.recursive.nifs.context` | construct the exact one-fresh, fourteen-running context with the stored parent | assumed setup plus computed wiring | `activeContext` |
| `fprime.zero_arity.transcript.delayed_roles` | derive the current context from both the folded and newly deposited claims | assumed setup function | `Setup` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uTranscriptState

/-- Fixed candidate verifier configuration for both recursive phases.

The derivation functions receive both delayed roles: `currentLatest` is folded
now, while `nextLatest` contributes to the current outer F-prime transcript
and is deposited for the following step. Production provenance is not assumed
by this structure. -/
structure Setup
    (shape : SemanticShape)
    (TranscriptState : Type uTranscriptState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  bootstrapTemplate :
    BootstrapContext.Template shape TranscriptState
      publicRingColumns publicFits verifierRows
  activeTemplate :
    Context.Template shape TranscriptState
      publicRingColumns publicFits verifierRows
  bootstrapPiCcsInput :
    Fresh shape publicRingColumns publicFits verifierRows ->
      Fresh shape publicRingColumns publicFits verifierRows ->
        PiCCS.SplitNc.Verifier.PublicInput shape
  bootstrapPriorState :
    Fresh shape publicRingColumns publicFits verifierRows ->
      Fresh shape publicRingColumns publicFits verifierRows -> TranscriptState
  activePiCcsInput :
    Accumulator shape publicRingColumns publicFits verifierRows ->
      Fresh shape publicRingColumns publicFits verifierRows ->
        Fresh shape publicRingColumns publicFits verifierRows ->
          PiCCS.SplitNc.Verifier.PublicInput shape
  activePriorState :
    Accumulator shape publicRingColumns publicFits verifierRows ->
      Fresh shape publicRingColumns publicFits verifierRows ->
        Fresh shape publicRingColumns publicFits verifierRows ->
          TranscriptState

/-- Construct the exact zero-running NIFS context for the first recursive
fold. -/
def bootstrapContext
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (currentLatest nextLatest :
      Fresh shape publicRingColumns publicFits verifierRows) :
    FixedBootstrap.Context shape TranscriptState publicRingColumns
      publicFits verifierRows :=
  setup.bootstrapTemplate.build {
    fresh := currentLatest
    piCcsInput := setup.bootstrapPiCcsInput currentLatest nextLatest
    priorState := setup.bootstrapPriorState currentLatest nextLatest
  }

@[simp] theorem bootstrapContext_runningParent
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (currentLatest nextLatest :
      Fresh shape publicRingColumns publicFits verifierRows) :
    (bootstrapContext setup currentLatest nextLatest).runningParent = none :=
  rfl

/-- Construct the exact fixed-active NIFS context for every later fold. -/
def activeContext
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (accumulator :
      Accumulator shape publicRingColumns publicFits verifierRows)
    (currentLatest nextLatest :
      Fresh shape publicRingColumns publicFits verifierRows) :
    FixedActive.Context shape TranscriptState publicRingColumns
      publicFits verifierRows :=
  setup.activeTemplate.build {
    fresh := currentLatest
    running := accumulator
    piCcsInput := setup.activePiCcsInput accumulator currentLatest nextLatest
    priorState := setup.activePriorState accumulator currentLatest nextLatest
  }

@[simp] theorem activeContext_runningParent
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (accumulator :
      Accumulator shape publicRingColumns publicFits verifierRows)
    (currentLatest nextLatest :
      Fresh shape publicRingColumns publicFits verifierRows) :
    (activeContext setup accumulator currentLatest nextLatest).runningParent =
      some accumulator.parent :=
  rfl

end Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle
