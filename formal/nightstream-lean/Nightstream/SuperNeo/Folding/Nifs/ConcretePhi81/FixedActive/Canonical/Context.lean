import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Canonical fixed-active public context for the concrete Phi81 NIFS.

Protocol: fixed SuperNeo NIFS `CE^k x CCS -> CE^k`.
Phase: verifier-owned setup plus one fresh and fourteen running public payloads
to the fixed production-domain transition context.
Constraint family: carrier construction only; this file emits no rows.

Owns: minimal public payloads for fresh/running/parent statements; installation
of one verifier-owned relation structure; canonical fresh/combined stages;
construction of the exact fixed-active source product and fixed 9/3/6
production-domain context; and proof that outgoing `Pi_RLC` source-structure
consistency is true by construction.

Does not own: semantic openings, incoming-parent recomposition, `Pi_CCS`
messages, challenge sampling, outgoing `Pi_DEC`, Rust/R1CS decoding, physical
rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: callers supply no relation structure or norm stage for an
individual source. `Context.materialize` installs `input.system` into every
fresh/running/parent statement, fixes source stages to `.fresh`, and fixes the
incoming checked-parent stage to `.combined`. A production implementation may
eliminate the corresponding comparisons only after its decoder is proved to
construct exactly this carrier.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.context.system` | one verifier-owned relation structure is shared by all statements | computed | `Input.materialize`, `Context.materialize` |
| `nifs.fixed_active.context.fresh_stage` | fresh CCS and all running CE sources are at `.fresh` | computed | payload `materialize` functions |
| `nifs.fixed_active.context.parent_stage` | incoming checked parent is at `.combined` | computed | `ParentPayload.materialize` |
| `nifs.fixed_active.context.parent_presence` | active context always carries the complete parent, never only a digest | computed | `Context.materialize` |
| `nifs.fixed_active.context.domain` | FE row and NC block/lane dimensions are the fixed production profile | computed | `PiCcsDomains.production` |
| `nifs.fixed_active.pi_rlc.source_structure` | every canonical source uses the verifier-owned system | derived/eliminated | `sourceStructures` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Fresh CCS payload after relation structure and norm stage are made
verifier-owned. -/
structure FreshPayload
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  commitment : CommitmentValue verifierRows
  publicInput :
    Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits)

namespace FreshPayload

def materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      FreshPayload shape publicRingColumns publicFits verifierRows)
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits)) :
    Phi81Relation.CCSStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) where
  constraintSystem := system
  commitment := payload.commitment
  publicInput := payload.publicInput
  stage := .fresh

end FreshPayload

/-- Running CE payload after relation structure and fresh stage are made
verifier-owned. -/
structure RunningPayload
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  commitment : CommitmentValue verifierRows
  publicInput :
    Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits)
  point :
    Phi81Relation.Point
      (RelationShape shape publicRingColumns publicFits)
  evaluations : Array Phi81Relation.Evaluation

namespace RunningPayload

def materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      RunningPayload shape publicRingColumns publicFits verifierRows)
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits)) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) where
  constraintSystem := system
  commitment := payload.commitment
  publicInput := payload.publicInput
  point := payload.point
  evaluations := payload.evaluations
  stage := .fresh

end RunningPayload

/-- Incoming checked-parent payload after relation structure and combined
stage are made verifier-owned. -/
structure ParentPayload
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  commitment : CommitmentValue verifierRows
  publicInput :
    Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits)
  point :
    Phi81Relation.Point
      (RelationShape shape publicRingColumns publicFits)
  evaluations : Array Phi81Relation.Evaluation

namespace ParentPayload

def materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      ParentPayload shape publicRingColumns publicFits verifierRows)
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits)) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) where
  constraintSystem := system
  commitment := payload.commitment
  publicInput := payload.publicInput
  point := payload.point
  evaluations := payload.evaluations
  stage := .combined

end ParentPayload

/-- Dynamic public statements in one fixed-active invocation. -/
structure Input
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  system :
    Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits)
  fresh : FreshPayload shape publicRingColumns publicFits verifierRows
  running : Fin productionGlobalParams.k ->
    RunningPayload shape publicRingColumns publicFits verifierRows
  parent : ParentPayload shape publicRingColumns publicFits verifierRows

namespace Input

/-- Install the sole structure and canonical stages into the exact one-plus-k
source product. -/
def materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input : Input shape publicRingColumns publicFits verifierRows) :
    SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity where
  fresh := fun _ => input.fresh.materialize input.system
  running := fun child => (input.running child).materialize input.system

end Input

/-- Complete fixed-active verifier context whose dynamic statement fields use
the canonical payload carrier. -/
structure Context
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  covers : PiCcsDomains.production.nc.Covers shape
  key : VerifierKey shape publicRingColumns publicFits verifierRows
  alignment : SourceAlignment shape productionGlobalParams arity
  input : Input shape publicRingColumns publicFits verifierRows
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : State
  piCcsSchedule :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Schedule
      (VerifierKey shape publicRingColumns publicFits verifierRows)
      (StatementInput shape publicRingColumns publicFits verifierRows arity)
      shape PiCcsDomains.production State
  piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine State
  profile : PiCCS.SplitNc.Verifier.Polynomial.Fe.SupportedProfile shape
    PiCcsDomains.production.fe
  challengeSetSize : Nat

namespace Context

/-- Project the canonical carrier into the existing independent transition
context. -/
def materialize
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :
    FixedActive.Context
      shape State publicRingColumns publicFits verifierRows where
  covers := context.covers
  key := context.key
  alignment := context.alignment
  input := context.input.materialize
  runningParent := some (context.input.parent.materialize context.input.system)
  piCcsInput := context.piCcsInput
  priorState := context.priorState
  piCcsSchedule := context.piCcsSchedule
  piRlcMachine := context.piRlcMachine
  profile := context.profile
  challengeSetSize := context.challengeSetSize

@[simp] theorem materialize_system
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :
    context.materialize.system = context.input.system := by
  rfl

/-- The formerly retained source-structure family is eliminated on the
canonical carrier: every source receives the same verifier-owned system by
construction. -/
theorem sourceStructures
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :
    DerivedPiRlc.SourceStructuresBound context.materialize := by
  refine PiCCS.InputProduct.sourceCases context.materialize.input
    (fun source =>
      source.constraintSystem = context.materialize.system) ?_ ?_
  · intro fresh
    simp [PiCCS.Source.constraintSystem, Context.materialize,
      Input.materialize, FreshPayload.materialize]
  · intro running
    simp [PiCCS.Source.constraintSystem, Context.materialize,
      Input.materialize, FreshPayload.materialize,
      RunningPayload.materialize]

end Context

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical
